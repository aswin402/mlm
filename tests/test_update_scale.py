import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import serialization

from mlm.config import Config
from mlm.hierarchy_run import make_routed_step, routed_objective
from mlm.model import initial_state
from mlm.training import create_state
from mlm.update_scale import SCALES, make_update_scale_probe


def test_scaled_updates_match_independent_parameter_construction():
    cfg = Config(
        nodes=4,
        width=8,
        groups=2,
        node_update="hierarchical",
        memory="delta",
        memory_keys=4,
        batch_size=2,
        length=8,
    )
    batch = {
        "inputs": jnp.array([[256, 65, 66, 67, 68, 69, 70, 71]] * 2),
        "targets": jnp.array([[65, 66, 67, 68, 69, 70, 71, 257]] * 2),
        "mask": jnp.ones((2, 8)),
        "loss_mask": jnp.array([[0, 0, 0, 0, 1, 1, 1, 1]] * 2),
        "resets": jnp.array([[True] + [False] * 7] * 2),
    }
    aggregate = {**batch, "targets": (batch["targets"] + 23) % 258, "loss_mask": jnp.ones((2, 8))}
    train = make_routed_step(cfg)
    source, _, _ = train(create_state(cfg), initial_state(cfg, 2), batch)
    before = serialization.to_bytes(source)
    full, _, _ = train(source, initial_state(cfg, 2), batch)
    zero = source.apply_gradients(grads=jax.tree.map(jnp.zeros_like, source.params))
    states, m = make_update_scale_probe(cfg)(source, aggregate, batch)
    assert len(states) == 10 and list(SCALES) == [0, 0.125, 0.25, 0.5, 1]
    for direction, endpoint in [("qa", full), ("zero", zero)]:
        for i, scale in enumerate(SCALES):
            name = f"{direction}_{i}"
            candidate = states[name]
            expected = (
                source.params
                if scale == 0
                else endpoint.params
                if scale == 1
                else jax.tree.map(lambda a, b: a + scale * (b - a), source.params, endpoint.params)
            )
            for a, b in zip(
                jax.tree.leaves(candidate.params), jax.tree.leaves(expected), strict=True
            ):
                np.testing.assert_array_equal(a, b)
            assert serialization.to_bytes(candidate.opt_state) == serialization.to_bytes(
                endpoint.opt_state
            )
            assert int(candidate.step) == int(source.step) + 1
            values = m["branches"][name]
            objective, (_, metrics) = routed_objective(
                candidate.params, candidate.apply_fn, batch, initial_state(cfg, 2), cfg
            )
            _, (_, agg) = routed_objective(
                candidate.params, candidate.apply_fn, aggregate, initial_state(cfg, 2), cfg
            )
            assert values["after"]["selected_language"] == pytest.approx(
                float(metrics["loss"]), abs=1e-6
            )
            assert values["after"]["aggregate_language"] == pytest.approx(
                float(agg["loss"]), abs=1e-6
            )
            assert values["after"]["configured_objective"] == pytest.approx(
                float(objective), abs=1e-6
            )
            if scale == 0:
                assert all(
                    values[k] == 0
                    for k in [
                        "selected_change",
                        "aggregate_change",
                        "objective_change",
                        "update_norm",
                    ]
                )
    for i in range(len(SCALES)):
        for metric in ["selected_change", "aggregate_change", "objective_change"]:
            assert m["qa_minus_zero"][str(i)][metric] == pytest.approx(
                m["branches"][f"qa_{i}"][metric] - m["branches"][f"zero_{i}"][metric], abs=1e-12
            )
    assert serialization.to_bytes(source) == before
    bad = {**batch, "loss_mask": jnp.zeros((2, 8))}
    for a, b in [(bad, batch), (aggregate, bad)]:
        with pytest.raises(ValueError, match="supervised"):
            make_update_scale_probe(cfg)(source, a, b)
    corrupt = source.replace(
        params=jax.tree.map(lambda x: jnp.full_like(x, jnp.nan), source.params)
    )
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        make_update_scale_probe(cfg)(corrupt, aggregate, batch)
