import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import serialization

from mlm.config import Config
from mlm.hierarchy_run import make_routed_step, routed_objective
from mlm.model import initial_state
from mlm.qa_controls import make_qa_controls
from mlm.training import create_state


def test_controls_match_direct_steps_and_preserve_source():
    cfg = Config(
        nodes=4,
        width=8,
        groups=2,
        node_update="hierarchical",
        router_balance=0.3,
        memory="delta",
        memory_keys=4,
        batch_size=2,
        length=8,
    )
    batch = {
        "inputs": jnp.array([[256, 65, 66, 67, 68, 69, 70, 71], [256, 81, 82, 83, 84, 85, 86, 87]]),
        "targets": jnp.array(
            [[65, 66, 67, 68, 69, 70, 71, 257], [81, 82, 83, 84, 85, 86, 87, 257]]
        ),
        "mask": jnp.ones((2, 8)),
        "loss_mask": jnp.array([[0, 0, 0, 0, 1, 1, 1, 1]] * 2),
        "resets": jnp.array([[True] + [False] * 7] * 2),
    }
    aggregate = {**batch, "targets": (batch["targets"] + 17) % 258, "loss_mask": jnp.ones((2, 8))}
    source = create_state(cfg)
    train = make_routed_step(cfg)
    probe = make_qa_controls(cfg)
    for count in range(2):
        before = serialization.to_bytes(source)
        states, metrics = probe(source, aggregate, batch)
        assert len(states) == 6
        for history in ["loaded", "fresh"]:
            base = (
                source
                if history == "loaded"
                else source.replace(opt_state=source.tx.init(source.params))
            )
            for mode in ["configured", "language"]:
                c = cfg if mode == "configured" else cfg.model_copy(update={"router_balance": 0.0})
                expected, _, _ = make_routed_step(c)(base, initial_state(c, 2), batch)
                name = history + "_" + mode
                for a, b in zip(
                    jax.tree.leaves(states[name]), jax.tree.leaves(expected), strict=True
                ):
                    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)
                m = metrics["branches"][name]
                total, (_, direct) = routed_objective(
                    expected.params, expected.apply_fn, batch, initial_state(c, 2), c
                )
                assert m["after"]["selected_language"] == pytest.approx(
                    float(direct["loss"]), abs=1e-6
                )
                assert m["after"]["active_objective"] == pytest.approx(float(total), abs=1e-6)
                assert m["after"]["configured_objective"] == pytest.approx(
                    m["after"]["selected_language"]
                    + cfg.router_balance * m["after"]["router_penalty"],
                    abs=1e-6,
                )
                _, (_, agg) = routed_objective(
                    expected.params, expected.apply_fn, aggregate, initial_state(c, 2), c
                )
                assert m["after"]["aggregate_language"] == pytest.approx(
                    float(agg["loss"]), abs=1e-6
                )
                baseline = (
                    metrics["before"]["selected_language"]
                    if mode == "language"
                    else metrics["before"]["configured_objective"]
                )
                assert m["active_objective_change"] == pytest.approx(
                    float(total) - baseline, abs=1e-6
                )
                assert m["step_after"] == int(source.step) + 1
            zero = base.apply_gradients(grads=jax.tree.map(jnp.zeros_like, base.params))
            for a, b in zip(
                jax.tree.leaves(states[history + "_zero"]), jax.tree.leaves(zero), strict=True
            ):
                np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)
        if count == 0:
            for mode in ["configured", "language", "zero"]:
                for a, b in zip(
                    jax.tree.leaves(states["loaded_" + mode]),
                    jax.tree.leaves(states["fresh_" + mode]),
                    strict=True,
                ):
                    np.testing.assert_array_equal(a, b)
        for metric in ["selected_change", "aggregate_change"]:
            values = {name: branch[metric] for name, branch in metrics["branches"].items()}
            c = metrics["contrasts"][metric]
            reg = {}
            for h in ["loaded", "fresh"]:
                reg[h] = values[h + "_configured"] - values[h + "_language"]
                assert c["regularizer_configured_minus_language"][h] == pytest.approx(
                    reg[h], abs=1e-12
                )
                for mode in ["configured", "language"]:
                    assert c["minus_zero"][h + "_" + mode] == pytest.approx(
                        values[h + "_" + mode] - values[h + "_zero"], abs=1e-12
                    )
            assert c["interaction"] == pytest.approx(reg["loaded"] - reg["fresh"], abs=1e-12)
            for mode in ["configured", "language", "zero"]:
                assert c["history_loaded_minus_fresh"][mode] == pytest.approx(
                    values["loaded_" + mode] - values["fresh_" + mode], abs=1e-12
                )
        assert serialization.to_bytes(source) == before
        source, _, _ = train(source, initial_state(cfg, 2), batch)
    bad = {**batch, "loss_mask": jnp.zeros((2, 8))}
    with pytest.raises(ValueError, match="supervised"):
        probe(source, batch, bad)
    with pytest.raises(ValueError, match="supervised"):
        probe(source, bad, batch)

    corrupt = source.replace(
        params=jax.tree.map(lambda x: jnp.full_like(x, jnp.nan), source.params)
    )
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        probe(corrupt, aggregate, batch)
