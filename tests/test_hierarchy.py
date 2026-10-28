import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pydantic import ValidationError

from mlm.config import Config
from mlm.model import initial_state
from mlm.training import create_state


def config(mode):
    return Config(nodes=4, width=8, groups=2, node_update=mode, batch_size=2, length=8)


def test_hierarchy_uniform_equivalence_causality_and_state(tmp_path):
    from mlm.binding_run import warm_start
    from mlm.checkpoint import load_checkpoint, save_checkpoint

    dense = Config(nodes=4, width=8, batch_size=2, length=8)
    source = tmp_path / "source.mlm"
    save_checkpoint(source, create_state(dense), dense, {})
    states = {mode: warm_start(source, config(mode)) for mode in ["uniform", "hierarchical"]}
    tokens = jnp.array([[256, 1, 2, 3], [256, 3, 2, 1]])
    resets = jnp.zeros_like(tokens, dtype=bool).at[:, 0].set(True)

    def apply(mode, t=tokens, h=None, r=resets, valid=None):
        state = states[mode]
        cfg = config(mode)
        return state.apply_fn(
            {"params": state.params},
            t,
            initial_state(cfg, 2) if h is None else h,
            r,
            valid=valid,
            return_routes=True,
        )

    a, ha, pa = apply("uniform")
    b, hb, pb = apply("hierarchical")
    np.testing.assert_allclose(a, b, atol=1e-6)
    np.testing.assert_allclose(ha, hb, atol=1e-6)
    np.testing.assert_allclose(pb, 0.25, atol=1e-6)
    np.testing.assert_allclose(pb.sum(-1), 1, atol=1e-6)
    changed, _, _ = apply("hierarchical", t=tokens.at[:, 2:].set(8))
    np.testing.assert_allclose(b[:, :2], changed[:, :2], atol=1e-6)
    prefix, carry, _ = apply("hierarchical", t=tokens[:, :2], r=resets[:, :2])
    suffix, carry, _ = apply("hierarchical", t=tokens[:, 2:], h=carry, r=resets[:, 2:])
    np.testing.assert_allclose(jnp.concatenate([prefix, suffix], 1), b, atol=1e-6)
    np.testing.assert_allclose(carry, hb, atol=1e-6)
    _, pad, prob = apply("hierarchical", valid=jnp.zeros_like(tokens))
    np.testing.assert_array_equal(pad, initial_state(config("hierarchical"), 2))
    np.testing.assert_array_equal(prob, 0)
    reset, _, _ = apply("hierarchical", h=initial_state(dense, 2) + 5)
    np.testing.assert_allclose(reset, b, atol=1e-6)
    path = tmp_path / "hierarchy.mlm"
    save_checkpoint(path, states["hierarchical"], config("hierarchical"), {})
    loaded, cfg, _ = load_checkpoint(path)
    replay, _ = loaded.apply_fn({"params": loaded.params}, tokens, initial_state(cfg, 2), resets)
    np.testing.assert_allclose(replay, b, atol=1e-6)
    for bad in [{"groups": 3}, {"groups": 4}, {"architecture": "gru"}]:
        with pytest.raises(ValidationError):
            Config(nodes=4, width=8, node_update="hierarchical", **bad)


def test_hierarchy_routing_gradients_and_real_context_dependence():
    from mlm.hierarchy_run import routed_objective

    cfg = config("hierarchical")
    state = create_state(cfg)
    tokens = jnp.array([[256, 1, 2, 3], [256, 3, 2, 1]])
    batch = {
        "inputs": tokens,
        "targets": (tokens + 1) % 256,
        "mask": jnp.ones_like(tokens),
        "resets": jnp.zeros_like(tokens, dtype=bool).at[:, 0].set(True),
    }

    def loss(params):
        return routed_objective(params, state.apply_fn, batch, initial_state(cfg, 2), cfg)[0]

    grads = jax.grad(loss)(state.params)
    for name in ["router_group_kernel", "router_child_kernel"]:
        assert np.isfinite(grads[name]).all() and float(jnp.linalg.norm(grads[name])) > 0
    changed = dict(state.params)
    changed["router_group_kernel"] = changed["router_group_kernel"].at[0, 0].set(2)
    _, _, routes = state.apply_fn(
        {"params": changed}, tokens, initial_state(cfg, 2), batch["resets"], return_routes=True
    )
    assert not np.allclose(routes, 0.25)
    np.testing.assert_allclose(routes.sum(-1), 1, atol=1e-6)


def test_fixed_slow_blend_and_two_level_factorization():
    cfg = config("hierarchical")
    state = create_state(cfg)
    params = dict(state.params)
    params["router_child_bias"] = params["router_child_bias"].at[0, 0].set(jnp.log(3.0))
    tokens = jnp.array([[1], [2]])
    resets = jnp.zeros_like(tokens, dtype=bool)
    hidden = jnp.ones((2, 4, 8)) * 0.1
    _, _, prob = state.apply_fn({"params": params}, tokens, hidden, resets, return_routes=True)
    np.testing.assert_allclose(prob[:, 0, :], [[0.375, 0.125, 0.25, 0.25]] * 2, atol=1e-6)
    dense_cfg = Config(nodes=4, width=8, batch_size=2, length=8)
    dense = create_state(dense_cfg)
    uniform = create_state(config("uniform")).replace(params=dense.params)
    _, candidate = dense.apply_fn({"params": dense.params}, tokens, hidden, resets)
    _, slowed = uniform.apply_fn({"params": uniform.params}, tokens, hidden, resets)
    np.testing.assert_allclose(slowed, hidden + (candidate - hidden) / 4, atol=1e-6)


def test_routed_runner_and_empty_batch_behavior(tmp_path):
    import hashlib
    import json
    from types import SimpleNamespace

    from mlm.checkpoint import load_checkpoint, save_checkpoint
    from mlm.confirmation_data import prepare
    from mlm.hierarchy_run import make_routed_step, run

    cfg = Config(nodes=4, width=8, groups=2, node_update="hierarchical", batch_size=4, length=128)
    source_cfg = Config(nodes=4, width=8, groups=2, batch_size=4, length=128)
    source = tmp_path / "source.mlm"
    save_checkpoint(source, create_state(source_cfg), source_cfg, {})
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    data = tmp_path / "data"
    prepare(data, seed=920231)
    config_file = tmp_path / "config.json"
    config_file.write_text(cfg.model_dump_json())
    english_train = tmp_path / "train.txt"
    english_train.write_text("A little story.")
    english_dev = tmp_path / "dev.txt"
    english_dev.write_text("Another tale.")
    args = SimpleNamespace(
        config=config_file,
        init=source,
        data=data,
        out=tmp_path / "out",
        steps=1,
        eval_every=1,
        english_train=english_train,
        english_dev=english_dev,
        rehearsal=True,
    )
    run(args)
    latest, _, _ = load_checkpoint(args.out / "latest.mlm")
    assert int(latest.step) == 2
    rows = [json.loads(line) for line in (args.out / "metrics.jsonl").read_text().splitlines()]
    assert sum(rows[-1]["routing"]["qa"]["mean_usage"]) == pytest.approx(1)
    assert rows[-1]["routing"]["qa"]["mean_entropy"] > 0
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    empty = {
        "inputs": jnp.zeros((4, 2), dtype=int),
        "targets": jnp.zeros((4, 2), dtype=int),
        "mask": jnp.zeros((4, 2)),
        "resets": jnp.zeros((4, 2), dtype=bool),
    }
    next_state, _, metrics = make_routed_step(cfg)(latest, initial_state(cfg, 4), empty)
    assert int(next_state.step) == int(latest.step)
    assert float(metrics["loss"]) == 0 and float(metrics["router_balance"]) == 0
    with pytest.raises(ValueError, match="empty"):
        run(args)


def test_timescale_surprise_routing_spectrum_and_step_zero_gradients(tmp_path):
    from mlm.binding_run import warm_start
    from mlm.checkpoint import save_checkpoint
    from mlm.hierarchy_run import routed_objective

    dense = Config(nodes=4, width=8, batch_size=2, length=8)
    source = tmp_path / "source.mlm"
    save_checkpoint(source, create_state(dense), dense, {})
    timescale_cfg = config("timescale")
    timescale = warm_start(source, timescale_cfg)
    tokens = jnp.array([[256, 1, 2, 3], [256, 3, 2, 1]])
    resets = jnp.zeros_like(tokens, dtype=bool).at[:, 0].set(True)
    t_logits, t_h, t_routes = timescale.apply_fn(
        {"params": timescale.params},
        tokens,
        initial_state(timescale_cfg, 2),
        resets,
        return_routes=True,
    )
    assert np.isfinite(t_logits).all() and np.isfinite(t_h).all()
    # Fast nodes have strictly higher initial routing share than slow nodes across the spectrum.
    assert (
        float(t_routes[0, 0, 0])
        > float(t_routes[0, 0, 1])
        > float(t_routes[0, 0, 2])
        > float(t_routes[0, 0, 3])
    )
    np.testing.assert_allclose(t_routes.sum(-1), 1.0, atol=1e-6)
    batch = {
        "inputs": tokens,
        "targets": (tokens + 1) % 256,
        "mask": jnp.ones_like(tokens),
        "resets": resets,
    }

    def loss(params):
        return routed_objective(
            params, timescale.apply_fn, batch, initial_state(timescale_cfg, 2), timescale_cfg
        )[0]

    grads = jax.grad(loss)(timescale.params)
    for name in ["router_timescale_kernel", "router_surprise_kernel", "router_timescale_bias"]:
        assert np.isfinite(grads[name]).all()
        assert float(jnp.linalg.norm(grads[name])) > 0
