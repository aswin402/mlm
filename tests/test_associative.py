import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pydantic import ValidationError

from mlm.associative import delta_step
from mlm.binding_run import warm_start
from mlm.checkpoint import load_checkpoint, save_checkpoint
from mlm.config import Config
from mlm.hierarchy_run import make_routed_step
from mlm.model import initial_state
from mlm.plasticity import parameter_roles
from mlm.training import create_state


def config(**kwargs):
    return Config(
        nodes=4,
        width=8,
        groups=2,
        node_update="hierarchical",
        memory="delta",
        memory_keys=4,
        **kwargs,
    )


def test_delta_memory_write_read_overwrite_and_noop():
    m = jnp.zeros((1, 2, 3))
    k = jnp.array([[1.0, 0.0]])
    other = jnp.array([[0.0, 1.0]])
    value = jnp.array([[2.0, 3.0, 4.0]])
    m, r = delta_step(m, k, value, k, jnp.ones(1))
    np.testing.assert_array_equal(r, 0)
    m, r = delta_step(m, other, -value, k, jnp.ones(1))
    np.testing.assert_array_equal(r, value)
    m, r = delta_step(m, k, value * 2, other, jnp.ones(1))
    np.testing.assert_array_equal(r, -value)
    unchanged, r = delta_step(m, other, value, k, jnp.zeros(1))
    np.testing.assert_array_equal(unchanged, m)
    np.testing.assert_array_equal(r, value * 2)
    # Rewriting an existing key replaces its mapping instead of adding another copy.
    np.testing.assert_array_equal(m[:, 0, :], value * 2)

    def loss(v):
        return jnp.square(delta_step(m, k, v, other, jnp.ones(1))[0]).sum()

    grad = jax.grad(loss)(value)
    assert np.isfinite(grad).all() and np.linalg.norm(grad) > 0
    overlap = jnp.ones((1, 2)) / jnp.sqrt(2.0)
    corrected, _ = delta_step(m, overlap, jnp.zeros_like(value), k, jnp.ones(1))
    _, recalled = delta_step(corrected, k, value, k, jnp.zeros(1))
    np.testing.assert_allclose(recalled, 1.5 * value, atol=1e-6)
    fractional, old_read = delta_step(m, overlap, jnp.zeros_like(value), k, jnp.full((1,), 0.5))
    np.testing.assert_allclose(old_read, 2 * value, atol=1e-6)
    np.testing.assert_allclose(fractional[:, 0, :], 1.75 * value, atol=1e-6)
    np.testing.assert_allclose(fractional[:, 1, :], -1.25 * value, atol=1e-6)
    # Overlapping addresses interfere; memory is not an exact symbolic dictionary.
    np.testing.assert_allclose(jnp.einsum("bk,bkv->bv", overlap, corrected), 0, atol=1e-6)
    # Per-key 2D liquid time-constant decay scales each key slot independently.
    per_key_alpha = jnp.array([[0.5, 0.25]])
    decayed_m, _ = delta_step(fractional, k, jnp.zeros_like(value), k, jnp.zeros(1), per_key_alpha)
    np.testing.assert_allclose(decayed_m[:, 0, :], 0.5 * fractional[:, 0, :], atol=1e-6)
    np.testing.assert_allclose(decayed_m[:, 1, :], 0.25 * fractional[:, 1, :], atol=1e-6)


def test_memory_zero_gate_causality_padding_chunking_and_gradients(tmp_path):
    cfg = config()
    base_cfg = cfg.model_copy(update={"memory": "none"})
    base = create_state(base_cfg)
    source = tmp_path / "base.mlm"
    save_checkpoint(source, base, base_cfg, {"messages": True})
    state = warm_start(source, cfg)
    for name, value in base.params.items():
        np.testing.assert_array_equal(value, state.params[name])
    tokens = jnp.array([[256, 12, 31, 45, 22, 11], [256, 3, 8, 7, 2, 1]])
    resets = jnp.zeros_like(tokens, dtype=bool).at[:, 0].set(True)

    def apply(p, t, h, r, v=None):
        return state.apply_fn({"params": p}, t, h, r, valid=v, return_routes=True)

    a, _ = base.apply_fn({"params": base.params}, tokens, initial_state(base_cfg, 2), resets)
    b, h, routes = apply(state.params, tokens, initial_state(cfg, 2), resets)
    # Packing extra recurrent state changes CPU fusion/rounding slightly.
    np.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-6)
    with jax.disable_jit():
        eager_base, _ = base.apply_fn(
            {"params": base.params}, tokens, initial_state(base_cfg, 2), resets
        )
        eager_memory, _, _ = apply(state.params, tokens, initial_state(cfg, 2), resets)
    np.testing.assert_array_equal(eager_base, eager_memory)
    assert h.shape == (2, 8, 8) and routes.shape == (2, 6, 4)
    assert np.linalg.norm(np.asarray(h[:, 4:])) > 0
    params = dict(state.params)
    params["memory_gain"] = jnp.array(0.25)
    logits, whole, _ = apply(params, tokens, initial_state(cfg, 2), resets)
    prefix, carry, _ = apply(params, tokens[:, :3], initial_state(cfg, 2), resets[:, :3])
    suffix, chunked, _ = apply(params, tokens[:, 3:], carry, resets[:, 3:])
    np.testing.assert_allclose(jnp.concatenate([prefix, suffix], 1), logits, atol=1e-6)
    np.testing.assert_allclose(chunked, whole, atol=1e-6)
    changed, _, _ = apply(params, tokens.at[:, 3:].set(9), initial_state(cfg, 2), resets)
    np.testing.assert_array_equal(changed[:, :3], logits[:, :3])
    dirty = initial_state(cfg, 2) + 0.75
    _, frozen, probs = apply(params, tokens, dirty, resets, jnp.zeros_like(tokens, dtype=bool))
    np.testing.assert_array_equal(frozen, dirty)
    np.testing.assert_array_equal(probs, 0)
    clean, _, _ = apply(params, tokens, dirty, resets)
    np.testing.assert_allclose(clean, logits, atol=1e-6)

    def loss(p):
        return jnp.square(apply(p, tokens, initial_state(cfg, 2), resets)[0]).mean()

    grads = jax.grad(loss)(params)
    roles = parameter_roles(params)
    for name in params:
        if roles[name] == "memory":
            assert np.isfinite(grads[name]).all() and np.linalg.norm(grads[name]) > 0, name
    first_grads = jax.grad(loss)(state.params)
    for name in state.params:
        if roles[name] == "memory":
            assert np.isfinite(first_grads[name]).all() and np.linalg.norm(first_grads[name]) > 0, (
                name
            )


def test_memory_checkpoint_update_and_validation(tmp_path):
    cfg = config(batch_size=2, length=8, memory_lr_scale=0.5, shared_lr_scale=0.1)
    state = create_state(cfg)
    params = dict(state.params)
    params["memory_gain"] = jnp.array(0.25)
    state = state.replace(params=params)
    tokens = jnp.array([[256, 1, 2, 3], [256, 3, 2, 1]])
    batch = {
        "inputs": tokens,
        "targets": (tokens + 1) % 256,
        "mask": jnp.ones_like(tokens),
        "resets": jnp.zeros_like(tokens, dtype=bool).at[:, 0].set(True),
    }
    step = make_routed_step(cfg)
    state, carry, metrics = step(state, initial_state(cfg, 2), batch)
    assert bool(metrics["updated"]) and bool(metrics["finite"])
    path = tmp_path / "memory.mlm"
    save_checkpoint(path, state, cfg, {"mode": "memory-test", "carry": np.asarray(carry).tolist()})
    loaded, restored, meta = load_checkpoint(path)
    assert restored == cfg
    resumed_carry = jnp.asarray(meta["carry"], dtype=jnp.float32)
    continuation = {**batch, "resets": jnp.zeros_like(tokens, dtype=bool)}
    remembered, _ = state.apply_fn({"params": state.params}, tokens, carry, continuation["resets"])
    blank_memory = carry.at[:, cfg.nodes :, :].set(0)
    forgotten, _ = state.apply_fn(
        {"params": state.params}, tokens, blank_memory, continuation["resets"]
    )
    assert not np.allclose(remembered, forgotten)
    a, ha, _ = step(state, carry, continuation)
    b, hb, _ = step(loaded, resumed_carry, continuation)
    for x, y in zip(
        jax.tree.leaves((a.params, a.opt_state, ha)),
        jax.tree.leaves((b.params, b.opt_state, hb)),
        strict=True,
    ):
        np.testing.assert_array_equal(x, y)
    for bad in [
        {"architecture": "gru"},
        {"node_readout": "attention"},
        {"memory_keys": 0},
        {"memory_lr_scale": 0},
    ]:
        args = {"memory": "delta", **bad}
        with pytest.raises(ValidationError):
            Config(**args)


def test_relational_memory_binds_across_separate_node_subspaces(tmp_path):
    cfg = config().model_copy(update={"memory": "relational"})
    base_cfg = cfg.model_copy(update={"memory": "none"})
    base = create_state(base_cfg)
    source = tmp_path / "base.mlm"
    save_checkpoint(source, base, base_cfg, {"messages": True})
    state = warm_start(source, cfg)
    assert state.params["memory_key_kernel"].shape == (cfg.nodes, cfg.width, cfg.memory_keys)
    assert state.params["memory_value_kernel"].shape == (cfg.nodes, cfg.width, cfg.width)
    tokens = jnp.array([[256, 12, 31, 45, 22, 11], [256, 3, 8, 7, 2, 1]])
    resets = jnp.zeros_like(tokens, dtype=bool).at[:, 0].set(True)
    ref_logits, _ = base.apply_fn(
        {"params": base.params}, tokens, initial_state(base_cfg, 2), resets
    )
    rel_logits, rel_state = state.apply_fn(
        {"params": state.params}, tokens, initial_state(cfg, 2), resets
    )
    np.testing.assert_allclose(ref_logits, rel_logits, atol=1e-6)
    assert np.linalg.norm(np.asarray(rel_state[:, cfg.nodes :])) > 0

    def loss(p):
        return jnp.square(
            state.apply_fn({"params": p}, tokens, initial_state(cfg, 2), resets)[0]
        ).mean()

    grads = jax.grad(loss)(state.params)
    roles = parameter_roles(state.params)
    for name in state.params:
        if roles[name] == "memory":
            assert np.isfinite(grads[name]).all() and np.linalg.norm(grads[name]) > 0, name
