import jax
import jax.numpy as jnp
import numpy as np

from mlm.checkpoint import load_checkpoint, save_checkpoint
from mlm.config import Config
from mlm.model import initial_state
from mlm.stream import DocumentStream, make_stream_step
from mlm.training import create_state


def test_gru_parameters_causality_padding_and_chunks(tmp_path):
    cfg = Config(architecture="gru", width=12, batch_size=2, length=8)
    state = create_state(cfg)
    h = initial_state(cfg, 2)
    assert h.shape == (2, 1, 12)
    assert "message_kernel" not in state.params
    tokens = jnp.array([[256, 10, 11, 12], [256, 20, 21, 22]])
    resets = jnp.zeros_like(tokens, dtype=bool).at[:, 0].set(True)

    def apply(x, c, r, **kw):
        return state.apply_fn({"params": state.params}, x, c, r, **kw)

    logits, carry = apply(tokens, h, resets)
    other, _ = apply(tokens.at[:, 3].set(99), h, resets)
    np.testing.assert_allclose(logits[:, :3], other[:, :3], atol=1e-6)
    reset_logits, _ = apply(tokens, h + 1, resets)
    np.testing.assert_allclose(logits, reset_logits, atol=1e-6)
    a, mid = apply(tokens[:, :2], h, resets[:, :2])
    b, end = apply(tokens[:, 2:], mid, resets[:, 2:])
    np.testing.assert_allclose(jnp.concatenate([a, b], 1), logits, atol=1e-6)
    np.testing.assert_allclose(end, carry, atol=1e-6)
    _, frozen = apply(tokens, h + 0.2, jnp.zeros_like(resets), valid=jnp.zeros_like(tokens))
    np.testing.assert_array_equal(frozen, h + 0.2)
    batch = DocumentStream(["abababab", "babababa"], 2, 8).next_batch()
    step = make_stream_step(cfg)
    trained, carry, metrics = step(state, h, jax.tree.map(jnp.asarray, batch))
    assert bool(metrics["finite"])
    assert float(metrics["grad_norm"]) > 0
    path = tmp_path / "gru.mlm"
    save_checkpoint(path, trained, cfg, {"kind": "gru"})
    restored, restored_cfg, _ = load_checkpoint(path)
    assert restored_cfg == cfg
    left = step(trained, carry, jax.tree.map(jnp.asarray, batch))
    right = step(restored, carry, jax.tree.map(jnp.asarray, batch))
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        np.testing.assert_array_equal(a, b)


def test_baseline_parameter_budget():
    graph = create_state(Config(width=64))
    gru = create_state(Config(architecture="gru", width=208))
    counts = [sum(p.size for p in jax.tree.leaves(s.params)) for s in (graph, gru)]
    assert abs(counts[0] - counts[1]) / counts[0] < 0.01
