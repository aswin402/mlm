import hashlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pydantic import ValidationError

from mlm.binding_run import warm_start
from mlm.checkpoint import load_checkpoint, save_checkpoint
from mlm.config import Config
from mlm.data import BOS
from mlm.model import initial_state
from mlm.stream import stream_objective
from mlm.training import create_state


def attention_config():
    return Config.model_validate(
        {"nodes": 4, "width": 8, "batch_size": 4, "length": 16, "node_readout": "attention"}
    )


def test_uniform_node_readout_preserves_source_and_causal_state(tmp_path):
    cfg = Config(nodes=4, width=8, batch_size=4, length=16)
    original = create_state(cfg)
    path = tmp_path / "source.mlm"
    save_checkpoint(path, original, cfg, {})
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    acfg = attention_config()
    updated = warm_start(path, acfg)
    tokens = jnp.array([[BOS, 1, 2, 3, 4], [BOS, 4, 3, 2, 1]])
    resets = jnp.zeros_like(tokens, dtype=bool).at[:, 0].set(True)
    hidden = initial_state(cfg, 2)
    reference, ref_final = original.apply_fn({"params": original.params}, tokens, hidden, resets)

    def apply(t, h=hidden, r=resets, valid=None):
        return updated.apply_fn({"params": updated.params}, t, h, r, valid=valid)

    logits, final = apply(tokens)
    np.testing.assert_allclose(logits, reference, atol=5e-6)
    np.testing.assert_array_equal(final, ref_final)
    changed, _ = apply(tokens.at[:, 3:].set(9))
    np.testing.assert_allclose(logits[:, :3], changed[:, :3], atol=1e-6)
    restarted, _ = apply(tokens, h=hidden + 10)
    np.testing.assert_allclose(logits, restarted, atol=1e-6)
    prefix, carry = apply(tokens[:, :2], r=resets[:, :2])
    suffix, carry = apply(tokens[:, 2:], h=carry, r=resets[:, 2:])
    np.testing.assert_allclose(jnp.concatenate([prefix, suffix], axis=1), logits, atol=1e-6)
    np.testing.assert_allclose(carry, final, atol=1e-6)
    valid = jnp.array([[1, 1, 0, 0, 0], [1, 1, 0, 0, 0]], dtype=bool)
    _, padded = apply(tokens, valid=valid)
    _, short = apply(tokens[:, :2], r=resets[:, :2])
    np.testing.assert_allclose(padded, short, atol=1e-6)
    saved = tmp_path / "attention.mlm"
    save_checkpoint(saved, updated, acfg, {})
    restored, loaded, _ = load_checkpoint(saved)
    assert loaded == acfg
    replay, _ = restored.apply_fn({"params": restored.params}, tokens, hidden, resets)
    np.testing.assert_array_equal(replay, logits)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert (
        sum(x.size for x in jax.tree.leaves(updated.params))
        - sum(x.size for x in jax.tree.leaves(original.params))
        == 2 * 8 * 8 + 8
    )


def test_attention_parameters_receive_gradients_and_validate_config():
    cfg = attention_config()
    state = create_state(cfg)
    params = dict(state.params)
    params["attention_vector"] = jnp.ones((cfg.width,)) * 0.2
    tokens = jnp.array([[BOS, 1, 2, 3], [BOS, 4, 3, 2], [BOS, 5, 6, 7], [BOS, 7, 6, 5]])
    batch = {
        "inputs": tokens,
        "targets": (tokens + 1) % 256,
        "mask": jnp.ones_like(tokens),
        "resets": jnp.zeros_like(tokens, dtype=bool).at[:, 0].set(True),
    }

    def loss(p):
        return stream_objective(p, state.apply_fn, batch, initial_state(cfg, 4))[0]

    first_grads = jax.grad(loss)(state.params)
    gradients = jax.grad(loss)(params)
    for name in ["attention_vector", "attention_key", "attention_query"]:
        assert np.isfinite(first_grads[name]).all()
        assert float(jnp.linalg.norm(first_grads[name])) > 0
        assert np.isfinite(gradients[name]).all()
        assert float(jnp.linalg.norm(gradients[name])) > 0
    with pytest.raises(ValidationError):
        Config(architecture="gru", node_readout="attention")
