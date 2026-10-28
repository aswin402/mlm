import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pydantic import ValidationError

from mlm.checkpoint import load_checkpoint, save_checkpoint
from mlm.config import Config
from mlm.data import BOS, EOS, encode, make_batch
from mlm.graph import aggregate, topology
from mlm.model import GraphLM, initial_state
from mlm.training import create_state, make_train_step, score


def small():
    return Config(nodes=4, width=8, degree=2, rounds=2, batch_size=2, length=12)


def test_config_rejects_invalid_values():
    for settings in ({"nodes": 1}, {"degree": 8}, {"rounds": 0}, {"typo": 3}):
        with pytest.raises(ValidationError):
            Config(**settings)


def test_bytes_and_padding():
    ids = encode("é")
    assert ids.tolist() == [BOS, 195, 169, EOS]
    batch = make_batch(["a", "hello"], [0, 1], 8)
    assert batch["mask"].sum() == 8
    assert batch["resets"][:, 0].all()
    assert not batch["resets"][:, 1:].any()
    assert np.all(batch["mask"][0, 2:] == 0)


def test_synchronous_aggregation_matches_reference():
    edges = topology(8, 2)
    assert edges.shape == (8, 2)
    assert all(len(set(row)) == 2 for row in edges.tolist())
    h = np.arange(48, dtype=np.float32).reshape(2, 8, 3)
    expected = np.stack([sum(h[:, j] for j in row) / np.sqrt(2) for row in edges], 1)
    np.testing.assert_allclose(aggregate(jnp.array(h), jnp.array(edges)), expected, rtol=1e-6)


def test_causality_reset_chunking_and_graph_effect():
    cfg = small()
    model = GraphLM(cfg)
    state = create_state(cfg)
    tokens = jnp.array([[BOS, 1, 2, 3, 4], [BOS, 4, 3, 2, 1]])
    resets = jnp.zeros_like(tokens, dtype=bool).at[:, 0].set(True)
    h = initial_state(cfg, 2)
    logits, final = model.apply({"params": state.params}, tokens, h, resets)
    changed = tokens.at[:, 3:].set(120)
    other, _ = model.apply({"params": state.params}, changed, h, resets)
    np.testing.assert_allclose(logits[:, :3], other[:, :3], atol=1e-6)
    reset_logits, _ = model.apply({"params": state.params}, tokens, h + 9, resets)
    np.testing.assert_allclose(logits, reset_logits, atol=1e-6)
    prefix, carry = model.apply({"params": state.params}, tokens[:, :2], h, resets[:, :2])
    suffix, carry = model.apply({"params": state.params}, tokens[:, 2:], carry, resets[:, 2:])
    np.testing.assert_allclose(jnp.concatenate([prefix, suffix], 1), logits, atol=1e-6)
    np.testing.assert_allclose(carry, final, atol=1e-6)
    no_edges, _ = model.apply({"params": state.params}, tokens, h, resets, messages=False)
    assert not np.allclose(logits, no_edges)


def test_learning_finite_gradients_and_exact_resume(tmp_path):
    cfg = small().model_copy(update={"learning_rate": 0.01})
    state = create_state(cfg)
    batch = jax.tree.map(jnp.asarray, make_batch(["ababababab", "bababababa"], [0, 1], 12))
    step = make_train_step(cfg)
    before = float(score(state, batch, cfg)["loss"])
    for _ in range(60):
        state, metrics = step(state, batch)
    assert bool(metrics["finite"])
    assert float(metrics["grad_norm"]) > 0
    assert float(score(state, batch, cfg)["loss"]) < before * 0.55
    path = tmp_path / "model.mlm"
    save_checkpoint(path, state, cfg, {"data_hash": "fixture", "next_batch": 60})
    restored, restored_cfg, metadata = load_checkpoint(path)
    assert restored_cfg == cfg
    assert metadata["next_batch"] == 60
    a, _ = step(state, batch)
    b, _ = step(restored, batch)
    for left, right in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_array_equal(left, right)


def test_decay_excludes_node_biases():
    from mlm.training import decay_mask

    mask = decay_mask(create_state(small()).params)
    assert not mask["gru_bias"]
    assert not mask["norm_scale"]
    assert mask["input_kernel"]
    delta_mask = decay_mask(create_state(small().model_copy(update={"memory": "delta"})).params)
    assert not delta_mask["memory_gain"]
    assert not delta_mask["memory_write_bias"]
    assert delta_mask["memory_key_kernel"]
    attn_mask = decay_mask(
        create_state(small().model_copy(update={"node_readout": "attention"})).params
    )
    assert not attn_mask["attention_vector"]
    assert attn_mask["attention_query"]


def test_complete_rounds_match_numpy_reference():
    cfg = small()
    state = create_state(cfg)
    params = jax.tree.map(np.asarray, state.params)
    h = np.random.default_rng(7).normal(size=(1, cfg.nodes, cfg.width)).astype(np.float32)
    tokens = jnp.array([[31]])
    logits, final = state.apply_fn(
        {"params": state.params}, tokens, jnp.array(h), jnp.array([[False]])
    )
    edges = topology(cfg.nodes, cfg.degree)
    x = params["embedding"][31]
    for _ in range(cfg.rounds):
        old = h.copy()
        projected = old @ params["message_kernel"]
        for i in range(cfg.nodes):
            message = sum(projected[0, j] for j in edges[i]) / np.sqrt(cfg.degree)
            gx = np.concatenate([x, message]) @ params["input_kernel"][i] + params["gru_bias"][i]
            gh = old[0, i] @ params["hidden_kernel"][i]
            xz, xr, xn = np.split(gx, 3)
            hz, hr, hn = np.split(gh, 3)
            z = 1 / (1 + np.exp(-(xz + hz)))
            r = 1 / (1 + np.exp(-(xr + hr)))
            h[0, i] = (1 - z) * np.tanh(xn + r * hn) + z * old[0, i]
    readout = sum(h[0, i] @ params["read_kernel"][i] for i in range(cfg.nodes))
    readout = readout / np.sqrt(cfg.nodes)
    readout = (readout - readout.mean()) / np.sqrt(readout.var() + 1e-5)
    expected = (readout * params["norm_scale"] + params["norm_bias"]) @ params["vocab_kernel"]
    expected += params["vocab_bias"]
    np.testing.assert_allclose(final, h, atol=1e-6)
    np.testing.assert_allclose(logits[0, 0], expected, atol=2e-6)


def test_learned_edge_attention_preserves_zero_gate_and_flows_step_zero_gradients():
    cfg = small()
    base = create_state(cfg)
    ecfg = cfg.model_copy(update={"edge_aggregation": "attention"})
    state = create_state(ecfg)
    params = {**state.params, **base.params}
    tokens = jnp.array([[BOS, 1, 2, 3, 4], [BOS, 4, 3, 2, 1]])
    resets = jnp.zeros_like(tokens, dtype=bool).at[:, 0].set(True)
    h = initial_state(cfg, 2)
    ref_logits, ref_final = base.apply_fn({"params": base.params}, tokens, h, resets)
    logits, final = state.apply_fn({"params": params}, tokens, h, resets)
    np.testing.assert_allclose(logits, ref_logits, atol=1e-6)
    np.testing.assert_allclose(final, ref_final, atol=1e-6)

    def loss(p):
        return jnp.square(state.apply_fn({"params": p}, tokens, h, resets)[0]).mean()

    grads = jax.grad(loss)(params)
    for name in ["edge_query", "edge_key", "edge_vector"]:
        assert np.isfinite(grads[name]).all()
        assert float(jnp.linalg.norm(grads[name])) > 0
    active = {**params, "edge_vector": jnp.ones((cfg.width,)) * 0.5}
    weighted_logits, _ = state.apply_fn({"params": active}, tokens, h, resets)
    assert not np.allclose(weighted_logits, ref_logits)
    with pytest.raises(ValidationError):
        Config(architecture="gru", edge_aggregation="attention")
