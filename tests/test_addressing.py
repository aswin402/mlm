import jax.numpy as jnp
import numpy as np
import pytest

from mlm.addressing import effective_coefficients, question_regions, read_disabled, trace_prompt
from mlm.associative import delta_step
from mlm.config import Config
from mlm.gradual_data import group
from mlm.model import initial_state
from mlm.training import create_state


def test_overwrite_aware_signed_coefficients_reconstruct_causal_reads():
    rng = np.random.default_rng(8)
    keys = rng.normal(size=(7, 4))
    keys /= np.linalg.norm(keys, axis=1, keepdims=True)
    queries = rng.normal(size=(7, 4))
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    values = rng.normal(size=(7, 3))
    beta = np.linspace(0.1, 0.8, 7)
    memory = jnp.zeros((1, 4, 3))
    for t in range(7):
        memory, actual = delta_step(
            memory,
            jnp.array(keys[t : t + 1]),
            jnp.array(values[t : t + 1]),
            jnp.array(queries[t : t + 1]),
            jnp.array(beta[t : t + 1]),
        )
        coefficients = effective_coefficients(keys, beta, queries[t], t)
        np.testing.assert_allclose(coefficients @ values, np.asarray(actual[0]), atol=1e-6)
        np.testing.assert_array_equal(coefficients[t:], 0)
        changed = keys.copy()
        changed[t:] *= -3
        np.testing.assert_array_equal(
            effective_coefficients(changed, beta, queries[t], t), coefficients
        )
    for bad in [-1, 8]:
        with pytest.raises(ValueError):
            effective_coefficients(keys, beta, queries[0], bad)


def test_prompt_trace_reconstructs_actual_model_and_ablation_is_read_only():
    cfg = Config(
        nodes=4, width=8, groups=2, node_update="hierarchical", memory="delta", memory_keys=4
    )
    state = create_state(cfg)
    params = dict(state.params)
    params["memory_gain"] = jnp.array(0.3)
    state = state.replace(params=params)
    prompt = "A small memory prompt."
    trace = trace_prompt(state, cfg, prompt)
    ids = jnp.array([[256, *prompt.encode()]], dtype=jnp.int32)
    resets = jnp.zeros_like(ids, dtype=bool).at[:, 0].set(True)
    logits, final = state.apply_fn({"params": state.params}, ids, initial_state(cfg, 1), resets)
    np.testing.assert_allclose(trace["last_logits"], logits[0, -1], atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(trace["final_state"], final[0], atol=1e-5)
    assert trace["max_memory_reconstruction_error"] < 1e-5
    assert trace["max_logit_reconstruction_error"] < 1e-5
    assert trace["read_reconstruction_error"] < 1e-5
    off = read_disabled(state)
    assert float(state.params["memory_gain"]) == pytest.approx(0.3)
    assert float(off.params["memory_gain"]) == 0
    off_trace = trace_prompt(off, cfg, prompt)
    for key in ["keys", "queries", "values", "betas", "final_state"]:
        np.testing.assert_allclose(trace[key], off_trace[key], atol=1e-6)
    assert not np.allclose(trace["last_logits"], off_trace["last_logits"])
    assert np.all((np.asarray(trace["betas"]) > 0) & (np.asarray(trace["betas"]) < 1))


def test_canonical_regions_cover_every_prompt_byte_once():
    _, questions = group("Dan", "Ned", ["gold", "blue"], False, "g", "training")
    q = questions[0]
    prompt = f"<|user|>\n{q['context']}\n{q['question']}\n<|assistant|>\n"
    regions = question_regions(q)
    coverage = np.zeros(1 + len(prompt.encode()), dtype=int)
    for start, end in regions.values():
        coverage[start:end] += 1
    np.testing.assert_array_equal(coverage, 1)
    lo, hi = regions["fact_1"]
    assert prompt.encode()[lo - 1 : hi - 1].decode() == "Dan's color is gold."
    bad = {**q, "context": "different text"}
    with pytest.raises(ValueError):
        question_regions(bad)
