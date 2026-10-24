"""Read-only memory diagnostics, with overwrite-aware conditional read decomposition."""

import math

import jax
import jax.numpy as jnp
import numpy as np

from mlm.associative import delta_step
from mlm.data import BOS
from mlm.model import initial_state


def effective_coefficients(keys, betas, query, read_index):
    keys = np.asarray(keys, dtype=np.float64)
    betas = np.asarray(betas, dtype=np.float64)
    survivor = np.asarray(query, dtype=np.float64).copy()
    if keys.ndim != 2 or betas.shape != (len(keys),) or survivor.shape != (keys.shape[1],):
        raise ValueError("Expected keys[T,K], betas[T] and query[K]")
    if not 0 <= read_index <= len(keys):
        raise ValueError("Read index outside write sequence")
    coefficients = np.zeros(len(keys), dtype=np.float64)
    for s in range(read_index - 1, -1, -1):
        coefficients[s] = betas[s] * np.dot(survivor, keys[s])
        survivor -= coefficients[s] * keys[s]
    return coefficients


def read_disabled(state):
    if "memory_gain" not in state.params:
        raise ValueError("Delta memory parameters required")
    params = dict(state.params)
    params["memory_gain"] = jnp.zeros_like(params["memory_gain"])
    return state.replace(params=params)


def question_regions(question):
    """Offline annotations for canonical synthetic facts; never passed into the network."""
    sentences = [f"{n}'s color is {v}." for n, v in question["facts"]]
    if len(sentences) != 2 or " ".join(sentences) != question["context"]:
        raise ValueError("Canonical two-fact context required for annotations")
    header_end = 1 + len("<|user|>\n".encode())
    fact1_end = header_end + len(sentences[0].encode())
    context_end = header_end + len(question["context"].encode())
    query_end = context_end + 1 + len(question["question"].encode())
    end = query_end + len("\n<|assistant|>\n".encode())
    return {
        "bos": (0, 1),
        "user_header": (1, header_end),
        "fact_1": (header_end, fact1_end),
        "fact_2": (fact1_end, context_end),
        "question": (context_end, query_end),
        "assistant_header": (query_end, end),
    }


def trace_prompt(state, config, prompt):
    """Trace frozen neural execution from prompt bytes; no facts/labels are accepted."""
    if config.memory != "delta" or config.node_readout != "sum":
        raise ValueError("Delta memory with sum readout required")
    ids = [BOS, *prompt.encode("utf-8")]
    if len(ids) > 1024:
        raise ValueError("Diagnostic prompt exceeds1024 input positions")

    @jax.jit
    def advance(params, tokens, carry, resets):
        logits, next_state = state.apply_fn({"params": params}, tokens, carry, resets)
        graph = next_state[:, : config.nodes, :]
        pooled = jnp.einsum("bni,nij->bj", graph, params["read_kernel"]) / math.sqrt(config.nodes)
        features = jnp.concatenate([params["embedding"][tokens[:, 0]], jnp.tanh(pooled)], -1)
        key = jnp.tanh(features @ params["memory_key_kernel"])
        query = jnp.tanh(features @ params["memory_query_kernel"])
        key *= jax.lax.rsqrt(jnp.square(key).sum(-1, keepdims=True) + 1e-6)
        query *= jax.lax.rsqrt(jnp.square(query).sum(-1, keepdims=True) + 1e-6)
        value = jnp.tanh(features @ params["memory_value_kernel"])
        beta = jax.nn.sigmoid(
            features @ params["memory_write_kernel"] + params["memory_write_bias"]
        )[:, 0]
        old = jnp.where(resets[:, :, None], 0.0, carry[:, config.nodes :, :])
        predicted_memory, recalled = delta_step(old, key, value, query, beta)
        residual = params["memory_gain"] * (recalled @ params["memory_output_kernel"])
        z = pooled + residual
        z = (z - z.mean(-1, keepdims=True)) * jax.lax.rsqrt(
            jnp.square(z - z.mean(-1, keepdims=True)).mean(-1, keepdims=True) + 1e-5
        )
        reconstructed = (z * params["norm_scale"] + params["norm_bias"]) @ params[
            "vocab_kernel"
        ] + params["vocab_bias"]
        aux = {
            "key": key[0],
            "query": query[0],
            "value": value[0],
            "beta": beta[0],
            "recalled": recalled[0],
            "base_norm": jnp.linalg.norm(pooled),
            "residual_norm": jnp.linalg.norm(residual),
            "memory_norm": jnp.linalg.norm(next_state[:, config.nodes :, :]),
            "memory_error": jnp.max(jnp.abs(predicted_memory - next_state[:, config.nodes :, :])),
            "logit_error": jnp.max(jnp.abs(reconstructed - logits[:, 0, :])),
        }
        return next_state, logits[0, 0], aux

    carry = initial_state(config, 1)
    rows = []
    for i, token in enumerate(ids):
        carry, logits, raw = advance(
            state.params,
            jnp.array([[token]], dtype=jnp.int32),
            carry,
            jnp.array([[i == 0]], dtype=bool),
        )
        rows.append(jax.device_get(raw))
    keys = np.stack([r["key"] for r in rows])
    queries = np.stack([r["query"] for r in rows])
    values = np.stack([r["value"] for r in rows])
    betas = np.array([r["beta"] for r in rows])
    coefficients = effective_coefficients(keys, betas, queries[-1], len(ids) - 1)
    recalled = np.asarray(rows[-1]["recalled"])
    reconstruction = coefficients @ values
    if not np.allclose(reconstruction, recalled, atol=1e-5, rtol=1e-5):
        raise AssertionError("Conditional read reconstruction failed")
    memory_error = max(float(r["memory_error"]) for r in rows)
    logit_error = max(float(r["logit_error"]) for r in rows)
    if memory_error > 1e-5 or logit_error > 1e-4:
        raise AssertionError(
            f"Model reconstruction mismatch: memory={memory_error}, logits={logit_error}"
        )
    full_tokens = jnp.array([ids], dtype=jnp.int32)
    full_resets = jnp.zeros_like(full_tokens, dtype=bool).at[:, 0].set(True)
    full_logits, full_state = state.apply_fn(
        {"params": state.params}, full_tokens, initial_state(config, 1), full_resets
    )
    if not np.allclose(np.asarray(logits), np.asarray(full_logits[0, -1]), atol=1e-5, rtol=1e-5):
        raise AssertionError("Step trace differs from full-prompt prediction")
    if not np.allclose(np.asarray(carry), np.asarray(full_state), atol=1e-5, rtol=1e-5):
        raise AssertionError("Step trace differs from full-prompt state")
    result = {
        "prompt": prompt,
        "token_ids": ids,
        "keys": keys.tolist(),
        "queries": queries.tolist(),
        "values": values.tolist(),
        "betas": betas.tolist(),
        "effective_read_coefficients": coefficients.tolist(),
        "conditional_write_contribution_l2": np.linalg.norm(
            coefficients[:, None] * values, axis=1
        ).tolist(),
        "last_recalled": recalled.tolist(),
        "last_logits": np.asarray(logits).tolist(),
        "final_state": np.asarray(carry[0]).tolist(),
        "memory_gain": float(state.params["memory_gain"]),
        "base_readout_norms": [float(r["base_norm"]) for r in rows],
        "memory_residual_norms": [float(r["residual_norm"]) for r in rows],
        "memory_norms": [float(r["memory_norm"]) for r in rows],
        "full_prompt_logit_max_difference": float(
            np.max(np.abs(np.asarray(logits) - np.asarray(full_logits[0, -1])))
        ),
        "full_prompt_state_max_difference": float(
            np.max(np.abs(np.asarray(carry) - np.asarray(full_state)))
        ),
        "max_memory_reconstruction_error": memory_error,
        "max_logit_reconstruction_error": logit_error,
        "read_reconstruction_error": float(np.max(np.abs(reconstruction - recalled))),
        "scope": (
            "conditional signed write decomposition; "
            "not probabilities or full input causal attribution"
        ),
    }
    return result
