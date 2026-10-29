"""Causal delta-rule fast-state memory; no persistent database or label access."""

import jax.numpy as jnp


def delta_step(memory, key, value, query, beta, alpha=None, rev_key=None, rev_value=None):
    """Read the old matrix and correct its key/value mapping for future tokens."""
    decayed = memory if alpha is None else jnp.reshape(alpha, (memory.shape[0], -1, 1)) * memory
    read = jnp.einsum("bk,bkv->bv", query, memory)
    prediction = jnp.einsum("bk,bkv->bv", key, decayed)
    error = value - prediction
    updated = decayed + beta[:, None, None] * key[:, :, None] * error[:, None, :]
    if rev_key is not None and rev_value is not None:
        rev_pred = jnp.einsum("bk,bkv->bv", rev_key, updated)
        rev_err = rev_value - rev_pred
        updated = updated + beta[:, None, None] * rev_key[:, :, None] * rev_err[:, None, :]
    return updated, read
