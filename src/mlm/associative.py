"""Causal delta-rule fast-state memory; no persistent database or label access."""

import jax.numpy as jnp


def delta_step(
    memory,
    key,
    value,
    query,
    beta,
    alpha=None,
    rev_key=None,
    rev_value=None,
    channel_alpha=None,
):
    """Read the old matrix and correct its key/value mapping for future tokens (KDA / DeltaNet)."""
    decayed = (
        memory
        if alpha is None
        else (
            alpha * memory
            if alpha.ndim == 3
            else jnp.reshape(alpha, (memory.shape[0], -1, 1)) * memory
        )
    )
    if channel_alpha is not None:
        decayed = decayed * jnp.reshape(channel_alpha, (memory.shape[0], 1, -1))
    read = jnp.einsum("bk,bkv->bv", query, memory)
    prediction = jnp.einsum("bk,bkv->bv", key, decayed)
    error = value - prediction
    updated = decayed + beta[:, None, None] * key[:, :, None] * error[:, None, :]
    if rev_key is not None and rev_value is not None:
        rev_pred = jnp.einsum("bk,bkv->bv", rev_key, updated)
        rev_err = rev_value - rev_pred
        updated = updated + beta[:, None, None] * rev_key[:, :, None] * rev_err[:, None, :]
    return updated, read
