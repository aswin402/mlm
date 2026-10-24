"""Causal delta-rule fast-state memory; no persistent database or label access."""

import jax.numpy as jnp


def delta_step(memory, key, value, query, beta):
    """Read the old matrix and correct its key/value mapping for future tokens."""
    read = jnp.einsum("bk,bkv->bv", query, memory)
    prediction = jnp.einsum("bk,bkv->bv", key, memory)
    error = value - prediction
    updated = memory + beta[:, None, None] * key[:, :, None] * error[:, None, :]
    return updated, read
