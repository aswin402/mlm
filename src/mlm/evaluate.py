"""Frozen-parameter document evaluation with fresh recurrent state."""

import math

import jax
import jax.numpy as jnp

from mlm.data import make_batch
from mlm.training import score


def evaluate(state, config, documents, messages=True):
    compute = jax.jit(lambda s, b: score(s, b, config, messages))
    positions = payload_bytes = 0.0
    total_loss = payload_nll = 0.0
    for start in range(0, len(documents), config.batch_size):
        indices = list(range(start, min(start + config.batch_size, len(documents))))
        metrics = jax.device_get(
            compute(state, jax.tree.map(jnp.asarray, make_batch(documents, indices, config.length)))
        )
        positions += float(metrics["positions"])
        total_loss += float(metrics["loss"]) * float(metrics["positions"])
        payload_bytes += float(metrics["bytes"])
        payload_nll += float(metrics["byte_nll"])
    if positions == 0 or payload_bytes == 0:
        raise ValueError("Evaluation requires nonempty payload bytes")
    loss = total_loss / positions
    return {
        "loss": loss,
        "perplexity_with_markers": math.exp(loss),
        "bits_per_byte": payload_nll / payload_bytes / math.log(2),
        "payload_bytes": int(payload_bytes),
        "positions": int(positions),
        "documents": len(documents),
        "prefix_limit": config.length,
    }
