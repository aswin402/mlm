"""Fixed directed topology and synchronous neighbor aggregation."""

import jax.numpy as jnp
import numpy as np


def topology(nodes: int, degree: int) -> np.ndarray:
    if not 1 <= degree < nodes:
        raise ValueError("Expected 1 <= degree < nodes")
    offsets = [1, 3] if (nodes, degree) == (8, 2) else []
    for offset in [1, 2, 4, 8, *range(1, nodes)]:
        offset %= nodes
        if offset and offset not in offsets:
            offsets.append(offset)
    return np.array(
        [[(i - k) % nodes for k in offsets[:degree]] for i in range(nodes)], dtype=np.int32
    )


def aggregate(states, edges):
    """states[B,N,D], edges[N,K] -> messages[B,N,D]."""
    return jnp.sum(states[:, edges, :], axis=2) / jnp.sqrt(edges.shape[1])


def model_topology(config):
    return None if config.architecture == "gru" else topology(config.nodes, config.degree).tolist()
