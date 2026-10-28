"""Fixed directed topology and synchronous neighbor aggregation."""

import jax.numpy as jnp
import numpy as np


def topology(nodes: int, degree: int) -> np.ndarray:
    if not 1 <= degree < nodes:
        raise ValueError("Expected 1 <= degree < nodes")
    offsets: list[int] = []
    strides = [*(1 << k for k in range(nodes.bit_length())), *range(1, nodes)]
    for offset in strides:
        offset %= nodes
        if offset and offset not in offsets:
            offsets.append(offset)
    return np.array(
        [[(i - k) % nodes for k in offsets[:degree]] for i in range(nodes)], dtype=np.int32
    )


def aggregate(states, edges, weights=None):
    """states[B,N,D], edges[N,K], optional weights[B,N,K] -> messages[B,N,D]."""
    neighbors = states[:, edges, :]
    if weights is None:
        return jnp.sum(neighbors, axis=2) / jnp.sqrt(edges.shape[1])
    return jnp.einsum("bnk,bnkd->bnd", weights, neighbors) * jnp.sqrt(edges.shape[1])


def model_topology(config):
    return None if config.architecture == "gru" else topology(config.nodes, config.degree).tolist()
