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


def sinkhorn_edge_weights(weights, edges, n_nodes: int, iters: int = 3):
    """Project positive edge weights [B,N,K] onto the doubly-stochastic manifold (mHC)."""
    import jax

    sender_one_hot = jax.nn.one_hot(edges, n_nodes, dtype=weights.dtype)
    w = jnp.maximum(weights, 1e-6)
    for _ in range(iters):
        col_sums = jnp.einsum("bnk,nks->bs", w, sender_one_hot)
        edge_col_norm = jnp.einsum("bs,nks->bnk", col_sums, sender_one_hot)
        w = w / jnp.maximum(edge_col_norm, 1e-6)
        w = w / jnp.maximum(w.sum(axis=-1, keepdims=True), 1e-6)
    return 0.5 * weights + 0.5 * w


def sinkhorn_stream_mix(streams, iters: int = 4):
    """GLM-5.3-Flash 4-stream mHC Birkhoff doubly-stochastic hyper-connection mixer.

    Given `streams` of shape [B, S, D], computes state-conditioned pairwise cosine energy
    biased toward the identity matrix, projects onto the SxS Birkhoff polytope via Sinkhorn-Knopp
    iterations, and returns `(mixed_streams, mix_matrix)`.
    """
    import jax

    if isinstance(streams, (list, tuple)):
        streams = jnp.stack(streams, axis=1)
    s_cnt = streams.shape[1]
    norms = jax.lax.rsqrt(jnp.sum(jnp.square(streams), axis=-1, keepdims=True) + 1e-6)
    unit_s = streams * norms
    sim = jnp.einsum("bsd,btd->bst", unit_s, unit_s)
    logits = 2.5 * jnp.eye(s_cnt, dtype=streams.dtype)[None, :, :] + 0.35 * sim
    w = jax.nn.softmax(logits, axis=-1)
    for _ in range(iters):
        w = w / jnp.maximum(w.sum(axis=-2, keepdims=True), 1e-6)
        w = w / jnp.maximum(w.sum(axis=-1, keepdims=True), 1e-6)
    mixed = jnp.einsum("bst,btd->bsd", w, streams)
    return mixed, w


def model_topology(config):
    if config.architecture == "gru":
        return None
    if config.edges is not None:
        return [list(row) for row in config.edges]
    return topology(config.nodes, config.degree).tolist()
