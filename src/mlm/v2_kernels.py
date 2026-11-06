"""Low-level stateless JAX operators and spectral utilities for MLM v2.

Includes:
- Root Mean Square Layer Normalization (`rms_norm`) & LFM3 Sandwich Sublayer Bounding
- SenseNova-Loop / mHC Sinkhorn-Knopp Birkhoff projection (`sinkhorn_knopp_doubly_stochastic`)
- Rotary Position Embeddings (`apply_rope`, `_apply_rope_pos_v2`)
- Causal ShortConv (`gated_short_conv`) & NCM Coarse-to-Fine Phi Bridge (`ncm_tied_codebook_logits`)
- Parallel Gated DeltaNet / KDA (`parallel_gated_deltanet`)
- GLM-5.3-Flash DSA Lightning Indexer (`compute_dsa_lightning_index`, `dsa_sparse_attention`)
- Google Research TurboQuant KV Cache Quantizer (`turboquant_kv_tensor_jax`)
- Grouped-Query Attention & SwiGLU FFN (`swiglu_ffn`)
- Von Neumann Spectral Effective Rank (`compute_von_neumann_effective_rank_v2`)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from mlm.model_v2 import ConfigV2


def compute_von_neumann_effective_rank_v2(x_mat: np.ndarray | jnp.ndarray) -> float:
    """Compute Von Neumann Spectral Effective Rank `R_eff(X) = exp(H_VN(X))` of a 2D matrix."""
    arr = np.asarray(x_mat, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    _, s_vals, _ = np.linalg.svd(arr, full_matrices=False)
    s_sq = np.square(s_vals.astype(np.float64))
    p_spec = s_sq / max(float(np.sum(s_sq)), 1e-12)
    vn_entropy = float(-np.sum(p_spec * np.log(p_spec + 1e-12)))
    return float(np.exp(vn_entropy))


def rms_norm(x: jnp.ndarray, weight: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
    """Root Mean Square Layer Normalization in fp32 for numerical stability."""
    x_f32 = x.astype(jnp.float32)
    variance = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
    normed = x_f32 * jax.lax.rsqrt(variance + eps)
    return (normed * weight.astype(jnp.float32)).astype(x.dtype)


def sandwich_rms_bound(
    dx: jnp.ndarray,
    post_scale: jnp.ndarray | None = None,
    clip_rms: float = 3.0,
    eps: float = 1e-5,
) -> jnp.ndarray:
    """LFM3 Sandwich Sublayer Bounding: suppress activation spikes while preserving identity."""
    out = dx.astype(jnp.float32)
    if post_scale is not None:
        out = out * post_scale.astype(jnp.float32)[None, None, :]
    if clip_rms > 0.0:
        rms = jnp.sqrt(jnp.mean(jnp.square(out), axis=-1, keepdims=True) + eps)
        factor = jnp.minimum(1.0, float(clip_rms) / rms)
        out = out * factor
    return out.astype(dx.dtype)


def sinkhorn_knopp_2x2(logits_2x2: jnp.ndarray, iters: int = 6) -> jnp.ndarray:
    """SenseNova-Loop / mHC (`arXiv:2512.24880`): Sinkhorn-Knopp Birkhoff polytope projection.

    Projects a raw `(..., 2, 2)` matrix onto the manifold of doubly stochastic matrices
    (non-negative entries with rows and columns summing to 1.0, guaranteeing spectral norm <= 1).
    """
    mat = jnp.exp(jnp.clip(logits_2x2.astype(jnp.float32), -8.0, 8.0))
    for _ in range(iters):
        mat = mat / jnp.maximum(jnp.sum(mat, axis=-1, keepdims=True), 1e-6)
        mat = mat / jnp.maximum(jnp.sum(mat, axis=-2, keepdims=True), 1e-6)
    return mat.astype(logits_2x2.dtype)


def mhc_read_stream(streams: jnp.ndarray, pre_logits: jnp.ndarray) -> jnp.ndarray:
    """Read width-`D` state `(B, T, D)` from 2-stream `mHC` state `(B, T, 2, D)`."""
    w_pre = jax.nn.softmax(pre_logits.astype(jnp.float32), axis=-1).astype(streams.dtype)
    return jnp.einsum("s,btsd->btd", w_pre, streams)


def mhc_write_stream(
    streams: jnp.ndarray,
    dy: jnp.ndarray,
    res_logits: jnp.ndarray,
    post_weights: jnp.ndarray,
    sinkhorn_iters: int = 6,
) -> jnp.ndarray:
    """Update 2-stream `mHC` state `(B, T, 2, D)` with Birkhoff stream mixing and write-back."""
    h_res = sinkhorn_knopp_2x2(res_logits, iters=sinkhorn_iters)
    mixed = jnp.einsum("oi,btid->btod", h_res, streams)
    w_post = post_weights.astype(streams.dtype)[None, None, :, None]
    return mixed + w_post * dy[:, :, None, :]


def build_layer_execution_schedule(config: ConfigV2) -> list[tuple[int, int]]:
    """Build `(layer_idx, pass_idx)` schedule with SenseNova-Loop Contiguous Hybrid Looping."""
    n = config.n_layers
    s = config.loop_start_layer
    e = config.loop_end_layer
    r = config.loop_repeats
    if r <= 1 or s >= e or s < 0 or e > n:
        return [(i, 0) for i in range(n)]
    schedule: list[tuple[int, int]] = [(i, 0) for i in range(s)]
    for p_idx in range(r):
        for i in range(s, e):
            schedule.append((i, p_idx))
    for i in range(e, n):
        schedule.append((i, 0))
    return schedule


def _default_mhc_tensors(
    max_passes: int = 2,
    symmetry_break: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Create `(res, pre, post)` mHC tensors of shapes `(P, 2, 2)`, `(P, 2)`, `(P, 2)`."""
    res_np = np.zeros((max_passes, 2, 2), dtype=np.float32)
    pre_np = np.zeros((max_passes, 2), dtype=np.float32)
    post_np = np.ones((max_passes, 2), dtype=np.float32)
    for p_idx in range(max_passes):
        res_np[p_idx] = np.array([[2.5, -2.5], [-2.5, 2.5]], dtype=np.float32)
        if symmetry_break:
            if p_idx % 2 == 0:
                pre_np[p_idx] = np.array([0.5, -0.5], dtype=np.float32)
                post_np[p_idx] = np.array([1.12, 0.88], dtype=np.float32)
            else:
                pre_np[p_idx] = np.array([-0.5, 0.5], dtype=np.float32)
                post_np[p_idx] = np.array([0.72, 0.92], dtype=np.float32)
    return jnp.asarray(res_np), jnp.asarray(pre_np), jnp.asarray(post_np)


def precompute_rope_freqs(
    seq_len: int,
    head_dim: int,
    theta: float = 10000.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Precompute cosine and sine Rotary Position Embedding tables `(seq_len, head_dim // 2)`."""
    half = head_dim // 2
    freqs = 1.0 / (theta ** (jnp.arange(0, half, dtype=jnp.float32) / float(half)))
    positions = jnp.arange(seq_len, dtype=jnp.float32)
    angles = jnp.outer(positions, freqs)
    return jnp.cos(angles), jnp.sin(angles)


def apply_rope(
    x: jnp.ndarray,
    cos: jnp.ndarray,
    sin: jnp.ndarray,
) -> jnp.ndarray:
    """Apply Rotary Position Embeddings to `x` of shape `(B, T, H, D_h)`."""
    t_len = x.shape[1]
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    c = cos[:t_len][None, :, None, :]
    s = sin[:t_len][None, :, None, :]
    out1 = x1 * c - x2 * s
    out2 = x2 * c + x1 * s
    return jnp.concatenate([out1, out2], axis=-1)


def causal_short_conv(x: jnp.ndarray, conv_weight: jnp.ndarray) -> jnp.ndarray:
    """Vectorized causal 1D depthwise short convolution over `(B, T, D)` with kernel `(K, D)`."""
    k_size = conv_weight.shape[0]
    padded = jnp.pad(x, ((0, 0), (k_size - 1, 0), (0, 0)))
    t_len = x.shape[1]
    acc = jnp.zeros_like(x)
    for k in range(k_size):
        acc = acc + padded[:, k : k + t_len, :] * conv_weight[k][None, None, :]
    return acc


def ncm_phi_bridge(
    x_in: jnp.ndarray,
    wle_vec: jnp.ndarray,
    phi_conv_w: jnp.ndarray,
    phi_ratio_logit: jnp.ndarray,
) -> jnp.ndarray:
    """NCM (`arXiv:2610.14162`) Coarse-to-Fine `Phi` Concept Bridge & Scale Embedding `wle`."""
    pooled = causal_short_conv(x_in, phi_conv_w) + wle_vec[None, None, :]
    r = (0.25 * jax.nn.sigmoid(phi_ratio_logit.astype(jnp.float32)))[None, None, :].astype(
        x_in.dtype
    )
    return (1.0 - r) * x_in + r * pooled


def ncm_tied_codebook_logits(
    hidden_normed: jnp.ndarray,
    out_w: jnp.ndarray,
    log_scale: jnp.ndarray | None = None,
    norm_alpha: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """NCM (`arXiv:2610.14162`) Norm-Corrected Tied Codebook Head (`_TiedCodebookHead`)."""
    raw_logits = jnp.matmul(hidden_normed, jnp.swapaxes(out_w, 0, 1))
    if log_scale is None or norm_alpha is None:
        return raw_logits
    e_sq = jnp.sum(jnp.square(out_w.astype(jnp.float32)), axis=-1)  # (V,)
    d_scale = jnp.maximum(jnp.mean(e_sq), 1e-6)
    alpha = 0.25 * jnp.tanh(norm_alpha.astype(jnp.float32))[0]
    inv_temp = jnp.exp(jnp.clip(log_scale.astype(jnp.float32)[0], -0.5, 0.5))
    debiased = raw_logits.astype(jnp.float32) - alpha * (e_sq / d_scale)[None, None, :]
    return (debiased * inv_temp).astype(raw_logits.dtype)


def parallel_gated_deltanet(
    x: jnp.ndarray,
    wq_d: jnp.ndarray,
    wk_d: jnp.ndarray,
    wv_d: jnp.ndarray,
    w_decay: jnp.ndarray,
    n_heads: int,
    w_beta: jnp.ndarray | None = None,
    channel_gate: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """GLM-5.3-Flash Kimi Delta Attention (`KDA`) / Gated DeltaNet operator over `(B, T, D)`."""
    bsz, t_len, d_model = x.shape
    head_dim = d_model // n_heads
    q_raw = x @ wq_d
    k_raw = x @ wk_d
    if channel_gate is not None:
        c_mod = jnp.clip(channel_gate.astype(x.dtype), 0.1, 2.5)[None, None, :]
        q_raw = q_raw * c_mod
        k_raw = k_raw * c_mod
    q = q_raw.reshape(bsz, t_len, n_heads, head_dim)
    k = k_raw.reshape(bsz, t_len, n_heads, head_dim)
    v = (x @ wv_d).reshape(bsz, t_len, n_heads, head_dim)

    # L2-normalize keys and queries for stable delta-rule eigen-spectrum
    q = q * jax.lax.rsqrt(jnp.sum(jnp.square(q), axis=-1, keepdims=True) + 1e-6)
    k = k * jax.lax.rsqrt(jnp.sum(jnp.square(k), axis=-1, keepdims=True) + 1e-6)

    # Log-space cumulative decay mask for parallel causal associative attention
    log_alpha = -jax.nn.softplus(x @ w_decay)  # (B, T, H)
    if w_beta is not None:
        beta = 2.0 * jax.nn.sigmoid(x @ w_beta)  # (B, T, H), 1.0 at zero init
        k_prev = jnp.pad(k[:, :-1, :, :], ((0, 0), (1, 0), (0, 0), (0, 0)))
        v_prev = jnp.pad(v[:, :-1, :, :], ((0, 0), (1, 0), (0, 0), (0, 0)))
        k_sim = jax.nn.relu(jnp.sum(k * k_prev, axis=-1)) * jnp.exp(log_alpha)
        v = beta[..., None] * (v - 0.20 * k_sim[..., None] * v_prev)
    cum_log_alpha = jnp.cumsum(log_alpha, axis=1)  # (B, T, H)
    # Pairwise decay difference: cum_log_alpha[:, i] - cum_log_alpha[:, j] for i >= j
    diff = cum_log_alpha[:, :, None, :] - cum_log_alpha[:, None, :, :]  # (B, T, T, H)
    causal_mask = jnp.tril(jnp.ones((t_len, t_len), dtype=jnp.bool_))[None, :, :, None]
    decay_matrix = jnp.where(causal_mask, jnp.exp(jnp.clip(diff, -30.0, 0.0)), 0.0)
    decay_matrix = jnp.transpose(decay_matrix, (0, 3, 1, 2))  # (B, H, T, T)

    q_h = jnp.transpose(q, (0, 2, 1, 3))  # (B, H, T, D_h)
    k_h = jnp.transpose(k, (0, 2, 1, 3))  # (B, H, T, D_h)
    v_h = jnp.transpose(v, (0, 2, 1, 3))  # (B, H, T, D_h)

    scores = jnp.matmul(q_h, jnp.swapaxes(k_h, -1, -2)) * decay_matrix  # (B, H, T, T)
    out_h = jnp.matmul(scores, v_h)  # (B, H, T, D_h)
    return jnp.transpose(out_h, (0, 2, 1, 3)).reshape(bsz, t_len, d_model)


def dsa_lightning_index_mask(
    logits: jnp.ndarray,
    sliding_window: int,
    dsa_top_k: int = 16,
    shared_index_scores: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """GLM-5.3-Flash `DSA` Lightning Indexer + `IndexShare` sparse attention mask."""
    t_len = logits.shape[-1]
    n_heads = logits.shape[1]
    idx = jnp.arange(t_len)
    dist = idx[:, None] - idx[None, :]  # (T, T)
    global_causal = (dist >= 0)[None, None, :, :]
    swa_causal = ((dist >= 0) & (dist < sliding_window))[None, None, :, :]

    causal_logits = jnp.where(global_causal, logits, -1e9)
    cur_scores = jnp.mean(causal_logits, axis=1)  # (B, T, T)
    index_scores = (
        0.5 * shared_index_scores + 0.5 * cur_scores
        if shared_index_scores is not None
        else cur_scores
    )
    if dsa_top_k > 0:
        k_sel = min(int(dsa_top_k), t_len)
        _, top_idx = jax.lax.top_k(index_scores, k_sel)  # (B, T, k_sel)
        dsa_oh = jnp.any(jax.nn.one_hot(top_idx, t_len, dtype=jnp.bool_), axis=-2)
        dsa_causal = dsa_oh[:, None, :, :] & global_causal
        sparse_causal = swa_causal | dsa_causal
    else:
        sparse_causal = swa_causal

    is_swa_head = (jnp.arange(n_heads) % 2 == 1)[None, :, None, None]
    mask = jnp.where(is_swa_head, sparse_causal, global_causal)
    return mask, index_scores


_TQ_4BIT_CENTROIDS = (
    -2.7326,
    -2.0690,
    -1.6180,
    -1.2562,
    -0.9423,
    -0.6568,
    -0.3880,
    -0.1284,
    0.1284,
    0.3880,
    0.6568,
    0.9423,
    1.2562,
    1.6180,
    2.0690,
    2.7326,
)
_TQ_2BIT_CENTROIDS = (-1.5104, -0.4528, 0.4528, 1.5104)


def _fwht_last_axis_jax(x: jnp.ndarray) -> jnp.ndarray:
    """Normalized Fast Walsh-Hadamard Transform along the last power-of-2 axis."""
    dim = int(x.shape[-1])
    prefix = x.shape[:-1]
    h = 1
    out = x
    while h < dim:
        out = out.reshape(*prefix, dim // (2 * h), 2, h)
        a = out[..., 0, :]
        b = out[..., 1, :]
        out = jnp.concatenate([a + b, a - b], axis=-1)
        h *= 2
    return out.reshape(*prefix, dim) * (1.0 / jnp.sqrt(float(dim)))


def turboquant_kv_tensor_jax(
    kv: jnp.ndarray,
    bits: int,
    seed_offset: int = 0,
) -> jnp.ndarray:
    """Google Research ICLR 2026 `TurboQuant` 4-Bit / 2-Bit KV Cache Quantizer (`arXiv:2504.19874`).

    Applies Fast Walsh-Hadamard Orthogonal Rotation (`FWHT`) + Gaussian Lloyd-Max 4-bit/2-bit
    scalar quantization + 1-Bit Quantized Johnson-Lindenstrauss (`QJL`) residual sign correction
    along the head dimension `D_h`.
    """
    if bits not in (1, 2, 4, 8):
        return kv
    orig_dim = int(kv.shape[-1])
    pad_dim = 1
    while pad_dim < orig_dim:
        pad_dim *= 2
    if pad_dim > orig_dim:
        kv_pad = jnp.pad(kv, [(0, 0)] * (kv.ndim - 1) + [(0, pad_dim - orig_dim)])
    else:
        kv_pad = kv

    idx_arr = np.arange(pad_dim, dtype=np.uint32) + np.uint32(seed_offset * 7919 + 1337)
    signs_np = np.where(((idx_arr * np.uint32(2654435761)) >> np.uint32(31)) == 0, 1.0, -1.0)
    signs = jnp.asarray(signs_np, dtype=kv.dtype)

    rot = _fwht_last_axis_jax(kv_pad * signs)
    if bits == 8:
        abs_max = jnp.maximum(jnp.max(jnp.abs(rot), axis=-1, keepdims=True), 1e-6)
        scale = abs_max / 127.0
        q_rot = jnp.clip(jnp.round(rot / scale), -127.0, 127.0) * scale
    elif bits == 1:
        # BiLLM + FWHT Sub-Group (G=8) L2-Preserving 1-Bit Sign + 1-Bit QJL Residual Sign
        g_shape = (*rot.shape[:-1], pad_dim // 8, 8)
        rot_g = rot.reshape(g_shape)
        rms = jnp.sqrt(jnp.mean(jnp.square(rot_g), axis=-1, keepdims=True) + 1e-12)
        a1_raw = 0.9816 * rms
        b1 = jnp.where(rot_g >= 0.0, 1.0, -1.0)
        res = rot_g - a1_raw * b1
        b2 = jnp.where(res >= 0.0, 1.0, -1.0)
        a2_raw = jnp.mean(res * b2, axis=-1, keepdims=True)
        raw_recon = a1_raw * b1 + a2_raw * b2
        recon_rms = jnp.sqrt(jnp.mean(jnp.square(raw_recon), axis=-1, keepdims=True) + 1e-12)
        q_rot = (raw_recon * (rms / recon_rms)).reshape(rot.shape)
    else:
        rms = jnp.sqrt(jnp.mean(jnp.square(rot), axis=-1, keepdims=True) + 1e-12)
        z = rot / rms
        centroids = jnp.asarray(
            _TQ_4BIT_CENTROIDS if bits == 4 else _TQ_2BIT_CENTROIDS,
            dtype=kv.dtype,
        )
        bounds = 0.5 * (centroids[:-1] + centroids[1:])
        q_idx = jnp.sum(z[..., None] > bounds, axis=-1)
        recon_z = centroids[q_idx]
        # 1-Bit QJL residual sign correction for unbiased inner products
        res = z - recon_z
        res_scale = jnp.mean(jnp.abs(res), axis=-1, keepdims=True)
        qjl_sign = jnp.where(res >= 0.0, 1.0, -1.0)
        q_rot = (recon_z + res_scale * qjl_sign) * rms

    inv_pad = _fwht_last_axis_jax(q_rot) * signs
    kv_hat = inv_pad[..., :orig_dim]
    return kv + jax.lax.stop_gradient(kv_hat - kv)


def grouped_query_attention(
    x: jnp.ndarray,
    wq: jnp.ndarray,
    wk: jnp.ndarray,
    wv: jnp.ndarray,
    wo: jnp.ndarray,
    q_norm_w: jnp.ndarray,
    k_norm_w: jnp.ndarray,
    cos: jnp.ndarray,
    sin: jnp.ndarray,
    n_heads: int,
    n_kv_heads: int,
    rms_eps: float = 1e-5,
    attn_gate_w: jnp.ndarray | None = None,
    attn_gate_b: jnp.ndarray | None = None,
    sliding_window: int = 0,
    dsa_top_k: int = 16,
    shared_dsa_scores: jnp.ndarray | None = None,
    kv_cache_bits: int = 0,
) -> jnp.ndarray:
    """LFM3 + GLM-5.3-Flash `DSA` Grouped-Query Attention with QK-RMSNorm + RoPE + TurboQuant KV."""
    bsz, t_len, d_model = x.shape
    head_dim = d_model // n_heads
    kv_group_size = n_heads // n_kv_heads

    q = (x @ wq).reshape(bsz, t_len, n_heads, head_dim)
    k = (x @ wk).reshape(bsz, t_len, n_kv_heads, head_dim)
    v = (x @ wv).reshape(bsz, t_len, n_kv_heads, head_dim)

    q = rms_norm(q, q_norm_w, rms_eps)
    k = rms_norm(k, k_norm_w, rms_eps)

    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)

    if kv_cache_bits in (1, 2, 4, 8):
        k = turboquant_kv_tensor_jax(k, bits=kv_cache_bits, seed_offset=1)
        v = turboquant_kv_tensor_jax(v, bits=kv_cache_bits, seed_offset=2)

    if kv_group_size > 1:
        k = jnp.repeat(k, kv_group_size, axis=2)
        v = jnp.repeat(v, kv_group_size, axis=2)

    q_t = jnp.transpose(q, (0, 2, 1, 3))  # (B, H, T, D_h)
    k_t = jnp.transpose(k, (0, 2, 1, 3))  # (B, H, T, D_h)
    v_t = jnp.transpose(v, (0, 2, 1, 3))  # (B, H, T, D_h)

    scale = 1.0 / jnp.sqrt(float(head_dim))
    logits = jnp.matmul(q_t, jnp.swapaxes(k_t, -1, -2)) * scale
    if sliding_window > 0:
        mask, _ = dsa_lightning_index_mask(
            logits,
            sliding_window=sliding_window,
            dsa_top_k=dsa_top_k,
            shared_index_scores=shared_dsa_scores,
        )
    else:
        mask = jnp.tril(jnp.ones((t_len, t_len), dtype=jnp.bool_))[None, None, :, :]
    logits = jnp.where(mask, logits, -1e9)
    weights = jax.nn.softmax(logits, axis=-1)
    attn_out = jnp.matmul(weights, v_t)  # (B, H, T, D_h)
    attn_heads = jnp.transpose(attn_out, (0, 2, 1, 3))  # (B, T, H, D_h)
    if attn_gate_w is not None:
        gate_logits = x @ attn_gate_w  # (B, T, H)
        if attn_gate_b is not None:
            gate_logits = gate_logits + attn_gate_b[None, None, :]
        head_gates = 2.0 * jax.nn.sigmoid(gate_logits)[..., None]  # (B, T, H, 1), 1.0 at zero init
        attn_heads = attn_heads * head_gates
    merged = attn_heads.reshape(bsz, t_len, d_model)
    return merged @ wo


def swiglu_ffn(
    x: jnp.ndarray,
    w_gate: jnp.ndarray,
    w_up: jnp.ndarray,
    w_down: jnp.ndarray,
) -> jnp.ndarray:
    """SwiGLU Feed-Forward Network `w_down(silu(x @ w_gate) * (x @ w_up))`."""
    gate = jax.nn.silu(x @ w_gate)
    up = x @ w_up
    return (gate * up) @ w_down
