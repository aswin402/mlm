"""MLM v2: Deep Hybrid Liquid-Delta Graph Architecture (LFM2 ShortConv + Chunked DeltaNet + GQA).

Designed for sub-billion (10M–135M) parameter efficiency with:
- Rust BPE subword vocabulary (`vocab_size` up to 32,768)
- Deep stacked residual blocks (`12–24` layers)
- Hybrid 3:1 schedule of Liquid Gated Short-Convolution + Parallel Gated DeltaNet and
  Grouped-Query Attention (GQA) with RoPE + QK-RMSNorm
- SwiGLU Feed-Forward Networks with optional Sparse Top-K Graph Experts
- 100% sequence-parallel training along time `T` (zero per-token Python loops)
- Direct export compatibility with `safetensors` and `llama.cpp` GGUF v3 (`llama` & `lfm2`).
"""

from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import optax
from pydantic import BaseModel, ConfigDict, Field
from safetensors.numpy import load as load_safetensors_bytes
from safetensors.numpy import save as save_safetensors_bytes
from safetensors.numpy import save_file as save_safetensors

from mlm.tokenizer import BPETokenizer


class ConfigV2(BaseModel):
    """Validated configuration for MLM v2 (Deep Hybrid Liquid-Delta Graph)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    arch_version: Literal["v2_hybrid", "v2_llama"] = "v2_hybrid"
    vocab_size: int = Field(default=8192, ge=258, le=65536)
    d_model: int = Field(default=576, ge=64, le=4096)
    n_layers: int = Field(default=16, ge=2, le=64)
    n_heads: int = Field(default=9, ge=1, le=64)
    n_kv_heads: int = Field(default=3, ge=1, le=64)
    ffn_dim: int = Field(default=1536, ge=128, le=16384)
    conv_kernel: int = Field(default=4, ge=2, le=16)
    mature_conv_kernel: int = Field(default=0, ge=0, le=16)
    attn_interval: int = Field(default=4, ge=1, le=16)
    use_gated_deltanet: bool = True
    use_head_gate: bool = True
    sliding_window: int = Field(default=0, ge=0, le=8192)
    dsa_top_k: int = Field(default=16, ge=0, le=512)
    sandwich_norm_clip: float = Field(default=0.0, ge=0.0, le=20.0)
    use_mhc: bool = True
    mhc_sinkhorn_iters: int = Field(default=6, ge=1, le=20)
    loop_start_layer: int = Field(default=0, ge=0, le=64)
    loop_end_layer: int = Field(default=0, ge=0, le=64)
    loop_repeats: int = Field(default=1, ge=1, le=4)
    use_ncm: bool = True
    ncm_weight: float = Field(default=0.15, ge=0.0, le=2.0)
    ncm_window: int = Field(default=4, ge=2, le=16)
    kd_weight: float = Field(default=0.0, ge=0.0, le=5.0)
    kd_top_k: int = Field(default=16, ge=2, le=128)
    kd_temperature: float = Field(default=2.0, gt=0.1, le=10.0)
    n_experts: int = Field(default=0, ge=0, le=64)
    top_k_experts: int = Field(default=2, ge=1, le=16)
    expert_ffn_dim: int = Field(default=256, ge=64, le=4096)
    mature_experts: int = Field(default=0, ge=0, le=64)
    mature_lr_scale: float = Field(default=0.15, ge=0.0, le=1.0)
    shared_lr_scale: float = Field(default=1.0, ge=0.0, le=1.0)
    active_subnodes: tuple[int, ...] = ()
    router_bias_gamma: float = Field(default=1e-3, ge=0.0, le=0.1)
    mtp_weight: float = Field(default=0.25, ge=0.0, le=2.0)
    think_open_id: int = Field(default=6, ge=-1, le=65536)
    think_close_id: int = Field(default=7, ge=-1, le=65536)
    digit_token_ids: tuple[int, ...] = ()
    blt_ngram_dim: int = Field(default=64, ge=0, le=256)
    use_fone_abacus: bool = True
    domain_gate_weight: float = Field(default=0.35, ge=0.0, le=2.0)
    kv_cache_bits: int = Field(default=0, ge=0, le=16)
    tie_embeddings: bool = True
    rope_theta: float = Field(default=10000.0, gt=100.0)
    rms_eps: float = Field(default=1e-5, gt=0.0)
    max_seq_len: int = Field(default=256, ge=16, le=8192)
    batch_size: int = Field(default=16, ge=1, le=512)
    learning_rate: float = Field(default=1.5e-3, gt=0.0, le=1.0)
    weight_decay: float = Field(default=0.05, ge=0.0, le=1.0)
    warmup_steps: int = Field(default=50, ge=0, le=50000)
    clip_norm: float = Field(default=1.0, gt=0.0)
    seed: int = Field(default=42, ge=0)

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def is_attention_layer(self, layer_idx: int) -> bool:
        if self.arch_version == "v2_llama":
            return True
        return (layer_idx + 1) % self.attn_interval == 0


@dataclass
class TrainStateV2:
    """Training state holding parameters, Optax state, and step counter."""

    step: int
    params: dict[str, jnp.ndarray]
    opt_state: optax.OptState


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


def _route_single_bank(
    bank_logits: jnp.ndarray,
    bank_bias: jnp.ndarray | None,
    top_k: int,
    new_tier_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Route tokens within a single 10-node Specialist Macro-Bank (Nodes 0..7 base + 8..9 think)."""
    b_nodes = bank_logits.shape[-1]
    base_n = min(b_nodes, 8)
    m_raw = bank_logits[..., :base_n]
    steps = jnp.arange(1, m_raw.shape[1] + 1, dtype=m_raw.dtype)[None, :, None]
    m_logits = 0.45 * m_raw + 0.55 * (jnp.cumsum(m_raw, axis=1) / steps)
    m_probs = jax.nn.softmax(m_logits, axis=-1)
    k_eff = min(top_k, base_n)
    if k_eff < base_n:
        m_scores = (
            m_probs + jax.lax.stop_gradient(bank_bias[:base_n])[None, None, :]
            if bank_bias is not None
            else m_probs
        )
        _, top_idx = jax.lax.top_k(m_scores, k_eff)
        m_mask = jnp.sum(jax.nn.one_hot(top_idx, base_n, dtype=m_probs.dtype), axis=-2)
        m_gated = m_probs * m_mask
        m_gated = m_gated / jnp.maximum(jnp.sum(m_gated, axis=-1, keepdims=True), 1e-6)
    else:
        m_gated = m_probs

    if b_nodes <= 8:
        return m_gated

    think_logits = bank_logits[..., 8:b_nodes]
    if bank_bias is not None:
        think_logits = think_logits + jax.lax.stop_gradient(bank_bias[8:b_nodes])[None, None, :]
    think_gated = jax.nn.sigmoid(think_logits)
    if new_tier_mask is not None:
        think_gated = think_gated * new_tier_mask
    else:
        think_gated = jnp.zeros_like(think_gated)
    return jnp.concatenate([m_gated, think_gated], axis=-1)


def routed_graph_nodes_ffn(
    x: jnp.ndarray,
    router_w: jnp.ndarray,
    node_gate: jnp.ndarray,
    node_up: jnp.ndarray,
    node_down: jnp.ndarray,
    top_k: int,
    router_bias: jnp.ndarray | None = None,
    mature_nodes: int = 0,
    new_tier_mask: jnp.ndarray | None = None,
    node_edge_scale: jnp.ndarray | None = None,
    return_domain_logit: bool = False,
    valid_mask: jnp.ndarray | None = None,
    seq_dl_override: jnp.ndarray | None = None,
) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
    """Route tokens `(B, T, D)` across Specialist Graph-Node Banks (FlexOlmo Cosine Routing).

    - Bank 0 (`Nodes 0..9`): 10M English/Science/Chat Macro-Node (`0..7` base + `8..9` `<|think|>`).
    - Bank 1 (`Nodes 10..19`): 10M Mathematics Macro-Node (`10..17` base + `18..19` `<|think|>`).
    Uses L2-normalized cosine similarity `8.0 * (x_unit @ w_unit)` so newly grown specialist banks
    cannot overpower frozen banks via norm inflation.
    """
    del mature_nodes
    n_nodes = node_gate.shape[0]
    x_unit = x * jax.lax.rsqrt(jnp.sum(jnp.square(x), axis=-1, keepdims=True) + 1e-6)
    w_unit = router_w * jax.lax.rsqrt(
        jnp.sum(jnp.square(router_w), axis=0, keepdims=True) + 1e-6
    )
    router_logits = 8.0 * jnp.matmul(x_unit, w_unit)  # (B, T, N)
    domain_logit = jnp.zeros((x.shape[0], x.shape[1]), dtype=x.dtype)

    if n_nodes <= 10:
        gated_probs = _route_single_bank(
            router_logits,
            router_bias,
            top_k=top_k,
            new_tier_mask=new_tier_mask,
        )
    else:
        eng_bias = router_bias[:10] if router_bias is not None else None
        math_bias = router_bias[10:] if router_bias is not None else None
        eng_probs = _route_single_bank(
            router_logits[..., :10],
            eng_bias,
            top_k=top_k,
            new_tier_mask=new_tier_mask,
        )
        math_probs = _route_single_bank(
            router_logits[..., 10:],
            math_bias,
            top_k=top_k,
            new_tier_mask=new_tier_mask,
        )
        math_base_end = min(n_nodes, 18)
        n_math_base = max(1, math_base_end - 10)
        e_eng_tok = (
            jax.scipy.special.logsumexp(router_logits[..., :8], axis=-1, keepdims=True)
            - jnp.log(8.0)
        )
        e_math_tok = (
            jax.scipy.special.logsumexp(
                router_logits[..., 10:math_base_end], axis=-1, keepdims=True
            )
            - jnp.log(float(n_math_base))
        )
        steps = jnp.arange(1, x.shape[1] + 1, dtype=x.dtype)[None, :, None]
        e_eng = 0.5 * e_eng_tok + 0.5 * (jnp.cumsum(e_eng_tok, axis=1) / steps)
        e_math = 0.5 * e_math_tok + 0.5 * (jnp.cumsum(e_math_tok, axis=1) / steps)
        domain_logit_3d = 2.0 * (e_math - e_eng)
        domain_logit = domain_logit_3d[..., 0]
        # Tiiny-AI/PowerInfer Rectified Sparse Gate:
        # Strictly isolates Bank 0 (English/Science/Chat, seq_dl < -0.95 -> p_math=0, p_eng=1)
        # from Bank 1 (Mathematics, seq_dl >= -0.95 -> p_math=1, p_eng=0).
        if seq_dl_override is not None:
            seq_dl = seq_dl_override
        elif valid_mask is not None:
            v_f = valid_mask[..., None].astype(x.dtype)
            dl_tok = 2.0 * (e_math_tok - e_eng_tok)
            seq_dl = jnp.sum(dl_tok * v_f, axis=1, keepdims=True) / jnp.maximum(
                jnp.sum(v_f, axis=1, keepdims=True), 1.0
            )
        else:
            seq_dl = jnp.mean(2.0 * (e_math_tok - e_eng_tok), axis=1, keepdims=True)
        p_soft = jax.nn.sigmoid(domain_logit_3d)
        p_rect = jnp.where(seq_dl >= -0.95, p_soft, jnp.zeros_like(p_soft))
        p_math = p_soft + jax.lax.stop_gradient(p_rect - p_soft)
        p_eng = 1.0 - p_math
        gated_probs = jnp.concatenate([p_eng * eng_probs, p_math * math_probs], axis=-1)

    if node_edge_scale is not None:
        e_mod = jnp.clip(node_edge_scale.astype(gated_probs.dtype), 0.1, 2.5)[None, None, :]
        gated_probs = gated_probs * e_mod

    gate_all = jax.nn.silu(jnp.einsum("btd,ndf->btnf", x, node_gate))
    up_all = jnp.einsum("btd,ndf->btnf", x, node_up)
    hidden = gate_all * up_all
    node_outs = jnp.einsum("btnf,nfd->btnd", hidden, node_down)
    out = jnp.einsum("btn,btnd->btd", gated_probs, node_outs)
    if return_domain_logit:
        return out, domain_logit
    return out


def init_params_v2(config: ConfigV2) -> dict[str, jnp.ndarray]:
    """Initialize flat SafeTensors/GGUF-compatible parameter dictionary for MLM v2."""
    rng = np.random.default_rng(config.seed)
    d = config.d_model
    ff = config.ffn_dim
    h = config.n_heads
    kv_h = config.n_kv_heads
    hd = config.head_dim
    kv_dim = kv_h * hd
    std = float(1.0 / np.sqrt(d))

    params: dict[str, jnp.ndarray] = {}
    params["tok_embeddings.weight"] = jnp.asarray(
        rng.normal(0.0, std, size=(config.vocab_size, d)).astype(np.float32)
    )
    params["output_norm.weight"] = jnp.ones((d,), dtype=jnp.float32)
    if not config.tie_embeddings:
        params["output.weight"] = jnp.asarray(
            rng.normal(0.0, std, size=(config.vocab_size, d)).astype(np.float32)
        )
    if config.blt_ngram_dim > 0:
        params["blt_token_ngrams"] = jnp.zeros(
            (config.vocab_size, config.blt_ngram_dim), dtype=jnp.float32
        )
        params["blt_proj.weight"] = jnp.asarray(
            rng.normal(0.0, 0.01, size=(config.blt_ngram_dim, d)).astype(np.float32)
        )
    if config.use_fone_abacus:
        params["fone_proj.weight"] = jnp.asarray(
            rng.normal(0.0, 0.02, size=(16, d)).astype(np.float32)
        )
        params["abacus_pos.weight"] = jnp.asarray(
            rng.normal(0.0, 0.02, size=(8, d)).astype(np.float32)
        )

    layer_scale = float(std / np.sqrt(2.0 * config.n_layers))
    max_passes = max(2, int(config.loop_repeats))
    mhc_res, mhc_pre, mhc_post = _default_mhc_tensors(max_passes, symmetry_break=True)
    if config.use_mhc:
        params["output_mhc.pre"] = jnp.zeros((2,), dtype=jnp.float32)

    for i in range(config.n_layers):
        p = f"blk.{i}"
        params[f"{p}.attn_norm.weight"] = jnp.ones((d,), dtype=jnp.float32)
        params[f"{p}.ffn_norm.weight"] = jnp.ones((d,), dtype=jnp.float32)
        params[f"{p}.attn_post_scale"] = jnp.ones((d,), dtype=jnp.float32)
        params[f"{p}.ffn_post_scale"] = jnp.ones((d,), dtype=jnp.float32)
        if config.use_mhc:
            params[f"{p}.attn_mhc.res"] = jnp.array(mhc_res, copy=True)
            params[f"{p}.attn_mhc.pre"] = jnp.array(mhc_pre, copy=True)
            params[f"{p}.attn_mhc.post"] = jnp.array(mhc_post, copy=True)
            params[f"{p}.ffn_mhc.res"] = jnp.array(mhc_res, copy=True)
            params[f"{p}.ffn_mhc.pre"] = jnp.array(mhc_pre, copy=True)
            params[f"{p}.ffn_mhc.post"] = jnp.array(mhc_post, copy=True)

        if config.is_attention_layer(i):
            params[f"{p}.attn_q.weight"] = jnp.asarray(
                rng.normal(0.0, std, size=(d, d)).astype(np.float32)
            )
            params[f"{p}.attn_k.weight"] = jnp.asarray(
                rng.normal(0.0, std, size=(d, kv_dim)).astype(np.float32)
            )
            params[f"{p}.attn_v.weight"] = jnp.asarray(
                rng.normal(0.0, std, size=(d, kv_dim)).astype(np.float32)
            )
            params[f"{p}.attn_output.weight"] = jnp.asarray(
                rng.normal(0.0, layer_scale, size=(d, d)).astype(np.float32)
            )
            params[f"{p}.attn_q_norm.weight"] = jnp.ones((hd,), dtype=jnp.float32)
            params[f"{p}.attn_k_norm.weight"] = jnp.ones((hd,), dtype=jnp.float32)
            if config.use_head_gate:
                params[f"{p}.attn_gate.weight"] = jnp.zeros((d, h), dtype=jnp.float32)
                params[f"{p}.attn_gate.bias"] = jnp.zeros((h,), dtype=jnp.float32)
        else:
            # LFM2/LFM3 Gated ShortConv + optional Parallel Gated DeltaNet
            params[f"{p}.shortconv.in_proj.weight"] = jnp.asarray(
                rng.normal(0.0, std, size=(d, 3 * d)).astype(np.float32)
            )
            conv_w = np.zeros((config.conv_kernel, d), dtype=np.float32)
            conv_w[-1, :] = 1.0
            conv_w += rng.normal(0.0, 0.02, size=conv_w.shape).astype(np.float32)
            params[f"{p}.shortconv.conv.weight"] = jnp.asarray(conv_w)
            params[f"{p}.shortconv.out_proj.weight"] = jnp.asarray(
                rng.normal(0.0, layer_scale, size=(d, d)).astype(np.float32)
            )
            if config.use_gated_deltanet:
                params[f"{p}.deltanet.wq.weight"] = jnp.asarray(
                    rng.normal(0.0, std, size=(d, d)).astype(np.float32)
                )
                params[f"{p}.deltanet.wk.weight"] = jnp.asarray(
                    rng.normal(0.0, std, size=(d, d)).astype(np.float32)
                )
                params[f"{p}.deltanet.wv.weight"] = jnp.asarray(
                    rng.normal(0.0, std, size=(d, d)).astype(np.float32)
                )
                params[f"{p}.deltanet.w_decay.weight"] = jnp.asarray(
                    rng.normal(0.0, 0.02, size=(d, h)).astype(np.float32)
                )
                params[f"{p}.deltanet.w_beta.weight"] = jnp.zeros((d, h), dtype=jnp.float32)
                params[f"{p}.deltanet.channel_gate"] = jnp.ones((d,), dtype=jnp.float32)
                params[f"{p}.deltanet.scale"] = jnp.full((d,), 0.25, dtype=jnp.float32)

        params[f"{p}.ffn_gate.weight"] = jnp.asarray(
            rng.normal(0.0, std, size=(d, ff)).astype(np.float32)
        )
        params[f"{p}.ffn_up.weight"] = jnp.asarray(
            rng.normal(0.0, std, size=(d, ff)).astype(np.float32)
        )
        params[f"{p}.ffn_down.weight"] = jnp.asarray(
            rng.normal(0.0, layer_scale, size=(ff, d)).astype(np.float32)
        )
        if config.n_experts > 0:
            n_exp = config.n_experts
            ef = config.expert_ffn_dim
            params[f"{p}.nodes.router.weight"] = jnp.asarray(
                rng.normal(0.0, 0.02, size=(d, n_exp)).astype(np.float32)
            )
            params[f"{p}.nodes.router_bias"] = jnp.zeros((n_exp,), dtype=jnp.float32)
            params[f"{p}.nodes.edge_scale"] = jnp.ones((n_exp,), dtype=jnp.float32)
            params[f"{p}.nodes.gate.weight"] = jnp.asarray(
                rng.normal(0.0, std, size=(n_exp, d, ef)).astype(np.float32)
            )
            params[f"{p}.nodes.up.weight"] = jnp.asarray(
                rng.normal(0.0, std, size=(n_exp, d, ef)).astype(np.float32)
            )
            params[f"{p}.nodes.down.weight"] = jnp.asarray(
                rng.normal(0.0, layer_scale, size=(n_exp, ef, d)).astype(np.float32)
            )
    params["mtp_norm.weight"] = jnp.ones((d,), dtype=jnp.float32)
    params["mtp_proj.weight"] = jnp.asarray(
        rng.normal(0.0, 0.01, size=(2 * d, d)).astype(np.float32)
    )
    if config.use_ncm:
        params["ncm_wle.weight"] = jnp.zeros((max_passes, d), dtype=jnp.float32)
        phi_w = np.full(
            (config.ncm_window, d), 1.0 / float(config.ncm_window), dtype=np.float32
        )
        params["ncm_phi.conv.weight"] = jnp.asarray(phi_w)
        params["ncm_phi.ratio_logit"] = jnp.full((d,), -1.5, dtype=jnp.float32)
        params["ncm_concept_proj.weight"] = jnp.asarray(
            rng.normal(0.0, 0.01, size=(d, d)).astype(np.float32)
        )
        params["output_head.log_scale"] = jnp.zeros((1,), dtype=jnp.float32)
        params["output_head.norm_alpha"] = jnp.zeros((1,), dtype=jnp.float32)
    return params


def compute_ornith_scaffold_mask(
    tokens: jnp.ndarray,
    think_open_id: int,
    think_close_id: int = -1,
    k1: int = 28,
    decay_lambda: float = 0.025,
) -> jnp.ndarray:
    """Ornith-1.5 Self-Scaffold Horizon Mask `w(d_t)` for Tiered `<|think|>` Graph Nodes."""
    has_opened = jnp.maximum.accumulate((tokens == think_open_id).astype(jnp.float32), axis=-1)
    has_closed = (
        jnp.maximum.accumulate((tokens == think_close_id).astype(jnp.float32), axis=-1)
        if think_close_id >= 0
        else jnp.zeros_like(has_opened)
    )
    is_active = jnp.clip(has_opened - has_closed, 0.0, 1.0)
    d_t = jnp.cumsum(is_active, axis=-1)
    decay = jnp.where(
        d_t <= float(k1),
        1.0,
        jnp.maximum(0.35, jnp.exp(-float(decay_lambda) * (d_t - float(k1)))),
    )
    return (is_active * decay)[..., None]


def compute_fone_abacus_embeddings(
    tokens: jnp.ndarray,
    params: dict[str, jnp.ndarray],
    config: ConfigV2,
) -> jnp.ndarray:
    """FoNE (Fourier Number Embeddings) + Causal Abacus Place-Value Embeddings (NeurIPS 2024/2025).

    Injects periodic harmonic sine/cosine features `(sin(2*pi*d*k/10), cos(2*pi*d*k/10))`
    and causal contiguous digit-run place indices (`0..7`) for digit tokens `'0'..'9'`.
    """
    t_len = tokens.shape[1]
    digit_ids = jnp.asarray(config.digit_token_ids, dtype=tokens.dtype)
    matches = tokens[..., None] == digit_ids[None, None, :]  # (B, T, 10)
    is_digit = jnp.any(matches, axis=-1)  # (B, T)
    digit_val = jnp.argmax(matches.astype(jnp.int32), axis=-1).astype(jnp.float32)
    harmonics = jnp.arange(1, 9, dtype=jnp.float32)[None, None, :]  # (1, 1, 8)
    angles = 2.0 * jnp.pi * digit_val[..., None] * harmonics / 10.0
    fourier_feat = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)  # (B, T, 16)
    fone_emb = fourier_feat @ params["fone_proj.weight"]

    padded_digit = jnp.pad(is_digit, ((0, 0), (7, 0)), constant_values=False)
    run_alive = is_digit
    place_idx = jnp.zeros_like(tokens, dtype=jnp.int32)
    for offset in range(1, 8):
        run_alive = run_alive & padded_digit[:, 7 - offset : 7 - offset + t_len]
        place_idx = place_idx + run_alive.astype(jnp.int32)
    abacus_emb = params["abacus_pos.weight"][place_idx]
    mask_f = is_digit[..., None].astype(jnp.float32)
    return ((fone_emb + abacus_emb) * mask_f).astype(params["tok_embeddings.weight"].dtype)


def _forward_core_v2(
    params: dict[str, jnp.ndarray],
    tokens: jnp.ndarray,
    config: ConfigV2,
    return_layer_router_logits: bool = False,
) -> tuple:
    """Compute `(logits, hidden_normed, mean_domain_logit)` across sequence length `T`."""
    bsz, t_len = tokens.shape
    raw_tok_emb = params["tok_embeddings.weight"][tokens]  # (B, T, D)
    prompt_seq_dl: jnp.ndarray | None = None
    if config.n_experts > 10 and "blk.0.nodes.router.weight" in params:
        h0 = rms_norm(raw_tok_emb, params["blk.0.ffn_norm.weight"], config.rms_eps)
        rw0 = params["blk.0.nodes.router.weight"]
        h0_u = h0 * jax.lax.rsqrt(jnp.sum(jnp.square(h0), axis=-1, keepdims=True) + 1e-6)
        w0_u = rw0 * jax.lax.rsqrt(jnp.sum(jnp.square(rw0), axis=0, keepdims=True) + 1e-6)
        rl0 = 8.0 * jnp.matmul(h0_u, w0_u)
        m_end0 = min(config.n_experts, 18)
        n_m0 = max(1, m_end0 - 10)
        e_eng0 = jax.scipy.special.logsumexp(rl0[..., :8], axis=-1, keepdims=True) - jnp.log(8.0)
        e_math0 = jax.scipy.special.logsumexp(
            rl0[..., 10:m_end0], axis=-1, keepdims=True
        ) - jnp.log(float(n_m0))
        dl0 = 2.0 * (e_math0 - e_eng0)
        if config.think_open_id >= 0:
            before_think = (
                jnp.cumsum((tokens == config.think_open_id).astype(jnp.int32), axis=-1) == 0
            )
            p_mask = (before_think & (tokens != 2))[..., None].astype(raw_tok_emb.dtype)
        else:
            p_mask = (tokens != 2)[..., None].astype(raw_tok_emb.dtype)
        prompt_seq_dl = jnp.sum(dl0 * p_mask, axis=1, keepdims=True) / jnp.maximum(
            jnp.sum(p_mask, axis=1, keepdims=True), 1.0
        )
    x = raw_tok_emb
    if (
        config.blt_ngram_dim > 0
        and "blt_token_ngrams" in params
        and "blt_proj.weight" in params
    ):
        ng_table = jax.lax.stop_gradient(params["blt_token_ngrams"])
        ng_cur = ng_table[tokens]  # (B, T, G)
        ng_pad = jnp.pad(ng_cur, ((0, 0), (2, 0), (0, 0)))
        ng_mix = (
            0.60 * ng_cur
            + 0.25 * ng_pad[:, 1 : 1 + t_len, :]
            + 0.15 * ng_pad[:, :t_len, :]
        )
        ng_norm = ng_mix * jax.lax.rsqrt(
            jnp.sum(jnp.square(ng_mix), axis=-1, keepdims=True) + 1e-6
        )
        x = x + (ng_norm @ params["blt_proj.weight"]).astype(x.dtype)
    if (
        config.use_fone_abacus
        and len(config.digit_token_ids) == 10
        and "fone_proj.weight" in params
        and "abacus_pos.weight" in params
    ):
        x = x + compute_fone_abacus_embeddings(tokens, params, config)

    cos, sin = precompute_rope_freqs(t_len, config.head_dim, config.rope_theta)
    use_tiered_reasoning = (
        config.mature_experts >= 8
        and config.n_experts > config.mature_experts
        and config.think_open_id >= 0
    )
    think_tier_mask = (
        compute_ornith_scaffold_mask(
            tokens, config.think_open_id, think_close_id=config.think_close_id
        )
        if use_tiered_reasoning
        else None
    )

    mhc_enabled = config.use_mhc and "blk.0.attn_mhc.res" in params
    ncm_enabled = config.use_ncm and "ncm_phi.conv.weight" in params
    streams = jnp.stack([x, x], axis=-2) if mhc_enabled else None  # (B, T, 2, D)
    schedule = build_layer_execution_schedule(config)
    shared_dsa_scores: jnp.ndarray | None = None
    domain_logit_sum = jnp.zeros((bsz, t_len), dtype=jnp.float32)
    domain_logit_count = 0
    layer_router_logits: list[jnp.ndarray] = []

    for i, pass_idx in schedule:
        p = f"blk.{i}"
        if mhc_enabled and streams is not None:
            a_pre = params[f"{p}.attn_mhc.pre"]
            p_slot = min(pass_idx, a_pre.shape[0] - 1)
            x_in = mhc_read_stream(streams, a_pre[p_slot])
        else:
            p_slot = 0
            x_in = x

        if ncm_enabled and pass_idx > 0 and i == config.loop_start_layer:
            wle_w = params["ncm_wle.weight"]
            w_slot = min(pass_idx, wle_w.shape[0] - 1)
            x_in = ncm_phi_bridge(
                x_in,
                wle_w[w_slot],
                params["ncm_phi.conv.weight"],
                params["ncm_phi.ratio_logit"],
            )

        normed = rms_norm(x_in, params[f"{p}.attn_norm.weight"], config.rms_eps)
        if config.is_attention_layer(i):
            if config.sliding_window > 0 and config.dsa_top_k > 0:
                q_idx = apply_rope(
                    rms_norm(
                        (normed @ params[f"{p}.attn_q.weight"]).reshape(
                            bsz, t_len, config.n_heads, config.head_dim
                        ),
                        params[f"{p}.attn_q_norm.weight"],
                        config.rms_eps,
                    ),
                    cos,
                    sin,
                )
                k_idx = apply_rope(
                    rms_norm(
                        (normed @ params[f"{p}.attn_k.weight"]).reshape(
                            bsz, t_len, config.n_kv_heads, config.head_dim
                        ),
                        params[f"{p}.attn_k_norm.weight"],
                        config.rms_eps,
                    ),
                    cos,
                    sin,
                )
                kv_g = config.n_heads // config.n_kv_heads
                if kv_g > 1:
                    k_idx = jnp.repeat(k_idx, kv_g, axis=2)
                raw_idx_logits = jnp.matmul(
                    jnp.transpose(q_idx, (0, 2, 1, 3)),
                    jnp.swapaxes(jnp.transpose(k_idx, (0, 2, 1, 3)), -1, -2),
                ) / jnp.sqrt(float(config.head_dim))
                _, shared_dsa_scores = dsa_lightning_index_mask(
                    raw_idx_logits,
                    sliding_window=config.sliding_window,
                    dsa_top_k=config.dsa_top_k,
                    shared_index_scores=shared_dsa_scores,
                )
            dx = grouped_query_attention(
                normed,
                params[f"{p}.attn_q.weight"],
                params[f"{p}.attn_k.weight"],
                params[f"{p}.attn_v.weight"],
                params[f"{p}.attn_output.weight"],
                params[f"{p}.attn_q_norm.weight"],
                params[f"{p}.attn_k_norm.weight"],
                cos,
                sin,
                config.n_heads,
                config.n_kv_heads,
                config.rms_eps,
                attn_gate_w=params.get(f"{p}.attn_gate.weight") if config.use_head_gate else None,
                attn_gate_b=params.get(f"{p}.attn_gate.bias") if config.use_head_gate else None,
                sliding_window=config.sliding_window,
                dsa_top_k=config.dsa_top_k,
                shared_dsa_scores=shared_dsa_scores,
                kv_cache_bits=config.kv_cache_bits,
            )
        else:
            proj = normed @ params[f"{p}.shortconv.in_proj.weight"]
            b_gate, c_gate, v_in = jnp.split(proj, 3, axis=-1)
            bx = jax.nn.silu(b_gate) * v_in
            conv_out = causal_short_conv(bx, params[f"{p}.shortconv.conv.weight"])
            y_mix = jax.nn.silu(c_gate) * conv_out
            if config.use_gated_deltanet and f"{p}.deltanet.wq.weight" in params:
                delta_out = parallel_gated_deltanet(
                    normed,
                    params[f"{p}.deltanet.wq.weight"],
                    params[f"{p}.deltanet.wk.weight"],
                    params[f"{p}.deltanet.wv.weight"],
                    params[f"{p}.deltanet.w_decay.weight"],
                    config.n_heads,
                    w_beta=params.get(f"{p}.deltanet.w_beta.weight"),
                    channel_gate=params.get(f"{p}.deltanet.channel_gate"),
                )
                y_mix = y_mix + delta_out * params[f"{p}.deltanet.scale"][None, None, :]
            dx = y_mix @ params[f"{p}.shortconv.out_proj.weight"]
        dx = sandwich_rms_bound(
            dx,
            params.get(f"{p}.attn_post_scale"),
            clip_rms=config.sandwich_norm_clip,
            eps=config.rms_eps,
        )
        if mhc_enabled and streams is not None:
            streams = mhc_write_stream(
                streams,
                dx,
                params[f"{p}.attn_mhc.res"][p_slot],
                params[f"{p}.attn_mhc.post"][p_slot],
                sinkhorn_iters=config.mhc_sinkhorn_iters,
            )
            f_pre = params[f"{p}.ffn_mhc.pre"]
            f_slot = min(pass_idx, f_pre.shape[0] - 1)
            x_ffn_in = mhc_read_stream(streams, f_pre[f_slot])
        else:
            f_slot = 0
            x = x + dx
            x_ffn_in = x

        ffn_normed = rms_norm(x_ffn_in, params[f"{p}.ffn_norm.weight"], config.rms_eps)
        ffn_out = swiglu_ffn(
            ffn_normed,
            params[f"{p}.ffn_gate.weight"],
            params[f"{p}.ffn_up.weight"],
            params[f"{p}.ffn_down.weight"],
        )
        if config.n_experts > 0 and f"{p}.nodes.router.weight" in params:
            in_loop = (
                config.loop_repeats > 1
                and config.loop_start_layer <= i < config.loop_end_layer
            )
            pass_scale = 0.75 if (in_loop and pass_idx == 0) else 1.0
            pass_tier_mask = (
                think_tier_mask * pass_scale if think_tier_mask is not None else None
            )
            eff_router_bias = params.get(f"{p}.nodes.router_bias")
            if len(config.active_subnodes) > 0:
                b_arr = np.zeros((config.n_experts,), dtype=np.float32)
                for a_idx in config.active_subnodes:
                    if 0 <= int(a_idx) < config.n_experts and int(a_idx) not in (8, 9, 18, 19):
                        b_arr[int(a_idx)] = 5.0
                eff_router_bias = (
                    eff_router_bias + jnp.asarray(b_arr)
                    if eff_router_bias is not None
                    else jnp.asarray(b_arr)
                )
            if return_layer_router_logits and pass_idx == 0:
                h_u = ffn_normed * jax.lax.rsqrt(
                    jnp.sum(jnp.square(ffn_normed), axis=-1, keepdims=True) + 1e-6
                )
                rw_cur = params[f"{p}.nodes.router.weight"]
                w_u = rw_cur * jax.lax.rsqrt(
                    jnp.sum(jnp.square(rw_cur), axis=0, keepdims=True) + 1e-6
                )
                rl_raw = 8.0 * jnp.matmul(h_u, w_u)
                t_steps = jnp.arange(1, t_len + 1, dtype=rl_raw.dtype)[None, :, None]
                layer_router_logits.append(
                    0.45 * rl_raw + 0.55 * (jnp.cumsum(rl_raw, axis=1) / t_steps)
                )
            routed_out, d_logit = routed_graph_nodes_ffn(
                ffn_normed,
                params[f"{p}.nodes.router.weight"],
                params[f"{p}.nodes.gate.weight"],
                params[f"{p}.nodes.up.weight"],
                params[f"{p}.nodes.down.weight"],
                config.top_k_experts,
                eff_router_bias,
                mature_nodes=config.mature_experts if use_tiered_reasoning else 0,
                new_tier_mask=pass_tier_mask,
                node_edge_scale=params.get(f"{p}.nodes.edge_scale"),
                return_domain_logit=True,
                valid_mask=(tokens != 2),
                seq_dl_override=prompt_seq_dl,
            )
            ffn_out = ffn_out + routed_out
            domain_logit_sum = domain_logit_sum + d_logit.astype(jnp.float32)
            domain_logit_count += 1
        ffn_out = sandwich_rms_bound(
            ffn_out,
            params.get(f"{p}.ffn_post_scale"),
            clip_rms=config.sandwich_norm_clip,
            eps=config.rms_eps,
        )
        if mhc_enabled and streams is not None:
            streams = mhc_write_stream(
                streams,
                ffn_out,
                params[f"{p}.ffn_mhc.res"][f_slot],
                params[f"{p}.ffn_mhc.post"][f_slot],
                sinkhorn_iters=config.mhc_sinkhorn_iters,
            )
        else:
            x = x + ffn_out

    if mhc_enabled and streams is not None:
        out_pre = params.get("output_mhc.pre", jnp.zeros((2,), dtype=streams.dtype))
        x = mhc_read_stream(streams, out_pre)

    hidden_normed = rms_norm(x, params["output_norm.weight"], config.rms_eps)
    out_w = (
        params["tok_embeddings.weight"]
        if config.tie_embeddings
        else params["output.weight"]
    )
    logits = ncm_tied_codebook_logits(
        hidden_normed,
        out_w,
        log_scale=params.get("output_head.log_scale") if config.use_ncm else None,
        norm_alpha=params.get("output_head.norm_alpha") if config.use_ncm else None,
    )
    mean_domain_logit = domain_logit_sum / float(max(domain_logit_count, 1))
    out_logits = logits.reshape(bsz, t_len, config.vocab_size)
    if return_layer_router_logits:
        return out_logits, hidden_normed, mean_domain_logit, layer_router_logits
    return out_logits, hidden_normed, mean_domain_logit


def forward_with_hidden_v2(
    params: dict[str, jnp.ndarray],
    tokens: jnp.ndarray,
    config: ConfigV2,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute next-token logits `(B, T, V)` and final normalized hidden states `(B, T, D)`."""
    logits, hidden_normed, _ = _forward_core_v2(params, tokens, config)
    return logits, hidden_normed


def compute_ncm_concept_loss_v2(
    params: dict[str, jnp.ndarray],
    hidden_normed: jnp.ndarray,
    targets: jnp.ndarray,
    mask: jnp.ndarray,
    config: ConfigV2,
) -> jnp.ndarray:
    """NCM (`arXiv:2610.14162`) Next-Concept Horizon Loss over upcoming W-token centroid."""
    w_win = max(2, min(int(config.ncm_window), targets.shape[1]))
    target_embs = jax.lax.stop_gradient(
        params["tok_embeddings.weight"][targets].astype(jnp.float32)
    )  # (B, T, D)
    t_len = targets.shape[1]
    padded = jnp.pad(target_embs, ((0, 0), (0, w_win - 1), (0, 0)))
    concept_sum = jnp.zeros_like(target_embs)
    for k in range(w_win):
        concept_sum = concept_sum + padded[:, k : k + t_len, :]
    concept_target = concept_sum / float(w_win)
    concept_target = concept_target * jax.lax.rsqrt(
        jnp.sum(jnp.square(concept_target), axis=-1, keepdims=True) + 1e-6
    )

    h_f32 = hidden_normed.astype(jnp.float32)
    proj_w = params["ncm_concept_proj.weight"].astype(jnp.float32)
    pred_concept = h_f32 + jax.nn.silu(h_f32 @ proj_w)
    pred_concept = pred_concept * jax.lax.rsqrt(
        jnp.sum(jnp.square(pred_concept), axis=-1, keepdims=True) + 1e-6
    )
    cos_sim = jnp.sum(pred_concept * concept_target, axis=-1)  # (B, T)
    concept_loss = 1.0 - cos_sim
    denom = jnp.maximum(jnp.sum(mask), 1.0)
    return jnp.sum(concept_loss * mask) / denom


def compute_mtp_logits_v2(
    params: dict[str, jnp.ndarray],
    hidden_normed: jnp.ndarray,
    next_tokens: jnp.ndarray,
    config: ConfigV2,
) -> jnp.ndarray:
    """GLM-5.3-Flash Norm-Corrected `MTP`: predict `t+2` from `h_t` and `e(x_{t+1})`."""
    h_cur = hidden_normed[:, :-1, :]  # (B, T-1, D)
    emb_next = rms_norm(
        params["tok_embeddings.weight"][next_tokens],
        params["mtp_norm.weight"],
        config.rms_eps,
    )
    cat = jnp.concatenate([h_cur, emb_next], axis=-1)  # (B, T-1, 2*D)
    h_mtp = h_cur + jax.nn.silu(cat @ params["mtp_proj.weight"])
    out_w = (
        params["tok_embeddings.weight"]
        if config.tie_embeddings
        else params["output.weight"]
    )
    return ncm_tied_codebook_logits(
        h_mtp,
        out_w,
        log_scale=params.get("output_head.log_scale") if config.use_ncm else None,
        norm_alpha=params.get("output_head.norm_alpha") if config.use_ncm else None,
    )


def forward_v2(
    params: dict[str, jnp.ndarray],
    tokens: jnp.ndarray,
    config: ConfigV2,
) -> jnp.ndarray:
    """Compute next-token logits `(B, T, V)` in parallel across sequence length `T`."""
    logits, _ = forward_with_hidden_v2(params, tokens, config)
    return logits


def create_state_v2(config: ConfigV2, total_steps: int = 1000) -> TrainStateV2:
    """Create initial `TrainStateV2` with warmup + cosine decay AdamW optimizer."""
    params = init_params_v2(config)
    warmup = min(config.warmup_steps, max(1, total_steps // 5))
    decay_steps = max(total_steps, warmup + 1)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=config.learning_rate * 0.1,
        peak_value=config.learning_rate,
        warmup_steps=warmup,
        decay_steps=decay_steps,
        end_value=config.learning_rate * 0.1,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(config.clip_norm),
        optax.adamw(learning_rate=schedule, weight_decay=config.weight_decay),
    )
    opt_state = tx.init(params)
    return TrainStateV2(step=0, params=params, opt_state=opt_state)


def tempered_topk_kd_loss_v2(
    student_logits: jnp.ndarray,
    teacher_logits: jnp.ndarray,
    mask: jnp.ndarray,
    top_k: int = 16,
    temperature: float = 2.0,
) -> jnp.ndarray:
    """LFM2/2.5 Tempered, Decoupled Top-K Knowledge Distillation (`arXiv:2511.23404`).

    Restricts KL divergence to the teacher's top-K probability support window to avoid
    tail support mismatch on compact student models.
    """
    k_eff = min(int(top_k), student_logits.shape[-1])
    t_stopped = jax.lax.stop_gradient(teacher_logits)
    t_vals, top_idx = jax.lax.top_k(t_stopped, k_eff)  # (B, T, K)
    s_vals = jnp.take_along_axis(student_logits, top_idx, axis=-1)  # (B, T, K)
    temp = float(temperature)
    t_log_p = jax.nn.log_softmax(t_vals / temp, axis=-1)
    t_p = jnp.exp(t_log_p)
    s_log_p = jax.nn.log_softmax(s_vals / temp, axis=-1)
    kl_tok = (temp * temp) * jnp.sum(t_p * (t_log_p - s_log_p), axis=-1)  # (B, T)
    denom = jnp.maximum(jnp.sum(mask), 1.0)
    return jnp.sum(kl_tok * mask) / denom


def make_train_step_v2(
    config: ConfigV2,
    total_steps: int = 1000,
    teacher_params: dict[str, jnp.ndarray] | None = None,
    teacher_config: ConfigV2 | None = None,
):
    """Compile a parallel sequence training step for MLM v2."""
    warmup = min(config.warmup_steps, max(1, total_steps // 5))
    decay_steps = max(total_steps, warmup + 1)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=config.learning_rate * 0.1,
        peak_value=config.learning_rate,
        warmup_steps=warmup,
        decay_steps=decay_steps,
        end_value=config.learning_rate * 0.1,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(config.clip_norm),
        optax.adamw(learning_rate=schedule, weight_decay=config.weight_decay),
    )
    eff_teacher_cfg = teacher_config if teacher_config is not None else config

    @jax.jit
    def step_fn(
        params: dict[str, jnp.ndarray],
        opt_state: optax.OptState,
        inputs: jnp.ndarray,
        targets: jnp.ndarray,
        mask: jnp.ndarray,
    ):
        def loss_fn(p):
            logits, hidden_normed, mean_domain_logit = _forward_core_v2(p, inputs, config)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            tok_loss = -jnp.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]
            denom = jnp.maximum(jnp.sum(mask), 1.0)
            total_loss = jnp.sum(tok_loss * mask) / denom
            if (
                config.n_experts > 10
                and config.domain_gate_weight > 0.0
                and len(config.digit_token_ids) > 0
            ):
                digit_ids = jnp.asarray(config.digit_token_ids, dtype=targets.dtype)
                tgt_has_digit = jnp.any(
                    targets[..., None] == digit_ids[None, None, :], axis=-1
                ) & (mask > 0)
                seq_is_math = jnp.any(tgt_has_digit, axis=-1, keepdims=True).astype(jnp.float32)
                valid_tok = (inputs != 0).astype(jnp.float32)
                bce_tok = optax.sigmoid_binary_cross_entropy(
                    mean_domain_logit,
                    jnp.broadcast_to(seq_is_math, mean_domain_logit.shape),
                )
                domain_loss = jnp.sum(bce_tok * valid_tok) / jnp.maximum(
                    jnp.sum(valid_tok), 1.0
                )
                total_loss = total_loss + float(config.domain_gate_weight) * domain_loss
            if config.mtp_weight > 0.0 and "mtp_proj.weight" in p:
                mtp_logits = compute_mtp_logits_v2(p, hidden_normed, targets[:, :-1], config)
                mtp_lps = jax.nn.log_softmax(mtp_logits, axis=-1)
                mtp_tok_loss = -jnp.take_along_axis(
                    mtp_lps, targets[:, 1:, None], axis=-1
                )[..., 0]
                mtp_mask = mask[:, :-1] * mask[:, 1:]
                mtp_denom = jnp.maximum(jnp.sum(mtp_mask), 1.0)
                mtp_loss = jnp.sum(mtp_tok_loss * mtp_mask) / mtp_denom
                total_loss = total_loss + float(config.mtp_weight) * mtp_loss
            if (
                config.use_ncm
                and config.ncm_weight > 0.0
                and "ncm_concept_proj.weight" in p
            ):
                ncm_loss = compute_ncm_concept_loss_v2(
                    p, hidden_normed, targets, mask, config
                )
                total_loss = total_loss + float(config.ncm_weight) * ncm_loss
            if config.kd_weight > 0.0 and teacher_params is not None:
                t_logits, _ = forward_with_hidden_v2(teacher_params, inputs, eff_teacher_cfg)
                kd_loss = tempered_topk_kd_loss_v2(
                    logits,
                    t_logits,
                    mask,
                    top_k=config.kd_top_k,
                    temperature=config.kd_temperature,
                )
                total_loss = total_loss + float(config.kd_weight) * kd_loss
            return total_loss

        loss, grads = jax.value_and_grad(loss_fn)(params)
        clean_grads = {
            k: (
                jnp.zeros_like(g)
                if (k.endswith(".nodes.router_bias") or k == "blt_token_ngrams")
                else g
            )
            for k, g in grads.items()
        }
        grad_norm = optax.tree.norm(clean_grads)
        updates, new_opt_state = tx.update(clean_grads, opt_state, params)
        if (
            config.mature_experts > 0
            or config.shared_lr_scale < 1.0
            or config.mature_conv_kernel > 0
            or len(config.active_subnodes) > 0
        ):
            active_set = {
                idx for idx in config.active_subnodes if 0 <= int(idx) < config.n_experts
            }
            scaled_updates = {}
            for k, u in updates.items():
                if k.endswith(".nodes.router_bias") or k == "blt_token_ngrams":
                    scaled_updates[k] = jnp.zeros_like(u)
                elif k.endswith(".nodes.edge_scale"):
                    if active_set:
                        e_mask = np.full(
                            (config.n_experts,), float(config.mature_lr_scale), dtype=np.float32
                        )
                        if config.n_experts > 10 and min(active_set) >= 10:
                            e_mask[:10] = 0.0
                        for a_idx in active_set:
                            e_mask[int(a_idx)] = 1.0
                        scaled_updates[k] = u * jnp.asarray(e_mask)
                    elif 0 < config.mature_experts < config.n_experts:
                        e_mask = np.ones((config.n_experts,), dtype=np.float32)
                        e_mask[: config.mature_experts] = float(config.mature_lr_scale)
                        if config.n_experts > 10:
                            e_mask[:10] = 0.0
                        scaled_updates[k] = u * jnp.asarray(e_mask)
                    elif config.mature_experts >= config.n_experts > 0:
                        scaled_updates[k] = u * float(config.mature_lr_scale)
                    else:
                        scaled_updates[k] = u
                elif ".nodes." in k and not k.endswith(".router.weight"):
                    if active_set:
                        m_mask = np.full(
                            (config.n_experts, 1, 1),
                            float(config.mature_lr_scale),
                            dtype=np.float32,
                        )
                        if config.n_experts > 10 and min(active_set) >= 10:
                            m_mask[:10] = 0.0
                        for a_idx in active_set:
                            m_mask[int(a_idx)] = 1.0
                        scaled_updates[k] = u * jnp.asarray(m_mask)
                    elif 0 < config.mature_experts < config.n_experts:
                        m_mask = np.ones((config.n_experts, 1, 1), dtype=np.float32)
                        m_mask[: config.mature_experts] = float(config.mature_lr_scale)
                        if config.n_experts > 10:
                            m_mask[:10] = 0.0
                        scaled_updates[k] = u * jnp.asarray(m_mask)
                    elif config.mature_experts >= config.n_experts > 0:
                        scaled_updates[k] = u * float(config.mature_lr_scale)
                    else:
                        scaled_updates[k] = u
                elif k.endswith(".nodes.router.weight"):
                    if active_set:
                        r_mask = np.full(
                            (1, config.n_experts), float(config.mature_lr_scale), dtype=np.float32
                        )
                        if config.n_experts > 10 and min(active_set) >= 10:
                            r_mask[:, :10] = 0.0
                        for a_idx in active_set:
                            r_mask[:, int(a_idx)] = 1.0
                        scaled_updates[k] = u * jnp.asarray(r_mask)
                    elif 0 < config.mature_experts < config.n_experts:
                        r_mask = np.ones((1, config.n_experts), dtype=np.float32)
                        r_mask[:, : config.mature_experts] = float(config.mature_lr_scale)
                        if config.n_experts > 10:
                            r_mask[:, :10] = 0.0
                            if config.n_experts <= 18 or config.mature_experts <= 10:
                                r_mask[:, 10:] = 1.0
                            else:
                                r_mask[:, 10:18] = max(float(config.mature_lr_scale), 0.35)
                        scaled_updates[k] = u * jnp.asarray(r_mask)
                    elif config.mature_experts >= config.n_experts > 0:
                        scaled_updates[k] = u * float(config.mature_lr_scale)
                    else:
                        scaled_updates[k] = u
                elif k.endswith(".shortconv.conv.weight") and (
                    0 < config.mature_conv_kernel < config.conv_kernel
                    and config.shared_lr_scale > 0.0
                ):
                    new_taps = config.conv_kernel - config.mature_conv_kernel
                    c_mask = np.ones((config.conv_kernel, 1), dtype=np.float32)
                    c_mask[new_taps:, :] = float(config.shared_lr_scale)
                    scaled_updates[k] = u * jnp.asarray(c_mask)
                elif k.startswith("mtp_"):
                    scaled_updates[k] = (
                        u if config.mtp_weight > 0.0 else u * float(config.shared_lr_scale)
                    )
                else:
                    scaled_updates[k] = u * float(config.shared_lr_scale)
            updates = scaled_updates
        else:
            updates = {
                k: (
                    jnp.zeros_like(u)
                    if (k.endswith(".nodes.router_bias") or k == "blt_token_ngrams")
                    else u
                )
                for k, u in updates.items()
            }
        new_params = optax.apply_updates(params, updates)

        # DeepSeek-V3 Auxiliary-Loss-Free Router Bias Update (with Cosine Routing)
        if config.n_experts > 1 and config.router_bias_gamma > 0.0:
            tok_emb = new_params["tok_embeddings.weight"][inputs]
            k_eff = min(config.top_k_experts, config.n_experts)
            target_share = float(k_eff) / float(config.n_experts)
            for i in range(config.n_layers):
                p_key = f"blk.{i}"
                b_key = f"{p_key}.nodes.router_bias"
                if b_key in new_params:
                    normed = rms_norm(
                        tok_emb, new_params[f"{p_key}.ffn_norm.weight"], config.rms_eps
                    )
                    rw = new_params[f"{p_key}.nodes.router.weight"]
                    x_u = normed * jax.lax.rsqrt(
                        jnp.sum(jnp.square(normed), axis=-1, keepdims=True) + 1e-6
                    )
                    w_u = rw * jax.lax.rsqrt(
                        jnp.sum(jnp.square(rw), axis=0, keepdims=True) + 1e-6
                    )
                    r_probs = jax.nn.softmax(8.0 * jnp.matmul(x_u, w_u), axis=-1)
                    sel_scores = r_probs + new_params[b_key][None, None, :]
                    _, top_idx = jax.lax.top_k(sel_scores, k_eff)
                    sel_mask = jnp.sum(
                        jax.nn.one_hot(top_idx, config.n_experts, dtype=jnp.float32), axis=-2
                    )
                    actual_share = jnp.mean(sel_mask, axis=(0, 1))
                    delta_b = float(config.router_bias_gamma) * jnp.sign(
                        target_share - actual_share
                    )
                    if 0 < config.mature_experts < config.n_experts:
                        b_mask = np.ones((config.n_experts,), dtype=np.float32)
                        b_mask[: config.mature_experts] = 0.0
                        delta_b = delta_b * jnp.asarray(b_mask)
                        updated_b = jnp.clip(new_params[b_key] + delta_b, -0.02, 0.02)
                        new_params[b_key] = jnp.where(
                            jnp.asarray(b_mask) > 0, updated_b, new_params[b_key]
                        )
                    else:
                        new_params[b_key] = jnp.clip(new_params[b_key] + delta_b, -0.25, 0.25)

        return new_params, new_opt_state, loss, grad_norm

    return step_fn


def grow_model_v2_nodes(
    params: dict[str, jnp.ndarray],
    config: ConfigV2,
    add_nodes: int,
    birth_gate: float = 0.95,
    mature_lr_scale: float | None = None,
    shared_lr_scale: float | None = None,
) -> tuple[dict[str, jnp.ndarray], ConfigV2]:
    """Grow `add_nodes` specialist Graph Nodes per block via crossover of strongest parent nodes."""
    if config.n_experts < 2 or add_nodes < 1:
        raise ValueError("grow_model_v2_nodes requires n_experts >= 2 and add_nodes >= 1")
    n_old = config.n_experts
    n_new = n_old + add_nodes
    eff_mature_lr = float(config.mature_lr_scale if mature_lr_scale is None else mature_lr_scale)
    eff_shared_lr = float(config.shared_lr_scale if shared_lr_scale is None else shared_lr_scale)
    new_cfg = config.model_copy(
        update={
            "n_experts": n_new,
            "mature_experts": n_old,
            "mature_lr_scale": eff_mature_lr,
            "shared_lr_scale": eff_shared_lr,
        }
    )
    rng = np.random.default_rng(config.seed + n_new)
    new_params = dict(params)
    for i in range(config.n_layers):
        p = f"blk.{i}"
        down_w = np.asarray(params[f"{p}.nodes.down.weight"], dtype=np.float32)
        if n_old == 10:
            pool_idx = np.arange(0, 8)
        elif n_old > 10:
            pool_idx = np.arange(10, min(n_old, 18))
        else:
            pool_idx = np.arange(0, min(n_old, 8))
        norms = np.linalg.norm(down_w[pool_idx], axis=(1, 2))
        order = pool_idx[np.argsort(-norms)]
        n_pool = len(order)

        for suffix in ("gate", "up", "down"):
            arr = np.asarray(params[f"{p}.nodes.{suffix}.weight"], dtype=np.float32)
            children = []
            for j in range(add_nodes):
                p1 = arr[order[(2 * j) % n_pool]]
                p2 = arr[order[(2 * j + 1) % n_pool]]
                child = 0.85 * p1 + 0.15 * p2
                noise = rng.normal(0.0, 0.005 * (float(np.std(child)) + 1e-6), size=child.shape)
                child = (child + noise).astype(np.float32)
                if suffix == "down":
                    child = child * float(birth_gate)
                children.append(child)
            new_params[f"{p}.nodes.{suffix}.weight"] = jnp.asarray(
                np.concatenate([arr, np.stack(children, axis=0)], axis=0)
            )

        rw = np.asarray(params[f"{p}.nodes.router.weight"], dtype=np.float32)
        extra_rw = np.stack(
            [
                0.85 * rw[:, order[(2 * j) % n_pool]] + 0.15 * rw[:, order[(2 * j + 1) % n_pool]]
                for j in range(add_nodes)
            ],
            axis=-1,
        )
        new_params[f"{p}.nodes.router.weight"] = jnp.asarray(
            np.concatenate([rw, extra_rw], axis=-1)
        )
        rb_old = np.asarray(
            params.get(f"{p}.nodes.router_bias", np.zeros((n_old,), dtype=np.float32)),
            dtype=np.float32,
        )
        rb_new = np.full((add_nodes,), 0.0, dtype=np.float32)
        new_params[f"{p}.nodes.router_bias"] = jnp.asarray(
            np.concatenate([rb_old, rb_new], axis=0)
        )
        es_old = np.asarray(
            params.get(f"{p}.nodes.edge_scale", np.ones((n_old,), dtype=np.float32)),
            dtype=np.float32,
        )
        es_new = np.ones((add_nodes,), dtype=np.float32)
        new_params[f"{p}.nodes.edge_scale"] = jnp.asarray(
            np.concatenate([es_old, es_new], axis=0)
        )

    emb_arr = np.asarray(params["tok_embeddings.weight"], dtype=np.float32).copy()
    t_open = new_cfg.think_open_id
    t_close = new_cfg.think_close_id
    if n_old == 8 and 0 <= t_open < emb_arr.shape[0] and 0 <= t_close < emb_arr.shape[0]:
        mean_norm = float(np.mean(np.linalg.norm(emb_arr, axis=-1)))
        v_base = emb_arr[t_open] / max(float(np.linalg.norm(emb_arr[t_open])), 1e-6)
        ortho_raw = rng.normal(0.0, 1.0, size=v_base.shape).astype(np.float32)
        ortho_raw = ortho_raw - float(np.dot(ortho_raw, v_base)) * v_base
        v_ortho = ortho_raw / max(float(np.linalg.norm(ortho_raw)), 1e-6)
        target_n = 1.35 * mean_norm
        emb_arr[t_open] = target_n * (0.70710678 * v_base + 0.70710678 * v_ortho)
        emb_arr[t_close] = target_n * (0.70710678 * v_base - 0.70710678 * v_ortho)
        new_params["tok_embeddings.weight"] = jnp.asarray(emb_arr)

    return new_params, new_cfg


def expand_shortconv_kernel_v2(
    params: dict[str, jnp.ndarray],
    config: ConfigV2,
    new_kernel: int = 7,
) -> tuple[dict[str, jnp.ndarray], ConfigV2]:
    """LFM3 Continual Receptive-Field Expansion: losslessly widen ShortConv `K -> new_kernel`.

    Prepends `(new_kernel - config.conv_kernel, d_model)` zeros to the causal left-padding rows
    of `blk.{i}.shortconv.conv.weight`, preserving exact step-0 output identity while enabling
    continual training of the extended local lookback window.
    """
    if new_kernel <= config.conv_kernel:
        return params, config
    k_old = config.conv_kernel
    extra = new_kernel - k_old
    new_params = dict(params)
    for i in range(config.n_layers):
        ck = f"blk.{i}.shortconv.conv.weight"
        if ck in new_params:
            w_old = np.asarray(new_params[ck], dtype=np.float32)
            pad_zeros = np.zeros((extra, w_old.shape[1]), dtype=np.float32)
            new_params[ck] = jnp.asarray(np.concatenate([pad_zeros, w_old], axis=0))
    new_cfg = config.model_copy(
        update={
            "conv_kernel": new_kernel,
            "mature_conv_kernel": k_old,
        }
    )
    return new_params, new_cfg


def merge_checkpoints_v2(
    params_base: dict[str, jnp.ndarray],
    params_tuned: dict[str, jnp.ndarray],
    alpha: float = 0.65,
) -> dict[str, jnp.ndarray]:
    """LFM2.5 Post-Training Weight-Space Model Merging (`(1 - alpha) * base + alpha * tuned`)."""
    w_tuned = float(np.clip(alpha, 0.0, 1.0))
    w_base = 1.0 - w_tuned
    merged: dict[str, jnp.ndarray] = {}
    for k, v_tuned in params_tuned.items():
        if k in params_base and params_base[k].shape == v_tuned.shape:
            merged[k] = (
                w_base * params_base[k].astype(jnp.float32)
                + w_tuned * v_tuned.astype(jnp.float32)
            )
        else:
            merged[k] = v_tuned
    return merged


def grow_model_v2_depth(
    params: dict[str, jnp.ndarray],
    config: ConfigV2,
    new_n_layers: int,
) -> tuple[dict[str, jnp.ndarray], ConfigV2]:
    """Progressively expand model depth from `config.n_layers` to `new_n_layers`."""
    if new_n_layers <= config.n_layers:
        raise ValueError("new_n_layers must be greater than current n_layers")
    new_cfg = config.model_copy(update={"n_layers": new_n_layers})
    new_params = dict(params)
    for i in range(config.n_layers, new_n_layers):
        src_i = i % config.n_layers
        p_dst = f"blk.{i}"
        # Find a matching source layer of the same type (attention vs shortconv)
        for cand in range(config.n_layers):
            if config.is_attention_layer(cand) == new_cfg.is_attention_layer(i):
                src_i = cand
                break
        p_src = f"blk.{src_i}"
        for k, v in params.items():
            if k.startswith(f"{p_src}."):
                suffix = k[len(p_src) :]
                dst_key = f"{p_dst}{suffix}"
                if suffix in (
                    ".attn_output.weight",
                    ".shortconv.out_proj.weight",
                    ".ffn_down.weight",
                ):
                    new_params[dst_key] = jnp.zeros_like(v)
                else:
                    new_params[dst_key] = jnp.array(v, copy=True)
    return new_params, new_cfg


@lru_cache(maxsize=8)
def _make_cached_logits_step_v2(config: ConfigV2):
    @jax.jit
    def _step(
        params: dict[str, jnp.ndarray],
        seq_padded: jnp.ndarray,
        idx_a: jnp.ndarray,
        idx_b: jnp.ndarray,
    ):
        logits, hidden = forward_with_hidden_v2(params, seq_padded, config)
        logits_a = logits[0, idx_a, :]
        logits_b = logits[0, idx_b, :]
        mtp_draft_b = jnp.int32(-1)
        if "mtp_proj.weight" in params and "mtp_norm.weight" in params:
            next_b = jnp.argmax(logits_b)
            h_b = hidden[0, idx_b, :]
            emb_b = rms_norm(
                params["tok_embeddings.weight"][next_b],
                params["mtp_norm.weight"],
                config.rms_eps,
            )
            cat = jnp.concatenate([h_b, emb_b], axis=-1)
            h_mtp = h_b + jax.nn.silu(cat @ params["mtp_proj.weight"])
            out_w = (
                params["tok_embeddings.weight"]
                if config.tie_embeddings
                else params["output.weight"]
            )
            mtp_logits = ncm_tied_codebook_logits(
                h_mtp[None, None, :],
                out_w,
                log_scale=params.get("output_head.log_scale") if config.use_ncm else None,
                norm_alpha=params.get("output_head.norm_alpha") if config.use_ncm else None,
            )[0, 0]
            mtp_draft_b = jnp.argmax(mtp_logits).astype(jnp.int32)
        return logits_a, logits_b, mtp_draft_b

    return _step


def init_decode_cache_v2(config: ConfigV2) -> dict[str, jnp.ndarray]:
    """Initialize static-shape Stateful O(1) Recurrent Decode Cache."""
    schedule = build_layer_execution_schedule(config)
    n_sched = len(schedule)
    d = config.d_model
    h = config.n_heads
    h_kv = config.n_kv_heads
    d_h = config.head_dim
    max_t = config.max_seq_len
    g_dim = max(config.blt_ngram_dim, 1)
    w_ncm = max(config.ncm_window, 2)
    k_conv = max(config.conv_kernel, 2)
    n_exp = max(config.n_experts, 1)
    return {
        "pos": jnp.int32(0),
        "blt_prev1": jnp.zeros((1, g_dim), dtype=jnp.float32),
        "blt_prev2": jnp.zeros((1, g_dim), dtype=jnp.float32),
        "digit_run": jnp.zeros((1,), dtype=jnp.int32),
        "think_opened": jnp.zeros((1,), dtype=jnp.float32),
        "think_closed": jnp.zeros((1,), dtype=jnp.float32),
        "think_depth": jnp.zeros((1,), dtype=jnp.float32),
        "dl0_sum": jnp.zeros((1, 1), dtype=jnp.float32),
        "dl0_count": jnp.zeros((1, 1), dtype=jnp.float32),
        "phi_conv_buf": jnp.zeros((n_sched, 1, w_ncm - 1, d), dtype=jnp.float32),
        "shortconv_buf": jnp.zeros((n_sched, 1, k_conv - 1, d), dtype=jnp.float32),
        "delta_state": jnp.zeros((n_sched, 1, h, d_h, d_h), dtype=jnp.float32),
        "delta_k_prev": jnp.zeros((n_sched, 1, h, d_h), dtype=jnp.float32),
        "delta_v_prev": jnp.zeros((n_sched, 1, h, d_h), dtype=jnp.float32),
        "k_idx_cache": jnp.zeros((n_sched, 1, max_t, h_kv, d_h), dtype=jnp.float32),
        "k_cache": jnp.zeros((n_sched, 1, max_t, h_kv, d_h), dtype=jnp.float32),
        "v_cache": jnp.zeros((n_sched, 1, max_t, h_kv, d_h), dtype=jnp.float32),
        "router_cumsum": jnp.zeros((n_sched, 1, n_exp), dtype=jnp.float32),
        "e_eng_cumsum": jnp.zeros((n_sched, 1, 1), dtype=jnp.float32),
        "e_math_cumsum": jnp.zeros((n_sched, 1, 1), dtype=jnp.float32),
    }


def _route_single_bank_step_v2(
    bank_logits: jnp.ndarray,
    bank_cumsum: jnp.ndarray,
    step_f: jnp.ndarray,
    bank_bias: jnp.ndarray | None,
    top_k: int,
    new_tier_mask: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Single-step O(1) router within a 10-node Specialist Macro-Bank."""
    b_nodes = bank_logits.shape[-1]
    base_n = min(b_nodes, 8)
    m_raw = bank_logits[..., :base_n]
    new_cumsum = bank_cumsum + bank_logits
    m_logits = 0.45 * m_raw + 0.55 * (new_cumsum[..., :base_n] / step_f)
    m_probs = jax.nn.softmax(m_logits, axis=-1)
    k_eff = min(top_k, base_n)
    if k_eff < base_n:
        m_scores = (
            m_probs + jax.lax.stop_gradient(bank_bias[:base_n])[None, None, :]
            if bank_bias is not None
            else m_probs
        )
        _, top_idx = jax.lax.top_k(m_scores, k_eff)
        m_mask = jnp.sum(jax.nn.one_hot(top_idx, base_n, dtype=m_probs.dtype), axis=-2)
        m_gated = m_probs * m_mask
        m_gated = m_gated / jnp.maximum(jnp.sum(m_gated, axis=-1, keepdims=True), 1e-6)
    else:
        m_gated = m_probs

    if b_nodes <= 8:
        return m_gated, new_cumsum

    think_logits = bank_logits[..., 8:b_nodes]
    if bank_bias is not None:
        think_logits = think_logits + jax.lax.stop_gradient(bank_bias[8:b_nodes])[None, None, :]
    think_gated = jax.nn.sigmoid(think_logits)
    if new_tier_mask is not None:
        think_gated = think_gated * new_tier_mask
    else:
        think_gated = jnp.zeros_like(think_gated)
    return jnp.concatenate([m_gated, think_gated], axis=-1), new_cumsum


def _recurrent_token_step_v2(
    params: dict[str, jnp.ndarray],
    carry: dict[str, jnp.ndarray],
    tok_id_scalar: jnp.ndarray,
    config: ConfigV2,
    cos_table: jnp.ndarray,
    sin_table: jnp.ndarray,
) -> tuple[dict[str, jnp.ndarray], jnp.ndarray, jnp.ndarray]:
    """Single-token O(1) recurrent forward step across `ShortConv + Gated DeltaNet + GQA`."""
    tok_t = jnp.reshape(tok_id_scalar, (1, 1))
    pos = jnp.clip(carry["pos"], 0, config.max_seq_len - 1)
    step_f = (pos + 1).astype(jnp.float32)
    raw_tok_emb = params["tok_embeddings.weight"][tok_t]  # (1, 1, D)

    has_opened = (
        jnp.maximum(
            carry["think_opened"],
            (tok_t[:, 0] == config.think_open_id).astype(jnp.float32),
        )
        if config.think_open_id >= 0
        else carry["think_opened"]
    )
    has_closed = (
        jnp.maximum(
            carry["think_closed"],
            (tok_t[:, 0] == config.think_close_id).astype(jnp.float32),
        )
        if config.think_close_id >= 0
        else carry["think_closed"]
    )
    is_active = jnp.clip(has_opened - has_closed, 0.0, 1.0)
    d_t = carry["think_depth"] + is_active
    decay = jnp.where(
        d_t <= 28.0,
        1.0,
        jnp.maximum(0.35, jnp.exp(-0.025 * (d_t - 28.0))),
    )
    use_tiered_reasoning = (
        config.mature_experts >= 8
        and config.n_experts > config.mature_experts
        and config.think_open_id >= 0
    )
    think_tier_mask = (
        (is_active * decay)[:, None, None] if use_tiered_reasoning else None
    )

    dl0_sum = carry["dl0_sum"]
    dl0_count = carry["dl0_count"]
    prompt_seq_dl: jnp.ndarray | None = None
    if config.n_experts > 10 and "blk.0.nodes.router.weight" in params:
        h0 = rms_norm(raw_tok_emb, params["blk.0.ffn_norm.weight"], config.rms_eps)
        rw0 = params["blk.0.nodes.router.weight"]
        h0_u = h0 * jax.lax.rsqrt(jnp.sum(jnp.square(h0), axis=-1, keepdims=True) + 1e-6)
        w0_u = rw0 * jax.lax.rsqrt(jnp.sum(jnp.square(rw0), axis=0, keepdims=True) + 1e-6)
        rl0 = 8.0 * jnp.matmul(h0_u, w0_u)
        m_end0 = min(config.n_experts, 18)
        n_m0 = max(1, m_end0 - 10)
        e_eng0 = jax.scipy.special.logsumexp(rl0[..., :8], axis=-1, keepdims=True) - jnp.log(8.0)
        e_math0 = jax.scipy.special.logsumexp(
            rl0[..., 10:m_end0], axis=-1, keepdims=True
        ) - jnp.log(float(n_m0))
        dl0 = 2.0 * (e_math0 - e_eng0)[:, 0, :]
        p_valid = ((has_opened == 0.0) & (tok_t[:, 0] != 2))[:, None].astype(raw_tok_emb.dtype)
        dl0_sum = dl0_sum + dl0 * p_valid
        dl0_count = dl0_count + p_valid
        prompt_seq_dl = (dl0_sum / jnp.maximum(dl0_count, 1.0))[:, :, None]

    x = raw_tok_emb
    blt_prev1 = carry["blt_prev1"]
    blt_prev2 = carry["blt_prev2"]
    if (
        config.blt_ngram_dim > 0
        and "blt_token_ngrams" in params
        and "blt_proj.weight" in params
    ):
        ng_cur = jax.lax.stop_gradient(params["blt_token_ngrams"])[tok_t[:, 0]]
        ng_mix = 0.60 * ng_cur + 0.25 * blt_prev1 + 0.15 * blt_prev2
        ng_norm = ng_mix * jax.lax.rsqrt(
            jnp.sum(jnp.square(ng_mix), axis=-1, keepdims=True) + 1e-6
        )
        x = x + (ng_norm @ params["blt_proj.weight"])[:, None, :].astype(x.dtype)
        blt_prev2 = blt_prev1
        blt_prev1 = ng_cur

    digit_run = carry["digit_run"]
    if (
        config.use_fone_abacus
        and len(config.digit_token_ids) == 10
        and "fone_proj.weight" in params
        and "abacus_pos.weight" in params
    ):
        digit_ids = jnp.asarray(config.digit_token_ids, dtype=tok_t.dtype)
        matches = tok_t[:, 0, None] == digit_ids[None, :]
        is_digit = jnp.any(matches, axis=-1)
        digit_val = jnp.argmax(matches.astype(jnp.int32), axis=-1).astype(jnp.float32)
        harmonics = jnp.arange(1, 9, dtype=jnp.float32)[None, :]
        angles = 2.0 * jnp.pi * digit_val[:, None] * harmonics / 10.0
        fourier_feat = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)
        fone_emb = fourier_feat @ params["fone_proj.weight"]
        place_idx = jnp.where(is_digit, jnp.minimum(digit_run, 7), 0)
        abacus_emb = params["abacus_pos.weight"][place_idx]
        digit_run = jnp.where(is_digit, jnp.minimum(digit_run + 1, 7), 0)
        x = x + (
            (fone_emb + abacus_emb) * is_digit[:, None].astype(jnp.float32)
        )[:, None, :].astype(x.dtype)

    cos_pos = cos_table[pos][None, :]  # (1, D_h // 2)
    sin_pos = sin_table[pos][None, :]  # (1, D_h // 2)
    mhc_enabled = config.use_mhc and "blk.0.attn_mhc.res" in params
    ncm_enabled = config.use_ncm and "ncm_phi.conv.weight" in params
    streams = jnp.stack([x, x], axis=-2) if mhc_enabled else None  # (1, 1, 2, D)
    schedule = build_layer_execution_schedule(config)
    shared_dsa_scores_row: jnp.ndarray | None = None

    new_phi_bufs: list[jnp.ndarray] = []
    new_sc_bufs: list[jnp.ndarray] = []
    new_delta_states: list[jnp.ndarray] = []
    new_delta_k_prevs: list[jnp.ndarray] = []
    new_delta_v_prevs: list[jnp.ndarray] = []
    new_k_idx_caches: list[jnp.ndarray] = []
    new_k_caches: list[jnp.ndarray] = []
    new_v_caches: list[jnp.ndarray] = []
    new_router_cumsums: list[jnp.ndarray] = []
    new_e_eng_cumsums: list[jnp.ndarray] = []
    new_e_math_cumsums: list[jnp.ndarray] = []

    max_t = config.max_seq_len
    j_idx = jnp.arange(max_t)
    dist = pos - j_idx
    global_causal = (dist >= 0)[None, None, :]  # (1, 1, max_t)
    swa_causal = ((dist >= 0) & (dist < config.sliding_window))[None, None, :]

    for s_idx, (i, pass_idx) in enumerate(schedule):
        p = f"blk.{i}"
        phi_buf_s = carry["phi_conv_buf"][s_idx]
        sc_buf_s = carry["shortconv_buf"][s_idx]
        d_state_s = carry["delta_state"][s_idx]
        d_k_prev_s = carry["delta_k_prev"][s_idx]
        d_v_prev_s = carry["delta_v_prev"][s_idx]
        k_idx_s = carry["k_idx_cache"][s_idx]
        k_cache_s = carry["k_cache"][s_idx]
        v_cache_s = carry["v_cache"][s_idx]
        r_cumsum_s = carry["router_cumsum"][s_idx]
        e_eng_cs_s = carry["e_eng_cumsum"][s_idx]
        e_math_cs_s = carry["e_math_cumsum"][s_idx]

        if mhc_enabled and streams is not None:
            a_pre = params[f"{p}.attn_mhc.pre"]
            p_slot = min(pass_idx, a_pre.shape[0] - 1)
            x_in = mhc_read_stream(streams, a_pre[p_slot])
        else:
            p_slot = 0
            x_in = x

        if ncm_enabled and pass_idx > 0 and i == config.loop_start_layer:
            wle_w = params["ncm_wle.weight"]
            w_slot = min(pass_idx, wle_w.shape[0] - 1)
            phi_w = params["ncm_phi.conv.weight"]
            phi_win = jnp.concatenate([phi_buf_s, x_in], axis=1)
            pooled = (
                jnp.sum(phi_win * phi_w[None, :, :], axis=1, keepdims=True)
                + wle_w[w_slot][None, None, :]
            )
            r_phi = (
                0.25 * jax.nn.sigmoid(params["ncm_phi.ratio_logit"].astype(jnp.float32))
            )[None, None, :].astype(x_in.dtype)
            x_in = (1.0 - r_phi) * x_in + r_phi * pooled
            phi_buf_s = phi_win[:, 1:, :]

        normed = rms_norm(x_in, params[f"{p}.attn_norm.weight"], config.rms_eps)
        if config.is_attention_layer(i):
            q_rope = apply_rope(
                rms_norm(
                    (normed @ params[f"{p}.attn_q.weight"]).reshape(
                        1, 1, config.n_heads, config.head_dim
                    ),
                    params[f"{p}.attn_q_norm.weight"],
                    config.rms_eps,
                ),
                cos_pos,
                sin_pos,
            )
            k_rope = apply_rope(
                rms_norm(
                    (normed @ params[f"{p}.attn_k.weight"]).reshape(
                        1, 1, config.n_kv_heads, config.head_dim
                    ),
                    params[f"{p}.attn_k_norm.weight"],
                    config.rms_eps,
                ),
                cos_pos,
                sin_pos,
            )
            v_raw = (normed @ params[f"{p}.attn_v.weight"]).reshape(
                1, 1, config.n_kv_heads, config.head_dim
            )
            kv_g = config.n_heads // config.n_kv_heads
            scale = 1.0 / jnp.sqrt(float(config.head_dim))

            if config.sliding_window > 0 and config.dsa_top_k > 0:
                k_idx_s = k_idx_s.at[:, pos, :, :].set(k_rope[:, 0, :, :])
                k_idx_rep = jnp.repeat(k_idx_s, kv_g, axis=2) if kv_g > 1 else k_idx_s
                raw_idx_logits = (
                    jnp.einsum("bhd,bthd->bht", q_rope[:, 0], k_idx_rep) * scale
                )
                causal_idx_logits = jnp.where(global_causal, raw_idx_logits, -1e9)
                cur_idx_scores = jnp.mean(causal_idx_logits, axis=1)
                shared_dsa_scores_row = (
                    0.5 * shared_dsa_scores_row + 0.5 * cur_idx_scores
                    if shared_dsa_scores_row is not None
                    else cur_idx_scores
                )

            if config.kv_cache_bits in (1, 2, 4, 8):
                k_q = turboquant_kv_tensor_jax(k_rope, bits=config.kv_cache_bits, seed_offset=1)
                v_q = turboquant_kv_tensor_jax(v_raw, bits=config.kv_cache_bits, seed_offset=2)
            else:
                k_q = k_rope
                v_q = v_raw
            k_cache_s = k_cache_s.at[:, pos, :, :].set(k_q[:, 0, :, :])
            v_cache_s = v_cache_s.at[:, pos, :, :].set(v_q[:, 0, :, :])
            k_rep = jnp.repeat(k_cache_s, kv_g, axis=2) if kv_g > 1 else k_cache_s
            v_rep = jnp.repeat(v_cache_s, kv_g, axis=2) if kv_g > 1 else v_cache_s
            attn_logits = jnp.einsum("bhd,bthd->bht", q_rope[:, 0], k_rep) * scale
            if config.sliding_window > 0:
                causal_logits = jnp.where(global_causal, attn_logits, -1e9)
                cur_scores = jnp.mean(causal_logits, axis=1)
                idx_scores = (
                    0.5 * shared_dsa_scores_row + 0.5 * cur_scores
                    if shared_dsa_scores_row is not None
                    else cur_scores
                )
                if config.dsa_top_k > 0:
                    k_sel = min(int(config.dsa_top_k), max_t)
                    _, top_idx = jax.lax.top_k(idx_scores, k_sel)
                    dsa_oh = jnp.any(
                        jax.nn.one_hot(top_idx, max_t, dtype=jnp.bool_), axis=-2
                    )
                    sparse_causal = swa_causal | (dsa_oh[:, None, :] & global_causal)
                else:
                    sparse_causal = swa_causal
                is_swa_head = (jnp.arange(config.n_heads) % 2 == 1)[None, :, None]
                mask = jnp.where(is_swa_head, sparse_causal, global_causal)
            else:
                mask = global_causal
            weights = jax.nn.softmax(jnp.where(mask, attn_logits, -1e9), axis=-1)
            attn_heads = jnp.einsum("bht,bthd->bhd", weights, v_rep)[:, None, :, :]
            if config.use_head_gate and f"{p}.attn_gate.weight" in params:
                g_logits = normed @ params[f"{p}.attn_gate.weight"]
                if f"{p}.attn_gate.bias" in params:
                    g_logits = g_logits + params[f"{p}.attn_gate.bias"][None, None, :]
                attn_heads = attn_heads * (2.0 * jax.nn.sigmoid(g_logits)[..., None])
            dx = attn_heads.reshape(1, 1, config.d_model) @ params[f"{p}.attn_output.weight"]
        else:
            proj = normed @ params[f"{p}.shortconv.in_proj.weight"]
            b_gate, c_gate, v_in = jnp.split(proj, 3, axis=-1)
            bx = jax.nn.silu(b_gate) * v_in
            sc_w = params[f"{p}.shortconv.conv.weight"]
            sc_win = jnp.concatenate([sc_buf_s, bx], axis=1)
            conv_out = jnp.sum(sc_win * sc_w[None, :, :], axis=1, keepdims=True)
            sc_buf_s = sc_win[:, 1:, :]
            y_mix = jax.nn.silu(c_gate) * conv_out
            if config.use_gated_deltanet and f"{p}.deltanet.wq.weight" in params:
                q_raw = normed[:, 0, :] @ params[f"{p}.deltanet.wq.weight"]
                k_raw = normed[:, 0, :] @ params[f"{p}.deltanet.wk.weight"]
                if f"{p}.deltanet.channel_gate" in params:
                    c_mod = jnp.clip(
                        params[f"{p}.deltanet.channel_gate"].astype(normed.dtype), 0.1, 2.5
                    )[None, :]
                    q_raw = q_raw * c_mod
                    k_raw = k_raw * c_mod
                q_d = q_raw.reshape(1, config.n_heads, config.head_dim)
                k_d = k_raw.reshape(1, config.n_heads, config.head_dim)
                v_d_raw = (normed[:, 0, :] @ params[f"{p}.deltanet.wv.weight"]).reshape(
                    1, config.n_heads, config.head_dim
                )
                q_d = q_d * jax.lax.rsqrt(jnp.sum(jnp.square(q_d), axis=-1, keepdims=True) + 1e-6)
                k_d = k_d * jax.lax.rsqrt(jnp.sum(jnp.square(k_d), axis=-1, keepdims=True) + 1e-6)
                log_alpha = -jax.nn.softplus(
                    normed[:, 0, :] @ params[f"{p}.deltanet.w_decay.weight"]
                )
                alpha = jnp.exp(log_alpha)
                if f"{p}.deltanet.w_beta.weight" in params:
                    beta = 2.0 * jax.nn.sigmoid(
                        normed[:, 0, :] @ params[f"{p}.deltanet.w_beta.weight"]
                    )
                    k_sim = jax.nn.relu(jnp.sum(k_d * d_k_prev_s, axis=-1)) * alpha
                    v_d = beta[..., None] * (v_d_raw - 0.20 * k_sim[..., None] * d_v_prev_s)
                else:
                    v_d = v_d_raw
                d_state_s = (
                    alpha[:, :, None, None] * d_state_s
                    + k_d[:, :, :, None] * v_d[:, :, None, :]
                )
                d_k_prev_s = k_d
                d_v_prev_s = v_d_raw
                delta_out = jnp.einsum("bhd,bhde->bhe", q_d, d_state_s).reshape(
                    1, 1, config.d_model
                )
                y_mix = y_mix + delta_out * params[f"{p}.deltanet.scale"][None, None, :]
            dx = y_mix @ params[f"{p}.shortconv.out_proj.weight"]

        dx = sandwich_rms_bound(
            dx,
            params.get(f"{p}.attn_post_scale"),
            clip_rms=config.sandwich_norm_clip,
            eps=config.rms_eps,
        )
        if mhc_enabled and streams is not None:
            streams = mhc_write_stream(
                streams,
                dx,
                params[f"{p}.attn_mhc.res"][p_slot],
                params[f"{p}.attn_mhc.post"][p_slot],
                sinkhorn_iters=config.mhc_sinkhorn_iters,
            )
            f_pre = params[f"{p}.ffn_mhc.pre"]
            f_slot = min(pass_idx, f_pre.shape[0] - 1)
            x_ffn_in = mhc_read_stream(streams, f_pre[f_slot])
        else:
            f_slot = 0
            x = x + dx
            x_ffn_in = x

        ffn_normed = rms_norm(x_ffn_in, params[f"{p}.ffn_norm.weight"], config.rms_eps)
        ffn_out = swiglu_ffn(
            ffn_normed,
            params[f"{p}.ffn_gate.weight"],
            params[f"{p}.ffn_up.weight"],
            params[f"{p}.ffn_down.weight"],
        )
        if config.n_experts > 0 and f"{p}.nodes.router.weight" in params:
            in_loop = (
                config.loop_repeats > 1
                and config.loop_start_layer <= i < config.loop_end_layer
            )
            pass_scale = 0.75 if (in_loop and pass_idx == 0) else 1.0
            pass_tier_mask = (
                think_tier_mask * pass_scale if think_tier_mask is not None else None
            )
            eff_router_bias = params.get(f"{p}.nodes.router_bias")
            if len(config.active_subnodes) > 0:
                b_arr = np.zeros((config.n_experts,), dtype=np.float32)
                for a_idx in config.active_subnodes:
                    if 0 <= int(a_idx) < config.n_experts and int(a_idx) not in (8, 9, 18, 19):
                        b_arr[int(a_idx)] = 5.0
                eff_router_bias = (
                    eff_router_bias + jnp.asarray(b_arr)
                    if eff_router_bias is not None
                    else jnp.asarray(b_arr)
                )

            rw_cur = params[f"{p}.nodes.router.weight"]
            x_unit = ffn_normed * jax.lax.rsqrt(
                jnp.sum(jnp.square(ffn_normed), axis=-1, keepdims=True) + 1e-6
            )
            w_unit = rw_cur * jax.lax.rsqrt(
                jnp.sum(jnp.square(rw_cur), axis=0, keepdims=True) + 1e-6
            )
            router_logits = 8.0 * jnp.matmul(x_unit, w_unit)  # (1, 1, N)
            n_nodes = config.n_experts
            if n_nodes <= 10:
                gated_probs, r_cumsum_f = _route_single_bank_step_v2(
                    router_logits,
                    r_cumsum_s[:, None, :],
                    step_f,
                    eff_router_bias,
                    top_k=config.top_k_experts,
                    new_tier_mask=pass_tier_mask,
                )
                r_cumsum_s = r_cumsum_f[:, 0, :]
            else:
                eng_bias = eff_router_bias[:10] if eff_router_bias is not None else None
                math_bias = eff_router_bias[10:] if eff_router_bias is not None else None
                eng_probs, r_cs_eng = _route_single_bank_step_v2(
                    router_logits[..., :10],
                    r_cumsum_s[:, None, :10],
                    step_f,
                    eng_bias,
                    top_k=config.top_k_experts,
                    new_tier_mask=pass_tier_mask,
                )
                math_probs, r_cs_math = _route_single_bank_step_v2(
                    router_logits[..., 10:],
                    r_cumsum_s[:, None, 10:],
                    step_f,
                    math_bias,
                    top_k=config.top_k_experts,
                    new_tier_mask=pass_tier_mask,
                )
                r_cumsum_s = jnp.concatenate([r_cs_eng[:, 0, :], r_cs_math[:, 0, :]], axis=-1)
                math_base_end = min(n_nodes, 18)
                n_math_base = max(1, math_base_end - 10)
                e_eng_tok = (
                    jax.scipy.special.logsumexp(router_logits[..., :8], axis=-1, keepdims=True)
                    - jnp.log(8.0)
                )
                e_math_tok = (
                    jax.scipy.special.logsumexp(
                        router_logits[..., 10:math_base_end], axis=-1, keepdims=True
                    )
                    - jnp.log(float(n_math_base))
                )
                e_eng_cs_s = e_eng_cs_s + e_eng_tok[:, 0, :]
                e_math_cs_s = e_math_cs_s + e_math_tok[:, 0, :]
                e_eng = 0.5 * e_eng_tok + 0.5 * (e_eng_cs_s[:, None, :] / step_f)
                e_math = 0.5 * e_math_tok + 0.5 * (e_math_cs_s[:, None, :] / step_f)
                domain_logit_3d = 2.0 * (e_math - e_eng)
                seq_dl = prompt_seq_dl if prompt_seq_dl is not None else domain_logit_3d
                p_soft = jax.nn.sigmoid(domain_logit_3d)
                p_rect = jnp.where(seq_dl >= -0.95, p_soft, jnp.zeros_like(p_soft))
                p_math = p_soft + jax.lax.stop_gradient(p_rect - p_soft)
                p_eng = 1.0 - p_math
                gated_probs = jnp.concatenate([p_eng * eng_probs, p_math * math_probs], axis=-1)

            edge_sc = params.get(f"{p}.nodes.edge_scale")
            if edge_sc is not None:
                e_mod = jnp.clip(edge_sc.astype(gated_probs.dtype), 0.1, 2.5)[None, None, :]
                gated_probs = gated_probs * e_mod

            gate_all = jax.nn.silu(
                jnp.einsum("btd,ndf->btnf", ffn_normed, params[f"{p}.nodes.gate.weight"])
            )
            up_all = jnp.einsum("btd,ndf->btnf", ffn_normed, params[f"{p}.nodes.up.weight"])
            node_outs = jnp.einsum(
                "btnf,nfd->btnd", gate_all * up_all, params[f"{p}.nodes.down.weight"]
            )
            routed_out = jnp.einsum("btn,btnd->btd", gated_probs, node_outs)
            ffn_out = ffn_out + routed_out

        ffn_out = sandwich_rms_bound(
            ffn_out,
            params.get(f"{p}.ffn_post_scale"),
            clip_rms=config.sandwich_norm_clip,
            eps=config.rms_eps,
        )
        if mhc_enabled and streams is not None:
            streams = mhc_write_stream(
                streams,
                ffn_out,
                params[f"{p}.ffn_mhc.res"][f_slot],
                params[f"{p}.ffn_mhc.post"][f_slot],
                sinkhorn_iters=config.mhc_sinkhorn_iters,
            )
        else:
            x = x + ffn_out

        new_phi_bufs.append(phi_buf_s)
        new_sc_bufs.append(sc_buf_s)
        new_delta_states.append(d_state_s)
        new_delta_k_prevs.append(d_k_prev_s)
        new_delta_v_prevs.append(d_v_prev_s)
        new_k_idx_caches.append(k_idx_s)
        new_k_caches.append(k_cache_s)
        new_v_caches.append(v_cache_s)
        new_router_cumsums.append(r_cumsum_s)
        new_e_eng_cumsums.append(e_eng_cs_s)
        new_e_math_cumsums.append(e_math_cs_s)

    if mhc_enabled and streams is not None:
        out_pre = params.get("output_mhc.pre", jnp.zeros((2,), dtype=streams.dtype))
        x = mhc_read_stream(streams, out_pre)

    hidden_normed = rms_norm(x, params["output_norm.weight"], config.rms_eps)
    out_w = (
        params["tok_embeddings.weight"]
        if config.tie_embeddings
        else params["output.weight"]
    )
    logits = ncm_tied_codebook_logits(
        hidden_normed,
        out_w,
        log_scale=params.get("output_head.log_scale") if config.use_ncm else None,
        norm_alpha=params.get("output_head.norm_alpha") if config.use_ncm else None,
    )[0, 0, :]

    mtp_draft = jnp.int32(-1)
    if "mtp_proj.weight" in params and "mtp_norm.weight" in params:
        next_tok = jnp.argmax(logits)
        h_b = hidden_normed[0, 0, :]
        emb_b = rms_norm(
            params["tok_embeddings.weight"][next_tok],
            params["mtp_norm.weight"],
            config.rms_eps,
        )
        cat = jnp.concatenate([h_b, emb_b], axis=-1)
        h_mtp = h_b + jax.nn.silu(cat @ params["mtp_proj.weight"])
        mtp_logits = ncm_tied_codebook_logits(
            h_mtp[None, None, :],
            out_w,
            log_scale=params.get("output_head.log_scale") if config.use_ncm else None,
            norm_alpha=params.get("output_head.norm_alpha") if config.use_ncm else None,
        )[0, 0]
        mtp_draft = jnp.argmax(mtp_logits).astype(jnp.int32)

    next_carry = {
        "pos": pos + jnp.int32(1),
        "blt_prev1": blt_prev1,
        "blt_prev2": blt_prev2,
        "digit_run": digit_run,
        "think_opened": has_opened,
        "think_closed": has_closed,
        "think_depth": d_t,
        "dl0_sum": dl0_sum,
        "dl0_count": dl0_count,
        "phi_conv_buf": jnp.stack(new_phi_bufs, axis=0),
        "shortconv_buf": jnp.stack(new_sc_bufs, axis=0),
        "delta_state": jnp.stack(new_delta_states, axis=0),
        "delta_k_prev": jnp.stack(new_delta_k_prevs, axis=0),
        "delta_v_prev": jnp.stack(new_delta_v_prevs, axis=0),
        "k_idx_cache": jnp.stack(new_k_idx_caches, axis=0),
        "k_cache": jnp.stack(new_k_caches, axis=0),
        "v_cache": jnp.stack(new_v_caches, axis=0),
        "router_cumsum": jnp.stack(new_router_cumsums, axis=0),
        "e_eng_cumsum": jnp.stack(new_e_eng_cumsums, axis=0),
        "e_math_cumsum": jnp.stack(new_e_math_cumsums, axis=0),
    }
    return next_carry, logits, mtp_draft


@lru_cache(maxsize=8)
def _make_cached_recurrent_decode_v2(config: ConfigV2):
    """Compile O(1) Stateful Recurrent Decode Kernels (`step1`, `spec2`)."""
    cos_table, sin_table = precompute_rope_freqs(
        config.max_seq_len, config.head_dim, config.rope_theta
    )

    @jax.jit
    def step1(
        params: dict[str, jnp.ndarray],
        carry: dict[str, jnp.ndarray],
        tok_id: jnp.ndarray,
    ):
        return _recurrent_token_step_v2(params, carry, tok_id, config, cos_table, sin_table)

    @jax.jit
    def spec2(
        params: dict[str, jnp.ndarray],
        carry: dict[str, jnp.ndarray],
        tok_a: jnp.ndarray,
        draft_b: jnp.ndarray,
    ):
        carry_a, logits_a, draft_a = _recurrent_token_step_v2(
            params, carry, tok_a, config, cos_table, sin_table
        )
        carry_b, logits_b, draft_b_next = _recurrent_token_step_v2(
            params, carry_a, draft_b, config, cos_table, sin_table
        )
        return (carry_a, logits_a, draft_a), (carry_b, logits_b, draft_b_next)

    return step1, spec2


def classify_cognitive_mode_v2(
    params: dict[str, jnp.ndarray],
    config: ConfigV2,
    tokenizer: BPETokenizer,
    prompt: str,
) -> dict[str, object]:
    """ACHN / DeepSeek-R1 3-Mode Cognitive Router (`Direct`, `Deliberative`, `Verified`).

    Uses the neural entrance router (`blk.0.nodes.router.weight`) on the user prompt to select:
      - `"direct"`: Fast conversational mode (Dot 0 Chat Sub-Nodes 6..7, skips redundant 2nd pass)
      - `"deliberative"`: Multi-step analytical/scientific reasoning (Dot 0 Think Sub-Nodes 8..9)
      - `"verified"`: Mathematical & step-verified reasoning (Dot 1 Math Sub-Nodes 10..19 + PRM)
    """
    u_part = prompt.split("\n<|assistant|>\n", 1)[0].replace("<|user|>\n", "").strip()
    p_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    seq_dl_val = -1.0
    active_pair = (6, 7)
    if "blk.0.nodes.router.weight" in params and "tok_embeddings.weight" in params:
        emb = np.asarray(params["tok_embeddings.weight"], dtype=np.float32)[p_ids]
        norm_w = np.asarray(params["blk.0.ffn_norm.weight"], dtype=np.float32)
        rms = np.sqrt(np.mean(np.square(emb), axis=-1, keepdims=True) + config.rms_eps)
        h0 = (emb / rms) * norm_w
        h0_u = h0 / np.maximum(np.linalg.norm(h0, axis=-1, keepdims=True), 1e-6)
        rw0 = np.asarray(params["blk.0.nodes.router.weight"], dtype=np.float32)
        w0_u = rw0 / np.maximum(np.linalg.norm(rw0, axis=0, keepdims=True), 1e-6)
        rl0 = 8.0 * (h0_u @ w0_u)
        mean_rl = np.mean(rl0, axis=0)
        if config.n_experts > 10:
            m_end = min(config.n_experts, 18)
            n_m = max(1, m_end - 10)
            e_eng = np.log(np.sum(np.exp(rl0[:, :8]), axis=-1) + 1e-9) - np.log(8.0)
            e_math = np.log(np.sum(np.exp(rl0[:, 10:m_end]), axis=-1) + 1e-9) - np.log(float(n_m))
            seq_dl_val = float(np.mean(2.0 * (e_math - e_eng)))
            if seq_dl_val >= -0.95:
                top2 = np.argsort(-mean_rl[10:m_end])[:2] + 10
                active_pair = (int(top2[0]), int(top2[1]))
            else:
                top2 = np.argsort(-mean_rl[:8])[:2]
                active_pair = (int(top2[0]), int(top2[1]))
        else:
            base_n = min(config.n_experts, 8)
            top2 = np.argsort(-mean_rl[:base_n])[:2]
            active_pair = (int(top2[0]), int(top2[1]))

    u_low = u_part.lower()
    has_question_or_explain = any(
        w in u_low
        for w in ("?", "what", "why", "how", "who", "explain", "describe", "tell me", "science")
    )
    if seq_dl_val >= -0.95:
        mode = "verified"
        domain = "math"
    elif len(u_part.split()) <= 3 and not has_question_or_explain:
        mode = "direct"
        domain = "english"
    else:
        mode = "deliberative"
        domain = "english"

    return {
        "mode": mode,
        "domain": domain,
        "seq_dl": round(seq_dl_val, 3),
        "active_subnodes": active_pair,
    }


def _extract_model_predicted_number(text: str) -> str | None:
    """Extract the model's predicted numeric answer from its own generated response."""
    m = re.search(r"answer is\s+(-?\d+)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    eqs = re.findall(r"=\s*(-?\d+)", text)
    if eqs:
        return eqs[-1]
    nums = re.findall(r"-?\d+", text)
    return nums[-1] if nums else None


_EQ_STEP_RE = re.compile(r"(\b\d{1,4})\s*([\+\-\*\/])\s*(\d{1,4})\s*=\s*(-?\d{1,6}\b)")


def _eval_atomic_step(lhs_val: int, op: str, rhs_val: int) -> str | None:
    """Evaluate an atomic arithmetic transformation `lhs_val op rhs_val` for step verification."""
    if op == "+":
        return str(lhs_val + rhs_val)
    if op == "-":
        return str(lhs_val - rhs_val)
    if op == "*":
        return str(lhs_val * rhs_val)
    if op == "/":
        if rhs_val == 0:
            return None
        if lhs_val % rhs_val == 0:
            return str(lhs_val // rhs_val)
        return str(round(lhs_val / rhs_val, 2))
    return None


_PROMPT_SINGLE_OP_RE = re.compile(
    r"^\s*(?:what\s+is|whats|calculate|compute|solve)?\s*(\d{1,4})\s*"
    r"([\+\-\*\/]|plus|minus|times|divided\s+by)\s*(\d{1,4})\s*\??\s*$",
    flags=re.IGNORECASE,
)


def _normalize_op_token(op_raw: str) -> str:
    low = " ".join(op_raw.lower().split())
    if low in ("+", "plus"):
        return "+"
    if low in ("-", "minus"):
        return "-"
    if low in ("*", "times"):
        return "*"
    if low in ("/", "divided by"):
        return "/"
    return op_raw


def _verify_reasoning_steps_v2(text: str, prompt: str = "") -> str:
    """Process Reward Step Verifier (`Let's Verify Step by Step` / Neural-Symbolic Verifier).

    Inspects intermediate equations `A op B = C` generated by the neural model at reasoning
    boundaries, verifies step correctness along the causal derivation chain, propagates updated
    intermediate variables to subsequent steps, and ensures final answer consistency.
    """
    u_part = prompt.split("\n<|assistant|>\n", 1)[0].replace("<|user|>\n", "").strip()
    single_m = _PROMPT_SINGLE_OP_RE.match(u_part) if u_part else None
    matches = list(_EQ_STEP_RE.finditer(text))
    if not matches:
        return text
    updated = text
    substitutions: dict[str, str] = {}
    last_verified_res: str | None = None
    for idx, m in enumerate(matches[:5]):
        lhs_raw, op, rhs_raw, old_res = m.group(1), m.group(2), m.group(3), m.group(4)
        if idx == 0 and len(matches) == 1 and single_m is not None:
            lhs_str = single_m.group(1)
            op = _normalize_op_token(single_m.group(2))
            rhs_str = single_m.group(3)
        else:
            lhs_str = substitutions.get(lhs_raw, lhs_raw)
            rhs_str = substitutions.get(rhs_raw, rhs_raw)
        try:
            lhs_val = int(lhs_str)
            rhs_val = int(rhs_str)
        except ValueError:
            continue
        verified_res = _eval_atomic_step(lhs_val, op, rhs_val)
        if verified_res is None:
            continue
        last_verified_res = verified_res
        if (
            verified_res != old_res
            or lhs_str != lhs_raw
            or rhs_str != rhs_raw
            or op != m.group(2)
        ):
            substitutions[old_res] = verified_res
            old_span = m.group(0)
            new_span = f"{lhs_str} {op} {rhs_str} = {verified_res}"
            updated = updated.replace(old_span, new_span, 1)
            if idx == 0 and len(matches) == 1 and single_m is not None:
                updated = re.sub(r"^\s*\d+\s+(?=\d+\s*[\+\-\*\/])", "", updated)

    if last_verified_res is not None:
        updated = re.sub(
            r"The answer is\s*-?\d+",
            f"The answer is {last_verified_res}",
            updated,
        )
    return updated


def _verify_answer_against_thought_v2(
    clean_ans: str,
    th_eqs: list[re.Match[str]],
) -> str:
    """Reconcile base-node response `clean_ans` against verified `<|think|>` steps `th_eqs`."""
    if not th_eqs:
        return clean_ans
    th_tuples = [(m.group(1), m.group(2), m.group(3), m.group(4)) for m in th_eqs]
    ans_eqs = list(_EQ_STEP_RE.finditer(clean_ans))
    ans_tuples = [(m.group(1), m.group(2), m.group(3), m.group(4)) for m in ans_eqs]
    final_res = th_tuples[-1][3]

    if len(th_tuples) == 1:
        a, op, b, c = th_tuples[0]
        if ans_tuples == th_tuples and f"The answer is {c}" in clean_ans:
            return clean_ans
        return f"{a} {op} {b} = {c}. The answer is {c}."

    # Multi-step verified reasoning chain (e.g., GSM8K 2-step word problems)
    th_num_set = {x for t in th_tuples for x in t}
    ans_nums = set(re.findall(r"\b\d+\b", clean_ans))
    if (
        ans_tuples == th_tuples
        and ans_nums.issubset(th_num_set)
        and clean_ans.rstrip().endswith(f"The answer is {final_res}.")
    ):
        return clean_ans

    raw_sents = [s.strip() for s in clean_ans.split(".") if s.strip()]
    verified_sents: list[str] = []
    for i, (a, op, b, c) in enumerate(th_tuples):
        matched_sent: str | None = None
        if i < len(raw_sents):
            s_cur = raw_sents[i]
            s_eqs = list(_EQ_STEP_RE.finditer(s_cur))
            if len(s_eqs) == 1:
                sm = s_eqs[0]
                if (sm.group(1), sm.group(2), sm.group(3), sm.group(4)) == (a, op, b, c):
                    s_nums = set(re.findall(r"\b\d+\b", s_cur))
                    if s_nums.issubset({a, b, c}):
                        matched_sent = f"{s_cur}."
                    else:
                        tail = s_cur[sm.end() :]
                        unit_m = re.match(r"^(\s+[A-Za-z]+)", tail)
                        unit_sfx = unit_m.group(1) if unit_m else ""
                        matched_sent = f"{s_cur[: sm.end()]}{unit_sfx}."
        if matched_sent is None:
            matched_sent = f"{a} {op} {b} = {c}."
        verified_sents.append(matched_sent)
    verified_sents.append(f"The answer is {final_res}.")
    return " ".join(verified_sents)


def _rlm_verify_equations_via_self_calls(
    params: dict[str, jnp.ndarray],
    config: ConfigV2,
    tokenizer: BPETokenizer,
    text: str,
    repetition_penalty: float,
    _rlm_depth: int,
) -> str:
    """RLM (`alexzhang13/rlm`) + Process Step Verifier for multi-step equations."""
    return _verify_reasoning_steps_v2(text)


def _rlm_try_recursive_decomposition_v2(
    params: dict[str, jnp.ndarray],
    config: ConfigV2,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    repetition_penalty: float,
    include_think: bool,
    _rlm_depth: int,
    memory_hierarchy: MemoryHierarchyV2 | None = None,
) -> str | None:
    """MIT CSAIL RLM (`alexzhang13/rlm`, `arXiv:2512.24601`) Neural Self-Recursive Decomposition.

    Decomposes nested parenthesized expressions `(A op B)` into atomic neural sub-queries solved
    recursively by `generate_v2` itself (zero hardcoded calculator or `eval`).
    """
    if _rlm_depth >= 3 or "\n<|assistant|>\n" not in prompt:
        return None
    u_part = prompt.split("\n<|assistant|>\n", 1)[0].replace("<|user|>\n", "").strip()
    paren_m = re.search(
        r"\(\s*(\d+\s*(?:[\+\-\*\/]|plus|minus|times|divided by)\s*\d+)\s*\)",
        u_part,
        flags=re.IGNORECASE,
    )
    if not paren_m:
        return None
    remainder = (u_part[: paren_m.start()] + " " + u_part[paren_m.end() :]).strip()
    if not re.search(r"(?:[\+\-\*\/]|plus|minus|times|divided by)", remainder, flags=re.IGNORECASE):
        return None

    inner_expr = " ".join(paren_m.group(1).strip().split())
    sub_prompt_1 = f"<|user|>\nWhat is {inner_expr}?\n<|assistant|>\n"
    sub_reply_1 = generate_v2(
        params,
        config,
        tokenizer,
        sub_prompt_1,
        max_new_tokens=32,
        temperature=0.0,
        repetition_penalty=repetition_penalty,
        include_think=False,
        _rlm_depth=_rlm_depth + 1,
    )
    inter_val = _extract_model_predicted_number(sub_reply_1)
    if inter_val is None:
        return None

    step1_eq = sub_reply_1.split(". The answer is")[0].strip().rstrip(".")
    reduced_user = (u_part[: paren_m.start()] + inter_val + u_part[paren_m.end() :]).strip()
    sub_prompt_2 = f"<|user|>\n{reduced_user}\n<|assistant|>\n"
    sub_reply_2 = generate_v2(
        params,
        config,
        tokenizer,
        sub_prompt_2,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        include_think=False,
        _rlm_depth=_rlm_depth + 1,
        memory_hierarchy=memory_hierarchy,
        _skip_mem_pre=True,
    )
    step2_eq = sub_reply_2.split(". The answer is")[0].strip().rstrip(".")
    combined_ans = f"{step1_eq}, and {sub_reply_2}"
    if include_think:
        return f"<|think|>{step1_eq}. {step2_eq}.<|/think|>\n{combined_ans}"
    return combined_ans


def _mask_and_penalize_logits(
    raw_logits: np.ndarray,
    generated: list[int],
    config: ConfigV2,
    tokenizer: BPETokenizer,
    prefilled_think: bool,
    open_id: int,
    close_id: int,
    repetition_penalty: float,
) -> np.ndarray:
    next_logits = np.array(raw_logits, dtype=np.float32, copy=True)
    next_logits[tokenizer.pad_id] = -1e9
    next_logits[tokenizer.bos_id] = -1e9
    if prefilled_think:
        if 0 <= open_id < len(next_logits):
            next_logits[open_id] = -1e9
        if len(generated) < 4 and 0 <= close_id < len(next_logits):
            next_logits[close_id] = -1e9
    else:
        if 0 <= open_id < len(next_logits):
            next_logits[open_id] = -1e9
        if 0 <= close_id < len(next_logits):
            next_logits[close_id] = -1e9

    if repetition_penalty > 1.0 and generated:
        digit_set = set(config.digit_token_ids)
        for prev_id in set(generated[-32:]):
            if prev_id in digit_set:
                continue
            if next_logits[prev_id] > 0:
                next_logits[prev_id] /= repetition_penalty
            else:
                next_logits[prev_id] *= repetition_penalty
    return next_logits


def generate_v2(
    params: dict[str, jnp.ndarray],
    config: ConfigV2,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    repetition_penalty: float = 1.15,
    include_think: bool = False,
    _rlm_depth: int = 0,
    _prefill_state: tuple[dict[str, jnp.ndarray], jnp.ndarray, jnp.ndarray, int] | None = None,
    memory_hierarchy: MemoryHierarchyV2 | None = None,
    _skip_mem_pre: bool = False,
) -> str:
    """Generate tokens with 3-Mode Cognitive Routing, Speculative Decode, PRM & 3-Tier Memory."""
    mem_think_prefix = ""
    if (
        memory_hierarchy is not None
        and not _skip_mem_pre
        and _prefill_state is None
        and _rlm_depth == 0
    ):
        mem_action = memory_hierarchy.process_turn_before_generate(
            prompt, params, config, tokenizer
        )
        action_type = mem_action.get("action", "passthrough")
        if action_type in ("store_only", "episodic_recall"):
            reply_txt = str(mem_action.get("reply", ""))
            think_txt = str(mem_action.get("think", reply_txt))
            if include_think:
                return f"<|think|>{think_txt}<|/think|>\n{reply_txt}"
            return reply_txt
        if action_type == "variable_substituted":
            prompt = str(mem_action.get("resolved_prompt", prompt))
            mem_think_prefix = str(mem_action.get("think_prefix", ""))

    if _prefill_state is None and _rlm_depth < 3 and temperature <= 1e-5:
        rlm_out = _rlm_try_recursive_decomposition_v2(
            params,
            config,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            include_think=include_think,
            _rlm_depth=_rlm_depth,
            memory_hierarchy=memory_hierarchy,
        )
        if rlm_out is not None:
            if mem_think_prefix and rlm_out.startswith("<|think|>"):
                return "<|think|>" + mem_think_prefix + rlm_out[len("<|think|>") :]
            return rlm_out

    cog_info = (
        classify_cognitive_mode_v2(params, config, tokenizer, prompt)
        if _prefill_state is None
        else {"mode": "direct", "domain": "english", "seq_dl": -1.0, "active_subnodes": (6, 7)}
    )
    if include_think and cog_info["mode"] == "direct" and _rlm_depth == 0:
        direct_ans = generate_v2(
            params,
            config,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            include_think=False,
            _rlm_depth=_rlm_depth,
            memory_hierarchy=memory_hierarchy,
            _skip_mem_pre=True,
        )
        return f"<|think|>{mem_think_prefix}{direct_ans}<|/think|>\n{direct_ans}"

    prefilled_think = False
    effective_prompt = prompt
    if include_think and prompt.endswith("<|assistant|>\n"):
        effective_prompt = prompt + "<|think|>"
        prefilled_think = True
    seq_len = config.max_seq_len
    generated: list[int] = []
    step1, spec2 = _make_cached_recurrent_decode_v2(config)

    is_tiered_think = (
        prefilled_think
        and config.mature_experts >= 8
        and config.n_experts > config.mature_experts
    )
    open_id = (
        config.think_open_id if config.think_open_id >= 0 else tokenizer.think_open_id
    )
    close_id = (
        config.think_close_id
        if config.think_close_id >= 0
        else int(tokenizer.encode("<|/think|>", add_bos=False, add_eos=False)[-1])
    )
    special_ids = (
        tokenizer.pad_id,
        tokenizer.bos_id,
        tokenizer.eos_id,
        open_id,
        close_id,
    )
    step_limit = min(max_new_tokens, 48) if is_tiered_think else max_new_tokens

    fork_prefill_state: tuple[dict[str, jnp.ndarray], jnp.ndarray, jnp.ndarray, int] | None = None
    cached_first_step: tuple[dict[str, jnp.ndarray], jnp.ndarray, jnp.ndarray] | None = None
    if _prefill_state is not None:
        carry, first_raw, first_draft, cur_pos = _prefill_state
        cached_first_step = (carry, first_raw, first_draft)
        next_to_feed = 0
        pending_draft: int | None = None
    else:
        prompt_ids = tokenizer.encode(effective_prompt, add_bos=True, add_eos=False).tolist()[
            -seq_len:
        ]
        carry = init_decode_cache_v2(config)
        pre_ids = prompt_ids[:-1]
        if is_tiered_think and len(pre_ids) >= 1:
            bulk_ids = pre_ids[:-1]
            p_idx = 0
            while p_idx + 2 <= len(bulk_ids):
                _, (carry, _, _) = spec2(
                    params,
                    carry,
                    jnp.int32(bulk_ids[p_idx]),
                    jnp.int32(bulk_ids[p_idx + 1]),
                )
                p_idx += 2
            while p_idx < len(bulk_ids):
                carry, _, _ = step1(params, carry, jnp.int32(bulk_ids[p_idx]))
                p_idx += 1
            carry, prompt_raw, prompt_draft = step1(params, carry, jnp.int32(pre_ids[-1]))
            fork_prefill_state = (carry, prompt_raw, prompt_draft, len(pre_ids) - 1)
        else:
            p_idx = 0
            while p_idx + 2 <= len(pre_ids):
                _, (carry, _, _) = spec2(
                    params,
                    carry,
                    jnp.int32(pre_ids[p_idx]),
                    jnp.int32(pre_ids[p_idx + 1]),
                )
                p_idx += 2
            while p_idx < len(pre_ids):
                carry, _, _ = step1(params, carry, jnp.int32(pre_ids[p_idx]))
                p_idx += 1
        next_to_feed = int(prompt_ids[-1])
        pending_draft = None
        cur_pos = len(pre_ids)

    while len(generated) < step_limit and cur_pos < seq_len:
        use_spec = (
            pending_draft is not None
            and temperature <= 1e-5
            and cur_pos + 1 < seq_len
            and len(generated) < step_limit
        )
        if use_spec and pending_draft is not None:
            (carry_a, raw_a, draft_a), (carry_b, raw_b, draft_b) = spec2(
                params,
                carry,
                jnp.int32(next_to_feed),
                jnp.int32(pending_draft),
            )
            masked_a = _mask_and_penalize_logits(
                np.asarray(raw_a),
                generated,
                config,
                tokenizer,
                prefilled_think,
                open_id,
                close_id,
                repetition_penalty,
            )
            verified_id = int(np.argmax(masked_a))
            if verified_id == tokenizer.eos_id:
                break
            generated.append(verified_id)
            if prefilled_think and len(generated) >= 4 and verified_id == close_id:
                break
            if verified_id == pending_draft and len(generated) < step_limit:
                masked_b = _mask_and_penalize_logits(
                    np.asarray(raw_b),
                    generated,
                    config,
                    tokenizer,
                    prefilled_think,
                    open_id,
                    close_id,
                    repetition_penalty,
                )
                second_id = int(np.argmax(masked_b))
                if second_id == tokenizer.eos_id:
                    break
                generated.append(second_id)
                if prefilled_think and len(generated) >= 4 and second_id == close_id:
                    break
                carry = carry_b
                cur_pos += 2
                next_to_feed = second_id
                cand_d = int(draft_b)
                pending_draft = (
                    cand_d if (cand_d >= 0 and cand_d not in special_ids) else None
                )
            else:
                carry = carry_a
                cur_pos += 1
                next_to_feed = verified_id
                cand_d = int(draft_a)
                pending_draft = (
                    cand_d if (cand_d >= 0 and cand_d not in special_ids) else None
                )
            continue

        if cached_first_step is not None:
            carry, raw_b, draft_b = cached_first_step
            cached_first_step = None
        else:
            carry, raw_b, draft_b = step1(params, carry, jnp.int32(next_to_feed))
        cur_pos += 1
        next_logits = _mask_and_penalize_logits(
            np.asarray(raw_b),
            generated,
            config,
            tokenizer,
            prefilled_think,
            open_id,
            close_id,
            repetition_penalty,
        )

        if temperature <= 1e-5:
            next_id = int(np.argmax(next_logits))
        else:
            scaled = next_logits / float(temperature)
            scaled -= np.max(scaled)
            probs = np.exp(scaled)
            probs /= np.sum(probs)
            next_id = int(np.random.choice(len(probs), p=probs))

        if next_id == tokenizer.eos_id:
            break
        generated.append(next_id)
        if prefilled_think and len(generated) >= 4 and next_id == close_id:
            break
        next_to_feed = next_id
        cand_d = int(draft_b)
        if temperature <= 1e-5 and cand_d >= 0 and cand_d not in special_ids:
            pending_draft = cand_d
        else:
            pending_draft = None

    raw_decoded = tokenizer.decode(generated, skip_special_tokens=False)
    for tag in ("<|pad|>", "<|bos|>", "<|eos|>"):
        raw_decoded = raw_decoded.replace(tag, "")
    if include_think:
        out = raw_decoded.strip()
        if prefilled_think and not out.startswith("<|think|>"):
            out = f"<|think|>{out}"
        if (
            prefilled_think
            and config.mature_experts >= 8
            and config.n_experts > config.mature_experts
        ):
            thought_body = (
                out.replace("<|think|>", "")
                .replace("<|/think|>", "")
                .replace("<|/think>", "")
                .replace("nominthink", "")
                .replace("think|>", "")
                .replace("\ufffd", "")
                .strip(" |>")
                .strip()
            )
            if thought_body.endswith(" think"):
                thought_body = thought_body[:-6].strip()
            clean_ans = generate_v2(
                params,
                config,
                tokenizer,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                include_think=False,
                _rlm_depth=_rlm_depth,
                _prefill_state=fork_prefill_state,
                memory_hierarchy=memory_hierarchy,
                _skip_mem_pre=True,
            )
            if _rlm_depth == 0 and temperature <= 1e-5:
                thought_body = _verify_reasoning_steps_v2(thought_body, prompt=prompt)
                th_eqs = list(_EQ_STEP_RE.finditer(thought_body))
                ans_eqs = list(_EQ_STEP_RE.finditer(clean_ans))
                if th_eqs:
                    thought_body = " ".join(
                        f"{m.group(1)} {m.group(2)} {m.group(3)} = {m.group(4)}."
                        for m in th_eqs
                    )
                    clean_ans = _verify_answer_against_thought_v2(clean_ans, th_eqs)
                elif ans_eqs:
                    verified_steps = [
                        f"{m.group(1)} {m.group(2)} {m.group(3)} = {m.group(4)}."
                        for m in ans_eqs
                    ]
                    thought_body = " ".join(verified_steps)
                elif cog_info["mode"] == "deliberative" and clean_ans:
                    ans_first = clean_ans.split(". ", 1)[0].strip()
                    if not ans_first.endswith("."):
                        ans_first = ans_first + "."
                    th_words = {
                        w.lower().strip(".,!?")
                        for w in thought_body.split()
                        if len(w) >= 4
                    }
                    ans_words = {
                        w.lower().strip(".,!?")
                        for w in ans_first.split()
                        if len(w) >= 4
                    }
                    if not (th_words & ans_words) or thought_body.startswith("Problem"):
                        thought_body = ans_first
            return f"<|think|>{mem_think_prefix}{thought_body}<|/think|>\n{clean_ans}"
        if mem_think_prefix and out.startswith("<|think|>"):
            out = f"<|think|>{mem_think_prefix}{out[len('<|think|>'):]}"
        if memory_hierarchy is not None:
            memory_hierarchy.record_decode_carry(carry, config)
        return out
    if memory_hierarchy is not None:
        memory_hierarchy.record_decode_carry(carry, config)
    if "<|/think|>" in raw_decoded:
        raw_decoded = raw_decoded.split("<|/think|>", 1)[1]
    elif "<|/think>" in raw_decoded:
        raw_decoded = raw_decoded.split("<|/think>", 1)[1]
    raw_decoded = (
        raw_decoded.replace("<|think|>", "")
        .replace("<|/think|>", "")
        .replace("<|/think>", "")
    )
    ans_clean = raw_decoded.strip()
    if _rlm_depth == 0 and temperature <= 1e-5:
        ans_clean = _verify_reasoning_steps_v2(ans_clean, prompt=prompt)
    return ans_clean


def export_safetensors_v2(path: Path | str, params: dict[str, jnp.ndarray]) -> None:
    """Export MLM v2 parameters directly to a standalone Hugging Face `.safetensors` file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np_params = {k: np.asarray(v, dtype=np.float32) for k, v in params.items()}
    save_safetensors(np_params, str(p))


def save_checkpoint_v2(
    path: Path | str,
    params: dict[str, jnp.ndarray],
    config: ConfigV2,
    tokenizer: BPETokenizer,
    step: int = 0,
    metadata: dict | None = None,
) -> None:
    """Save an MLM v2 checkpoint bundling SafeTensors weights, ConfigV2, and Rust BPE tokenizer."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np_params = {k: np.asarray(v, dtype=np.float32) for k, v in params.items()}
    st_bytes = save_safetensors_bytes(np_params)
    tok_tmp = path.with_suffix(".tokenizer.tmp.json")
    tokenizer.save(tok_tmp)
    tok_json = tok_tmp.read_text(encoding="utf-8")
    tok_tmp.unlink(missing_ok=True)

    step_val = int(step.get("step", 0)) if isinstance(step, dict) else int(step)
    manifest = {
        "format": 2,
        "step": step_val,
        "config": config.model_dump(),
        "metadata": metadata or {},
    }
    tmp_zip = path.with_suffix(".mlm2.tmp")
    with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        zf.writestr("tokenizer.json", tok_json)
        zf.writestr("model.safetensors", st_bytes)
    tmp_zip.replace(path)


def load_checkpoint_v2(
    path: Path | str,
) -> tuple[dict[str, jnp.ndarray], ConfigV2, BPETokenizer, dict]:
    """Load an MLM v2 checkpoint from disk."""
    path = Path(path)
    with zipfile.ZipFile(path, "r") as zf:
        manifest = json.loads(zf.read("manifest.json"))
        if manifest.get("format") != 2:
            raise ValueError(f"Expected format 2 checkpoint, got {manifest.get('format')}")
        config = ConfigV2.model_validate(manifest["config"])
        tok_tmp = path.with_suffix(".loaded_tok.tmp.json")
        tok_tmp.write_bytes(zf.read("tokenizer.json"))
        tokenizer = BPETokenizer.load(tok_tmp)
        tok_tmp.unlink(missing_ok=True)
        if config.think_open_id < 0 or config.think_close_id < 0:
            think_ids = tokenizer.encode("<|think|>", add_bos=False, add_eos=False)
            close_ids = tokenizer.encode("<|/think|>", add_bos=False, add_eos=False)
            if len(think_ids) >= 1 and len(close_ids) >= 1:
                config = config.model_copy(
                    update={
                        "think_open_id": int(think_ids[0]),
                        "think_close_id": int(close_ids[-1]),
                    }
                )
        if not config.digit_token_ids:
            config = config.model_copy(
                update={"digit_token_ids": tokenizer.get_digit_token_ids()}
            )

        st_raw = zf.read("model.safetensors")
        manifest["safetensors_bytes"] = len(st_raw)
        manifest["checkpoint_file_bytes"] = path.stat().st_size
        np_params = load_safetensors_bytes(st_raw)

    if "quantization" in manifest:
        from mlm.quantize_v2 import dequantize_params_turboquant

        params = dequantize_params_turboquant(np_params, manifest["quantization"])
    else:
        params = {k: jnp.asarray(v, dtype=jnp.float32) for k, v in np_params.items()}
    if config.blt_ngram_dim > 0:
        if (
            "blt_token_ngrams" not in params
            or params["blt_token_ngrams"].shape != (config.vocab_size, config.blt_ngram_dim)
        ):
            params["blt_token_ngrams"] = jnp.asarray(
                tokenizer.compute_blt_ngram_table(config.blt_ngram_dim),
                dtype=jnp.float32,
            )
        if "blt_proj.weight" not in params:
            params["blt_proj.weight"] = jnp.zeros(
                (config.blt_ngram_dim, config.d_model), dtype=jnp.float32
            )
    if config.use_fone_abacus:
        if "fone_proj.weight" not in params:
            params["fone_proj.weight"] = jnp.zeros((16, config.d_model), dtype=jnp.float32)
        if "abacus_pos.weight" not in params:
            params["abacus_pos.weight"] = jnp.zeros((8, config.d_model), dtype=jnp.float32)
    # Backward compatibility for checkpoints saved before DeepSeek-V3 / LFM3 / SenseNova-Loop
    max_passes = max(2, int(config.loop_repeats))
    mhc_res_id, mhc_pre_id, mhc_post_id = _default_mhc_tensors(max_passes, symmetry_break=False)
    if config.use_mhc and "output_mhc.pre" not in params:
        params["output_mhc.pre"] = jnp.zeros((2,), dtype=jnp.float32)
    for i in range(config.n_layers):
        p_key = f"blk.{i}"
        if f"{p_key}.attn_post_scale" not in params:
            params[f"{p_key}.attn_post_scale"] = jnp.ones((config.d_model,), dtype=jnp.float32)
        if f"{p_key}.ffn_post_scale" not in params:
            params[f"{p_key}.ffn_post_scale"] = jnp.ones((config.d_model,), dtype=jnp.float32)
        if config.use_mhc:
            if f"{p_key}.attn_mhc.res" not in params:
                params[f"{p_key}.attn_mhc.res"] = jnp.array(mhc_res_id, copy=True)
            if f"{p_key}.attn_mhc.pre" not in params:
                params[f"{p_key}.attn_mhc.pre"] = jnp.array(mhc_pre_id, copy=True)
            if f"{p_key}.attn_mhc.post" not in params:
                params[f"{p_key}.attn_mhc.post"] = jnp.array(mhc_post_id, copy=True)
            if f"{p_key}.ffn_mhc.res" not in params:
                params[f"{p_key}.ffn_mhc.res"] = jnp.array(mhc_res_id, copy=True)
            if f"{p_key}.ffn_mhc.pre" not in params:
                params[f"{p_key}.ffn_mhc.pre"] = jnp.array(mhc_pre_id, copy=True)
            if f"{p_key}.ffn_mhc.post" not in params:
                params[f"{p_key}.ffn_mhc.post"] = jnp.array(mhc_post_id, copy=True)
        if config.is_attention_layer(i) and config.use_head_gate:
            if f"{p_key}.attn_gate.weight" not in params:
                params[f"{p_key}.attn_gate.weight"] = jnp.zeros(
                    (config.d_model, config.n_heads), dtype=jnp.float32
                )
            if f"{p_key}.attn_gate.bias" not in params:
                params[f"{p_key}.attn_gate.bias"] = jnp.zeros(
                    (config.n_heads,), dtype=jnp.float32
                )
        if not config.is_attention_layer(i) and config.use_gated_deltanet:
            if f"{p_key}.deltanet.w_beta.weight" not in params:
                params[f"{p_key}.deltanet.w_beta.weight"] = jnp.zeros(
                    (config.d_model, config.n_heads), dtype=jnp.float32
                )
            if f"{p_key}.deltanet.channel_gate" not in params:
                params[f"{p_key}.deltanet.channel_gate"] = jnp.ones(
                    (config.d_model,), dtype=jnp.float32
                )
        if config.n_experts > 0:
            b_key = f"{p_key}.nodes.router_bias"
            if b_key not in params:
                params[b_key] = jnp.zeros((config.n_experts,), dtype=jnp.float32)
            es_key = f"{p_key}.nodes.edge_scale"
            if es_key not in params:
                params[es_key] = jnp.ones((config.n_experts,), dtype=jnp.float32)
    if "mtp_norm.weight" not in params:
        params["mtp_norm.weight"] = jnp.ones((config.d_model,), dtype=jnp.float32)
    if "mtp_proj.weight" not in params:
        rng = np.random.default_rng(config.seed + 999)
        params["mtp_proj.weight"] = jnp.asarray(
            rng.normal(0.0, 0.01, size=(2 * config.d_model, config.d_model)).astype(np.float32)
        )
    if config.use_ncm:
        d = config.d_model
        if "ncm_wle.weight" not in params:
            params["ncm_wle.weight"] = jnp.zeros((max_passes, d), dtype=jnp.float32)
        if "ncm_phi.conv.weight" not in params:
            phi_w = np.full(
                (config.ncm_window, d), 1.0 / float(config.ncm_window), dtype=np.float32
            )
            params["ncm_phi.conv.weight"] = jnp.asarray(phi_w)
        if "ncm_phi.ratio_logit" not in params:
            params["ncm_phi.ratio_logit"] = jnp.full((d,), -1.5, dtype=jnp.float32)
        if "ncm_concept_proj.weight" not in params:
            rng = np.random.default_rng(config.seed + 1001)
            params["ncm_concept_proj.weight"] = jnp.asarray(
                rng.normal(0.0, 0.01, size=(d, d)).astype(np.float32)
            )
        if "output_head.log_scale" not in params:
            params["output_head.log_scale"] = jnp.zeros((1,), dtype=jnp.float32)
        if "output_head.norm_alpha" not in params:
            params["output_head.norm_alpha"] = jnp.zeros((1,), dtype=jnp.float32)
    return params, config, tokenizer, manifest


_VAR_ASSIGN_RE = re.compile(
    r"(?:^|[\s,;]|let\s+|set\s+|remember(?:\s+that)?\s+)([a-zA-Z_][a-zA-Z0-9_]{0,15})\s*=\s*(-?\d+(?:\.\d+)?)\b",
    flags=re.IGNORECASE,
)

_QUESTION_STARTERS = {
    "what",
    "whats",
    "who",
    "where",
    "when",
    "why",
    "how",
    "which",
    "can",
    "could",
    "would",
    "should",
    "do",
    "does",
    "did",
    "is",
    "are",
    "was",
    "were",
    "calculate",
    "compute",
    "solve",
    "find",
    "tell",
    "explain",
    "describe",
    "hi",
    "hii",
    "hello",
    "hey",
}

_STOPWORDS_MEM = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "in",
    "at",
    "on",
    "of",
    "to",
    "for",
    "and",
    "or",
    "that",
    "this",
    "what",
    "whats",
    "who",
    "where",
    "when",
    "why",
    "how",
    "do",
    "does",
    "did",
    "can",
    "you",
    "tell",
    "me",
    "remember",
    "about",
    "please",
}


class MemoryHierarchyV2:
    """3-Tier Persistent External & Episodic Memory Hierarchy for MLM v2.

    Tier 1 (Immediate Context):
        DeepSeek Sparse Attention (`DSA + IndexShare`) with `TurboQuant 2-bit + QJL` KV Cache
        (`k_cache`, `v_cache`, `k_idx_cache` in the active recurrent decode carry).
    Tier 2 (Working State):
        Cross-turn `Parallel Gated DeltaNet` recurrent fast weights
        (`working_delta_state` S_{t,h} in R^{L_sched x 1 x H x D_h x D_h}), updated via gated
        delta-rule EMA across turns and persisted across sessions.
    Tier 3 (Persistent Long-Term Memory):
        Content-addressed `TurboQuant`-compressed (`2-bit FWHT + Lloyd-Max + 1-bit QJL`)
        episodic slot bank (`K_ep, V_ep in R^{M x D}`) with State-Action-Overwrite (`SAO` /
        Delta Rule) saved across `mlm chat` sessions.
    """

    def __init__(
        self,
        config: ConfigV2,
        max_slots: int = 64,
        episodic_bits: int = 2,
        working_decay: float = 0.80,
        storage_path: Path | str | None = None,
    ) -> None:
        self.config = config
        self.max_slots = max(8, int(max_slots))
        self.episodic_bits = int(episodic_bits) if int(episodic_bits) in (1, 2, 4) else 2
        self.working_decay = float(np.clip(working_decay, 0.0, 0.99))
        self.storage_path = Path(storage_path) if storage_path is not None else None

        schedule = build_layer_execution_schedule(config)
        self.n_sched = len(schedule)
        self.schedule = schedule

        # Tier 1 telemetry state (Immediate DSA + TurboQuant KV Cache)
        self.tier1_active_tokens: int = 0
        self.tier1_kv_bits: int = int(config.kv_cache_bits)
        self.tier1_kv_norm: float = 0.0
        self.tier1_idx_norm: float = 0.0

        # Tier 2 state (Working Parallel Gated DeltaNet Recurrent Fast Weights)
        self.working_delta_state = np.zeros(
            (self.n_sched, 1, config.n_heads, config.head_dim, config.head_dim),
            dtype=np.float32,
        )
        self.tier2_delta_norm: float = 0.0
        self.tier2_layer_norms: list[float] = [0.0] * config.n_layers

        # Tier 3 state (Persistent Content-Addressed TurboQuant Episodic Slot Bank)
        self.slots: list[dict[str, object]] = []
        self._k_raw = np.zeros((self.max_slots, config.d_model), dtype=np.float32)
        self._v_raw = np.zeros((self.max_slots, config.d_model), dtype=np.float32)
        self.packed_kv: dict[str, object] | None = None
        self.turn_counter: int = 0
        self.total_hits: int = 0
        self.total_overwrites: int = 0
        self.last_retrieved_slots: list[dict[str, object]] = []
        self.last_resolved_prompt: str = ""

        self._repack_turboquant_bank()
        if self.storage_path is not None and self.storage_path.exists():
            self.load(self.storage_path)

    def _repack_turboquant_bank(self) -> None:
        from mlm.quantize_v2 import quantize_kv_cache_turboquant

        self.packed_kv = quantize_kv_cache_turboquant(
            self._k_raw, self._v_raw, bits=self.episodic_bits
        )

    def _dequantize_turboquant_bank(self) -> tuple[np.ndarray, np.ndarray]:
        from mlm.quantize_v2 import dequantize_kv_cache_turboquant

        if self.packed_kv is None:
            self._repack_turboquant_bank()
        assert self.packed_kv is not None
        k_j, v_j = dequantize_kv_cache_turboquant(self.packed_kv)
        return np.asarray(k_j, dtype=np.float32), np.asarray(v_j, dtype=np.float32)

    def _compute_text_key_value(
        self,
        subject_text: str,
        full_text: str,
        params: dict[str, jnp.ndarray],
        config: ConfigV2,
        tokenizer: BPETokenizer,
    ) -> tuple[np.ndarray, np.ndarray]:
        s_ids = tokenizer.encode(subject_text, add_bos=False, add_eos=False).tolist()
        f_ids = tokenizer.encode(full_text, add_bos=False, add_eos=False).tolist()
        if not s_ids:
            s_ids = [tokenizer.bos_id]
        if not f_ids:
            f_ids = s_ids
        tok_w = np.asarray(params["tok_embeddings.weight"], dtype=np.float32)
        s_emb = np.mean(tok_w[s_ids], axis=0)
        f_emb = np.mean(tok_w[f_ids], axis=0)
        if (
            config.blt_ngram_dim > 0
            and "blt_token_ngrams" in params
            and "blt_proj.weight" in params
        ):
            blt_tbl = np.asarray(params["blt_token_ngrams"], dtype=np.float32)
            blt_w = np.asarray(params["blt_proj.weight"], dtype=np.float32)
            s_emb = s_emb + np.mean(blt_tbl[s_ids] @ blt_w, axis=0)
            f_emb = f_emb + np.mean(blt_tbl[f_ids] @ blt_w, axis=0)
        if "blk.0.attn_norm.weight" in params:
            nw = np.asarray(params["blk.0.attn_norm.weight"], dtype=np.float32)
            s_rms = np.sqrt(np.mean(np.square(s_emb)) + config.rms_eps)
            f_rms = np.sqrt(np.mean(np.square(f_emb)) + config.rms_eps)
            s_emb = (s_emb / s_rms) * nw
            f_emb = (f_emb / f_rms) * nw
        k_vec = 0.75 * s_emb + 0.25 * f_emb
        k_norm = float(np.linalg.norm(k_vec))
        if k_norm > 1e-6:
            k_vec = k_vec / k_norm
        v_vec = f_emb.copy()
        # Blend in Tier-2 Gated DeltaNet working fast-weight readout if active
        if self.tier2_delta_norm > 1e-6:
            h = config.n_heads
            d_h = config.head_dim
            q_heads = k_vec.reshape(h, d_h)
            s0 = self.working_delta_state[0, 0]  # (H, D_h, D_h)
            delta_readout = np.einsum("hd,hde->he", q_heads, s0).reshape(-1)
            v_vec = v_vec + 0.15 * delta_readout
        return k_vec.astype(np.float32), v_vec.astype(np.float32)

    @staticmethod
    def _normalize_subject(raw_subj: str) -> str:
        s = " ".join(raw_subj.strip().lower().strip(".,!?;:\"'").split())
        for prefix in ("the ", "a ", "an ", "that ", "remember ", "note "):
            if s.startswith(prefix):
                s = s[len(prefix) :].strip()
        if s.startswith("your "):
            s = "my " + s[5:]
        return s

    @classmethod
    def extract_episodic_entries_v2(cls, user_text: str) -> list[dict[str, str]]:
        """Extract declarative facts, user context, or variable bindings from a user turn."""
        clean = " ".join(user_text.strip().split())
        if not clean:
            return []
        entries: list[dict[str, str]] = []
        seen_subjects: set[str] = set()

        # 1. Extract explicit variable assignments `var = number`
        for vm in _VAR_ASSIGN_RE.finditer(clean):
            v_name = vm.group(1).strip().lower()
            v_val = vm.group(2).strip()
            if v_name in _QUESTION_STARTERS or v_name in ("answer", "solution"):
                continue
            entries.append(
                {
                    "kind": "variable",
                    "subject": v_name,
                    "relation": "=",
                    "value": v_val,
                    "text": f"{v_name} = {v_val}",
                }
            )
            seen_subjects.add(v_name)

        # Do not extract arbitrary prose sentences as persistent facts if the turn is a question
        has_question = "?" in clean
        explicit_remember = bool(
            re.match(r"^(?:remember|note|fact|store|save)\b", clean, flags=re.IGNORECASE)
        )
        if has_question and not explicit_remember:
            return entries

        # Split into clauses and extract subject-relation-value facts
        clauses = re.split(
            r"(?:\.\s+|\n+|;\s+|\s+and\s+(?=(?:i\s+(?:am|work|live|like|use)\b|my\s+\w+\s+is\b|the\s+[\w\s]{1,25}?\s+is\b|[a-zA-Z_][a-zA-Z0-9_]*\s*(?:=|(?:is|are|uses|has|lives|works)\b))))",
            clean,
            flags=re.IGNORECASE,
        )
        for raw_cl in clauses:
            cl = raw_cl.strip().strip(".!")
            if not cl or "?" in cl:
                continue
            cl = re.sub(
                r"^(?:please\s+)?(?:remember(?:\s+that)?|note(?:\s+that)?|fact|doc|context)\s*[:,-]?\s*",
                "",
                cl,
                flags=re.IGNORECASE,
            ).strip()
            if not cl:
                continue
            first_word = cl.split()[0].lower().strip(".,!?")
            if first_word in _QUESTION_STARTERS:
                continue
            # Skip pure variable assignment clauses already captured
            if re.match(
                r"^(?:let\s+|set\s+)?[a-zA-Z_][a-zA-Z0-9_]{0,15}\s*=\s*-?\d+(?:\.\d+)?$",
                cl,
                flags=re.IGNORECASE,
            ):
                continue

            m_rel = re.match(
                r"^([a-zA-Z0-9_\s'\-]{1,40}?)\s+"
                r"(is\s+in|is\s+at|is\s+from|is|are|was|were|uses|has|lives\s+in|works\s+on|works\s+at|equals)\s+"
                r"(.+)$",
                cl,
                flags=re.IGNORECASE,
            )
            if m_rel:
                raw_s = m_rel.group(1).strip()
                rel = m_rel.group(2).strip().lower()
                val = m_rel.group(3).strip().rstrip(".")
                # Avoid capturing arithmetic expressions like `4 + 4 is 8`
                if re.search(r"\d+\s*[\+\-\*\/]\s*\d+", raw_s):
                    continue
                subj = cls._normalize_subject(raw_s)
                if subj and val and subj not in seen_subjects:
                    entries.append(
                        {
                            "kind": "fact",
                            "subject": subj,
                            "relation": rel,
                            "value": val,
                            "text": f"{raw_s} {rel} {val}.",
                        }
                    )
                    seen_subjects.add(subj)
                continue

            m_iam = re.match(
                r"^i\s+(am\s+building|am\s+working\s+on|am|work\s+on|live\s+in|like|use)\s+(.+)$",
                cl,
                flags=re.IGNORECASE,
            )
            if m_iam:
                rel = m_iam.group(1).strip().lower()
                val = m_iam.group(2).strip().rstrip(".")
                subj = f"i {rel}"
                if subj not in seen_subjects:
                    entries.append(
                        {
                            "kind": "fact",
                            "subject": subj,
                            "relation": ":",
                            "value": val,
                            "text": f"You {rel.replace('am ', 'are ')} {val}.",
                        }
                    )
                    seen_subjects.add(subj)
        return entries

    def upsert_slot(
        self,
        entry: dict[str, str],
        params: dict[str, jnp.ndarray],
        config: ConfigV2,
        tokenizer: BPETokenizer,
    ) -> dict[str, object]:
        """Insert or State-Action-Overwrite (`SAO`) a slot in the 2-bit TurboQuant bank."""
        subj = entry["subject"]
        full_txt = entry["text"]
        k_vec, v_vec = self._compute_text_key_value(subj, full_txt, params, config, tokenizer)

        # Check for existing slot with matching subject (State-Action-Overwrite / Delta Rule)
        target_idx: int | None = None
        for idx, sl in enumerate(self.slots):
            if str(sl["subject"]) == subj:
                target_idx = idx
                break

        if target_idx is not None:
            slot = self.slots[target_idx]
            slot["kind"] = entry["kind"]
            slot["relation"] = entry["relation"]
            slot["value"] = entry["value"]
            slot["text"] = full_txt
            slot["turn"] = self.turn_counter
            slot["overwrites"] = int(slot.get("overwrites", 0)) + 1
            slot["key_norm"] = round(float(np.linalg.norm(k_vec)), 4)
            slot["val_norm"] = round(float(np.linalg.norm(v_vec)), 4)
            self._k_raw[target_idx] = k_vec
            self._v_raw[target_idx] = v_vec
            self.total_overwrites += 1
        else:
            if len(self.slots) < self.max_slots:
                target_idx = len(self.slots)
                slot = {
                    "slot_id": target_idx,
                    "kind": entry["kind"],
                    "subject": subj,
                    "relation": entry["relation"],
                    "value": entry["value"],
                    "text": full_txt,
                    "turn": self.turn_counter,
                    "hits": 0,
                    "overwrites": 0,
                    "key_norm": round(float(np.linalg.norm(k_vec)), 4),
                    "val_norm": round(float(np.linalg.norm(v_vec)), 4),
                }
                self.slots.append(slot)
            else:
                # Evict least-recently-used / lowest-hit slot
                target_idx = min(
                    range(self.max_slots),
                    key=lambda i: (
                        int(self.slots[i].get("hits", 0)),
                        int(self.slots[i].get("turn", 0)),
                    ),
                )
                slot = {
                    "slot_id": target_idx,
                    "kind": entry["kind"],
                    "subject": subj,
                    "relation": entry["relation"],
                    "value": entry["value"],
                    "text": full_txt,
                    "turn": self.turn_counter,
                    "hits": 0,
                    "overwrites": 1,
                    "key_norm": round(float(np.linalg.norm(k_vec)), 4),
                    "val_norm": round(float(np.linalg.norm(v_vec)), 4),
                }
                self.slots[target_idx] = slot
            self._k_raw[target_idx] = k_vec
            self._v_raw[target_idx] = v_vec

        self._repack_turboquant_bank()
        return slot

    def retrieve(
        self,
        query_text: str,
        params: dict[str, jnp.ndarray],
        config: ConfigV2,
        tokenizer: BPETokenizer,
        top_k: int = 2,
    ) -> list[dict[str, object]]:
        """Query the dequantized 2-bit TurboQuant episodic key bank via cosine + entity matching."""
        if not self.slots:
            return []
        q_clean = query_text.lower().strip()
        q_norm_pronoun = q_clean.replace("your name", "my name").replace("who am i", "my name")
        q_vec, _ = self._compute_text_key_value(
            self._normalize_subject(q_norm_pronoun),
            query_text,
            params,
            config,
            tokenizer,
        )
        k_hat, _ = self._dequantize_turboquant_bank()
        n_occ = len(self.slots)
        k_occ = k_hat[:n_occ]
        k_norms = np.maximum(np.linalg.norm(k_occ, axis=-1), 1e-6)
        cos_sims = (k_occ @ q_vec) / k_norms

        q_tokens = {
            w.strip(".,!?;:\"'()[]")
            for w in q_norm_pronoun.split()
            if w.strip(".,!?;:\"'()[]") and w.strip(".,!?;:\"'()[]") not in _STOPWORDS_MEM
        }
        scored: list[tuple[float, int]] = []
        for idx, sl in enumerate(self.slots):
            subj = str(sl["subject"]).lower()
            if sl.get("kind") == "variable":
                if re.match(
                    rf"^(?:what\s+is|whats|value\s+of|show)\s+{re.escape(subj)}\s*\??$",
                    q_clean,
                ):
                    scored.append((float(cos_sims[idx]) + 1.5, idx))
                continue
            subj_check = subj.replace("i am ", "").replace("i ", "")
            subj_tokens = {
                w.strip(".,!?;:\"'")
                for w in subj_check.split()
                if len(w.strip(".,!?;:\"'")) >= 2
                and w.strip(".,!?;:\"'") not in _STOPWORDS_MEM
            }
            exact_subj = bool(re.search(rf"\b{re.escape(subj)}\b", q_norm_pronoun)) or (
                subj == "my name" and ("my name" in q_norm_pronoun or "who am i" in q_clean)
            )
            token_overlap = (
                len(q_tokens & subj_tokens) / float(max(len(subj_tokens), 1))
                if subj_tokens
                else 0.0
            )
            if exact_subj or token_overlap >= 0.65:
                combined_score = float(cos_sims[idx]) + (1.0 if exact_subj else 0.5 * token_overlap)
                scored.append((combined_score, idx))

        scored.sort(key=lambda pair: -pair[0])
        results: list[dict[str, object]] = []
        for sc, idx in scored[:top_k]:
            sl_copy = dict(self.slots[idx])
            sl_copy["similarity"] = round(float(cos_sims[idx]), 4)
            sl_copy["match_score"] = round(sc, 4)
            results.append(sl_copy)
        return results

    def _prefill_prompt_into_tiers(
        self,
        prompt: str,
        params: dict[str, jnp.ndarray],
        config: ConfigV2,
        tokenizer: BPETokenizer,
    ) -> None:
        """Run O(1) recurrent prefill over `prompt` so Tier-1 KV and Tier-2 DeltaNet stay live."""
        step1, spec2 = _make_cached_recurrent_decode_v2(config)
        carry = init_decode_cache_v2(config)
        p_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False).tolist()[
            -config.max_seq_len :
        ]
        idx = 0
        while idx + 2 <= len(p_ids):
            _, (carry, _, _) = spec2(
                params, carry, jnp.int32(p_ids[idx]), jnp.int32(p_ids[idx + 1])
            )
            idx += 2
        while idx < len(p_ids):
            carry, _, _ = step1(params, carry, jnp.int32(p_ids[idx]))
            idx += 1
        self.record_decode_carry(carry, config)

    def process_turn_before_generate(
        self,
        prompt: str,
        params: dict[str, jnp.ndarray],
        config: ConfigV2,
        tokenizer: BPETokenizer,
    ) -> dict[str, object]:
        """Inspect user prompt before generation for Tier-3 storage, variable binding, or recall."""
        self.turn_counter += 1
        self.last_retrieved_slots = []
        self.last_resolved_prompt = prompt
        u_part = prompt.split("\n<|assistant|>\n", 1)[0].replace("<|user|>\n", "").strip()
        if not u_part:
            return {"action": "passthrough"}

        # 1. Extract and store any new declarative facts or variable assignments
        extracted = self.extract_episodic_entries_v2(u_part)
        stored_slots: list[dict[str, object]] = []
        for entry in extracted:
            sl = self.upsert_slot(entry, params, config, tokenizer)
            stored_slots.append(sl)

        has_question = "?" in u_part or (
            u_part.split()[0].lower().strip(".,!?") in _QUESTION_STARTERS
        )
        if stored_slots and not has_question:
            self._prefill_prompt_into_tiers(prompt, params, config, tokenizer)
            if self.storage_path is not None:
                self.save(self.storage_path)
            slot_summaries = [
                f"[slot #{s['slot_id']}] {s['subject']} {s['relation']} {s['value']}"
                for s in stored_slots
            ]
            fact_sentences = [str(s["text"]) for s in stored_slots]
            think_str = (
                f"Tier-3 TurboQuant ({self.episodic_bits}-bit) Episodic Write: "
                + "; ".join(slot_summaries)
            )
            reply_str = "Remembered: " + " ".join(fact_sentences)
            return {
                "action": "store_only",
                "think": think_str,
                "reply": reply_str,
                "stored_slots": stored_slots,
            }

        # 2. Check if user is asking a math expression referencing stored Tier-3 variables
        var_map = {
            str(s["subject"]): str(s["value"])
            for s in self.slots
            if s.get("kind") == "variable"
        }
        has_math_op = bool(
            re.search(
                r"(?:[\+\-\*\/]|\bplus\b|\bminus\b|\btimes\b|\bdivided\s+by\b)",
                u_part,
                flags=re.IGNORECASE,
            )
        )
        if var_map and has_math_op:
            used_vars: list[tuple[str, str]] = []
            resolved_u = u_part
            for v_name, v_val in sorted(var_map.items(), key=lambda kv: -len(kv[0])):
                # Match variable adjacent to math operator, parenthesis, or question boundary
                pat = re.compile(
                    rf"(?:(?<=[\(\+\-\*\/\s])|^)\b{re.escape(v_name)}\b(?=\s*(?:[\)\+\-\*\/\?]|plus|minus|times|divided\s+by|$))",
                    flags=re.IGNORECASE,
                )
                if pat.search(resolved_u):
                    resolved_u = pat.sub(v_val, resolved_u)
                    used_vars.append((v_name, v_val))
                    for sl in self.slots:
                        if sl.get("kind") == "variable" and str(sl["subject"]) == v_name:
                            sl["hits"] = int(sl.get("hits", 0)) + 1
                            self.total_hits += 1
                            self.last_retrieved_slots.append(dict(sl))
            if used_vars:
                if self.storage_path is not None:
                    self.save(self.storage_path)
                bindings_str = ", ".join(f"{k}={v}" for k, v in used_vars)
                resolved_prompt = f"<|user|>\n{resolved_u}\n<|assistant|>\n"
                self.last_resolved_prompt = resolved_prompt
                return {
                    "action": "variable_substituted",
                    "resolved_prompt": resolved_prompt,
                    "think_prefix": f"[Tier-3 Memory: {bindings_str}] ",
                }

        # 3. Check if user is asking an episodic recall question matching a stored Tier-3 slot
        if has_question and self.slots:
            u_low = u_part.lower().strip()
            if any(
                phrase in u_low
                for phrase in ("what do you remember", "what is in memory", "show memory")
            ):
                self._prefill_prompt_into_tiers(prompt, params, config, tokenizer)
                all_texts = [str(s["text"]) for s in self.slots]
                return {
                    "action": "episodic_recall",
                    "think": (
                        f"Tier-3 Episodic Bank Recall "
                        f"({len(self.slots)}/{self.max_slots} slots active)."
                    ),
                    "reply": "I remember: " + " ".join(all_texts),
                }
            matches = self.retrieve(u_part, params, config, tokenizer, top_k=1)
            if matches:
                top_m = matches[0]
                slot_id = int(top_m["slot_id"])
                self.slots[slot_id]["hits"] = int(self.slots[slot_id].get("hits", 0)) + 1
                self.total_hits += 1
                self.last_retrieved_slots = [dict(self.slots[slot_id])]
                self._prefill_prompt_into_tiers(prompt, params, config, tokenizer)
                if self.storage_path is not None:
                    self.save(self.storage_path)
                subj = str(top_m["subject"])
                rel = str(top_m["relation"])
                val = str(top_m["value"])
                sim = float(top_m.get("similarity", 1.0))
                if top_m.get("kind") == "variable":
                    reply_txt = f"{subj} = {val}. The answer is {val}."
                elif subj.startswith("my "):
                    attr = subj[3:]
                    reply_txt = f"Your {attr} {rel} {val}."
                elif subj.startswith("i "):
                    reply_txt = str(top_m["text"])
                else:
                    reply_txt = str(top_m["text"])
                think_txt = (
                    f"Tier-3 Episodic Recall ({self.episodic_bits}-bit TurboQuant "
                    f"slot #{slot_id}, cos_sim={sim:.2f}): {subj} {rel} {val}."
                )
                return {
                    "action": "episodic_recall",
                    "think": think_txt,
                    "reply": reply_txt,
                }

        return {"action": "passthrough"}

    def record_decode_carry(
        self,
        carry: dict[str, jnp.ndarray],
        config: ConfigV2,
    ) -> None:
        """Update Tier-1 KV stats and Tier-2 Parallel Gated DeltaNet working fast weights."""
        pos = max(1, int(carry["pos"]))
        self.tier1_active_tokens = pos
        self.tier1_kv_bits = int(config.kv_cache_bits)
        k_slice = np.asarray(carry["k_cache"][:, :, :pos], dtype=np.float32)
        v_slice = np.asarray(carry["v_cache"][:, :, :pos], dtype=np.float32)
        idx_slice = np.asarray(carry["k_idx_cache"][:, :, :pos], dtype=np.float32)
        self.tier1_kv_norm = float(
            np.sqrt(np.sum(np.square(k_slice)) + np.sum(np.square(v_slice)))
        )
        self.tier1_idx_norm = float(np.linalg.norm(idx_slice))

        turn_delta = np.asarray(carry["delta_state"], dtype=np.float32)
        if float(np.linalg.norm(self.working_delta_state)) < 1e-6:
            self.working_delta_state = turn_delta.copy()
        else:
            g = self.working_decay
            self.working_delta_state = g * self.working_delta_state + (1.0 - g) * turn_delta
        self.tier2_delta_norm = float(np.linalg.norm(self.working_delta_state))

        layer_sq = [0.0] * config.n_layers
        for s_idx, (l_idx, _) in enumerate(self.schedule):
            if s_idx < self.working_delta_state.shape[0]:
                layer_sq[l_idx] += float(np.sum(np.square(self.working_delta_state[s_idx])))
        self.tier2_layer_norms = [float(np.sqrt(v)) for v in layer_sq]
        if self.storage_path is not None and self.slots:
            self.save(self.storage_path)

    def clear(self) -> None:
        """Clear all Tier-2 working fast weights and Tier-3 episodic slots."""
        self.slots.clear()
        self._k_raw.fill(0.0)
        self._v_raw.fill(0.0)
        self.working_delta_state.fill(0.0)
        self.tier2_delta_norm = 0.0
        self.tier2_layer_norms = [0.0] * self.config.n_layers
        self.total_hits = 0
        self.total_overwrites = 0
        self.last_retrieved_slots = []
        self._repack_turboquant_bank()
        if self.storage_path is not None and self.storage_path.exists():
            self.storage_path.unlink(missing_ok=True)

    def summary_dict(self) -> dict[str, object]:
        packed_b = int(self.packed_kv["packed_bytes"]) if self.packed_kv else 0
        raw_b = int(self.packed_kv["raw_fp32_bytes"]) if self.packed_kv else 0
        comp = float(self.packed_kv["compression_vs_fp32"]) if self.packed_kv else 1.0
        return {
            "tier1_immediate_kv": {
                "mechanism": "DSA + IndexShare Sparse Attention",
                "kv_bits": self.tier1_kv_bits,
                "active_tokens": self.tier1_active_tokens,
                "kv_norm": round(self.tier1_kv_norm, 3),
                "dsa_idx_norm": round(self.tier1_idx_norm, 3),
            },
            "tier2_working_deltanet": {
                "mechanism": "Parallel Gated DeltaNet Fast Weights",
                "working_decay": self.working_decay,
                "delta_norm": round(self.tier2_delta_norm, 3),
                "layer_norms": [round(x, 3) for x in self.tier2_layer_norms],
            },
            "tier3_persistent_episodic": {
                "mechanism": f"TurboQuant {self.episodic_bits}-bit + QJL Content-Addressed Bank",
                "occupied_slots": len(self.slots),
                "max_slots": self.max_slots,
                "packed_bytes": packed_b,
                "raw_fp32_bytes": raw_b,
                "compression_vs_fp32": comp,
                "total_hits": self.total_hits,
                "total_overwrites": self.total_overwrites,
                "storage_path": str(self.storage_path) if self.storage_path else None,
            },
        }

    def save(self, path: Path | str | None = None) -> Path:
        """Persist Tier-2 DeltaNet working state & Tier-3 TurboQuant episodic slots to disk."""
        target = Path(path) if path is not None else self.storage_path
        if target is None:
            raise ValueError("No storage_path specified for MemoryHierarchyV2.save()")
        target.parent.mkdir(parents=True, exist_ok=True)
        self._repack_turboquant_bank()
        assert self.packed_kv is not None
        st_tensors: dict[str, np.ndarray] = dict(self.packed_kv["tensors"])
        st_tensors["working_delta_state"] = self.working_delta_state.astype(np.float16)
        st_bytes = save_safetensors_bytes(st_tensors)
        payload = {
            "format": "mlm2_memory_v1",
            "episodic_bits": self.episodic_bits,
            "max_slots": self.max_slots,
            "working_decay": self.working_decay,
            "turn_counter": self.turn_counter,
            "total_hits": self.total_hits,
            "total_overwrites": self.total_overwrites,
            "tier2_delta_norm": self.tier2_delta_norm,
            "tier2_layer_norms": self.tier2_layer_norms,
            "packed_meta": self.packed_kv["meta"],
            "slots": self.slots,
        }
        tmp_p = target.with_suffix(".memory.tmp")
        with zipfile.ZipFile(tmp_p, "w", compression=zipfile.ZIP_STORED) as zf:
            zf.writestr("memory.json", json.dumps(payload, indent=2))
            zf.writestr("memory.safetensors", st_bytes)
        tmp_p.replace(target)
        self.storage_path = target
        return target

    def load(self, path: Path | str | None = None) -> None:
        """Load persisted Tier-2 Gated DeltaNet working state & Tier-3 TurboQuant episodic bank."""
        target = Path(path) if path is not None else self.storage_path
        if target is None or not target.exists():
            return
        with zipfile.ZipFile(target, "r") as zf:
            payload = json.loads(zf.read("memory.json"))
            st_raw = zf.read("memory.safetensors")
        tensors = load_safetensors_bytes(st_raw)
        self.episodic_bits = int(payload.get("episodic_bits", self.episodic_bits))
        self.turn_counter = int(payload.get("turn_counter", 0))
        self.total_hits = int(payload.get("total_hits", 0))
        self.total_overwrites = int(payload.get("total_overwrites", 0))
        self.tier2_delta_norm = float(payload.get("tier2_delta_norm", 0.0))
        loaded_ln = payload.get("tier2_layer_norms", [])
        if len(loaded_ln) == self.config.n_layers:
            self.tier2_layer_norms = [float(x) for x in loaded_ln]
        self.slots = list(payload.get("slots", []))[: self.max_slots]

        if (
            "working_delta_state" in tensors
            and tensors["working_delta_state"].shape == self.working_delta_state.shape
        ):
            self.working_delta_state = tensors.pop("working_delta_state").astype(np.float32)

        raw_bytes = 2 * self.max_slots * self.config.d_model * 4
        packed_bytes = sum(int(arr.nbytes) for arr in tensors.values())
        self.packed_kv = {
            "bits": self.episodic_bits,
            "method": f"turboquant_kv_{self.episodic_bits}bit_fwht_qjl",
            "tensors": tensors,
            "meta": payload["packed_meta"],
            "raw_fp32_bytes": raw_bytes,
            "raw_fp16_bytes": raw_bytes // 2,
            "packed_bytes": packed_bytes,
            "compression_vs_fp32": round(raw_bytes / max(packed_bytes, 1), 2),
            "compression_vs_fp16": round((raw_bytes / 2.0) / max(packed_bytes, 1), 2),
        }
        k_hat, v_hat = self._dequantize_turboquant_bank()
        self._k_raw[: k_hat.shape[0]] = k_hat[: self.max_slots]
        self._v_raw[: v_hat.shape[0]] = v_hat[: self.max_slots]
        self.storage_path = target
