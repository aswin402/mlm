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
    router_bias_gamma: float = Field(default=1e-3, ge=0.0, le=0.1)
    mtp_weight: float = Field(default=0.25, ge=0.0, le=2.0)
    think_open_id: int = Field(default=6, ge=-1, le=65536)
    think_close_id: int = Field(default=7, ge=-1, le=65536)
    digit_token_ids: tuple[int, ...] = ()
    blt_ngram_dim: int = Field(default=64, ge=0, le=256)
    use_fone_abacus: bool = True
    domain_gate_weight: float = Field(default=0.35, ge=0.0, le=2.0)
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
) -> jnp.ndarray:
    """LFM3 + GLM-5.3-Flash `DSA` Grouped-Query Attention (GQA) with QK-RMSNorm + RoPE."""
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
    m_logits = bank_logits[..., :base_n]
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
        p_math = jax.nn.sigmoid(domain_logit_3d)
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
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute `(logits, hidden_normed, mean_domain_logit)` across sequence length `T`."""
    bsz, t_len = tokens.shape
    x = params["tok_embeddings.weight"][tokens]  # (B, T, D)
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
            routed_out, d_logit = routed_graph_nodes_ffn(
                ffn_normed,
                params[f"{p}.nodes.router.weight"],
                params[f"{p}.nodes.gate.weight"],
                params[f"{p}.nodes.up.weight"],
                params[f"{p}.nodes.down.weight"],
                config.top_k_experts,
                params.get(f"{p}.nodes.router_bias"),
                mature_nodes=config.mature_experts if use_tiered_reasoning else 0,
                new_tier_mask=pass_tier_mask,
                node_edge_scale=params.get(f"{p}.nodes.edge_scale"),
                return_domain_logit=True,
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
    return logits.reshape(bsz, t_len, config.vocab_size), hidden_normed, mean_domain_logit


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
        ):
            scaled_updates = {}
            for k, u in updates.items():
                if k.endswith(".nodes.router_bias") or k == "blt_token_ngrams":
                    scaled_updates[k] = jnp.zeros_like(u)
                elif k.endswith(".nodes.edge_scale"):
                    if 0 < config.mature_experts < config.n_experts:
                        e_mask = np.ones((config.n_experts,), dtype=np.float32)
                        e_mask[: config.mature_experts] = float(config.mature_lr_scale)
                        if config.n_experts > 10:
                            e_mask[:10] = 0.0
                        scaled_updates[k] = u * jnp.asarray(e_mask)
                    else:
                        scaled_updates[k] = u
                elif ".nodes." in k and not k.endswith(".router.weight"):
                    if 0 < config.mature_experts < config.n_experts:
                        m_mask = np.ones((config.n_experts, 1, 1), dtype=np.float32)
                        m_mask[: config.mature_experts] = float(config.mature_lr_scale)
                        if config.n_experts > 10:
                            m_mask[:10] = 0.0
                        scaled_updates[k] = u * jnp.asarray(m_mask)
                    else:
                        scaled_updates[k] = u
                elif k.endswith(".nodes.router.weight"):
                    if 0 < config.mature_experts < config.n_experts:
                        r_mask = np.ones((1, config.n_experts), dtype=np.float32)
                        r_mask[:, : config.mature_experts] = float(config.mature_lr_scale)
                        if config.n_experts > 10:
                            r_mask[:, :10] = 0.0
                            if config.n_experts <= 18:
                                r_mask[:, 10:] = 1.0
                            else:
                                r_mask[:, 10:18] = max(float(config.mature_lr_scale), 0.35)
                        scaled_updates[k] = u * jnp.asarray(r_mask)
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
                    scaled_updates[k] = u
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
                        new_params[b_key] = jnp.clip(new_params[b_key] + delta_b, -0.02, 0.02)
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
    def _step(params: dict[str, jnp.ndarray], seq_padded: jnp.ndarray, last_idx: jnp.ndarray):
        logits = forward_v2(params, seq_padded, config)
        return logits[0, last_idx, :]

    return _step


def generate_v2(
    params: dict[str, jnp.ndarray],
    config: ConfigV2,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    repetition_penalty: float = 1.15,
    include_think: bool = False,
) -> str:
    """Autoregressively generate subword tokens with MLM v2 using a fixed-shape JIT kernel."""
    prefilled_think = False
    effective_prompt = prompt
    if include_think and prompt.endswith("<|assistant|>\n"):
        effective_prompt = prompt + "<|think|>"
        prefilled_think = True
    prompt_ids = tokenizer.encode(effective_prompt, add_bos=True, add_eos=False).tolist()
    generated: list[int] = []
    logits_step = _make_cached_logits_step_v2(config)
    seq_len = config.max_seq_len
    buf = np.full((1, seq_len), tokenizer.pad_id, dtype=np.int32)

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
    step_limit = min(max_new_tokens, 48) if is_tiered_think else max_new_tokens
    for _ in range(step_limit):
        ctx = (prompt_ids + generated)[-seq_len:]
        cur_len = len(ctx)
        buf.fill(tokenizer.pad_id)
        buf[0, :cur_len] = ctx
        next_logits = np.array(
            logits_step(params, jnp.asarray(buf), jnp.int32(cur_len - 1)),
            dtype=np.float32,
            copy=True,
        )
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
            for prev_id in set(generated[-32:]):
                if next_logits[prev_id] > 0:
                    next_logits[prev_id] /= repetition_penalty
                else:
                    next_logits[prev_id] *= repetition_penalty

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
            )
            return f"<|think|>{thought_body}<|/think|>\n{clean_ans}"
        return out
    if "<|/think|>" in raw_decoded:
        raw_decoded = raw_decoded.split("<|/think|>", 1)[1]
    elif "<|/think>" in raw_decoded:
        raw_decoded = raw_decoded.split("<|/think>", 1)[1]
    raw_decoded = (
        raw_decoded.replace("<|think|>", "")
        .replace("<|/think|>", "")
        .replace("<|/think>", "")
    )
    return raw_decoded.strip()


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

    manifest = {
        "format": 2,
        "step": int(step),
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

        np_params = load_safetensors_bytes(zf.read("model.safetensors"))

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
