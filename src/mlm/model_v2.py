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
    sandwich_norm_clip: float = Field(default=0.0, ge=0.0, le=20.0)
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
    think_open_id: int = Field(default=-1, ge=-1, le=65536)
    think_close_id: int = Field(default=-1, ge=-1, le=65536)
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


def parallel_gated_deltanet(
    x: jnp.ndarray,
    wq_d: jnp.ndarray,
    wk_d: jnp.ndarray,
    wv_d: jnp.ndarray,
    w_decay: jnp.ndarray,
    n_heads: int,
) -> jnp.ndarray:
    """Parallel causal Gated DeltaNet associative fast-weight operator over `(B, T, D)`."""
    bsz, t_len, d_model = x.shape
    head_dim = d_model // n_heads
    q = (x @ wq_d).reshape(bsz, t_len, n_heads, head_dim)
    k = (x @ wk_d).reshape(bsz, t_len, n_heads, head_dim)
    v = (x @ wv_d).reshape(bsz, t_len, n_heads, head_dim)

    # L2-normalize keys and queries for stable delta-rule eigen-spectrum
    q = q * jax.lax.rsqrt(jnp.sum(jnp.square(q), axis=-1, keepdims=True) + 1e-6)
    k = k * jax.lax.rsqrt(jnp.sum(jnp.square(k), axis=-1, keepdims=True) + 1e-6)

    # Log-space cumulative decay mask for parallel causal associative attention
    log_alpha = -jax.nn.softplus(x @ w_decay)  # (B, T, H)
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
) -> jnp.ndarray:
    """LFM3 Head-Wise Gated & Tri-Hybrid Grouped-Query Attention (GQA) with QK-RMSNorm + RoPE."""
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
        idx = jnp.arange(t_len)
        dist = idx[:, None] - idx[None, :]  # (T, T)
        global_causal = (dist >= 0)[None, None, :, :]
        swa_causal = ((dist >= 0) & (dist < sliding_window))[None, None, :, :]
        is_swa_head = (jnp.arange(n_heads) % 2 == 1)[None, :, None, None]
        mask = jnp.where(is_swa_head, swa_causal, global_causal)
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
) -> jnp.ndarray:
    """Route tokens `(B, T, D)` across `N` specialist Graph-Node SwiGLU experts (top-`k` active).

    Implements DeepSeek-V3 Auxiliary-Loss-Free Load Balancing (`router_bias`) and Tiered
    Specialist Routing when expanding beyond `mature_nodes` so mature nodes are never evicted.
    """
    n_nodes = node_gate.shape[0]
    router_logits = x @ router_w  # (B, T, N)
    if 0 < mature_nodes < n_nodes:
        m_logits = router_logits[..., :mature_nodes]
        m_probs = jax.nn.softmax(m_logits, axis=-1)
        k_eff = min(top_k, mature_nodes)
        if k_eff < mature_nodes:
            m_scores = (
                m_probs + jax.lax.stop_gradient(router_bias[:mature_nodes])[None, None, :]
                if router_bias is not None
                else m_probs
            )
            _, top_idx = jax.lax.top_k(m_scores, k_eff)
            m_mask = jnp.sum(jax.nn.one_hot(top_idx, mature_nodes, dtype=m_probs.dtype), axis=-2)
            m_gated = m_probs * m_mask
            m_gated = m_gated / jnp.maximum(jnp.sum(m_gated, axis=-1, keepdims=True), 1e-6)
        else:
            m_gated = m_probs

        new_logits = router_logits[..., mature_nodes:]
        if router_bias is not None:
            new_logits = new_logits + jax.lax.stop_gradient(router_bias[mature_nodes:])[
                None, None, :
            ]
        new_gated = jax.nn.sigmoid(new_logits)
        if new_tier_mask is not None:
            new_gated = new_gated * new_tier_mask
        gated_probs = jnp.concatenate([m_gated, new_gated], axis=-1)
    else:
        k_eff = min(top_k, n_nodes)
        probs = jax.nn.softmax(router_logits, axis=-1)
        if k_eff < n_nodes:
            sel_scores = (
                probs + jax.lax.stop_gradient(router_bias)[None, None, :]
                if router_bias is not None
                else probs
            )
            _, top_idx = jax.lax.top_k(sel_scores, k_eff)
            mask = jnp.sum(jax.nn.one_hot(top_idx, n_nodes, dtype=probs.dtype), axis=-2)
            gated_probs = probs * mask
            gated_probs = gated_probs / jnp.maximum(
                jnp.sum(gated_probs, axis=-1, keepdims=True), 1e-6
            )
        else:
            gated_probs = probs

    gate_all = jax.nn.silu(jnp.einsum("btd,ndf->btnf", x, node_gate))
    up_all = jnp.einsum("btd,ndf->btnf", x, node_up)
    hidden = gate_all * up_all
    node_outs = jnp.einsum("btnf,nfd->btnd", hidden, node_down)
    return jnp.einsum("btn,btnd->btd", gated_probs, node_outs)


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

    layer_scale = float(std / np.sqrt(2.0 * config.n_layers))
    for i in range(config.n_layers):
        p = f"blk.{i}"
        params[f"{p}.attn_norm.weight"] = jnp.ones((d,), dtype=jnp.float32)
        params[f"{p}.ffn_norm.weight"] = jnp.ones((d,), dtype=jnp.float32)
        params[f"{p}.attn_post_scale"] = jnp.ones((d,), dtype=jnp.float32)
        params[f"{p}.ffn_post_scale"] = jnp.ones((d,), dtype=jnp.float32)

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
    return params


def forward_with_hidden_v2(
    params: dict[str, jnp.ndarray],
    tokens: jnp.ndarray,
    config: ConfigV2,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute next-token logits `(B, T, V)` and final normalized hidden states `(B, T, D)`."""
    bsz, t_len = tokens.shape
    x = params["tok_embeddings.weight"][tokens]  # (B, T, D)
    cos, sin = precompute_rope_freqs(t_len, config.head_dim, config.rope_theta)
    use_tiered_reasoning = (
        config.mature_experts >= 8
        and config.n_experts > config.mature_experts
        and config.think_open_id >= 0
    )
    think_tier_mask = (
        jnp.maximum.accumulate((tokens == config.think_open_id).astype(jnp.float32), axis=-1)[
            ..., None
        ]
        if use_tiered_reasoning
        else None
    )

    for i in range(config.n_layers):
        p = f"blk.{i}"
        normed = rms_norm(x, params[f"{p}.attn_norm.weight"], config.rms_eps)
        if config.is_attention_layer(i):
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
                )
                y_mix = y_mix + delta_out * params[f"{p}.deltanet.scale"][None, None, :]
            dx = y_mix @ params[f"{p}.shortconv.out_proj.weight"]
        dx = sandwich_rms_bound(
            dx,
            params.get(f"{p}.attn_post_scale"),
            clip_rms=config.sandwich_norm_clip,
            eps=config.rms_eps,
        )
        x = x + dx

        ffn_normed = rms_norm(x, params[f"{p}.ffn_norm.weight"], config.rms_eps)
        ffn_out = swiglu_ffn(
            ffn_normed,
            params[f"{p}.ffn_gate.weight"],
            params[f"{p}.ffn_up.weight"],
            params[f"{p}.ffn_down.weight"],
        )
        if config.n_experts > 0 and f"{p}.nodes.router.weight" in params:
            ffn_out = ffn_out + routed_graph_nodes_ffn(
                ffn_normed,
                params[f"{p}.nodes.router.weight"],
                params[f"{p}.nodes.gate.weight"],
                params[f"{p}.nodes.up.weight"],
                params[f"{p}.nodes.down.weight"],
                config.top_k_experts,
                params.get(f"{p}.nodes.router_bias"),
                mature_nodes=config.mature_experts if use_tiered_reasoning else 0,
                new_tier_mask=think_tier_mask,
            )
        ffn_out = sandwich_rms_bound(
            ffn_out,
            params.get(f"{p}.ffn_post_scale"),
            clip_rms=config.sandwich_norm_clip,
            eps=config.rms_eps,
        )
        x = x + ffn_out

    hidden_normed = rms_norm(x, params["output_norm.weight"], config.rms_eps)
    out_w = (
        params["tok_embeddings.weight"]
        if config.tie_embeddings
        else params["output.weight"]
    )
    logits = jnp.matmul(hidden_normed, jnp.swapaxes(out_w, 0, 1))
    return logits.reshape(bsz, t_len, config.vocab_size), hidden_normed


def compute_mtp_logits_v2(
    params: dict[str, jnp.ndarray],
    hidden_normed: jnp.ndarray,
    next_tokens: jnp.ndarray,
    config: ConfigV2,
) -> jnp.ndarray:
    """DeepSeek-V3 Multi-Token Prediction (`MTP`): predict `t+2` from `h_t` and `e(x_{t+1})`."""
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
    return jnp.matmul(h_mtp, jnp.swapaxes(out_w, 0, 1))


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
            logits, hidden_normed = forward_with_hidden_v2(p, inputs, config)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            tok_loss = -jnp.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]
            denom = jnp.maximum(jnp.sum(mask), 1.0)
            total_loss = jnp.sum(tok_loss * mask) / denom
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
            k: (jnp.zeros_like(g) if k.endswith(".nodes.router_bias") else g)
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
                if k.endswith(".nodes.router_bias"):
                    scaled_updates[k] = jnp.zeros_like(u)
                elif ".nodes." in k and not k.endswith(".router.weight"):
                    if 0 < config.mature_experts < config.n_experts:
                        m_mask = np.ones((config.n_experts, 1, 1), dtype=np.float32)
                        m_mask[: config.mature_experts] = float(config.mature_lr_scale)
                        scaled_updates[k] = u * jnp.asarray(m_mask)
                    else:
                        scaled_updates[k] = u
                elif k.endswith(".nodes.router.weight"):
                    if 0 < config.mature_experts < config.n_experts:
                        r_mask = np.ones((1, config.n_experts), dtype=np.float32)
                        r_mask[:, : config.mature_experts] = float(config.mature_lr_scale)
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
                k: (jnp.zeros_like(u) if k.endswith(".nodes.router_bias") else u)
                for k, u in updates.items()
            }
        new_params = optax.apply_updates(params, updates)

        # DeepSeek-V3 Auxiliary-Loss-Free Router Bias Update
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
                    r_probs = jax.nn.softmax(
                        normed @ new_params[f"{p_key}.nodes.router.weight"], axis=-1
                    )
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
        norms = np.linalg.norm(down_w, axis=(1, 2))
        order = np.argsort(-norms)

        for suffix in ("gate", "up", "down"):
            arr = np.asarray(params[f"{p}.nodes.{suffix}.weight"], dtype=np.float32)
            children = []
            for j in range(add_nodes):
                p1 = arr[order[(2 * j) % n_old]]
                p2 = arr[order[(2 * j + 1) % n_old]]
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
                0.85 * rw[:, order[(2 * j) % n_old]] + 0.15 * rw[:, order[(2 * j + 1) % n_old]]
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

    emb_arr = np.asarray(params["tok_embeddings.weight"], dtype=np.float32).copy()
    t_open = new_cfg.think_open_id
    t_close = new_cfg.think_close_id
    if n_old >= 8 and 0 <= t_open < emb_arr.shape[0] and 0 <= t_close < emb_arr.shape[0]:
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
    close_id = (
        config.think_close_id
        if config.think_close_id >= 0
        else int(tokenizer.encode("<|think|>", add_bos=False, add_eos=False)[-1])
    )
    step_limit = min(max_new_tokens, 36) if is_tiered_think else max_new_tokens
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
        if prefilled_think and len(generated) >= 5 and next_id == close_id:
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
                .strip()
            )
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
            if len(think_ids) >= 2:
                config = config.model_copy(
                    update={
                        "think_open_id": int(think_ids[0]),
                        "think_close_id": int(think_ids[-1]),
                    }
                )

        np_params = load_safetensors_bytes(zf.read("model.safetensors"))

    params = {k: jnp.asarray(v, dtype=jnp.float32) for k, v in np_params.items()}
    # Backward compatibility for checkpoints saved before DeepSeek-V3 / LFM3 upgrades
    for i in range(config.n_layers):
        p_key = f"blk.{i}"
        if f"{p_key}.attn_post_scale" not in params:
            params[f"{p_key}.attn_post_scale"] = jnp.ones((config.d_model,), dtype=jnp.float32)
        if f"{p_key}.ffn_post_scale" not in params:
            params[f"{p_key}.ffn_post_scale"] = jnp.ones((config.d_model,), dtype=jnp.float32)
        if config.is_attention_layer(i) and config.use_head_gate:
            if f"{p_key}.attn_gate.weight" not in params:
                params[f"{p_key}.attn_gate.weight"] = jnp.zeros(
                    (config.d_model, config.n_heads), dtype=jnp.float32
                )
            if f"{p_key}.attn_gate.bias" not in params:
                params[f"{p_key}.attn_gate.bias"] = jnp.zeros(
                    (config.n_heads,), dtype=jnp.float32
                )
        if config.n_experts > 0:
            b_key = f"{p_key}.nodes.router_bias"
            if b_key not in params:
                params[b_key] = jnp.zeros((config.n_experts,), dtype=jnp.float32)
    if "mtp_norm.weight" not in params:
        params["mtp_norm.weight"] = jnp.ones((config.d_model,), dtype=jnp.float32)
    if "mtp_proj.weight" not in params:
        rng = np.random.default_rng(config.seed + 999)
        params["mtp_proj.weight"] = jnp.asarray(
            rng.normal(0.0, 0.01, size=(2 * config.d_model, config.d_model)).astype(np.float32)
        )
    return params, config, tokenizer, manifest
