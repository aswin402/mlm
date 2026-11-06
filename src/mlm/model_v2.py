"""MLM v2 (SpiderNet) Core Model Definition, Forward Pass, Training Step & Checkpoint I/O.

Submodules (re-exported here for 100% backward compatibility):
  - `mlm.v2_kernels`: Stateless JAX attention, ShortConv, Gated DeltaNet, NCM, and mHC kernels.
  - `mlm.v2_graph`: SpiderNet directed graph routing, message passing, FDA calibration & telemetry.
  - `mlm.v2_growth`: Structural growth, orthogonal splitting, CUSUM controller & ablations.
  - `mlm.v2_verifier`: Neural-symbolic step verifiers, safe AST evaluator & RLM decomposition.
  - `mlm.v2_decode`: Stepwise recurrent decoding, LMCacheHybridV2, O(1) Ring-Buffer KV & generate.
  - `mlm.v2_memory`: 3-Tier persistent episodic/working/KV MemoryHierarchyV2.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import optax
from pydantic import BaseModel, ConfigDict, Field
from safetensors.numpy import (
    load as load_safetensors_bytes,
)
from safetensors.numpy import (
    save as save_safetensors_bytes,
)
from safetensors.numpy import (
    save_file as save_safetensors,
)

from mlm.tokenizer import BPETokenizer
from mlm.v2_decode import (  # noqa: F401
    _CODE_LOGIC_PATTERNS,
    _DEFAULT_LMCACHE_V2,
    LMCacheHybridV2,
    _execute_dense_nodes_step_v2,
    _execute_sparse_nodes_step_v2,
    _make_cached_logits_step_v2,
    _make_cached_recurrent_decode_v2,
    _recurrent_token_step_v2,
    _route_single_bank_step_v2,
    cacheblend_fuse_kv_chunks_v2,
    cacheblend_rope_shift_kv_v2,
    classify_cognitive_mode_v2,
    generate_v2,
    init_decode_cache_v2,
    init_o1_ring_decode_cache_v2,
    profile_o1_memory_scaling_v2,
    profile_sparse_execution_v2,
    step_o1_ring_decode_v2,
)
from mlm.v2_graph import (  # noqa: F401
    _calibrate_fda_layer_edges_v2,
    _compute_layer_specialist_outputs_v2,
    _route_single_bank,
    _route_single_bank_from_combined_logits_v2,
    adaptive_multi_hop_message_passing_v2,
    compute_directed_edge_weights_v2,
    compute_spidernet_graph_regularizers_v2,
    directed_graph_message_passing_v2,
    get_spidernet_node_role_v2,
    inject_and_calibrate_spidernet_graph_edges_v2,
    inspect_directed_graph_edges_v2,
    routed_graph_nodes_ffn,
    trace_compositional_3hop_chain_v2,
    trace_reading_node_collaborator_switch_v2,
)
from mlm.v2_growth import (  # noqa: F401
    AutonomousGrowthControllerV2,
    _orthogonal_perturbation_like,
    compute_node_domain_mutual_info_v2,
    evaluate_continual_learning_metrics_v2,
    expand_shortconv_kernel_v2,
    grow_dot2_code_logic_v2,
    grow_model_v2_depth,
    grow_model_v2_nodes,
    mask_specialist_updates_v2,
    merge_checkpoints_v2,
    run_full_spidernet_research_benchmark_v2,
    run_spidernet_ablation_suite_v2,
    split_specialist_nodes_orthogonal_v2,
)
from mlm.v2_kernels import (  # noqa: F401
    _TQ_2BIT_CENTROIDS,
    _TQ_4BIT_CENTROIDS,
    _default_mhc_tensors,
    _fwht_last_axis_jax,
    apply_rope,
    build_layer_execution_schedule,
    causal_short_conv,
    compute_von_neumann_effective_rank_v2,
    dsa_lightning_index_mask,
    grouped_query_attention,
    mhc_read_stream,
    mhc_write_stream,
    ncm_phi_bridge,
    ncm_tied_codebook_logits,
    parallel_gated_deltanet,
    precompute_rope_freqs,
    rms_norm,
    sandwich_rms_bound,
    sinkhorn_knopp_2x2,
    swiglu_ffn,
    turboquant_kv_tensor_jax,
)
from mlm.v2_memory import (  # noqa: F401
    _QUESTION_STARTERS,
    _STOPWORDS_MEM,
    _VAR_ASSIGN_RE,
    MemoryHierarchyV2,
)
from mlm.v2_verifier import (  # noqa: F401
    _EQ_STEP_RE,
    _PROMPT_SINGLE_OP_RE,
    _eval_atomic_step,
    _extract_model_predicted_number,
    _mask_and_penalize_logits,
    _normalize_op_token,
    _rlm_try_recursive_decomposition_v2,
    _rlm_verify_equations_via_self_calls,
    _safe_eval_python_ast,
    _verify_answer_against_thought_v2,
    _verify_code_and_logic_v2,
    _verify_reasoning_steps_v2,
)


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
    graph_edge_rank: int = Field(default=16, ge=0, le=128)
    graph_edge_top_k: int = Field(default=2, ge=1, le=16)
    graph_edge_temp: float = Field(default=1.0, gt=0.05, le=10.0)
    graph_reg_weight: float = Field(default=0.02, ge=0.0, le=2.0)
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
            if n_exp >= 2 and config.graph_edge_rank > 0:
                er = int(config.graph_edge_rank)
                params[f"{p}.nodes.edge_u"] = jnp.asarray(
                    rng.normal(0.0, 0.02, size=(n_exp, er)).astype(np.float32)
                )
                params[f"{p}.nodes.edge_v"] = jnp.asarray(
                    rng.normal(0.0, 0.02, size=(n_exp, er)).astype(np.float32)
                )
                params[f"{p}.nodes.edge_proj_h"] = jnp.asarray(
                    rng.normal(0.0, std, size=(d, er)).astype(np.float32)
                )
                params[f"{p}.nodes.edge_proj_x"] = jnp.asarray(
                    rng.normal(0.0, std, size=(d, er)).astype(np.float32)
                )
                params[f"{p}.nodes.edge_bias"] = jnp.zeros((n_exp, n_exp), dtype=jnp.float32)
                params[f"{p}.nodes.msg_down"] = jnp.asarray(
                    rng.normal(0.0, std, size=(d, er)).astype(np.float32)
                )
                params[f"{p}.nodes.msg_up"] = jnp.asarray(
                    rng.normal(0.0, layer_scale, size=(er, d)).astype(np.float32)
                )
                params[f"{p}.nodes.collab_gate"] = jnp.zeros((n_exp,), dtype=jnp.float32)
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
                    if 0 <= int(a_idx) < config.n_experts and int(a_idx) not in (
                        8,
                        9,
                        18,
                        19,
                        28,
                        29,
                    ):
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
                edge_u=params.get(f"{p}.nodes.edge_u"),
                edge_v=params.get(f"{p}.nodes.edge_v"),
                edge_proj_h=params.get(f"{p}.nodes.edge_proj_h"),
                edge_proj_x=params.get(f"{p}.nodes.edge_proj_x"),
                edge_bias=params.get(f"{p}.nodes.edge_bias"),
                msg_down=params.get(f"{p}.nodes.msg_down"),
                msg_up=params.get(f"{p}.nodes.msg_up"),
                collab_gate=params.get(f"{p}.nodes.collab_gate"),
                top_k_edges=config.graph_edge_top_k,
                edge_temp=config.graph_edge_temp,
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
            if config.n_experts >= 2 and config.graph_reg_weight > 0.0:
                reg_dict = compute_spidernet_graph_regularizers_v2(p, inputs, mask, config)
                total_loss = total_loss + float(config.graph_reg_weight) * reg_dict["total_reg"]
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
        updates = mask_specialist_updates_v2(updates, config)
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
    if (
        config.n_experts > 0
        and config.graph_edge_rank > 0
        and "blk.0.nodes.edge_u" not in params
    ):
        params = inject_and_calibrate_spidernet_graph_edges_v2(
            params, config, collab_strength=0.008
        )
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
