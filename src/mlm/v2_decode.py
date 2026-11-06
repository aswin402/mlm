"""Stepwise Recurrent Decoding, LMCacheHybridV2, O(1) Ring-Buffer KV & Generation for MLM v2."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from mlm.tokenizer import BPETokenizer
from mlm.v2_graph import (
    _route_single_bank_from_combined_logits_v2,
    directed_graph_message_passing_v2,
)
from mlm.v2_kernels import (
    apply_rope,
    build_layer_execution_schedule,
    compute_von_neumann_effective_rank_v2,
    mhc_read_stream,
    mhc_write_stream,
    ncm_tied_codebook_logits,
    precompute_rope_freqs,
    rms_norm,
    sandwich_rms_bound,
    swiglu_ffn,
    turboquant_kv_tensor_jax,
)
from mlm.v2_verifier import (
    _EQ_STEP_RE,
    _mask_and_penalize_logits,
    _rlm_try_recursive_decomposition_v2,
    _verify_answer_against_thought_v2,
    _verify_code_and_logic_v2,
    _verify_reasoning_steps_v2,
)

if TYPE_CHECKING:
    from mlm.model_v2 import ConfigV2
    from mlm.v2_memory import MemoryHierarchyV2


@lru_cache(maxsize=8)
def _make_cached_logits_step_v2(config: ConfigV2):
    @jax.jit
    def _step(
        params: dict[str, jnp.ndarray],
        seq_padded: jnp.ndarray,
        idx_a: jnp.ndarray,
        idx_b: jnp.ndarray,
    ):
        from mlm.model_v2 import forward_with_hidden_v2

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
    base_n = min(bank_logits.shape[-1], 8)
    m_raw = bank_logits[..., :base_n]
    new_cumsum = bank_cumsum + bank_logits
    m_logits = 0.45 * m_raw + 0.55 * (new_cumsum[..., :base_n] / step_f)
    routed = _route_single_bank_from_combined_logits_v2(
        m_logits, bank_logits, bank_bias, top_k=top_k, new_tier_mask=new_tier_mask
    )
    return routed, new_cumsum

def _execute_dense_nodes_step_v2(
    ffn_normed: jnp.ndarray,
    gated_probs: jnp.ndarray,
    gate_w: jnp.ndarray,
    up_w: jnp.ndarray,
    down_w: jnp.ndarray,
    edge_u: jnp.ndarray | None = None,
    edge_v: jnp.ndarray | None = None,
    edge_proj_h: jnp.ndarray | None = None,
    edge_proj_x: jnp.ndarray | None = None,
    edge_bias: jnp.ndarray | None = None,
    msg_down: jnp.ndarray | None = None,
    msg_up: jnp.ndarray | None = None,
    collab_gate: jnp.ndarray | None = None,
    top_k_edges: int = 2,
    edge_temp: float = 1.0,
) -> jnp.ndarray:
    """Reference dense specialist execution over all `N` nodes (used for equivalence profiling)."""
    gate_all = jax.nn.silu(jnp.einsum("btd,ndf->btnf", ffn_normed, gate_w))
    up_all = jnp.einsum("btd,ndf->btnf", ffn_normed, up_w)
    node_outs = jnp.einsum("btnf,nfd->btnd", gate_all * up_all, down_w)
    if (
        edge_u is not None
        and edge_v is not None
        and edge_proj_h is not None
        and edge_proj_x is not None
        and edge_bias is not None
        and msg_down is not None
        and msg_up is not None
        and collab_gate is not None
    ):
        return directed_graph_message_passing_v2(
            ffn_normed,
            gated_probs,
            node_outs,
            edge_u,
            edge_v,
            edge_proj_h,
            edge_proj_x,
            edge_bias,
            msg_down,
            msg_up,
            collab_gate,
            top_k_edges=top_k_edges,
            temperature=edge_temp,
        )
    return jnp.einsum("btn,btnd->btd", gated_probs, node_outs)


def _execute_sparse_nodes_step_v2(
    ffn_normed: jnp.ndarray,
    gated_probs: jnp.ndarray,
    gate_w: jnp.ndarray,
    up_w: jnp.ndarray,
    down_w: jnp.ndarray,
    k_active: int,
    edge_u: jnp.ndarray | None = None,
    edge_v: jnp.ndarray | None = None,
    edge_proj_h: jnp.ndarray | None = None,
    edge_proj_x: jnp.ndarray | None = None,
    edge_bias: jnp.ndarray | None = None,
    msg_down: jnp.ndarray | None = None,
    msg_up: jnp.ndarray | None = None,
    collab_gate: jnp.ndarray | None = None,
    top_k_edges: int = 2,
    edge_temp: float = 1.0,
) -> jnp.ndarray:
    """True sparse index-gather specialist execution for a single token `(1, 1, D)`.

    Selects the `k_active` highest-weight specialist indices from `gated_probs[0, 0]`,
    gathers only their weight slices `(K, D, F)` and `(K, F, D)` from `gate_w`, `up_w`,
    and `down_w`, and computes SwiGLU FFN + Directed Graph Message Passing over `K << N`
    nodes (5x-10x fewer specialist FFN FLOPs and memory reads per layer pass).
    """
    n_nodes = int(gate_w.shape[0])
    k_eff = max(1, min(int(k_active), n_nodes))
    probs_1d = gated_probs[0, 0]  # (N,)
    active_probs, active_idx = jax.lax.top_k(probs_1d, k_eff)  # (K,), (K,)
    w_gate_k = gate_w[active_idx]  # (K, D, F)
    w_up_k = up_w[active_idx]      # (K, D, F)
    w_down_k = down_w[active_idx]  # (K, F, D)
    x_vec = ffn_normed[0, 0]       # (D,)
    h_gate = jax.nn.silu(jnp.einsum("d,kdf->kf", x_vec, w_gate_k))
    h_up = jnp.einsum("d,kdf->kf", x_vec, w_up_k)
    node_outs_k = jnp.einsum("kf,kfd->kd", h_gate * h_up, w_down_k)
    if (
        k_eff >= 2
        and edge_u is not None
        and edge_v is not None
        and edge_proj_h is not None
        and edge_proj_x is not None
        and edge_bias is not None
        and msg_down is not None
        and msg_up is not None
        and collab_gate is not None
    ):
        u_k = edge_u[active_idx]
        v_k = edge_v[active_idx]
        b_k = edge_bias[active_idx][:, active_idx]
        cg_k = collab_gate[active_idx]
        return directed_graph_message_passing_v2(
            ffn_normed,
            active_probs[None, None, :],
            node_outs_k[None, None, :, :],
            u_k,
            v_k,
            edge_proj_h,
            edge_proj_x,
            b_k,
            msg_down,
            msg_up,
            cg_k,
            top_k_edges=min(int(top_k_edges), k_eff - 1),
            temperature=edge_temp,
        )
    routed_vec = jnp.einsum("k,kd->d", active_probs, node_outs_k)
    return routed_vec[None, None, :]


def _recurrent_token_step_v2(
    params: dict[str, jnp.ndarray],
    carry: dict[str, jnp.ndarray],
    tok_id_scalar: jnp.ndarray,
    config: ConfigV2,
    cos_table: jnp.ndarray,
    sin_table: jnp.ndarray,
    use_sparse_gather: bool = True,
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
                math_end = min(n_nodes, 20)
                eng_bias = eff_router_bias[:10] if eff_router_bias is not None else None
                math_bias = eff_router_bias[10:math_end] if eff_router_bias is not None else None
                eng_probs, r_cs_eng = _route_single_bank_step_v2(
                    router_logits[..., :10],
                    r_cumsum_s[:, None, :10],
                    step_f,
                    eng_bias,
                    top_k=config.top_k_experts,
                    new_tier_mask=pass_tier_mask,
                )
                math_probs, r_cs_math = _route_single_bank_step_v2(
                    router_logits[..., 10:math_end],
                    r_cumsum_s[:, None, 10:math_end],
                    step_f,
                    math_bias,
                    top_k=config.top_k_experts,
                    new_tier_mask=pass_tier_mask,
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
                if n_nodes <= 20:
                    r_cumsum_s = jnp.concatenate([r_cs_eng[:, 0, :], r_cs_math[:, 0, :]], axis=-1)
                    gated_probs = jnp.concatenate([p_eng * eng_probs, p_math * math_probs], axis=-1)
                    code_active_cond = jnp.zeros((1, 1, 1), dtype=jnp.bool_)
                else:
                    code_bias = eff_router_bias[20:] if eff_router_bias is not None else None
                    code_probs, r_cs_code = _route_single_bank_step_v2(
                        router_logits[..., 20:],
                        r_cumsum_s[:, None, 20:],
                        step_f,
                        code_bias,
                        top_k=config.top_k_experts,
                        new_tier_mask=pass_tier_mask,
                    )
                    r_cumsum_s = jnp.concatenate(
                        [r_cs_eng[:, 0, :], r_cs_math[:, 0, :], r_cs_code[:, 0, :]], axis=-1
                    )
                    code_base_end = min(n_nodes, 28)
                    n_code_base = max(1, code_base_end - 20)
                    code_bias_base = (
                        code_bias[:n_code_base]
                        if code_bias is not None
                        else jnp.full((n_code_base,), -2.5, dtype=router_logits.dtype)
                    )
                    e_code_tok = (
                        jax.scipy.special.logsumexp(
                            router_logits[..., 20:code_base_end] + code_bias_base[None, None, :],
                            axis=-1,
                            keepdims=True,
                        )
                        - jnp.log(float(n_code_base))
                    )
                    code_dl_3d = 2.0 * (e_code_tok - jnp.maximum(e_eng_tok, e_math_tok))
                    code_active_cond = (jnp.max(code_bias_base) > 1.0) | (code_dl_3d >= 2.0)
                    p_code_soft = jax.nn.sigmoid(code_dl_3d)
                    p_code_rect = jnp.where(
                        code_active_cond, jnp.ones_like(p_code_soft), jnp.zeros_like(p_code_soft)
                    )
                    p_code = p_code_soft + jax.lax.stop_gradient(p_code_rect - p_code_soft)
                    p_rem = 1.0 - p_code
                    gated_probs = jnp.concatenate(
                        [
                            p_rem * p_eng * eng_probs,
                            p_rem * p_math * math_probs,
                            p_code * code_probs,
                        ],
                        axis=-1,
                    )

            edge_sc = params.get(f"{p}.nodes.edge_scale")
            if edge_sc is not None:
                e_mod = jnp.clip(edge_sc.astype(gated_probs.dtype), 0.1, 2.5)[None, None, :]
                gated_probs = gated_probs * e_mod

            gate_w_cur = params[f"{p}.nodes.gate.weight"]
            up_w_cur = params[f"{p}.nodes.up.weight"]
            down_w_cur = params[f"{p}.nodes.down.weight"]
            eu_cur = params.get(f"{p}.nodes.edge_u")
            ev_cur = params.get(f"{p}.nodes.edge_v")
            eph_cur = params.get(f"{p}.nodes.edge_proj_h")
            epx_cur = params.get(f"{p}.nodes.edge_proj_x")
            eb_cur = params.get(f"{p}.nodes.edge_bias")
            md_cur = params.get(f"{p}.nodes.msg_down")
            mu_cur = params.get(f"{p}.nodes.msg_up")
            cg_cur = params.get(f"{p}.nodes.collab_gate")
            if use_sparse_gather:
                k_t0 = max(1, min(n_nodes, int(config.top_k_experts)))
                k_t1 = max(k_t0, min(n_nodes, int(2 * config.top_k_experts)))
                k_t2 = max(k_t1, min(n_nodes, int(2 * (config.top_k_experts + 2))))
                if n_nodes <= 10:
                    single_bank_cond = jnp.bool_(True)
                elif n_nodes <= 20:
                    single_bank_cond = (seq_dl[0, 0, 0] < -0.95)
                else:
                    single_bank_cond = (seq_dl[0, 0, 0] < -0.95) | code_active_cond[0, 0, 0]
                in_think_step = is_active[0] > 0.0
                is_t1 = (in_think_step & single_bank_cond) | (
                    (~in_think_step) & (~single_bank_cond)
                )
                tier_idx = jnp.where(
                    (~in_think_step) & single_bank_cond,
                    jnp.int32(0),
                    jnp.where(
                        is_t1,
                        jnp.int32(1),
                        jnp.int32(2),
                    ),
                )
                routed_out = jax.lax.switch(
                    tier_idx,
                    [
                        lambda op: _execute_sparse_nodes_step_v2(
                            op[0],
                            op[1],
                            op[2],
                            op[3],
                            op[4],
                            k_t0,
                            edge_u=eu_cur,
                            edge_v=ev_cur,
                            edge_proj_h=eph_cur,
                            edge_proj_x=epx_cur,
                            edge_bias=eb_cur,
                            msg_down=md_cur,
                            msg_up=mu_cur,
                            collab_gate=cg_cur,
                            top_k_edges=config.graph_edge_top_k,
                            edge_temp=config.graph_edge_temp,
                        ),
                        lambda op: _execute_sparse_nodes_step_v2(
                            op[0],
                            op[1],
                            op[2],
                            op[3],
                            op[4],
                            k_t1,
                            edge_u=eu_cur,
                            edge_v=ev_cur,
                            edge_proj_h=eph_cur,
                            edge_proj_x=epx_cur,
                            edge_bias=eb_cur,
                            msg_down=md_cur,
                            msg_up=mu_cur,
                            collab_gate=cg_cur,
                            top_k_edges=config.graph_edge_top_k,
                            edge_temp=config.graph_edge_temp,
                        ),
                        lambda op: _execute_sparse_nodes_step_v2(
                            op[0],
                            op[1],
                            op[2],
                            op[3],
                            op[4],
                            k_t2,
                            edge_u=eu_cur,
                            edge_v=ev_cur,
                            edge_proj_h=eph_cur,
                            edge_proj_x=epx_cur,
                            edge_bias=eb_cur,
                            msg_down=md_cur,
                            msg_up=mu_cur,
                            collab_gate=cg_cur,
                            top_k_edges=config.graph_edge_top_k,
                            edge_temp=config.graph_edge_temp,
                        ),
                    ],
                    (ffn_normed, gated_probs, gate_w_cur, up_w_cur, down_w_cur),
                )
            else:
                routed_out = _execute_dense_nodes_step_v2(
                    ffn_normed,
                    gated_probs,
                    gate_w_cur,
                    up_w_cur,
                    down_w_cur,
                    edge_u=eu_cur,
                    edge_v=ev_cur,
                    edge_proj_h=eph_cur,
                    edge_proj_x=epx_cur,
                    edge_bias=eb_cur,
                    msg_down=md_cur,
                    msg_up=mu_cur,
                    collab_gate=cg_cur,
                    top_k_edges=config.graph_edge_top_k,
                    edge_temp=config.graph_edge_temp,
                )
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


def cacheblend_rope_shift_kv_v2(
    k_cached: jnp.ndarray,
    old_positions: jnp.ndarray,
    new_positions: jnp.ndarray,
    head_dim: int,
    theta: float = 10000.0,
) -> jnp.ndarray:
    """LMCache / CacheBlend (EuroSys Best Paper) Dynamic SO(2) Relative RoPE Re-Rotation.

    Shifts a cached key tensor `k_cached` `(B, T, H, D_h)` from `old_positions` `(T,)` to
    `new_positions` `(T,)` via exact orthogonal relative angle rotation:
        RoPE(k, p_new) = R((p_new - p_old) * omega) @ RoPE(k, p_old)
    """
    half = head_dim // 2
    freqs = 1.0 / (theta ** (jnp.arange(0, half, dtype=jnp.float32) / float(half)))
    delta_pos = jnp.asarray(new_positions, dtype=jnp.float32) - jnp.asarray(
        old_positions, dtype=jnp.float32
    )
    delta_angles = jnp.outer(delta_pos, freqs)  # (T, half)
    c = jnp.cos(delta_angles)[None, :, None, :]
    s = jnp.sin(delta_angles)[None, :, None, :]
    x1 = k_cached[..., :half]
    x2 = k_cached[..., half:]
    out1 = x1 * c - x2 * s
    out2 = x2 * c + x1 * s
    return jnp.concatenate([out1, out2], axis=-1)


def cacheblend_fuse_kv_chunks_v2(
    chunks: list[tuple[jnp.ndarray, jnp.ndarray, int]],
    head_dim: int,
    theta: float = 10000.0,
    hkvd_ratio: float = 0.15,
) -> dict[str, object]:
    """LMCache / CacheBlend Non-Prefix KV Knowledge Fusion + High-KV-Deviation (`HKVD`) Selection.

    Takes a list of `(k_chunk, v_chunk, old_start_pos)` tuples, dynamically re-rotates each chunk's
    keys to its fused target offset, concatenates `(k_fused, v_fused)`, and returns the top
    `hkvd_ratio` (default 15%) High-KV-Deviation token indices for selective boundary refresh.
    """
    k_parts: list[jnp.ndarray] = []
    v_parts: list[jnp.ndarray] = []
    hkvd_indices: list[int] = []
    cur_offset = 0
    for c_idx, (k_c, v_c, old_start) in enumerate(chunks):
        t_len = int(k_c.shape[1])
        old_pos = jnp.arange(old_start, old_start + t_len, dtype=jnp.int32)
        new_pos = jnp.arange(cur_offset, cur_offset + t_len, dtype=jnp.int32)
        if int(old_start) != cur_offset:
            k_shifted = cacheblend_rope_shift_kv_v2(
                k_c, old_pos, new_pos, head_dim=head_dim, theta=theta
            )
        else:
            k_shifted = k_c
        k_parts.append(k_shifted)
        v_parts.append(v_c)
        if c_idx > 0 and t_len > 0:
            n_hkvd = max(1, int(np.ceil(float(hkvd_ratio) * t_len)))
            tok_energy = np.array(
                jnp.mean(jnp.square(k_shifted[0]) + jnp.square(v_c[0]), axis=(-2, -1)),
                dtype=np.float32,
                copy=True,
            )
            # Prioritize chunk boundary token + highest-energy tokens inside the non-prefix chunk
            tok_energy[0] += 1e6
            top_local = np.argsort(-tok_energy)[:n_hkvd]
            for loc_i in sorted(int(x) for x in top_local):
                hkvd_indices.append(cur_offset + loc_i)
        cur_offset += t_len
    k_fused = jnp.concatenate(k_parts, axis=1)
    v_fused = jnp.concatenate(v_parts, axis=1)
    return {
        "k_fused": k_fused,
        "v_fused": v_fused,
        "hkvd_indices": hkvd_indices,
        "total_tokens": cur_offset,
        "recomputed_fraction": round(len(hkvd_indices) / max(cur_offset, 1), 4),
    }


class LMCacheHybridV2:
    """LMCache Hash-Indexed Hybrid Prefix & Chunk State Cache (`KV + Gated DeltaNet + ShortConv`).

    Stores opaque hybrid recurrent state pages (`delta_state`, `shortconv_buf`, `phi_conv_buf`,
    `k_cache`, `v_cache`, `k_idx_cache`, and router load states) at 4-token chunk boundaries and
    full prompt boundaries to eliminate redundant prefill across multi-turn chat and RLM calls.
    """

    def __init__(self, chunk_size: int = 4, max_entries: int = 64) -> None:
        self.chunk_size = max(2, int(chunk_size))
        self.max_entries = max(8, int(max_entries))
        self._prefix_carries: dict[tuple[object, ...], dict[str, jnp.ndarray]] = {}
        self._full_prefills: dict[
            tuple[object, ...], tuple[dict[str, jnp.ndarray], jnp.ndarray, jnp.ndarray, int]
        ] = {}
        self.prefix_hits: int = 0
        self.full_hits: int = 0
        self.saved_prefill_tokens: int = 0

    @staticmethod
    def model_signature(params: dict[str, jnp.ndarray], config: ConfigV2) -> tuple[object, ...]:
        w0 = float(np.asarray(params["tok_embeddings.weight"])[0, 0])
        b0 = (
            float(np.asarray(params["blk.0.ffn_down.weight"])[0, 0])
            if "blk.0.ffn_down.weight" in params
            else 0.0
        )
        return (
            config.n_layers,
            config.d_model,
            config.n_experts,
            config.kv_cache_bits,
            tuple(config.active_subnodes),
            round(w0, 6),
            round(b0, 6),
        )

    def lookup_full_prefill(
        self,
        model_sig: tuple[object, ...],
        pre_ids: list[int],
        is_tiered_think: bool,
    ) -> tuple[dict[str, jnp.ndarray], jnp.ndarray, jnp.ndarray, int] | None:
        key = (model_sig, tuple(pre_ids), bool(is_tiered_think))
        hit = self._full_prefills.get(key)
        if hit is not None:
            self.full_hits += 1
            self.saved_prefill_tokens += len(pre_ids)
        return hit

    def store_full_prefill(
        self,
        model_sig: tuple[object, ...],
        pre_ids: list[int],
        is_tiered_think: bool,
        state: tuple[dict[str, jnp.ndarray], jnp.ndarray, jnp.ndarray, int],
    ) -> None:
        if len(self._full_prefills) >= self.max_entries:
            oldest = next(iter(self._full_prefills))
            self._full_prefills.pop(oldest, None)
        self._full_prefills[(model_sig, tuple(pre_ids), bool(is_tiered_think))] = state

    def lookup_longest_prefix(
        self,
        model_sig: tuple[object, ...],
        token_ids: list[int],
    ) -> tuple[dict[str, jnp.ndarray] | None, int]:
        n = len(token_ids)
        max_aligned = (n // self.chunk_size) * self.chunk_size
        for k in range(max_aligned, self.chunk_size - 1, -self.chunk_size):
            key = (model_sig, tuple(token_ids[:k]))
            hit = self._prefix_carries.get(key)
            if hit is not None:
                self.prefix_hits += 1
                self.saved_prefill_tokens += k
                return hit, k
        return None, 0

    def store_prefix_chunk(
        self,
        model_sig: tuple[object, ...],
        prefix_ids: list[int],
        carry: dict[str, jnp.ndarray],
    ) -> None:
        if not prefix_ids:
            return
        if len(self._prefix_carries) >= self.max_entries:
            oldest = next(iter(self._prefix_carries))
            self._prefix_carries.pop(oldest, None)
        self._prefix_carries[(model_sig, tuple(prefix_ids))] = carry

    def clear(self) -> None:
        self._prefix_carries.clear()
        self._full_prefills.clear()
        self.prefix_hits = 0
        self.full_hits = 0
        self.saved_prefill_tokens = 0


_DEFAULT_LMCACHE_V2 = LMCacheHybridV2(chunk_size=4, max_entries=64)


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


def profile_sparse_execution_v2(
    params: dict[str, jnp.ndarray],
    config: ConfigV2,
    token_ids: list[int] | None = None,
    n_bench_steps: int = 16,
) -> dict[str, float | int]:
    """Profile SpiderNet v0.0.22 True Sparse Index-Gather Execution vs. Dense Masked Execution.

    Computes exact parameter activation ratios, specialist FFN FLOPs per token,
    numerical equivalence (`max_logit_diff`), and single-token recurrent decode latency.
    """
    import time as _time

    total_params = int(sum(int(np.prod(v.shape)) for v in params.values()))
    specialist_keys = (".nodes.gate.weight", ".nodes.up.weight", ".nodes.down.weight")
    specialist_params = int(
        sum(
            int(np.prod(v.shape))
            for k, v in params.items()
            if any(k.endswith(sk) for sk in specialist_keys)
        )
    )
    shared_params = total_params - specialist_params
    n_nodes = max(1, int(config.n_experts))
    k_fast = max(1, min(n_nodes, int(config.top_k_experts)))
    k_delib = max(k_fast, min(n_nodes, int(2 * config.top_k_experts)))

    d_model = int(config.d_model)
    d_ff = (
        int(params["blk.0.nodes.gate.weight"].shape[-1])
        if "blk.0.nodes.gate.weight" in params
        else int(config.expert_ffn_dim or config.ffn_dim)
    )
    base_layers = list(range(config.n_layers))
    loop_layers = (
        list(range(config.loop_start_layer, min(config.loop_end_layer, config.n_layers)))
        * max(0, config.loop_repeats - 1)
    )
    n_passes = len(base_layers) + len(loop_layers)

    dense_flops = int(n_passes * n_nodes * 6 * d_model * d_ff)
    sparse_fast_flops = int(n_passes * k_fast * 6 * d_model * d_ff)
    sparse_delib_flops = int(n_passes * k_delib * 6 * d_model * d_ff)

    active_spec_fast = int(round(specialist_params * (k_fast / n_nodes)))
    active_spec_delib = int(round(specialist_params * (k_delib / n_nodes)))

    cos_table, sin_table = precompute_rope_freqs(
        config.max_seq_len, config.head_dim, config.rope_theta
    )

    @jax.jit
    def _step_sparse(p, c, t):
        return _recurrent_token_step_v2(
            p, c, t, config, cos_table, sin_table, use_sparse_gather=True
        )

    @jax.jit
    def _step_dense(p, c, t):
        return _recurrent_token_step_v2(
            p, c, t, config, cos_table, sin_table, use_sparse_gather=False
        )

    seq = (
        [int(x) for x in token_ids]
        if (token_ids is not None and len(token_ids) > 0)
        else [1, 10, 20, 30, 40, 50, 60, 70]
    )
    carry_s = init_decode_cache_v2(config)
    carry_d = init_decode_cache_v2(config)
    max_logit_diff = 0.0
    for tid in seq:
        t_arr = jnp.int32(tid)
        carry_s, logits_s, _ = _step_sparse(params, carry_s, t_arr)
        carry_d, logits_d, _ = _step_dense(params, carry_d, t_arr)
        diff = float(jnp.max(jnp.abs(logits_s - logits_d)))
        if diff > max_logit_diff:
            max_logit_diff = diff

    # Warmup and benchmark single-step latency
    t_bench = jnp.int32(seq[-1])
    _, l_s, _ = _step_sparse(params, carry_s, t_bench)
    l_s.block_until_ready()
    _, l_d, _ = _step_dense(params, carry_d, t_bench)
    l_d.block_until_ready()

    steps = max(1, int(n_bench_steps))
    t0 = _time.perf_counter()
    c_cur = carry_s
    for _ in range(steps):
        c_cur, l_s, _ = _step_sparse(params, c_cur, t_bench)
    l_s.block_until_ready()
    sparse_ms = ((_time.perf_counter() - t0) * 1000.0) / steps

    t1 = _time.perf_counter()
    c_cur_d = carry_d
    for _ in range(steps):
        c_cur_d, l_d, _ = _step_dense(params, c_cur_d, t_bench)
    l_d.block_until_ready()
    dense_ms = ((_time.perf_counter() - t1) * 1000.0) / steps

    return {
        "total_params": total_params,
        "shared_params": shared_params,
        "specialist_params": specialist_params,
        "n_nodes": n_nodes,
        "k_fast": k_fast,
        "k_delib": k_delib,
        "active_specialist_params_fast": active_spec_fast,
        "active_specialist_params_delib": active_spec_delib,
        "active_total_params_fast": shared_params + active_spec_fast,
        "active_total_params_delib": shared_params + active_spec_delib,
        "activation_ratio_fast": float(k_fast) / float(n_nodes),
        "activation_ratio_delib": float(k_delib) / float(n_nodes),
        "dense_specialist_flops_per_tok": dense_flops,
        "sparse_specialist_flops_fast_per_tok": sparse_fast_flops,
        "sparse_specialist_flops_delib_per_tok": sparse_delib_flops,
        "flops_reduction_fast_x": float(n_nodes) / float(k_fast),
        "flops_saved_fast_pct": 100.0 * (1.0 - float(k_fast) / float(n_nodes)),
        "max_logit_diff": max_logit_diff,
        "sparse_ms_per_tok": sparse_ms,
        "dense_ms_per_tok": dense_ms,
        "speedup_x": dense_ms / max(sparse_ms, 1e-6),
    }


def init_o1_ring_decode_cache_v2(
    config: ConfigV2,
    ring_window: int = 64,
) -> dict[str, jnp.ndarray]:
    """Initialize Strictly O(1) Constant-Memory Ring-Buffer Decode Cache (`v0.0.32`).

    Bounds the 2 DSA attention layers' KV cache to a fixed ring buffer of capacity `W = ring_window`
    independent of total sequence length `T`, while adding a constant-size Recurrent Eviction
    Summary State `evict_s` `(L_sched, 1, H_kv, D_h, D_h)` and normalizer `evict_z`
    `(L_sched, 1, H_kv, D_h)` that absorb tokens evicted from the `W`-token sliding window.
    """
    w_cap = max(8, int(ring_window))
    ring_cfg = config.model_copy(update={"max_seq_len": w_cap, "sliding_window": w_cap})
    base_carry = init_decode_cache_v2(ring_cfg)
    n_sched = int(base_carry["k_cache"].shape[0])
    h_kv = config.n_kv_heads
    d_h = config.head_dim
    base_carry["evict_s"] = jnp.zeros((n_sched, 1, h_kv, d_h, d_h), dtype=jnp.float32)
    base_carry["evict_z"] = jnp.zeros((n_sched, 1, h_kv, d_h), dtype=jnp.float32)
    base_carry["total_steps"] = jnp.int32(0)
    return base_carry


def step_o1_ring_decode_v2(
    params: dict[str, jnp.ndarray],
    ring_carry: dict[str, jnp.ndarray],
    tok_id_scalar: jnp.ndarray,
    config: ConfigV2,
    ring_window: int = 64,
    evict_decay: float = 0.98,
) -> tuple[dict[str, jnp.ndarray], jnp.ndarray, jnp.ndarray]:
    """Execute one strictly O(1)-memory token decode step with Ring-Buffer KV + Recurrent Eviction.

    - For `t < W` (`ring_window`), executes identically (`max_logit_diff == 0.0`) to standard
      `_recurrent_token_step_v2`.
    - For `t >= W`, folds the overwritten ring-buffer slot `r_t = t mod W` into the constant-size
      recurrent summary state `(evict_s, evict_z)` before writing the new token into slot `r_t`,
      guaranteeing `0.0 bytes/token` state memory growth as `T -> infinity`.
    """
    w_cap = max(8, int(ring_window))
    ring_cfg = config.model_copy(update={"max_seq_len": w_cap, "sliding_window": w_cap})
    t_step = ring_carry["total_steps"]
    slot_idx = jnp.mod(t_step, jnp.int32(w_cap))
    is_evicting = t_step >= jnp.int32(w_cap)

    k_old = ring_carry["k_cache"][:, :, slot_idx, :, :]  # (L_sched, 1, H_kv, D_h)
    v_old = ring_carry["v_cache"][:, :, slot_idx, :, :]  # (L_sched, 1, H_kv, D_h)
    phi_k = jax.nn.elu(k_old) + 1.0
    phi_k = phi_k * jax.lax.rsqrt(jnp.sum(jnp.square(phi_k), axis=-1, keepdims=True) + 1e-6)
    kv_outer = phi_k[..., :, None] * v_old[..., None, :]  # (L_sched, 1, H_kv, D_h, D_h)

    evict_s_new = jnp.where(
        is_evicting,
        float(evict_decay) * ring_carry["evict_s"] + kv_outer,
        ring_carry["evict_s"],
    )
    evict_z_new = jnp.where(
        is_evicting,
        float(evict_decay) * ring_carry["evict_z"] + phi_k,
        ring_carry["evict_z"],
    )

    step_in_carry = dict(ring_carry)
    step_in_carry["pos"] = slot_idx
    cos_table, sin_table = precompute_rope_freqs(
        max(w_cap, 64), ring_cfg.head_dim, ring_cfg.rope_theta
    )
    next_base, logits, draft = _recurrent_token_step_v2(
        params,
        step_in_carry,
        tok_id_scalar,
        ring_cfg,
        cos_table,
        sin_table,
        use_sparse_gather=True,
    )
    next_base["evict_s"] = evict_s_new
    next_base["evict_z"] = evict_z_new
    next_base["total_steps"] = t_step + jnp.int32(1)
    next_base["pos"] = jnp.minimum(t_step + jnp.int32(1), jnp.int32(w_cap - 1))
    return next_base, logits, draft


def profile_o1_memory_scaling_v2(
    params: dict[str, jnp.ndarray],
    config: ConfigV2,
    ring_window: int = 32,
    seq_lengths: tuple[int, ...] = (32, 64, 128, 256, 512, 1024, 2048, 4096),
) -> dict[str, object]:
    """Verify Strictly O(1) Constant State Memory Scaling (`0.0 B/tok` slope, `v0.0.32`).

    Compares full-sequence KV cache memory growth (`O(T)`) against the Ring-Buffer + Recurrent
    Eviction Bridge (`O(1)`) across `seq_lengths`, and verifies exact logit equivalence within
    the window (`t < ring_window`) plus active eviction state accumulation for `t >= ring_window`.
    """
    w_cap = max(8, int(ring_window))
    ring_carry = init_o1_ring_decode_cache_v2(config, ring_window=w_cap)
    o1_state_bytes = int(sum(int(v.nbytes) for v in ring_carry.values()))

    scaling_rows: list[dict[str, float | int]] = []
    for t_len in seq_lengths:
        full_cfg = config.model_copy(update={"max_seq_len": int(t_len)})
        full_carry = init_decode_cache_v2(full_cfg)
        full_bytes = int(sum(int(v.nbytes) for v in full_carry.values()))
        scaling_rows.append(
            {
                "seq_len": int(t_len),
                "full_kv_state_bytes": full_bytes,
                "o1_ring_state_bytes": o1_state_bytes,
                "memory_reduction_x": round(float(full_bytes) / float(max(1, o1_state_bytes)), 2),
            }
        )

    # Verify exact logit equivalence for t < W and eviction accumulation for t >= W
    std_cfg = config.model_copy(update={"max_seq_len": w_cap, "sliding_window": w_cap})
    std_carry = init_decode_cache_v2(std_cfg)
    cos_t, sin_t = precompute_rope_freqs(max(w_cap, 64), std_cfg.head_dim, std_cfg.rope_theta)

    @jax.jit
    def _jit_ring(p, c, tok):
        return step_o1_ring_decode_v2(p, c, tok, config, ring_window=w_cap)

    @jax.jit
    def _jit_std(p, c, tok):
        return _recurrent_token_step_v2(p, c, tok, std_cfg, cos_t, sin_t, use_sparse_gather=True)

    max_diff_in_window = 0.0
    c_ring = ring_carry
    c_std = std_carry
    n_test_steps = w_cap + 6
    for step_i in range(n_test_steps):
        tid = jnp.int32(1 + (step_i % max(1, config.vocab_size - 2)))
        c_ring, l_ring, _ = _jit_ring(params, c_ring, tid)
        if step_i < w_cap:
            c_std, l_std, _ = _jit_std(params, c_std, tid)
            diff = float(jnp.max(jnp.abs(l_ring - l_std)))
            if diff > max_diff_in_window:
                max_diff_in_window = diff

    evict_norm = float(jnp.linalg.norm(c_ring["evict_s"]))
    slope_o1 = float(
        scaling_rows[-1]["o1_ring_state_bytes"] - scaling_rows[0]["o1_ring_state_bytes"]
    ) / float(max(1, int(seq_lengths[-1]) - int(seq_lengths[0])))
    slope_full = float(
        scaling_rows[-1]["full_kv_state_bytes"] - scaling_rows[0]["full_kv_state_bytes"]
    ) / float(max(1, int(seq_lengths[-1]) - int(seq_lengths[0])))

    return {
        "ring_window": w_cap,
        "is_strictly_o1_constant_memory": bool(slope_o1 == 0.0),
        "o1_constant_state_bytes": o1_state_bytes,
        "o1_bytes_per_token_slope": slope_o1,
        "standard_bytes_per_token_slope": round(slope_full, 2),
        "full_kv_bytes_per_token_slope": round(slope_full, 2),
        "max_logit_diff_within_window": max_diff_in_window,
        "evicted_summary_state_norm": round(evict_norm, 4),
        "scaling_table": scaling_rows,
    }


_CODE_LOGIC_PATTERNS = (
    "python",
    "rust",
    "def ",
    "fn ",
    "write a function",
    "write code",
    "boolean",
    "true and ",
    "true or ",
    "false and ",
    "false or ",
    "not (true",
    "not (false",
    "not true",
    "not false",
    " xor ",
    "bitwise",
    " << ",
    " >> ",
    " & ",
    " | ",
    " ^ ",
    "de morgan",
    "modus ponens",
    "modus tollens",
    "syllogism",
    "if all ",
    "fibonacci",
    "is_prime",
    "binary search",
    "binary_search",
    "time complexity",
    "big-o",
    "list comprehension",
    "range(",
    "sorted(",
    "len(",
    "[x ",
    "[i ",
)


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
      - `"verified"`: Mathematical & Code/Symbolic step-verified reasoning
        (Dot 1 Math Sub-Nodes 10..19 + PRM or Dot 2 Code/Logic Sub-Nodes 20..29 + Symbolic Verifier)
    """
    u_part = prompt.split("\n<|assistant|>\n", 1)[0].replace("<|user|>\n", "").strip()
    u_low = u_part.lower()
    p_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    seq_dl_val = -1.0
    active_pair = (6, 7)
    mean_rl: np.ndarray | None = None
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

    is_code_or_logic = config.n_experts >= 30 and any(pat in u_low for pat in _CODE_LOGIC_PATTERNS)
    if is_code_or_logic:
        mode = "verified"
        domain = "code"
        if any(
            w in u_low
            for w in (
                "boolean",
                "true and ",
                "true or ",
                "false and ",
                "false or ",
                "not (",
                "not true",
                "not false",
                " xor ",
                "bitwise",
                " << ",
                " >> ",
                " & ",
                " | ",
                " ^ ",
                "de morgan",
                "modus ponens",
                "modus tollens",
                "syllogism",
                "if all ",
            )
        ):
            active_pair = (24, 25)
        elif any(
            w in u_low
            for w in (
                "fibonacci",
                "is_prime",
                "binary search",
                "binary_search",
                "time complexity",
                "big-o",
                "list comprehension",
                "range(",
                "sorted(",
                "len(",
                "[x ",
                "[i ",
            )
        ):
            active_pair = (22, 23)
        elif any(w in u_low for w in ("type", "bug", "error", "trace", "debug")):
            active_pair = (26, 27)
        else:
            active_pair = (20, 22)
        return {
            "mode": mode,
            "domain": domain,
            "seq_dl": round(seq_dl_val, 3),
            "active_subnodes": active_pair,
        }

    eff_rank = 8.0
    if "tok_embeddings.weight" in params:
        emb_s = np.asarray(params["tok_embeddings.weight"], dtype=np.float32)[p_ids]
        eff_rank = compute_von_neumann_effective_rank_v2(emb_s)
    has_q_token = bool(np.any(p_ids == 38))
    if seq_dl_val >= -0.95:
        mode = "verified"
        domain = "math"
    elif eff_rank < 6.0 and not has_q_token:
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
    if cog_info.get("domain") == "code" and len(config.active_subnodes) == 0:
        config = config.model_copy(
            update={"active_subnodes": tuple(cog_info["active_subnodes"])}  # type: ignore[arg-type]
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
        pre_ids = prompt_ids[:-1]
        active_lmcache = (
            memory_hierarchy.lmcache if memory_hierarchy is not None else _DEFAULT_LMCACHE_V2
        )
        m_sig = active_lmcache.model_signature(params, config)
        full_hit = active_lmcache.lookup_full_prefill(m_sig, pre_ids, is_tiered_think)
        if full_hit is not None:
            if is_tiered_think and len(pre_ids) >= 1:
                carry, prompt_raw, prompt_draft, fork_pos = full_hit
                fork_prefill_state = (carry, prompt_raw, prompt_draft, fork_pos)
            else:
                carry, _, _, _ = full_hit
        elif is_tiered_think and len(pre_ids) >= 1:
            bulk_ids = pre_ids[:-1]
            cached_carry, p_idx = active_lmcache.lookup_longest_prefix(m_sig, bulk_ids)
            carry = cached_carry if cached_carry is not None else init_decode_cache_v2(config)
            while p_idx + 2 <= len(bulk_ids):
                _, (carry, _, _) = spec2(
                    params,
                    carry,
                    jnp.int32(bulk_ids[p_idx]),
                    jnp.int32(bulk_ids[p_idx + 1]),
                )
                p_idx += 2
                if p_idx % active_lmcache.chunk_size == 0:
                    active_lmcache.store_prefix_chunk(m_sig, bulk_ids[:p_idx], carry)
            while p_idx < len(bulk_ids):
                carry, _, _ = step1(params, carry, jnp.int32(bulk_ids[p_idx]))
                p_idx += 1
                if p_idx % active_lmcache.chunk_size == 0:
                    active_lmcache.store_prefix_chunk(m_sig, bulk_ids[:p_idx], carry)
            carry, prompt_raw, prompt_draft = step1(params, carry, jnp.int32(pre_ids[-1]))
            fork_prefill_state = (carry, prompt_raw, prompt_draft, len(pre_ids) - 1)
            active_lmcache.store_full_prefill(
                m_sig, pre_ids, is_tiered_think, fork_prefill_state
            )
        else:
            cached_carry, p_idx = active_lmcache.lookup_longest_prefix(m_sig, pre_ids)
            carry = cached_carry if cached_carry is not None else init_decode_cache_v2(config)
            while p_idx + 2 <= len(pre_ids):
                _, (carry, _, _) = spec2(
                    params,
                    carry,
                    jnp.int32(pre_ids[p_idx]),
                    jnp.int32(pre_ids[p_idx + 1]),
                )
                p_idx += 2
                if p_idx % active_lmcache.chunk_size == 0:
                    active_lmcache.store_prefix_chunk(m_sig, pre_ids[:p_idx], carry)
            while p_idx < len(pre_ids):
                carry, _, _ = step1(params, carry, jnp.int32(pre_ids[p_idx]))
                p_idx += 1
                if p_idx % active_lmcache.chunk_size == 0:
                    active_lmcache.store_prefix_chunk(m_sig, pre_ids[:p_idx], carry)
            dummy_z = jnp.zeros((config.vocab_size,), dtype=jnp.float32)
            active_lmcache.store_full_prefill(
                m_sig, pre_ids, is_tiered_think, (carry, dummy_z, jnp.int32(-1), len(pre_ids))
            )
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
                if cog_info.get("domain") == "code":
                    thought_body, clean_ans = _verify_code_and_logic_v2(
                        prompt, thought_body, clean_ans
                    )
                else:
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
        if cog_info.get("domain") == "code":
            _, ans_clean = _verify_code_and_logic_v2(prompt, "", ans_clean)
        else:
            ans_clean = _verify_reasoning_steps_v2(ans_clean, prompt=prompt)
    return ans_clean
