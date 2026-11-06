"""Stepwise Recurrent Decoding, LMCacheHybridV2, O(1) Ring-Buffer KV & Generation for MLM v2."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from mlm.v2_cache import (  # noqa: F401
    _DEFAULT_LMCACHE_V2,
    LMCacheHybridV2,
    _execute_dense_nodes_step_v2,
    _execute_sparse_nodes_step_v2,
    cacheblend_fuse_kv_chunks_v2,
    cacheblend_rope_shift_kv_v2,
    init_decode_cache_v2,
    init_o1_ring_decode_cache_v2,
    profile_o1_memory_scaling_v2,
    profile_sparse_execution_v2,
    step_o1_ring_decode_v2,
)
from mlm.v2_graph import (  # noqa: F401
    _route_single_bank_from_combined_logits_v2,
    _route_single_bank_step_v2,
)
from mlm.v2_kernels import (
    apply_rope,
    build_layer_execution_schedule,
    mhc_read_stream,
    mhc_write_stream,
    ncm_tied_codebook_logits,
    precompute_rope_freqs,
    rms_norm,
    sandwich_rms_bound,
    swiglu_ffn,
    turboquant_kv_tensor_jax,
)

if TYPE_CHECKING:
    from mlm.model_v2 import ConfigV2


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


from mlm.v2_generate import (  # noqa: E402, F401
    classify_cognitive_mode_v2,
    generate_v2,
)

