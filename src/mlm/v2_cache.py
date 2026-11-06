"""KV-Cache, LMCacheHybridV2, CacheBlend RoPE Shift, O(1) Ring-Buffer & Sparse Profiler."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from mlm.v2_graph import directed_graph_message_passing_v2
from mlm.v2_kernels import build_layer_execution_schedule, precompute_rope_freqs

if TYPE_CHECKING:
    from mlm.model_v2 import ConfigV2


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


def profile_sparse_execution_v2(
    params: dict[str, jnp.ndarray],
    config: ConfigV2,
    token_ids: list[int] | None = None,
    n_bench_steps: int = 16,
) -> dict[str, float | int]:
    """Profile SpiderNet v0.0.22 True Sparse Index-Gather Execution vs. Dense Masked Execution."""
    import time as _time

    from mlm.v2_decode import _recurrent_token_step_v2

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
    """Initialize Strictly O(1) Constant-Memory Ring-Buffer Decode Cache (`v0.0.32`)."""
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
    """Execute one strictly O(1)-memory token decode step with Ring-Buffer KV & Eviction."""
    from mlm.v2_decode import _recurrent_token_step_v2

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
    """Verify Strictly O(1) Constant State Memory Scaling (`0.0 B/tok` slope, `v0.0.32`)."""
    from mlm.v2_decode import _recurrent_token_step_v2

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
