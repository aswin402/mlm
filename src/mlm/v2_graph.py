"""SpiderNet Directed Graph Routing, Message Passing, FDA Calibration & Multi-Hop Diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from mlm.v2_kernels import rms_norm

if TYPE_CHECKING:
    from mlm.model_v2 import ConfigV2


def _route_single_bank_from_combined_logits_v2(
    m_logits: jnp.ndarray,
    bank_logits: jnp.ndarray,
    bank_bias: jnp.ndarray | None,
    top_k: int,
    new_tier_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Shared top-k base + sigmoid `<|think|>` router given combined base logits `m_logits`."""
    b_nodes = bank_logits.shape[-1]
    base_n = min(b_nodes, 8)
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


def _route_single_bank(
    bank_logits: jnp.ndarray,
    bank_bias: jnp.ndarray | None,
    top_k: int,
    new_tier_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Route tokens within a single 10-node Specialist Macro-Bank (Nodes 0..7 base + 8..9 think)."""
    base_n = min(bank_logits.shape[-1], 8)
    m_raw = bank_logits[..., :base_n]
    steps = jnp.arange(1, m_raw.shape[1] + 1, dtype=m_raw.dtype)[None, :, None]
    m_logits = 0.45 * m_raw + 0.55 * (jnp.cumsum(m_raw, axis=1) / steps)
    return _route_single_bank_from_combined_logits_v2(
        m_logits, bank_logits, bank_bias, top_k=top_k, new_tier_mask=new_tier_mask
    )


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


def compute_directed_edge_weights_v2(
    x: jnp.ndarray,
    node_outs: jnp.ndarray,
    edge_u: jnp.ndarray,
    edge_v: jnp.ndarray,
    edge_proj_h: jnp.ndarray,
    edge_proj_x: jnp.ndarray,
    edge_bias: jnp.ndarray,
    top_k_edges: int = 2,
    temperature: float = 1.0,
    active_node_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Compute Input-Conditioned Low-Rank Bilinear Directed Edge Weights `a_ij(x_t)`.

    Implements SpiderNet v0.0.23/v0.0.24 Directed Graph Edge Routing:
      `e_ij(x_t) = ((q_i(x_t) @ k_j(x_t)^T) + u_i^T r(x_t) + v_j^T r(x_t)) / sqrt(r) + c_ij`
    with strict zero self-loop constraint `A_ii = 0` (`e_ii = -1e9`) and top-k outgoing
    collaborator selection per primary node `i`.

    Args:
        x: Input token states `(B, T, D)`
        node_outs: Hop-1 specialist outputs `H^(1)` of shape `(B, T, M, D)`
        edge_u: Source node embeddings `(M, r)`
        edge_v: Target collaborator embeddings `(M, r)`
        edge_proj_h: Hop-1 state projection `(D, r)`
        edge_proj_x: Token conditioning projection `(D, r)`
        edge_bias: Directed adjacency prior bias `(M, M)`
        active_node_mask: Optional boolean mask `(B, T, M)` restricting collaborator targets
    Returns:
        Sparse directed adjacency weights `a_ij(x_t)` of shape `(B, T, M, M)` with zero diagonal.
    """
    m_nodes = int(node_outs.shape[2])
    if m_nodes <= 1:
        return jnp.zeros(
            (x.shape[0], x.shape[1], m_nodes, m_nodes), dtype=x.dtype
        )
    rank = int(edge_u.shape[-1])
    inv_sqrt_r = 1.0 / jnp.sqrt(float(max(1, rank)))

    h_proj = jnp.einsum("btmd,dr->btmr", node_outs, edge_proj_h)
    q_node = h_proj + edge_u[None, None, :, :]  # (B, T, M, r)
    k_node = h_proj + edge_v[None, None, :, :]  # (B, T, M, r)
    r_tok = jnp.einsum("btd,dr->btr", x, edge_proj_x)  # (B, T, r)

    bilinear = jnp.einsum("btir,btjr->btij", q_node, k_node)
    src_ctx = jnp.einsum("ir,btr->bti", edge_u, r_tok)[..., :, None]
    dst_ctx = jnp.einsum("jr,btr->btj", edge_v, r_tok)[..., None, :]
    edge_logits = (bilinear + src_ctx + dst_ctx) * inv_sqrt_r + edge_bias[None, None, :, :]

    diag_mask = jnp.eye(m_nodes, dtype=jnp.bool_)[None, None, :, :]
    valid_edge_mask = ~diag_mask
    if active_node_mask is not None:
        valid_edge_mask = valid_edge_mask & active_node_mask[..., None, :]
    edge_logits = jnp.where(valid_edge_mask, edge_logits, -1e9)
    scaled_logits = edge_logits / float(max(0.05, temperature))

    k_eff = max(1, min(int(top_k_edges), m_nodes - 1))
    if k_eff < m_nodes - 1:
        _, top_j = jax.lax.top_k(scaled_logits, k_eff)  # (B, T, M, k_eff)
        top_mask = (
            jnp.sum(jax.nn.one_hot(top_j, m_nodes, dtype=jnp.int32), axis=-2) > 0
        ) & valid_edge_mask
        masked_logits = jnp.where(top_mask, scaled_logits, -1e9)
        edge_probs = jax.nn.softmax(masked_logits, axis=-1)
        edge_probs = jnp.where(top_mask, edge_probs, 0.0)
        denom = jnp.maximum(jnp.sum(edge_probs, axis=-1, keepdims=True), 1e-6)
        return edge_probs / denom

    edge_probs = jax.nn.softmax(scaled_logits, axis=-1)
    edge_probs = jnp.where(valid_edge_mask, edge_probs, 0.0)
    denom = jnp.maximum(jnp.sum(edge_probs, axis=-1, keepdims=True), 1e-6)
    return edge_probs / denom


def directed_graph_message_passing_v2(
    x: jnp.ndarray,
    gated_probs: jnp.ndarray,
    node_outs: jnp.ndarray,
    edge_u: jnp.ndarray,
    edge_v: jnp.ndarray,
    edge_proj_h: jnp.ndarray,
    edge_proj_x: jnp.ndarray,
    edge_bias: jnp.ndarray,
    msg_down: jnp.ndarray,
    msg_up: jnp.ndarray,
    collab_gate: jnp.ndarray,
    top_k_edges: int = 2,
    temperature: float = 1.0,
    return_edge_weights: bool = False,
) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
    """Execute Hop-1 + Hop-2 Directed Inter-Node Graph Message Passing (`N_i -> N_j`).

    1. Primary nodes `i` compute Hop-1 outputs `h_i^(1)` and emit directed messages
       `m_i = (h_i^(1) @ msg_down) @ msg_up`.
    2. Input-conditioned directed edges `a_ij(x_t)` route messages from primary node `i`
       to collaborator node `j != i`: `M_j = sum_{i != j} p_i(x_t) * a_ij(x_t) * m_i`.
    3. Collaborator node `j` refines its representation via gated residual modulation:
       `delta_h_j^(2) = tanh(collab_gate_j) * (silu(M_j) * norm(h_j^(1)) + M_j)`.
    4. Combined graph readout:
       `y_t = sum_i p_i(x_t) h_i^(1) + sum_j q_j(x_t) delta_h_j^(2)`,
       where `q_j(x_t) = sum_{i != j} p_i(x_t) a_ij(x_t)`.
    """
    base_out = jnp.einsum("btn,btnd->btd", gated_probs, node_outs)
    m_nodes = int(node_outs.shape[2])
    if m_nodes <= 1:
        if return_edge_weights:
            z_edges = jnp.zeros((x.shape[0], x.shape[1], m_nodes, m_nodes), dtype=x.dtype)
            return base_out, z_edges
        return base_out

    edge_weights = compute_directed_edge_weights_v2(
        x,
        node_outs,
        edge_u,
        edge_v,
        edge_proj_h,
        edge_proj_x,
        edge_bias,
        top_k_edges=top_k_edges,
        temperature=temperature,
        active_node_mask=(gated_probs > 0.0),
    )  # (B, T, M, M)

    msg_lat = jnp.einsum("btmd,dr->btmr", node_outs, msg_down)
    msg_emitted = jnp.einsum("btmr,rd->btmd", msg_lat, msg_up)  # (B, T, M, D)
    flow_ij = gated_probs[..., :, None] * edge_weights  # (B, T, M_src, M_dst)
    inc_msg = jnp.einsum("btij,btid->btjd", flow_ij, msg_emitted)  # (B, T, M_dst, D)
    inc_lat = jnp.einsum("btij,btir->btjr", flow_ij, msg_lat)  # (B, T, M_dst, r)
    inc_weight = jnp.sum(flow_ij, axis=-2)  # (B, T, M_dst)

    h_norm = node_outs * jax.lax.rsqrt(
        jnp.mean(jnp.square(node_outs), axis=-1, keepdims=True) + 1e-5
    )
    h_lat = jnp.einsum("btmd,dr->btmr", h_norm, edge_proj_h)
    inv_sqrt_r = 1.0 / jnp.sqrt(float(max(1, int(edge_proj_h.shape[-1]))))
    gate_logit = (
        jnp.einsum("btjr,btjr->btj", h_lat, inc_lat)[..., None] * inv_sqrt_r
        + h_norm * inc_msg
    )
    sigma_gate = 2.0 * jax.nn.sigmoid(gate_logit)
    gamma_j = jnp.tanh(collab_gate.astype(node_outs.dtype))[None, None, :, None] * sigma_gate
    delta_h2 = gamma_j * (jax.nn.silu(inc_msg) * h_norm + inc_msg)
    collab_out = jnp.einsum("btj,btjd->btd", inc_weight, delta_h2)
    total_out = base_out + collab_out
    if return_edge_weights:
        return total_out, edge_weights
    return total_out


def adaptive_multi_hop_message_passing_v2(
    x: jnp.ndarray,
    gated_probs: jnp.ndarray,
    node_outs: jnp.ndarray,
    edge_u: jnp.ndarray,
    edge_v: jnp.ndarray,
    edge_proj_h: jnp.ndarray,
    edge_proj_x: jnp.ndarray,
    edge_bias: jnp.ndarray,
    msg_down: jnp.ndarray,
    msg_up: jnp.ndarray,
    collab_gate: jnp.ndarray,
    max_hops: int = 3,
    top_k_edges: int = 2,
    temperature: float = 1.0,
    hop2_mask: jnp.ndarray | None = None,
    hop3_mask: jnp.ndarray | None = None,
    return_diagnostics: bool = False,
) -> jnp.ndarray | tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Execute Adaptive Multi-Hop (`1-Hop` / `2-Hop` / `3-Hop`) Directed Graph Message Passing."""
    base_out = jnp.einsum("btn,btnd->btd", gated_probs, node_outs)
    m_nodes = int(node_outs.shape[2])
    hops_eff = max(1, min(3, int(max_hops)))
    if m_nodes <= 1 or hops_eff == 1:
        if return_diagnostics:
            z_edges = jnp.zeros(
                (x.shape[0], x.shape[1], m_nodes, m_nodes), dtype=node_outs.dtype
            )
            return base_out, {
                "a_hop1": z_edges,
                "a_hop2": z_edges,
                "q_hop2": jnp.zeros_like(gated_probs),
                "q_hop3": jnp.zeros_like(gated_probs),
            }
        return base_out

    inv_sqrt_r = 1.0 / jnp.sqrt(float(max(1, int(edge_proj_h.shape[-1]))))
    cg_tanh = jnp.tanh(collab_gate.astype(node_outs.dtype))[None, None, :, None]

    # Hop 1 -> Hop 2 (`N_i -> N_j`)
    mask_h2 = hop2_mask if hop2_mask is not None else (gated_probs > 0.0)
    a_hop1 = compute_directed_edge_weights_v2(
        x,
        node_outs,
        edge_u,
        edge_v,
        edge_proj_h,
        edge_proj_x,
        edge_bias,
        top_k_edges=top_k_edges,
        temperature=temperature,
        active_node_mask=mask_h2,
    )
    msg_lat1 = jnp.einsum("btid,dr->btir", node_outs, msg_down)
    weighted_src1 = gated_probs[:, :, :, None] * a_hop1
    inc_lat1 = jnp.einsum("btij,btir->btjr", weighted_src1, msg_lat1)
    inc_msg1 = jnp.einsum("btjr,rd->btjd", inc_lat1, msg_up)
    q_hop2 = jnp.sum(weighted_src1, axis=2)  # (B, T, M)

    h1_norm = node_outs * jax.lax.rsqrt(
        jnp.sum(jnp.square(node_outs), axis=-1, keepdims=True) + 1e-6
    )
    h1_lat = jnp.einsum("btmd,dr->btmr", h1_norm, edge_proj_h)
    gate_logit1 = (
        jnp.einsum("btjr,btjr->btj", h1_lat, inc_lat1)[..., None] * inv_sqrt_r
        + h1_norm * inc_msg1
    )
    gamma1 = cg_tanh * (2.0 * jax.nn.sigmoid(gate_logit1))
    delta_h2 = gamma1 * (jax.nn.silu(inc_msg1) * h1_norm + inc_msg1)
    h2_outs = node_outs + delta_h2
    out_hop2 = base_out + jnp.einsum("btj,btjd->btd", q_hop2, delta_h2)

    if hops_eff == 2:
        if return_diagnostics:
            return out_hop2, {
                "a_hop1": a_hop1,
                "a_hop2": jnp.zeros_like(a_hop1),
                "q_hop2": q_hop2,
                "q_hop3": jnp.zeros_like(q_hop2),
            }
        return out_hop2

    # Hop 2 -> Hop 3 (`N_j -> N_k` using updated Hop-2 representations `h2_outs`)
    mask_h3 = hop3_mask if hop3_mask is not None else (mask_h2 | (q_hop2 > 0.0))
    a_hop2 = compute_directed_edge_weights_v2(
        x,
        h2_outs,
        edge_u,
        edge_v,
        edge_proj_h,
        edge_proj_x,
        edge_bias,
        top_k_edges=top_k_edges,
        temperature=temperature,
        active_node_mask=mask_h3,
    )
    msg_lat2 = jnp.einsum("btjd,dr->btjr", h2_outs, msg_down)
    weighted_src2 = q_hop2[:, :, :, None] * a_hop2
    inc_lat2 = jnp.einsum("btjk,btjr->btkr", weighted_src2, msg_lat2)
    inc_msg2 = jnp.einsum("btkr,rd->btkd", inc_lat2, msg_up)
    q_hop3 = jnp.sum(weighted_src2, axis=2)  # (B, T, M)

    h2_norm = h2_outs * jax.lax.rsqrt(
        jnp.sum(jnp.square(h2_outs), axis=-1, keepdims=True) + 1e-6
    )
    h2_lat = jnp.einsum("btmd,dr->btmr", h2_norm, edge_proj_h)
    gate_logit2 = (
        jnp.einsum("btkr,btkr->btk", h2_lat, inc_lat2)[..., None] * inv_sqrt_r
        + h2_norm * inc_msg2
    )
    gamma2 = cg_tanh * (2.0 * jax.nn.sigmoid(gate_logit2))
    delta_h3 = gamma2 * (jax.nn.silu(inc_msg2) * h2_norm + inc_msg2)
    out_hop3 = out_hop2 + jnp.einsum("btk,btkd->btd", q_hop3, delta_h3)

    if return_diagnostics:
        return out_hop3, {
            "a_hop1": a_hop1,
            "a_hop2": a_hop2,
            "q_hop2": q_hop2,
            "q_hop3": q_hop3,
        }
    return out_hop3


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
) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
    """Route tokens `(B, T, D)` across Specialist Graph-Node Banks (FlexOlmo Cosine Routing)."""
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
        math_end = min(n_nodes, 20)
        eng_bias = router_bias[:10] if router_bias is not None else None
        math_bias = router_bias[10:math_end] if router_bias is not None else None
        eng_probs = _route_single_bank(
            router_logits[..., :10],
            eng_bias,
            top_k=top_k,
            new_tier_mask=new_tier_mask,
        )
        math_probs = _route_single_bank(
            router_logits[..., 10:math_end],
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
        if n_nodes <= 20:
            gated_probs = jnp.concatenate([p_eng * eng_probs, p_math * math_probs], axis=-1)
        else:
            code_bias = router_bias[20:] if router_bias is not None else None
            code_probs = _route_single_bank(
                router_logits[..., 20:],
                code_bias,
                top_k=top_k,
                new_tier_mask=new_tier_mask,
            )
            code_base_end = min(n_nodes, 28)
            n_code_base = max(1, code_base_end - 20)
            code_bias_base = (
                code_bias[:n_code_base]
                if code_bias is not None
                else jnp.full((n_code_base,), -2.5, dtype=x.dtype)
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
            code_active_cond = (jnp.max(code_bias_base) > 1.0) | (
                jnp.mean(code_dl_3d, axis=1, keepdims=True) >= 2.0
            )
            p_code_soft = jax.nn.sigmoid(code_dl_3d)
            p_code_rect = jnp.where(
                code_active_cond, jnp.ones_like(p_code_soft), jnp.zeros_like(p_code_soft)
            )
            p_code = p_code_soft + jax.lax.stop_gradient(p_code_rect - p_code_soft)
            p_rem = 1.0 - p_code
            gated_probs = jnp.concatenate(
                [p_rem * p_eng * eng_probs, p_rem * p_math * math_probs, p_code * code_probs],
                axis=-1,
            )

    if node_edge_scale is not None:
        e_mod = jnp.clip(node_edge_scale.astype(gated_probs.dtype), 0.1, 2.5)[None, None, :]
        gated_probs = gated_probs * e_mod

    gate_all = jax.nn.silu(jnp.einsum("btd,ndf->btnf", x, node_gate))
    up_all = jnp.einsum("btd,ndf->btnf", x, node_up)
    hidden = gate_all * up_all
    node_outs = jnp.einsum("btnf,nfd->btnd", hidden, node_down)
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
        out = directed_graph_message_passing_v2(
            x,
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
    else:
        out = jnp.einsum("btn,btnd->btd", gated_probs, node_outs)
    if return_domain_logit:
        return out, domain_logit
    return out


def compute_spidernet_graph_regularizers_v2(
    params: dict[str, jnp.ndarray],
    inputs: jnp.ndarray,
    mask: jnp.ndarray,
    config: ConfigV2,
) -> dict[str, jnp.ndarray]:
    """Compute SpiderNet v0.0.24 Anti-Collapse Graph Regularizers (`L_load`, `L_edge`, `L_div`)."""
    n_exp = int(config.n_experts)
    if n_exp < 2:
        z = jnp.float32(0.0)
        return {"total_reg": z, "load_loss": z, "edge_loss": z, "diversity_loss": z}

    tok_emb = params["tok_embeddings.weight"][inputs]
    valid_w = (mask > 0).astype(jnp.float32)[..., None]  # (B, T, 1)
    valid_denom = jnp.maximum(jnp.sum(valid_w), 1.0)
    k_eff = max(1, min(int(config.top_k_experts), n_exp))

    load_acc = jnp.float32(0.0)
    edge_acc = jnp.float32(0.0)
    div_acc = jnp.float32(0.0)
    n_layers_used = 0
    off_diag = 1.0 - jnp.eye(n_exp, dtype=jnp.float32)
    off_denom = float(max(1, n_exp * (n_exp - 1)))

    for i in range(config.n_layers):
        p = f"blk.{i}"
        rw_key = f"{p}.nodes.router.weight"
        if rw_key not in params:
            continue
        n_layers_used += 1
        normed = rms_norm(tok_emb, params[f"{p}.ffn_norm.weight"], config.rms_eps)
        rw = params[rw_key]
        x_u = normed * jax.lax.rsqrt(jnp.sum(jnp.square(normed), axis=-1, keepdims=True) + 1e-6)
        w_u = rw * jax.lax.rsqrt(jnp.sum(jnp.square(rw), axis=0, keepdims=True) + 1e-6)
        r_logits = 8.0 * jnp.matmul(x_u, w_u)
        r_probs = jax.nn.softmax(r_logits, axis=-1)  # (B, T, N)

        _, top_idx = jax.lax.top_k(r_probs, k_eff)
        sel_mask = (
            jnp.sum(jax.nn.one_hot(top_idx, n_exp, dtype=jnp.float32), axis=-2) / float(k_eff)
        )
        f_i = jnp.sum(sel_mask * valid_w, axis=(0, 1)) / valid_denom
        p_i = jnp.sum(r_probs * valid_w, axis=(0, 1)) / valid_denom
        load_acc = load_acc + float(n_exp) * jnp.sum(f_i * p_i)

        r_gram = jnp.matmul(jnp.transpose(w_u), w_u)  # (N, N)
        r_div = jnp.sum(jnp.square(r_gram) * off_diag) / off_denom
        down_w = params[f"{p}.nodes.down.weight"]  # (N, F, D)
        d_sig = jnp.mean(down_w, axis=1)  # (N, D)
        d_u = d_sig * jax.lax.rsqrt(jnp.sum(jnp.square(d_sig), axis=-1, keepdims=True) + 1e-6)
        d_gram = jnp.matmul(d_u, jnp.transpose(d_u))
        d_div = jnp.sum(jnp.square(d_gram) * off_diag) / off_denom
        div_acc = div_acc + 0.5 * (r_div + d_div)

        if f"{p}.nodes.edge_u" in params and f"{p}.nodes.edge_v" in params:
            eu = params[f"{p}.nodes.edge_u"]  # (N, r)
            ev = params[f"{p}.nodes.edge_v"]  # (N, r)
            eb = params[f"{p}.nodes.edge_bias"]  # (N, N)
            eph = params[f"{p}.nodes.edge_proj_h"]  # (D, r)
            epx = params[f"{p}.nodes.edge_proj_x"]  # (D, r)
            node_proto = jnp.einsum("dn,dr->nr", w_u, eph)[None, None, :, :]
            r_tok = jnp.einsum("btd,dr->btr", normed, epx)
            q_n = node_proto + eu[None, None, :, :]
            k_n = node_proto + ev[None, None, :, :]
            inv_sr = 1.0 / jnp.sqrt(float(max(1, eu.shape[-1])))
            bilin = jnp.einsum("btir,btjr->btij", q_n, k_n)
            s_c = jnp.einsum("ir,btr->bti", eu, r_tok)[..., :, None]
            d_c = jnp.einsum("jr,btr->btj", ev, r_tok)[..., None, :]
            e_logits = (bilin + s_c + d_c) * inv_sr + eb[None, None, :, :]
            diag_m = jnp.eye(n_exp, dtype=jnp.bool_)[None, None, :, :]
            e_logits = jnp.where(diag_m, -1e9, e_logits)
            a_ij = jax.nn.softmax(e_logits / float(max(0.05, config.graph_edge_temp)), axis=-1)
            a_ij = jnp.where(diag_m, 0.0, a_ij)
            a_ij = a_ij / jnp.maximum(jnp.sum(a_ij, axis=-1, keepdims=True), 1e-6)

            cond_ent = -jnp.sum(
                a_ij * jnp.log(jnp.maximum(a_ij, 1e-8)), axis=-1
            ) / jnp.log(float(max(2, n_exp - 1)))
            mean_cond_ent = (
                jnp.sum(jnp.mean(cond_ent, axis=-1, keepdims=True) * valid_w) / valid_denom
            )
            mean_in_j = jnp.sum(
                jnp.mean(a_ij, axis=-2) * valid_w, axis=(0, 1)
            ) / valid_denom  # (N,)
            hub_balance = float(n_exp) * jnp.sum(jnp.square(mean_in_j))
            edge_acc = edge_acc + (0.25 * mean_cond_ent + 0.75 * hub_balance)

    inv_l = 1.0 / float(max(1, n_layers_used))
    l_load = load_acc * inv_l
    l_edge = edge_acc * inv_l
    l_div = div_acc * inv_l
    total_reg = l_load + l_edge + l_div
    return {
        "total_reg": total_reg,
        "load_loss": l_load,
        "edge_loss": l_edge,
        "diversity_loss": l_div,
    }


from mlm.v2_graph_trace import (  # noqa: E402, F401
    _calibrate_fda_layer_edges_v2,
    _compute_layer_specialist_outputs_v2,
    get_spidernet_node_role_v2,
    inject_and_calibrate_spidernet_graph_edges_v2,
    inspect_directed_graph_edges_v2,
    trace_compositional_3hop_chain_v2,
    trace_reading_node_collaborator_switch_v2,
)

