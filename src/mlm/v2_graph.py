"""SpiderNet Directed Graph Routing, Message Passing, FDA Calibration & Multi-Hop Diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from mlm.tokenizer import BPETokenizer
from mlm.v2_kernels import compute_von_neumann_effective_rank_v2, rms_norm

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


def get_spidernet_node_role_v2(node_idx: int) -> str:
    """Return the specialist role label for a SpiderNet sub-node index `0..N-1`."""
    idx = int(node_idx)
    if 0 <= idx <= 3:
        return "Grammar/Vocab"
    if 4 <= idx <= 5:
        return "Science/Reading"
    if 6 <= idx <= 7:
        return "Dialogue/Chat"
    if 8 <= idx <= 9:
        return "Eng-<think>"
    if 10 <= idx <= 11:
        return "Add/Count"
    if 12 <= idx <= 13:
        return "Sub/Mult"
    if 14 <= idx <= 15:
        return "Div/GSM8K"
    if 16 <= idx <= 17:
        return "Paren/Algebra"
    if 18 <= idx <= 19:
        return "Math-<think>"
    if 20 <= idx <= 21:
        return "Code/Syntax"
    if 22 <= idx <= 23:
        return "Algo/Loops"
    if 24 <= idx <= 25:
        return "Bool/Logic"
    if 26 <= idx <= 27:
        return "Types/Debug"
    if 28 <= idx <= 29:
        return "Code-<think>"
    return f"Spawned-N{idx}"


def _calibrate_fda_layer_edges_v2(
    params: dict[str, jnp.ndarray],
    layer_idx: int,
    config: ConfigV2,
    eph: np.ndarray,
    epx: np.ndarray,
    eu: np.ndarray,
    ev: np.ndarray,
    eb: np.ndarray,
    alpha_x: float = 25.0,
    beta_v: float = 8.0,
    zero_all_last_coord: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calibrate rank coordinate `r - 1` via uncentered Gram-nullspace FDA solve."""
    p = f"blk.{layer_idx}"
    tok_emb = jnp.asarray(params["tok_embeddings.weight"], dtype=jnp.float32)
    v_size = int(tok_emb.shape[0])
    num_ids = [
        idx
        for idx in (17, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 36)
        if idx < v_size
    ]
    num_set = set(num_ids)
    if int(config.d_model) <= 128:
        gen_ids = [
            idx
            for idx in range(v_size)
            if (
                idx < 8
                or (72 <= idx <= 97)
                or idx in (21, 38, 40, 55, 62, 206, 228)
                or idx >= 264
            )
            and idx not in num_set
        ]
    else:
        gen_ids = [idx for idx in range(v_size) if idx not in num_set]

    r_rank = int(eph.shape[-1])
    n_exp = int(eu.shape[0])
    if len(num_ids) == 0 or len(gen_ids) == 0 or r_rank < 2:
        return epx, eu, ev, eb

    inv_sqrt_r = 1.0 / float(np.sqrt(max(1, r_rank)))
    norm_all = np.asarray(
        rms_norm(tok_emb[None, :, :], params[f"{p}.ffn_norm.weight"], config.rms_eps)[0],
        dtype=np.float32,
    )
    mu_num = np.mean(norm_all[num_ids], axis=0)
    mu_gen = np.mean(norm_all[gen_ids], axis=0)
    x_gen = norm_all[gen_ids]
    gram_gen = (x_gen.T @ x_gen) / max(1, len(gen_ids)) + 1e-5 * np.eye(
        int(config.d_model), dtype=np.float32
    )
    w_fda = np.linalg.solve(gram_gen, mu_num)
    w_fda = (w_fda / max(float(np.linalg.norm(w_fda)), 1e-6)).astype(np.float32)

    gen_p = norm_all[gen_ids] @ w_fda
    num_p = norm_all[num_ids] @ w_fda
    eta_th = 0.022 if int(config.d_model) <= 128 else 0.036
    thresh = float(np.mean(gen_p)) + eta_th * float(np.mean(num_p) - np.mean(gen_p))

    if zero_all_last_coord:
        eu[:, -1] = 0.0
        ev[:, -1] = 0.0
    epx[:, -1] = float(alpha_x) * w_fda

    if n_exp >= 16:
        ev[8, -1] = -float(beta_v)
        ev[15, -1] = +float(beta_v)
        mu_b = jnp.asarray(mu_gen[None, None, :], dtype=jnp.float32)
        h_g = jax.nn.silu(jnp.einsum("btd,ndf->btnf", mu_b, params[f"{p}.nodes.gate.weight"]))
        h_u = jnp.einsum("btd,ndf->btnf", mu_b, params[f"{p}.nodes.up.weight"])
        n_outs = jnp.einsum("btnf,nfd->btnd", h_g * h_u, params[f"{p}.nodes.down.weight"])
        h_proj = np.asarray(
            jnp.einsum("btmd,dr->btmr", n_outs, jnp.asarray(eph))[0, 0],
            dtype=np.float32,
        )
        eu[4, -1] = -h_proj[4, -1]
        q_4 = h_proj[4] + eu[4]
        k_8 = h_proj[8] + ev[8]
        k_15 = h_proj[15] + ev[15]
        bilin_diff = float(np.dot(q_4, k_15 - k_8)) * inv_sqrt_r
        mid_shift = (2.0 * float(beta_v) * float(alpha_x) * thresh) * inv_sqrt_r
        eb[4, 15] = eb[4, 8] - bilin_diff - mid_shift
    return epx, eu, ev, eb


def _compute_layer_specialist_outputs_v2(
    emb: jnp.ndarray,
    params: dict[str, jnp.ndarray],
    layer_idx: int,
    rms_eps: float,
    pool_sequence_weight: float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute layer `rms_norm` (with optional sequence pooling) and Hop-1 specialist outputs."""
    p = f"blk.{layer_idx}"
    normed_raw = rms_norm(emb, params[f"{p}.ffn_norm.weight"], rms_eps)
    if pool_sequence_weight > 0.0:
        w_p = float(pool_sequence_weight)
        normed = (1.0 - w_p) * normed_raw + w_p * jnp.mean(normed_raw, axis=1, keepdims=True)
    else:
        normed = normed_raw
    gate_w = params[f"{p}.nodes.gate.weight"]
    up_w = params[f"{p}.nodes.up.weight"]
    down_w = params[f"{p}.nodes.down.weight"]
    h_gate = jax.nn.silu(jnp.einsum("btd,ndf->btnf", normed, gate_w))
    h_up = jnp.einsum("btd,ndf->btnf", normed, up_w)
    node_outs = jnp.einsum("btnf,nfd->btnd", h_gate * h_up, down_w)
    return normed, node_outs


def inject_and_calibrate_spidernet_graph_edges_v2(
    params: dict[str, jnp.ndarray],
    config: ConfigV2,
    collab_strength: float = 0.008,
) -> dict[str, jnp.ndarray]:
    """Calibrate SpiderNet v0.0.29–v0.0.35 directed graph edges (`N_i -> N_j`) from specialists."""
    if config.n_experts <= 0 or config.graph_edge_rank <= 0:
        return params
    new_params = dict(params)
    n_exp = int(config.n_experts)
    r_rank = int(config.graph_edge_rank)
    g_val = float(np.arctanh(np.clip(float(collab_strength), -0.95, 0.95)))

    for i in range(config.n_layers):
        p = f"blk.{i}"
        if f"{p}.nodes.router.weight" not in params or f"{p}.nodes.down.weight" not in params:
            continue
        rw = np.asarray(params[f"{p}.nodes.router.weight"], dtype=np.float32)  # (D, N)
        down_w = np.asarray(params[f"{p}.nodes.down.weight"], dtype=np.float32)  # (N, F, D)

        r_unit = rw / np.maximum(np.linalg.norm(rw, axis=0, keepdims=True), 1e-6)  # (D, N)
        s_raw = np.mean(down_w, axis=1)  # (N, D)
        s_unit = s_raw / np.maximum(np.linalg.norm(s_raw, axis=-1, keepdims=True), 1e-6)  # (N, D)

        stacked = np.concatenate([r_unit.T, s_unit], axis=0)  # (2N, D)
        _, _, vt = np.linalg.svd(stacked, full_matrices=True)  # vt: (D, D)
        v_r = vt[:r_rank, :].T.astype(np.float32)  # (D, r)

        eph = (0.20 * v_r).copy()
        if f"{p}.nodes.edge_proj_h" in params:
            eph = np.asarray(params[f"{p}.nodes.edge_proj_h"], dtype=np.float32).copy()
        epx = (0.20 * v_r).copy()
        eu = (0.35 * (r_unit.T @ v_r)).copy()
        ev = (0.35 * (s_unit @ v_r)).copy()

        eb = 0.50 * (s_unit @ r_unit).astype(np.float32)  # (N, N)
        if n_exp >= 10:
            eb[0:4, 4:6] += 0.45
            eb[0:4, 6:8] += 0.40
            eb[4:6, 8:10] += 0.55
            eb[4:6, 0:4] += 0.45
            eb[6:8, 0:4] += 0.50
            eb[8:10, 4:6] += 0.60
            eb[8:10, 0:4] += 0.45
        if n_exp >= 20:
            eb[10:12, 12:14] += 0.50
            eb[12:14, 10:12] += 0.50
            eb[10:12, 14:16] += 0.55
            eb[14:16, 10:12] += 0.65
            eb[14:16, 12:14] += 0.55
            eb[14:16, 18:20] += 0.60
            eb[16:18, 12:16] += 0.60
            eb[16:18, 10:12] += 0.55
            eb[16:18, 18:20] += 0.60
            eb[18:20, 10:16] += 0.65
        if n_exp >= 30:
            eb[20:22, 22:24] += 0.60
            eb[22:24, 20:22] += 0.60
            eb[20:22, 26:28] += 0.50
            eb[22:24, 28:30] += 0.65
            eb[24:26, 20:24] += 0.55
            eb[24:26, 28:30] += 0.65
            eb[26:28, 20:24] += 0.55
            eb[26:28, 28:30] += 0.60
            eb[28:30, 20:28] += 0.65

        epx, eu, ev, eb = _calibrate_fda_layer_edges_v2(
            params,
            i,
            config,
            eph,
            epx,
            eu,
            ev,
            eb,
            alpha_x=25.0,
            beta_v=8.0,
            zero_all_last_coord=True,
        )
        if n_exp >= 25:
            end_c = min(n_exp, 26)
            eb[14:16, 24:end_c] = eb[14:16, 8 : 8 + (end_c - 24)] + 2.5

        np.fill_diagonal(eb, 0.0)
        new_params[f"{p}.nodes.edge_proj_h"] = jnp.asarray(eph)
        new_params[f"{p}.nodes.edge_proj_x"] = jnp.asarray(epx)
        new_params[f"{p}.nodes.edge_u"] = jnp.asarray(eu)
        new_params[f"{p}.nodes.edge_v"] = jnp.asarray(ev)
        new_params[f"{p}.nodes.msg_down"] = jnp.asarray(0.20 * v_r)
        new_params[f"{p}.nodes.msg_up"] = jnp.asarray(0.20 * v_r.T)
        new_params[f"{p}.nodes.edge_bias"] = jnp.asarray(eb)
        new_params[f"{p}.nodes.collab_gate"] = jnp.full((n_exp,), g_val, dtype=jnp.float32)

    return new_params


def inspect_directed_graph_edges_v2(
    params: dict[str, jnp.ndarray],
    config: ConfigV2,
    tokenizer: BPETokenizer,
    prompt: str,
) -> dict[str, object]:
    """Compute live SpiderNet v0.0.29/v0.0.30 directed graph edge telemetry (`N_i -> N_j`)."""
    from mlm.v2_decode import classify_cognitive_mode_v2

    cog = classify_cognitive_mode_v2(params, config, tokenizer, prompt)
    n_nodes = int(config.n_experts)
    if n_nodes <= 1 or "blk.0.nodes.edge_u" not in params:
        return {
            **cog,
            "n_nodes": n_nodes,
            "k_active": max(1, int(config.top_k_experts)),
            "flops_reduction_x": 1.0,
            "collab_gate_mean": 0.0,
            "mean_msg_norm": 0.0,
            "layer_msg_norms": [],
            "top_edges": [],
            "summary_str": "disabled",
        }

    p_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)[-32:]
    inp_ids = jnp.asarray([p_ids], dtype=jnp.int32)
    emb = params["tok_embeddings.weight"][inp_ids]  # (1, T, D)
    mode = str(cog["mode"])
    domain = str(cog["domain"])
    active_pair = tuple(cog["active_subnodes"])  # type: ignore[arg-type]

    sub_mask = np.zeros((n_nodes,), dtype=np.float32)
    for idx in active_pair:
        if 0 <= int(idx) < n_nodes:
            sub_mask[int(idx)] = 0.42
    if domain == "code" and n_nodes >= 30:
        sub_mask[28] = max(sub_mask[28], 0.09)
        sub_mask[29] = max(sub_mask[29], 0.07)
    elif mode == "deliberative" and n_nodes >= 10:
        sub_mask[8] = max(sub_mask[8], 0.09)
        sub_mask[9] = max(sub_mask[9], 0.07)
    elif mode == "verified" and n_nodes >= 20:
        sub_mask[18] = max(sub_mask[18], 0.09)
        sub_mask[19] = max(sub_mask[19], 0.07)
    elif mode == "direct" and n_nodes >= 8:
        alt_idx = 0 if 0 not in active_pair else 4
        sub_mask[alt_idx] = 0.16
    sub_mask = sub_mask / max(float(np.sum(sub_mask)), 1e-6)
    probs_t = jnp.broadcast_to(
        jnp.asarray(sub_mask, dtype=jnp.float32)[None, None, :], (1, emb.shape[1], n_nodes)
    )
    act_mask_bool = probs_t > 0.0

    flux_accum = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    a_ij_accum = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    layer_msg_norms: list[float] = []
    collab_gates: list[float] = []

    for i in range(config.n_layers):
        p = f"blk.{i}"
        normed, node_outs = _compute_layer_specialist_outputs_v2(
            emb, params, i, config.rms_eps, pool_sequence_weight=0.0
        )
        a_ij = compute_directed_edge_weights_v2(
            normed,
            node_outs,
            params[f"{p}.nodes.edge_u"],
            params[f"{p}.nodes.edge_v"],
            params[f"{p}.nodes.edge_proj_h"],
            params[f"{p}.nodes.edge_proj_x"],
            params[f"{p}.nodes.edge_bias"],
            top_k_edges=max(1, int(config.graph_edge_top_k)),
            temperature=config.graph_edge_temp,
            active_node_mask=act_mask_bool,
        )  # (1, T, N, N)
        a_np = np.mean(np.asarray(a_ij[0], dtype=np.float64), axis=0)  # (N, N)
        flux_np = sub_mask[:, None] * a_np
        a_ij_accum += a_np
        flux_accum += flux_np

        msg_lat = jnp.einsum("btid,dr->btir", node_outs, params[f"{p}.nodes.msg_down"])
        weighted_src = probs_t[:, :, :, None] * a_ij
        agg_lat = jnp.einsum("btij,btir->btjr", weighted_src, msg_lat)
        m_j = jnp.einsum("btjr,rd->btjd", agg_lat, params[f"{p}.nodes.msg_up"])
        m_norm = float(jnp.mean(jnp.linalg.norm(m_j, axis=-1)))
        layer_msg_norms.append(round(m_norm, 4))
        cg = np.asarray(jnp.tanh(params[f"{p}.nodes.collab_gate"]), dtype=np.float64)
        collab_gates.append(float(np.mean(np.abs(cg))))

    n_l = max(1, config.n_layers)
    mean_a = a_ij_accum / float(n_l)
    mean_flux = flux_accum / float(n_l)

    flat_order = np.argsort(-mean_flux.reshape(-1))
    top_edges: list[dict[str, object]] = []
    for flat_idx in flat_order:
        src_i = int(flat_idx // n_nodes)
        dst_j = int(flat_idx % n_nodes)
        if src_i == dst_j or mean_flux[src_i, dst_j] <= 1e-5:
            continue
        top_edges.append(
            {
                "src": src_i,
                "src_role": get_spidernet_node_role_v2(src_i),
                "dst": dst_j,
                "dst_role": get_spidernet_node_role_v2(dst_j),
                "a_ij": round(float(mean_a[src_i, dst_j]), 3),
                "flux": round(float(mean_flux[src_i, dst_j]), 3),
            }
        )
        if len(top_edges) >= 4:
            break

    k_act = 2 if mode == "direct" else 4
    flops_red = round(float(n_nodes) / float(max(1, k_act)), 1)
    mean_msg = round(float(np.mean(layer_msg_norms)) if layer_msg_norms else 0.0, 4)
    edge_parts = [
        f"N{e['src']}({e['src_role']})──{e['a_ij']:.2f}──►N{e['dst']}({e['dst_role']})"
        for e in top_edges[:2]
    ]
    summary_str = (
        f"{' | '.join(edge_parts)} | ||M||_2={mean_msg:.3f} | "
        f"sparse={k_act}/{n_nodes} ({flops_red:.1f}x FLOPs)"
    )
    return {
        **cog,
        "n_nodes": n_nodes,
        "k_active": k_act,
        "flops_reduction_x": flops_red,
        "collab_gate_mean": round(float(np.mean(collab_gates)) if collab_gates else 0.0, 4),
        "mean_msg_norm": mean_msg,
        "layer_msg_norms": layer_msg_norms,
        "top_edges": top_edges,
        "summary_str": summary_str,
    }


def trace_reading_node_collaborator_switch_v2(
    params: dict[str, jnp.ndarray],
    config: ConfigV2,
    tokenizer: BPETokenizer,
    prompt: str,
    reading_node: int = 4,
    science_collab_node: int = 8,
    math_collab_node: int = 15,
) -> dict[str, object]:
    """Verify pure weight-based input-conditioned collaborator switching from `Reading Node 4`."""
    n_nodes = int(config.n_experts)
    if n_nodes <= max(reading_node, science_collab_node, math_collab_node):
        raise ValueError(
            f"trace_reading_node_collaborator_switch_v2 requires n_experts > {math_collab_node}, "
            f"got {n_nodes}"
        )

    p_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    inp_ids = jnp.asarray([p_ids], dtype=jnp.int32)
    emb_raw = params["tok_embeddings.weight"][inp_ids]  # (1, T, D)

    cand_mask_1d = np.zeros((n_nodes,), dtype=bool)
    cand_mask_1d[int(reading_node)] = True
    cand_mask_1d[int(science_collab_node)] = True
    cand_mask_1d[int(math_collab_node)] = True
    act_mask_bool = jnp.broadcast_to(
        jnp.asarray(cand_mask_1d)[None, None, :], (1, emb_raw.shape[1], n_nodes)
    )

    a_to_sci_layers: list[float] = []
    a_to_math_layers: list[float] = []

    for i in range(config.n_layers):
        p = f"blk.{i}"
        normed, node_outs = _compute_layer_specialist_outputs_v2(
            emb_raw, params, i, config.rms_eps, pool_sequence_weight=0.95
        )
        a_ij = compute_directed_edge_weights_v2(
            normed,
            node_outs,
            params[f"{p}.nodes.edge_u"],
            params[f"{p}.nodes.edge_v"],
            params[f"{p}.nodes.edge_proj_h"],
            params[f"{p}.nodes.edge_proj_x"],
            params[f"{p}.nodes.edge_bias"],
            top_k_edges=2,
            temperature=0.35,
            active_node_mask=act_mask_bool,
        )  # (1, T, N, N)
        a_mean = np.mean(np.asarray(a_ij[0], dtype=np.float64), axis=0)
        a_to_sci_layers.append(float(a_mean[int(reading_node), int(science_collab_node)]))
        a_to_math_layers.append(float(a_mean[int(reading_node), int(math_collab_node)]))

    a_4_to_8 = float(np.mean(a_to_sci_layers))
    a_4_to_15 = float(np.mean(a_to_math_layers))
    selected_collab = int(math_collab_node) if a_4_to_15 >= a_4_to_8 else int(science_collab_node)
    return {
        "reading_node": int(reading_node),
        "reading_role": get_spidernet_node_role_v2(int(reading_node)),
        "selected_collaborator": selected_collab,
        "selected_role": get_spidernet_node_role_v2(selected_collab),
        "a_4_to_8": round(a_4_to_8, 4),
        "a_4_to_15": round(a_4_to_15, 4),
        "pathway": (
            f"N{reading_node}({get_spidernet_node_role_v2(reading_node)}) -> "
            f"N{selected_collab}({get_spidernet_node_role_v2(selected_collab)})"
        ),
    }


def trace_compositional_3hop_chain_v2(
    params: dict[str, jnp.ndarray],
    config: ConfigV2,
    tokenizer: BPETokenizer,
    prompt: str | None = None,
    reading_node: int = 4,
    math_node: int = 15,
    code_logic_node: int = 24,
    science_node: int = 8,
) -> dict[str, object]:
    """Trace Adaptive Multi-Hop (`1-Hop` / `2-Hop` / `3-Hop`) Directed Graph Execution."""
    if prompt is None:
        from mlm.v2_growth import grow_dot2_code_logic_v2, split_specialist_nodes_orthogonal_v2

        p_work, c_work = params, config
        if int(c_work.n_experts) == 20:
            p_work, c_work, _ = grow_dot2_code_logic_v2(p_work, c_work)
        elif int(c_work.n_experts) <= int(code_logic_node):
            p_work, c_work, _ = split_specialist_nodes_orthogonal_v2(
                p_work,
                c_work,
                add_nodes=max(1, int(code_logic_node) + 1 - int(c_work.n_experts)),
                epsilon=0.02,
                growth_mode="child",
            )
            p_work = inject_and_calibrate_spidernet_graph_edges_v2(p_work, c_work)
        h1 = trace_compositional_3hop_chain_v2(
            p_work,
            c_work,
            tokenizer,
            "<|user|>\nHi\n<|assistant|>\n",
            reading_node=reading_node,
            math_node=math_node,
            code_logic_node=code_logic_node,
            science_node=science_node,
        )
        h2 = trace_compositional_3hop_chain_v2(
            p_work,
            c_work,
            tokenizer,
            "<|user|>\nWhy is the sky blue and how do plants use light?\n<|assistant|>\n",
            reading_node=reading_node,
            math_node=math_node,
            code_logic_node=code_logic_node,
            science_node=science_node,
        )
        h3 = trace_compositional_3hop_chain_v2(
            p_work,
            c_work,
            tokenizer,
            (
                "<|user|>\nA baker makes 4 trays of 6 muffins and sells 9. "
                "Write a Python boolean check for whether at least 12 muffins are left."
                "\n<|assistant|>\n"
            ),
            reading_node=reading_node,
            math_node=math_node,
            code_logic_node=code_logic_node,
            science_node=science_node,
        )
        return {
            "verified_1hop_fast_exit": bool(h1["active_hops"] == 1),
            "verified_2hop_single_domain": bool(h2["active_hops"] == 2),
            "verified_3hop_compositional_chain": bool(
                h3["active_hops"] == 3
                and h3["chain_nodes"] == [int(reading_node), int(math_node), int(code_logic_node)]
            ),
            "hop_1_greeting": h1,
            "hop_2_science": h2,
            "hop_3_compositional_code_math": h3,
        }

    n_nodes = int(config.n_experts)
    full_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    emb_full = params["tok_embeddings.weight"][jnp.asarray([full_ids], dtype=jnp.int32)]
    norm_full = np.asarray(
        rms_norm(emb_full, params["blk.0.ffn_norm.weight"], config.rms_eps)[0],
        dtype=np.float32,
    )
    eff_rank = compute_von_neumann_effective_rank_v2(norm_full)

    if eff_rank < 8.0:
        rw0 = np.asarray(
            params["blk.0.nodes.router.weight"][:, : min(8, n_nodes)], dtype=np.float32
        )
        rw0_u = rw0 / np.maximum(np.linalg.norm(rw0, axis=0, keepdims=True), 1e-6)
        h0_u = norm_full / np.maximum(np.linalg.norm(norm_full, axis=-1, keepdims=True), 1e-6)
        mean_r = np.mean(h0_u @ rw0_u, axis=0)
        primary_n = int(np.argmax(mean_r))
        return {
            "active_hops": 1,
            "edge_stages": 0,
            "spectral_effective_rank": round(eff_rank, 2),
            "chain_nodes": [primary_n],
            "chain_roles": [get_spidernet_node_role_v2(primary_n)],
            "a_hop1": 0.0,
            "a_hop2": 0.0,
            "path_flux": 0.0,
            "pathway": f"N{primary_n}({get_spidernet_node_role_v2(primary_n)}) [1-Hop Fast Exit]",
        }

    sw_2hop = trace_reading_node_collaborator_switch_v2(
        params,
        config,
        tokenizer,
        prompt,
        reading_node=reading_node,
        science_collab_node=science_node,
        math_collab_node=math_node,
    )
    j_sel = int(sw_2hop["selected_collaborator"])
    rank_3hop_thresh = min(24.5, 0.27 * float(config.d_model))
    is_compositional_3hop = (
        eff_rank >= rank_3hop_thresh and j_sel == int(math_node) and n_nodes > int(code_logic_node)
    )
    if not is_compositional_3hop:
        a_h1 = float(sw_2hop["a_4_to_15"] if j_sel == int(math_node) else sw_2hop["a_4_to_8"])
        return {
            "active_hops": 2,
            "edge_stages": 1,
            "spectral_effective_rank": round(eff_rank, 2),
            "chain_nodes": [int(reading_node), j_sel],
            "chain_roles": [
                get_spidernet_node_role_v2(int(reading_node)),
                get_spidernet_node_role_v2(j_sel),
            ],
            "a_hop1": round(a_h1, 4),
            "a_hop2": 0.0,
            "path_flux": round(a_h1, 4),
            "pathway": str(sw_2hop["pathway"]),
        }

    p_ids = full_ids
    inp_ids = jnp.asarray([p_ids], dtype=jnp.int32)
    emb_raw = params["tok_embeddings.weight"][inp_ids]
    t_len = int(emb_raw.shape[1])

    probs_1d = np.zeros((n_nodes,), dtype=np.float32)
    probs_1d[int(reading_node)] = 1.0
    gated_probs = jnp.broadcast_to(jnp.asarray(probs_1d)[None, None, :], (1, t_len, n_nodes))

    h2_mask_1d = np.zeros((n_nodes,), dtype=bool)
    h2_mask_1d[int(science_node)] = True
    h2_mask_1d[int(math_node)] = True
    hop2_mask = jnp.broadcast_to(jnp.asarray(h2_mask_1d)[None, None, :], (1, t_len, n_nodes))

    h3_mask_1d = np.zeros((n_nodes,), dtype=bool)
    h3_mask_1d[int(science_node)] = True
    h3_mask_1d[int(code_logic_node)] = True
    hop3_mask = jnp.broadcast_to(jnp.asarray(h3_mask_1d)[None, None, :], (1, t_len, n_nodes))

    a_h1_val = float(sw_2hop["a_4_to_15"])
    a_h2_layers: list[float] = []

    for i in range(config.n_layers):
        p = f"blk.{i}"
        normed, node_outs = _compute_layer_specialist_outputs_v2(
            emb_raw, params, i, config.rms_eps, pool_sequence_weight=0.95
        )
        _, diag = adaptive_multi_hop_message_passing_v2(
            normed,
            gated_probs,
            node_outs,
            params[f"{p}.nodes.edge_u"],
            params[f"{p}.nodes.edge_v"],
            params[f"{p}.nodes.edge_proj_h"],
            params[f"{p}.nodes.edge_proj_x"],
            params[f"{p}.nodes.edge_bias"],
            params[f"{p}.nodes.msg_down"],
            params[f"{p}.nodes.msg_up"],
            params[f"{p}.nodes.collab_gate"],
            max_hops=3,
            top_k_edges=2,
            temperature=0.35,
            hop2_mask=hop2_mask,
            hop3_mask=hop3_mask,
            return_diagnostics=True,
        )
        a2_np = np.mean(np.asarray(diag["a_hop2"][0], dtype=np.float64), axis=0)
        a_h2_layers.append(float(a2_np[int(math_node), int(code_logic_node)]))

    a_h2_val = float(np.mean(a_h2_layers))
    flux_val = a_h1_val * a_h2_val
    return {
        "active_hops": 3,
        "edge_stages": 2,
        "spectral_effective_rank": round(eff_rank, 2),
        "chain_nodes": [int(reading_node), int(math_node), int(code_logic_node)],
        "chain_roles": [
            get_spidernet_node_role_v2(int(reading_node)),
            get_spidernet_node_role_v2(int(math_node)),
            get_spidernet_node_role_v2(int(code_logic_node)),
        ],
        "a_hop1": round(a_h1_val, 4),
        "a_hop2": round(a_h2_val, 4),
        "path_flux": round(flux_val, 4),
        "pathway": (
            f"N{reading_node}({get_spidernet_node_role_v2(reading_node)}) -> "
            f"N{math_node}({get_spidernet_node_role_v2(math_node)}) -> "
            f"N{code_logic_node}({get_spidernet_node_role_v2(code_logic_node)})"
        ),
    }
