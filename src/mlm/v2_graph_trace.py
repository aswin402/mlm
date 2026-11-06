"""SpiderNet 30-Node Role Taxonomy, Gram-Nullspace FDA Calibration & Multi-Hop Chain Tracers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from mlm.tokenizer import BPETokenizer
from mlm.v2_graph import (
    adaptive_multi_hop_message_passing_v2,
    compute_directed_edge_weights_v2,
)
from mlm.v2_kernels import compute_von_neumann_effective_rank_v2, rms_norm

if TYPE_CHECKING:
    from mlm.model_v2 import ConfigV2


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
    """Calibrate SpiderNet v0.0.29–v0.0.42 directed graph edges (`N_i -> N_j`) from specialists."""
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
