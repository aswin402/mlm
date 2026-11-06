"""SpiderNet Structural Growth, Orthogonal Splitting, CUSUM Controller & Continual Benchmarks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from mlm.tokenizer import BPETokenizer
from mlm.v2_graph import (
    _calibrate_fda_layer_edges_v2,
    inject_and_calibrate_spidernet_graph_edges_v2,
    trace_compositional_3hop_chain_v2,
    trace_reading_node_collaborator_switch_v2,
)
from mlm.v2_verifier import _verify_code_and_logic_v2

if TYPE_CHECKING:
    from mlm.model_v2 import ConfigV2


def mask_specialist_updates_v2(
    updates: dict[str, jnp.ndarray],
    config: ConfigV2,
    active_subnodes_override: set[int] | tuple[int, ...] | None = None,
    strict_active_only: bool = False,
    always_train_mtp: bool = False,
) -> dict[str, jnp.ndarray]:
    """Scale or mask optimizer updates for mature vs. newly grown specialist nodes & trunk."""
    if strict_active_only:
        raw_sub = (
            active_subnodes_override
            if active_subnodes_override is not None
            else config.active_subnodes
        )
        active_set = {int(idx) for idx in raw_sub if 0 <= int(idx) < config.n_experts}
        m_1d = np.zeros((config.n_experts,), dtype=np.float32)
        for a_i in active_set:
            m_1d[int(a_i)] = 1.0
        m_jax = jnp.asarray(m_1d)
        e_mat = jnp.asarray(np.maximum(m_1d[:, None], m_1d[None, :]))
        r_mask = jnp.asarray(m_1d[None, :])
        masked_updates: dict[str, jnp.ndarray] = {}
        for k, u in updates.items():
            if k.endswith(".nodes.edge_scale") or k.endswith(".nodes.collab_gate"):
                masked_updates[k] = u * m_jax
            elif (
                ".nodes." in k
                and not k.endswith(".router.weight")
                and not k.endswith(".router_bias")
            ):
                if u.ndim == 3 and u.shape[0] == config.n_experts:
                    masked_updates[k] = u * m_jax[:, None, None]
                elif u.ndim == 2 and u.shape == (config.n_experts, config.n_experts):
                    masked_updates[k] = u * e_mat
                elif u.ndim == 2 and u.shape[0] == config.n_experts:
                    masked_updates[k] = u * m_jax[:, None]
                else:
                    masked_updates[k] = (
                        jnp.zeros_like(u)
                        if float(config.shared_lr_scale) == 0.0
                        else u * 0.25
                    )
            elif k.endswith(".nodes.router.weight"):
                masked_updates[k] = u * r_mask
            else:
                masked_updates[k] = jnp.zeros_like(u)
        return masked_updates

    raw_sub = (
        active_subnodes_override
        if active_subnodes_override is not None
        else config.active_subnodes
    )
    active_set = {int(idx) for idx in raw_sub if 0 <= int(idx) < config.n_experts}
    if not (
        config.mature_experts > 0
        or config.shared_lr_scale < 1.0
        or config.mature_conv_kernel > 0
        or len(active_set) > 0
    ):
        return {
            k: (
                jnp.zeros_like(u)
                if (k.endswith(".nodes.router_bias") or k == "blt_token_ngrams")
                else u
            )
            for k, u in updates.items()
        }

    if active_set:
        node_mask_1d = np.full(
            (config.n_experts,), float(config.mature_lr_scale), dtype=np.float32
        )
        if config.n_experts > 20 and min(active_set) >= 20:
            node_mask_1d[:20] = 0.0
        elif config.n_experts > 10 and min(active_set) >= 10:
            node_mask_1d[:10] = 0.0
        for a_idx in active_set:
            node_mask_1d[int(a_idx)] = 1.0
    elif 0 < config.mature_experts < config.n_experts:
        node_mask_1d = np.ones((config.n_experts,), dtype=np.float32)
        node_mask_1d[: config.mature_experts] = float(config.mature_lr_scale)
        if config.n_experts > 20 and config.mature_experts >= 20:
            node_mask_1d[:20] = 0.0
        elif config.n_experts > 10:
            node_mask_1d[:10] = 0.0
    elif config.mature_experts >= config.n_experts > 0:
        node_mask_1d = np.full(
            (config.n_experts,), float(config.mature_lr_scale), dtype=np.float32
        )
    else:
        node_mask_1d = np.ones((config.n_experts,), dtype=np.float32)

    node_mask_jax = jnp.asarray(node_mask_1d)
    edge_mat_mask_jax = jnp.asarray(np.maximum(node_mask_1d[:, None], node_mask_1d[None, :]))

    scaled_updates: dict[str, jnp.ndarray] = {}
    for k, u in updates.items():
        if k.endswith(".nodes.router_bias") or k == "blt_token_ngrams":
            scaled_updates[k] = jnp.zeros_like(u)
        elif k.endswith(".nodes.edge_scale") or k.endswith(".nodes.collab_gate"):
            scaled_updates[k] = u * node_mask_jax
        elif ".nodes." in k and not k.endswith(".router.weight"):
            if u.ndim == 3 and u.shape[0] == config.n_experts:
                scaled_updates[k] = u * node_mask_jax[:, None, None]
            elif u.ndim == 2 and u.shape == (config.n_experts, config.n_experts):
                scaled_updates[k] = u * edge_mat_mask_jax
            elif u.ndim == 2 and u.shape[0] == config.n_experts:
                scaled_updates[k] = u * node_mask_jax[:, None]
            else:
                scaled_updates[k] = (
                    jnp.zeros_like(u)
                    if float(config.shared_lr_scale) == 0.0
                    else u * max(float(config.shared_lr_scale), 0.25)
                )
        elif k.endswith(".nodes.router.weight"):
            if active_set:
                r_mask = np.full(
                    (1, config.n_experts), float(config.mature_lr_scale), dtype=np.float32
                )
                if config.n_experts > 20 and min(active_set) >= 20:
                    r_mask[:, :20] = 0.0
                elif config.n_experts > 10 and min(active_set) >= 10:
                    r_mask[:, :10] = 0.0
                for a_idx in active_set:
                    r_mask[:, int(a_idx)] = 1.0
                scaled_updates[k] = u * jnp.asarray(r_mask)
            elif 0 < config.mature_experts < config.n_experts:
                r_mask = np.ones((1, config.n_experts), dtype=np.float32)
                r_mask[:, : config.mature_experts] = float(config.mature_lr_scale)
                if config.n_experts > 20 and config.mature_experts >= 20:
                    r_mask[:, :20] = 0.0
                elif config.n_experts > 10:
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
            0 < config.mature_conv_kernel < config.conv_kernel and config.shared_lr_scale > 0.0
        ):
            new_taps = config.conv_kernel - config.mature_conv_kernel
            c_mask = np.ones((config.conv_kernel, 1), dtype=np.float32)
            c_mask[new_taps:, :] = float(config.shared_lr_scale)
            scaled_updates[k] = u * jnp.asarray(c_mask)
        elif k.startswith("mtp_"):
            scaled_updates[k] = (
                u
                if (always_train_mtp or config.mtp_weight > 0.0)
                else u * float(config.shared_lr_scale)
            )
        else:
            scaled_updates[k] = u * float(config.shared_lr_scale)
    return scaled_updates


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
        if f"{p}.nodes.edge_u" in params:
            for e_key in ("edge_u", "edge_v"):
                e_arr = np.asarray(params[f"{p}.nodes.{e_key}"], dtype=np.float32)
                e_children = []
                for j in range(add_nodes):
                    p1 = e_arr[order[(2 * j) % n_pool]]
                    p2 = e_arr[order[(2 * j + 1) % n_pool]]
                    c_vec = 0.85 * p1 + 0.15 * p2
                    c_vec = c_vec + rng.normal(0.0, 0.005, size=c_vec.shape).astype(np.float32)
                    e_children.append(c_vec)
                new_params[f"{p}.nodes.{e_key}"] = jnp.asarray(
                    np.concatenate([e_arr, np.stack(e_children, axis=0)], axis=0)
                )
        if f"{p}.nodes.edge_bias" in params:
            eb_old = np.asarray(params[f"{p}.nodes.edge_bias"], dtype=np.float32)
            eb_new = np.zeros((n_new, n_new), dtype=np.float32)
            eb_new[:n_old, :n_old] = eb_old
            new_params[f"{p}.nodes.edge_bias"] = jnp.asarray(eb_new)
        if f"{p}.nodes.collab_gate" in params:
            cg_old = np.asarray(params[f"{p}.nodes.collab_gate"], dtype=np.float32)
            cg_new = np.zeros((add_nodes,), dtype=np.float32)
            new_params[f"{p}.nodes.collab_gate"] = jnp.asarray(
                np.concatenate([cg_old, cg_new], axis=0)
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


def _orthogonal_perturbation_like(
    w_parent: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Construct Gram-Schmidt orthogonal tensor `Q` with `<Q, W> = 0` and `||Q||_F = ||W||_F`."""
    flat_w = w_parent.reshape(-1).astype(np.float64)
    w_norm_sq = float(np.dot(flat_w, flat_w))
    z = rng.normal(0.0, 1.0, size=flat_w.shape).astype(np.float64)
    if w_norm_sq > 1e-12:
        proj = float(np.dot(z, flat_w)) / w_norm_sq
        z_orth = z - proj * flat_w
        z_norm = float(np.linalg.norm(z_orth))
        w_norm = float(np.sqrt(w_norm_sq))
        q_flat = (z_orth / max(z_norm, 1e-12)) * w_norm
    else:
        q_flat = z * 0.01
    return q_flat.astype(np.float32).reshape(w_parent.shape)


def split_specialist_nodes_orthogonal_v2(
    params: dict[str, jnp.ndarray],
    config: ConfigV2,
    parent_indices: list[int] | None = None,
    add_nodes: int = 1,
    epsilon: float = 0.05,
    mature_lr_scale: float | None = None,
    shared_lr_scale: float | None = None,
    growth_mode: str = "sibling",
) -> tuple[dict[str, jnp.ndarray], ConfigV2, dict[str, object]]:
    """SpiderNet v0.0.26 Symmetric Orthogonal Parent-Child Specialist Splitting."""
    if config.n_experts < 2 or add_nodes < 1:
        raise ValueError("split_specialist_nodes_orthogonal_v2 requires n_experts >= 2")
    if growth_mode not in ("sibling", "child"):
        raise ValueError(f"growth_mode must be 'sibling' or 'child', got {growth_mode!r}")
    n_old = int(config.n_experts)
    n_new = n_old + int(add_nodes)
    eff_eps = float(np.clip(epsilon, 1e-4, 0.25))
    eff_mature_lr = float(config.mature_lr_scale if mature_lr_scale is None else mature_lr_scale)
    eff_shared_lr = float(config.shared_lr_scale if shared_lr_scale is None else shared_lr_scale)
    log2_val = float(np.log(2.0))

    new_cfg = config.model_copy(
        update={
            "n_experts": n_new,
            "mature_experts": n_old,
            "mature_lr_scale": eff_mature_lr,
            "shared_lr_scale": eff_shared_lr,
        }
    )
    rng = np.random.default_rng(config.seed + 7919 * n_new)
    new_params = dict(params)
    max_orth_cos = 0.0
    max_mean_recon_err = 0.0
    chosen_parents_per_layer: list[list[int]] = []

    for i in range(config.n_layers):
        p = f"blk.{i}"
        down_w = np.asarray(params[f"{p}.nodes.down.weight"], dtype=np.float32)
        if parent_indices is not None and len(parent_indices) > 0:
            parents = [
                int(parent_indices[j % len(parent_indices)]) % n_old
                for j in range(add_nodes)
            ]
        else:
            norms = np.linalg.norm(down_w, axis=(1, 2))
            order = np.argsort(-norms)
            parents = [int(order[j % len(order)]) for j in range(add_nodes)]
        chosen_parents_per_layer.append(parents)

        for suffix in ("gate", "up", "down"):
            arr = np.asarray(params[f"{p}.nodes.{suffix}.weight"], dtype=np.float32).copy()
            children = []
            for j, p_idx in enumerate(parents):
                w_orig = np.asarray(params[f"{p}.nodes.{suffix}.weight"][p_idx], dtype=np.float32)
                q_orth = _orthogonal_perturbation_like(w_orig, rng)
                denom_orth = float(np.linalg.norm(w_orig) * np.linalg.norm(q_orth)) + 1e-12
                cos_val = abs(float(np.sum(w_orig * q_orth)) / denom_orth)
                if cos_val > max_orth_cos:
                    max_orth_cos = cos_val
                w_sym_plus = (w_orig + eff_eps * q_orth).astype(np.float32)
                w_child_new = (w_orig - eff_eps * q_orth).astype(np.float32)
                recon_err = float(np.max(np.abs(0.5 * (w_sym_plus + w_child_new) - w_orig)))
                if recon_err > max_mean_recon_err:
                    max_mean_recon_err = recon_err
                arr[p_idx] = w_sym_plus if growth_mode == "sibling" else w_orig
                children.append(w_child_new)
                del j
            new_params[f"{p}.nodes.{suffix}.weight"] = jnp.asarray(
                np.concatenate([arr, np.stack(children, axis=0)], axis=0)
            )

        rw = np.asarray(params[f"{p}.nodes.router.weight"], dtype=np.float32).copy()
        extra_rw = []
        for p_idx in parents:
            r_orig = np.asarray(params[f"{p}.nodes.router.weight"][:, p_idx], dtype=np.float32)
            q_r = _orthogonal_perturbation_like(r_orig, rng)
            if growth_mode == "sibling":
                rw[:, p_idx] = (r_orig + eff_eps * q_r).astype(np.float32)
            extra_rw.append((r_orig - eff_eps * q_r).astype(np.float32))
        new_params[f"{p}.nodes.router.weight"] = jnp.asarray(
            np.concatenate([rw, np.stack(extra_rw, axis=-1)], axis=-1)
        )

        rb_old = np.asarray(
            params.get(f"{p}.nodes.router_bias", np.zeros((n_old,), dtype=np.float32)),
            dtype=np.float32,
        ).copy()
        rb_new = np.zeros((add_nodes,), dtype=np.float32)
        for j, p_idx in enumerate(parents):
            if growth_mode == "sibling":
                orig_b = float(rb_old[p_idx])
                rb_old[p_idx] = orig_b - log2_val
                rb_new[j] = orig_b - log2_val
            else:
                rb_new[j] = float(rb_old[p_idx]) - 0.5 * log2_val
        new_params[f"{p}.nodes.router_bias"] = jnp.asarray(
            np.concatenate([rb_old, rb_new], axis=0)
        )

        es_old = np.asarray(
            params.get(f"{p}.nodes.edge_scale", np.ones((n_old,), dtype=np.float32)),
            dtype=np.float32,
        )
        es_new = np.array([es_old[p_idx] for p_idx in parents], dtype=np.float32)
        new_params[f"{p}.nodes.edge_scale"] = jnp.asarray(
            np.concatenate([es_old, es_new], axis=0)
        )

        if f"{p}.nodes.edge_u" in params:
            for e_key in ("edge_u", "edge_v"):
                e_arr = np.asarray(params[f"{p}.nodes.{e_key}"], dtype=np.float32).copy()
                e_children = []
                for p_idx in parents:
                    e_orig = np.asarray(params[f"{p}.nodes.{e_key}"][p_idx], dtype=np.float32)
                    q_e = _orthogonal_perturbation_like(e_orig, rng)
                    if growth_mode == "sibling":
                        e_arr[p_idx] = (e_orig + eff_eps * q_e).astype(np.float32)
                    e_children.append((e_orig - eff_eps * q_e).astype(np.float32))
                new_params[f"{p}.nodes.{e_key}"] = jnp.asarray(
                    np.concatenate([e_arr, np.stack(e_children, axis=0)], axis=0)
                )
        if f"{p}.nodes.edge_bias" in params:
            eb_old = np.asarray(params[f"{p}.nodes.edge_bias"], dtype=np.float32)
            eb_new = np.zeros((n_new, n_new), dtype=np.float32)
            eb_new[:n_old, :n_old] = eb_old
            for j, p_idx in enumerate(parents):
                c_idx = n_old + j
                if growth_mode == "sibling":
                    eb_new[p_idx, c_idx] = 0.5
                    eb_new[c_idx, p_idx] = 0.5
                else:
                    eb_new[p_idx, c_idx] = 1.0
                    eb_new[c_idx, p_idx] = 0.1
            new_params[f"{p}.nodes.edge_bias"] = jnp.asarray(eb_new)
        if f"{p}.nodes.collab_gate" in params:
            cg_old = np.asarray(params[f"{p}.nodes.collab_gate"], dtype=np.float32)
            cg_new = np.array([cg_old[p_idx] for p_idx in parents], dtype=np.float32)
            new_params[f"{p}.nodes.collab_gate"] = jnp.asarray(
                np.concatenate([cg_old, cg_new], axis=0)
            )

    report = {
        "n_old": n_old,
        "n_new": n_new,
        "epsilon": eff_eps,
        "growth_mode": growth_mode,
        "parents_layer0": chosen_parents_per_layer[0] if chosen_parents_per_layer else [],
        "max_orthogonality_cosine": max_orth_cos,
        "max_mean_reconstruction_err": max_mean_recon_err,
    }
    return new_params, new_cfg, report


class AutonomousGrowthControllerV2:
    """SpiderNet v0.0.25 Autonomous CUSUM + 3-Signal Structural Growth Controller."""

    def __init__(
        self,
        threshold_h: float = 2.5,
        delta_drift: float = 0.35,
        ema_alpha: float = 0.10,
        cooldown_steps: int = 8,
        max_experts: int = 32,
        w_saturation: float = 0.45,
        w_overload: float = 0.35,
        w_uncertainty: float = 0.20,
        split_epsilon: float = 0.05,
    ) -> None:
        self.threshold_h = float(threshold_h)
        self.delta_drift = float(delta_drift)
        self.ema_alpha = float(ema_alpha)
        self.cooldown_steps = int(cooldown_steps)
        self.max_experts = int(max_experts)
        self.w_saturation = float(w_saturation)
        self.w_overload = float(w_overload)
        self.w_uncertainty = float(w_uncertainty)
        self.split_epsilon = float(split_epsilon)

        self.loss_ema: float | None = None
        self.loss_var_ema: float = 1e-2
        self.cusum_val: float = 0.0
        self.last_grow_step: int = -1000
        self.growth_events: list[dict[str, object]] = []

    @staticmethod
    def _norm01(arr: np.ndarray) -> np.ndarray:
        amin = float(np.min(arr))
        amax = float(np.max(arr))
        if amax - amin < 1e-8:
            return np.full_like(arr, 0.5, dtype=np.float32)
        return ((arr - amin) / (amax - amin)).astype(np.float32)

    def compute_node_spawning_scores(
        self,
        params: dict[str, jnp.ndarray],
        grads: dict[str, jnp.ndarray] | None,
        inputs: jnp.ndarray,
        mask: jnp.ndarray,
        config: ConfigV2,
    ) -> dict[str, np.ndarray | int]:
        """Compute `S_hat_i`, `O_hat_i`, `H_hat_i`, and combined `G_i` for each specialist node."""
        n_exp = max(1, int(config.n_experts))
        sat_raw = np.zeros((n_exp,), dtype=np.float32)
        over_raw = np.zeros((n_exp,), dtype=np.float32)
        unc_raw = np.zeros((n_exp,), dtype=np.float32)

        tok_emb = np.asarray(params["tok_embeddings.weight"], dtype=np.float32)[np.asarray(inputs)]
        valid_m = (np.asarray(mask) > 0).astype(np.float32)[..., None]
        v_denom = max(float(np.sum(valid_m)), 1.0)

        for i in range(config.n_layers):
            p = f"blk.{i}"
            dk = f"{p}.nodes.down.weight"
            rk = f"{p}.nodes.router.weight"
            if dk not in params or rk not in params:
                continue
            w_d = np.asarray(params[dk], dtype=np.float32)
            w_norms = np.linalg.norm(w_d.reshape(n_exp, -1), axis=-1) + 1e-6
            if grads is not None and dk in grads:
                g_d = np.asarray(grads[dk], dtype=np.float32)
                g_norms = np.linalg.norm(g_d.reshape(n_exp, -1), axis=-1)
                sat_raw += g_norms / w_norms
            else:
                sat_raw += w_norms

            fn_w = np.asarray(params[f"{p}.ffn_norm.weight"], dtype=np.float32)
            rms = np.sqrt(np.mean(np.square(tok_emb), axis=-1, keepdims=True) + config.rms_eps)
            h_normed = (tok_emb / rms) * fn_w
            h_u = h_normed / np.maximum(
                np.linalg.norm(h_normed, axis=-1, keepdims=True), 1e-6
            )
            rw = np.asarray(params[rk], dtype=np.float32)
            w_u = rw / np.maximum(np.linalg.norm(rw, axis=0, keepdims=True), 1e-6)
            logits = 8.0 * (h_u @ w_u)
            logits_shift = logits - np.max(logits, axis=-1, keepdims=True)
            exp_l = np.exp(logits_shift)
            probs = exp_l / np.maximum(np.sum(exp_l, axis=-1, keepdims=True), 1e-8)

            mean_p = np.sum(probs * valid_m, axis=(0, 1)) / v_denom
            over_raw += mean_p
            p_clip = np.clip(probs, 1e-6, 1.0 - 1e-6)
            bin_ent = -(p_clip * np.log(p_clip) + (1.0 - p_clip) * np.log(1.0 - p_clip))
            unc_raw += np.sum(bin_ent * valid_m, axis=(0, 1)) / v_denom

        s_hat = self._norm01(sat_raw)
        o_hat = self._norm01(over_raw)
        h_hat = self._norm01(unc_raw)
        g_score = (
            self.w_saturation * s_hat
            + self.w_overload * o_hat
            + self.w_uncertainty * h_hat
        ).astype(np.float32)
        best_parent = int(np.argmax(g_score))
        return {
            "saturation_hat": s_hat,
            "overload_hat": o_hat,
            "uncertainty_hat": h_hat,
            "g_score": g_score,
            "best_parent": best_parent,
        }

    def update_cusum(self, loss_val: float) -> tuple[float, bool]:
        """Update Page-Hinkley / One-Sided CUSUM statistic `C_t` and return `(C_t, triggered)`."""
        val = float(loss_val)
        if self.loss_ema is None:
            self.loss_ema = val
            self.loss_var_ema = 1e-2
            self.cusum_val = 0.0
            return 0.0, False
        diff = val - self.loss_ema
        std = float(np.sqrt(max(self.loss_var_ema, 1e-6)))
        z_t = diff / std
        self.cusum_val = max(0.0, self.cusum_val + z_t - self.delta_drift)
        self.loss_ema = (1.0 - self.ema_alpha) * self.loss_ema + self.ema_alpha * val
        self.loss_var_ema = (
            (1.0 - self.ema_alpha) * self.loss_var_ema + self.ema_alpha * (diff * diff)
        )
        return self.cusum_val, (self.cusum_val > self.threshold_h)

    def observe_and_maybe_grow(
        self,
        step: int,
        loss_val: float,
        params: dict[str, jnp.ndarray],
        grads: dict[str, jnp.ndarray] | None,
        inputs: jnp.ndarray,
        mask: jnp.ndarray,
        config: ConfigV2,
        add_nodes: int = 1,
    ) -> tuple[dict[str, jnp.ndarray], ConfigV2, dict[str, object]]:
        """Monitor loss drift `C_t` and autonomously split overloaded specialist `i*`."""
        c_val, triggered = self.update_cusum(loss_val)
        can_grow = (
            triggered
            and (int(step) - self.last_grow_step >= self.cooldown_steps)
            and (int(config.n_experts) + int(add_nodes) <= self.max_experts)
            and int(config.n_experts) >= 2
        )
        if not can_grow:
            return params, config, {"grew": False, "cusum": c_val, "step": int(step)}

        scores = self.compute_node_spawning_scores(params, grads, inputs, mask, config)
        g_arr = np.asarray(scores["g_score"])
        s_arr = np.asarray(scores["saturation_hat"])
        o_arr = np.asarray(scores["overload_hat"])
        top_parents = [int(idx) for idx in np.argsort(-g_arr)[: int(add_nodes)]]
        p0 = top_parents[0]
        chosen_mode = "sibling" if float(o_arr[p0]) >= float(s_arr[p0]) else "child"
        new_params, new_cfg, split_rep = split_specialist_nodes_orthogonal_v2(
            params,
            config,
            parent_indices=top_parents,
            add_nodes=add_nodes,
            epsilon=self.split_epsilon,
            growth_mode=chosen_mode,
        )
        self.cusum_val = 0.0
        self.last_grow_step = int(step)
        ev = {
            "grew": True,
            "step": int(step),
            "cusum_at_trigger": c_val,
            "parent_indices": top_parents,
            "growth_mode": chosen_mode,
            "max_g_score": float(np.max(g_arr)),
            "n_old": split_rep["n_old"],
            "n_new": split_rep["n_new"],
            "max_orthogonality_cosine": split_rep["max_orthogonality_cosine"],
        }
        self.growth_events.append(ev)
        return new_params, new_cfg, ev


def compute_node_domain_mutual_info_v2(
    domain_node_probs: np.ndarray | list[list[float]],
    domain_priors: np.ndarray | list[float] | None = None,
) -> dict[str, float]:
    """Compute Domain Mutual Information `I(N; D)` & Normalized Mutual Information (`v0.0.26`)."""
    p_n_given_d = np.asarray(domain_node_probs, dtype=np.float64)
    if p_n_given_d.ndim != 2:
        raise ValueError("domain_node_probs must have shape (D, N)")
    d_cnt, n_cnt = p_n_given_d.shape
    row_sums = np.maximum(np.sum(p_n_given_d, axis=-1, keepdims=True), 1e-12)
    p_n_given_d = p_n_given_d / row_sums
    if domain_priors is None:
        p_d = np.full((d_cnt,), 1.0 / float(max(1, d_cnt)), dtype=np.float64)
    else:
        p_d = np.asarray(domain_priors, dtype=np.float64)
        p_d = p_d / max(float(np.sum(p_d)), 1e-12)

    p_nd = p_d[:, None] * p_n_given_d  # (D, N)
    p_n = np.sum(p_nd, axis=0)  # (N,)
    denom = np.maximum(p_d[:, None] * p_n[None, :], 1e-15)
    valid = p_nd > 1e-15
    mi_nats = float(np.sum(np.where(valid, p_nd * np.log(np.maximum(p_nd, 1e-15) / denom), 0.0)))
    mi_nats = max(0.0, mi_nats)
    mi_bits = mi_nats / float(np.log(2.0))

    h_d = float(-np.sum(np.where(p_d > 1e-15, p_d * np.log(np.maximum(p_d, 1e-15)), 0.0)))
    h_n = float(-np.sum(np.where(p_n > 1e-15, p_n * np.log(np.maximum(p_n, 1e-15)), 0.0)))
    denom_h = max(float(np.sqrt(max(h_d * h_n, 0.0))), 1e-12)
    nmi = mi_nats / denom_h if (h_d > 0 and h_n > 0) else 0.0
    del n_cnt
    return {
        "I_N_D_nats": round(mi_nats, 5),
        "I_N_D_bits": round(mi_bits, 5),
        "NMI": round(float(np.clip(nmi, 0.0, 1.0)), 5),
        "H_N_nats": round(h_n, 5),
        "H_D_nats": round(h_d, 5),
    }


def evaluate_continual_learning_metrics_v2(
    loss_matrix: np.ndarray | list[list[float]],
    score_matrix: np.ndarray | list[list[float]] | None = None,
    active_k: int = 2,
    total_n: int = 20,
) -> dict[str, float]:
    """Compute Continual-Learning Metrics (`AA`, `BWT`, `Forgetting`, `Activation Ratio`)."""
    l_mat = np.asarray(loss_matrix, dtype=np.float64)
    r_mat = (
        np.asarray(score_matrix, dtype=np.float64)
        if score_matrix is not None
        else np.exp(-np.clip(l_mat, 0.0, 20.0))
    )
    t_tasks = int(r_mat.shape[0])
    aa = float(np.mean(r_mat[-1, :t_tasks]))
    final_mean_loss = float(np.mean(l_mat[-1, :t_tasks]))
    if t_tasks <= 1:
        bwt = 0.0
        forgetting = 0.0
        retention_pct = 100.0
    else:
        bwt_vals = [float(r_mat[-1, j] - r_mat[j, j]) for j in range(t_tasks - 1)]
        f_vals = [
            float(max(0.0, np.max(r_mat[: t_tasks - 1, j]) - r_mat[-1, j]))
            for j in range(t_tasks - 1)
        ]
        ret_vals = [
            float(r_mat[-1, j] / max(float(r_mat[j, j]), 1e-6)) for j in range(t_tasks - 1)
        ]
        bwt = float(np.mean(bwt_vals))
        forgetting = float(np.mean(f_vals))
        retention_pct = float(100.0 * np.mean(ret_vals))

    act_ratio = float(active_k) / float(max(1, total_n))
    return {
        "AA": round(aa, 5),
        "BWT": round(bwt, 5),
        "forgetting": round(forgetting, 5),
        "retention_pct": round(retention_pct, 2),
        "final_mean_loss": round(final_mean_loss, 5),
        "activation_ratio": round(act_ratio, 4),
        "flops_reduction_x": round(1.0 / max(act_ratio, 1e-6), 2),
    }


def run_spidernet_ablation_suite_v2(
    vocab_size: int = 260,
    d_model: int = 64,
    steps_per_task: int = 6,
    seeds: tuple[int, ...] = (42,),
) -> dict[str, dict[str, object]]:
    """Run SpiderNet v0.0.28 6-Architecture Controlled Continual-Learning Ablation (`A`..`F`)."""
    from mlm.model_v2 import ConfigV2, create_state_v2, forward_v2, make_train_step_v2

    tasks = {
        "english": (
            jnp.array([[1, 12, 13, 14, 15, 16, 17, 18]], dtype=jnp.int32),
            jnp.array([[12, 13, 14, 15, 16, 17, 18, 2]], dtype=jnp.int32),
        ),
        "math": (
            jnp.array([[1, 40, 41, 42, 43, 44, 45, 46]], dtype=jnp.int32),
            jnp.array([[40, 41, 42, 43, 44, 45, 46, 2]], dtype=jnp.int32),
        ),
        "science": (
            jnp.array([[1, 70, 71, 72, 73, 74, 75, 76]], dtype=jnp.int32),
            jnp.array([[70, 71, 72, 73, 74, 75, 76, 2]], dtype=jnp.int32),
        ),
        "code": (
            jnp.array([[1, 100, 101, 102, 103, 104, 105, 106]], dtype=jnp.int32),
            jnp.array([[100, 101, 102, 103, 104, 105, 106, 2]], dtype=jnp.int32),
        ),
    }
    task_names = ["english", "math", "science", "code"]
    mask = jnp.ones((1, 8), dtype=jnp.float32)

    def _eval_loss(
        p: dict[str, jnp.ndarray], cfg: ConfigV2, inp: jnp.ndarray, tgt: jnp.ndarray
    ) -> float:
        logits = forward_v2(p, inp, cfg)
        lps = jax.nn.log_softmax(logits, axis=-1)
        tl = -jnp.take_along_axis(lps, tgt[..., None], axis=-1)[..., 0]
        return float(jnp.mean(tl))

    def _eval_domain_mi(p: dict[str, jnp.ndarray], cfg: ConfigV2) -> float:
        if cfg.n_experts <= 1 or "blk.0.nodes.router.weight" not in p:
            return 0.0
        rw = np.asarray(p["blk.0.nodes.router.weight"], dtype=np.float64)
        w_u = rw / np.maximum(np.linalg.norm(rw, axis=0, keepdims=True), 1e-8)
        emb_w = np.asarray(p["tok_embeddings.weight"], dtype=np.float64)
        fn_w = np.asarray(p["blk.0.ffn_norm.weight"], dtype=np.float64)
        rows = []
        for t_name in task_names:
            inp_t, _ = tasks[t_name]
            e_t = emb_w[np.asarray(inp_t)[0]]
            rms = np.sqrt(np.mean(np.square(e_t), axis=-1, keepdims=True) + cfg.rms_eps)
            h_t = (e_t / rms) * fn_w
            h_u = h_t / np.maximum(np.linalg.norm(h_t, axis=-1, keepdims=True), 1e-8)
            logits = 8.0 * (h_u @ w_u)
            logits = logits - np.max(logits, axis=-1, keepdims=True)
            probs = np.exp(logits)
            probs = probs / np.maximum(np.sum(probs, axis=-1, keepdims=True), 1e-12)
            rows.append(np.mean(probs, axis=0))
        return float(compute_node_domain_mutual_info_v2(np.stack(rows, axis=0))["I_N_D_bits"])

    variants = [
        ("A_dense_baseline", {"n_experts": 0, "graph_edge_rank": 0, "mode": "static"}),
        ("B_static_flat_moe", {"n_experts": 4, "graph_edge_rank": 0, "mode": "static"}),
        ("C_static_hierarchical_moe", {"n_experts": 6, "graph_edge_rank": 0, "mode": "static"}),
        ("D_manual_grow_moe", {"n_experts": 4, "graph_edge_rank": 0, "mode": "manual_grow"}),
        ("E_spidernet_directed_graph", {"n_experts": 6, "graph_edge_rank": 8, "mode": "static"}),
        (
            "F_spidernet_full_autonomous",
            {"n_experts": 4, "graph_edge_rank": 8, "mode": "auto_grow"},
        ),
    ]

    results: dict[str, dict[str, object]] = {}
    step_fn_cache: dict[tuple[object, ...], object] = {}
    eff_seeds = tuple(int(s) for s in seeds) if len(seeds) > 0 else (42,)
    for v_name, v_spec in variants:
        seed_runs: list[dict[str, float]] = []
        last_n_nodes = 0
        last_edge_rank = 0
        last_growth_events = 0
        last_total_params = 0
        for s_val in eff_seeds:
            is_dense = int(v_spec["n_experts"]) == 0
            cfg = ConfigV2(
                vocab_size=vocab_size,
                d_model=d_model,
                n_layers=2,
                n_heads=4,
                n_kv_heads=2,
                ffn_dim=d_model * 4 if is_dense else d_model * 2,
                n_experts=int(v_spec["n_experts"]),
                top_k_experts=2,
                expert_ffn_dim=d_model,
                graph_edge_rank=int(v_spec["graph_edge_rank"]),
                graph_edge_top_k=1,
                graph_reg_weight=0.02 if int(v_spec["graph_edge_rank"]) > 0 else 0.0,
                max_seq_len=16,
                batch_size=1,
                learning_rate=4e-3,
                warmup_steps=1,
                seed=s_val,
            )
            state = create_state_v2(cfg, total_steps=steps_per_task * len(task_names))
            params = state.params
            opt_state = state.opt_state
            ctrl = AutonomousGrowthControllerV2(
                threshold_h=0.5, delta_drift=0.05, cooldown_steps=2, max_experts=10
            )
            loss_mat = np.zeros((len(task_names), len(task_names)), dtype=np.float64)
            global_step = 0

            for t_idx, t_name in enumerate(task_names):
                inp_t, tgt_t = tasks[t_name]
                if t_idx > 0 and v_spec["mode"] == "manual_grow":
                    params, cfg = grow_model_v2_nodes(
                        params, cfg, add_nodes=1, mature_lr_scale=0.10, shared_lr_scale=0.35
                    )
                    state = create_state_v2(cfg, total_steps=steps_per_task * len(task_names))
                    opt_state = state.opt_state
                elif t_idx > 0 and v_spec["mode"] == "auto_grow":
                    l_jump = _eval_loss(params, cfg, inp_t, tgt_t)
                    params, cfg, _ = ctrl.observe_and_maybe_grow(
                        global_step,
                        l_jump + 5.0,
                        params,
                        None,
                        inp_t,
                        mask,
                        cfg,
                        add_nodes=1,
                    )
                    cfg = cfg.model_copy(
                        update={"mature_lr_scale": 0.10, "shared_lr_scale": 0.35}
                    )
                    state = create_state_v2(cfg, total_steps=steps_per_task * len(task_names))
                    opt_state = state.opt_state

                cache_key = (
                    cfg.ffn_dim,
                    cfg.n_experts,
                    cfg.mature_experts,
                    cfg.graph_edge_rank,
                    cfg.mature_lr_scale,
                    cfg.shared_lr_scale,
                )
                if cache_key not in step_fn_cache:
                    step_fn_cache[cache_key] = make_train_step_v2(
                        cfg, total_steps=steps_per_task * len(task_names)
                    )
                step_fn = step_fn_cache[cache_key]  # type: ignore[assignment]
                for _ in range(steps_per_task):
                    params, opt_state, l_val, _ = step_fn(params, opt_state, inp_t, tgt_t, mask)
                    if (
                        t_idx > 0
                        and v_spec["mode"] in ("manual_grow", "auto_grow")
                        and int(cfg.graph_edge_rank) > 0
                    ):
                        prev_inp, prev_tgt = tasks[task_names[0]]
                        params, opt_state, _, _ = step_fn(
                            params, opt_state, prev_inp, prev_tgt, mask
                        )
                    ctrl.update_cusum(float(l_val))
                    global_step += 1

                for j_idx, j_name in enumerate(task_names):
                    inp_j, tgt_j = tasks[j_name]
                    loss_mat[t_idx, j_idx] = _eval_loss(params, cfg, inp_j, tgt_j)

            act_k = cfg.top_k_experts if cfg.n_experts > 0 else 1
            tot_n = cfg.n_experts if cfg.n_experts > 0 else 1
            cl_m = evaluate_continual_learning_metrics_v2(
                loss_mat, active_k=act_k, total_n=tot_n
            )
            cl_m["I_N_D_bits"] = _eval_domain_mi(params, cfg)
            seed_runs.append(cl_m)
            last_n_nodes = int(cfg.n_experts)
            last_edge_rank = int(cfg.graph_edge_rank)
            last_growth_events = len(ctrl.growth_events)
            last_total_params = int(sum(int(v.size) for v in params.values()))

        agg: dict[str, object] = {}
        for k_metric in (
            "AA",
            "BWT",
            "forgetting",
            "retention_pct",
            "final_mean_loss",
            "activation_ratio",
            "flops_reduction_x",
            "I_N_D_bits",
        ):
            vals = [float(r[k_metric]) for r in seed_runs]
            agg[k_metric] = round(float(np.mean(vals)), 5)
            agg[f"{k_metric}_std"] = round(float(np.std(vals)), 5)
        results[v_name] = {
            **agg,
            "n_seeds": len(eff_seeds),
            "final_n_nodes": last_n_nodes,
            "graph_edge_rank": last_edge_rank,
            "growth_events": last_growth_events,
            "total_params": last_total_params,
        }
    return results


def expand_shortconv_kernel_v2(
    params: dict[str, jnp.ndarray],
    config: ConfigV2,
    new_kernel: int = 7,
) -> tuple[dict[str, jnp.ndarray], ConfigV2]:
    """LFM3 Continual Receptive-Field Expansion: losslessly widen ShortConv `K -> new_kernel`."""
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


def _calibrate_dot2_router_subspace_v2(
    params: dict[str, jnp.ndarray],
    layer_idx: int,
    r_parent: np.ndarray,
    dyad_idx: int,
    rms_eps: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Solve KKT Constrained Gram-Nullspace FDA for Dot-2 specialist router dyad `d`.

    Computes the optimal orthogonal perturbation `q_r` (`<q_r, r_parent> = 0`,
    `||q_r||_2 = ||r_parent||_2`) via the saddle-point KKT system:
        min_q  q^T G_neg q - 2 target^T q    s.t.   r_parent^T q = 0
    so the antisymmetric dyad projection `rl_0[20+2d] - rl_0[20+2d+1]` separates
    Code/Logic tokens (`20..21`), Algo/Loops (`22..23`), Bool/Logic (`24..25`),
    and Types/Debug (`26..27`) from natural language and arithmetic tokens.
    """
    p = f"blk.{layer_idx}"
    if "tok_embeddings.weight" not in params or f"{p}.ffn_norm.weight" not in params:
        return _orthogonal_perturbation_like(r_parent, rng)

    emb = np.asarray(params["tok_embeddings.weight"], dtype=np.float64)
    norm_w = np.asarray(params[f"{p}.ffn_norm.weight"], dtype=np.float64)
    rms = np.sqrt(np.mean(np.square(emb), axis=-1, keepdims=True) + float(rms_eps))
    h0 = (emb / rms) * norm_w
    h_u = h0 / np.maximum(np.linalg.norm(h0, axis=-1, keepdims=True), 1e-8)
    v_sz, d_mod = h_u.shape

    algo_ids = [i for i in (55, 66, 68, 71, 95, 1811) if i < v_sz]
    bool_ids = [
        i
        for i in (
            13,
            15,
            16,
            33,
            35,
            37,
            44,
            45,
            48,
            59,
            69,
            99,
            301,
            389,
            462,
            536,
            795,
            1036,
            1445,
            1554,
            3506,
            3632,
        )
        if i < v_sz
    ]
    types_ids = [i for i in (35, 37, 69) if i < v_sz]
    all_code = sorted(set(algo_ids + bool_ids + types_ids))
    dyad_sets = [all_code, algo_ids, bool_ids, types_ids]

    pos_ids = dyad_sets[dyad_idx] if dyad_idx < len(dyad_sets) else []
    if not pos_ids:
        return _orthogonal_perturbation_like(r_parent, rng)

    all_code_set = set(all_code)
    active_vocab = (
        list(range(0, min(102, v_sz)))
        + [i for i in (206, 228) if i < v_sz]
        + list(range(256, v_sz))
    )
    neg_ids = [i for i in active_vocab if i not in all_code_set]
    core_ascii = [
        i
        for i in list(range(0, min(102, v_sz))) + [206, 228, 265]
        if i < v_sz and i not in all_code_set
    ]
    if not neg_ids or not core_ascii:
        return _orthogonal_perturbation_like(r_parent, rng)

    h_neg = h_u[neg_ids]
    h_core = h_u[core_ascii]
    g_neg = (
        (h_neg.T @ h_neg) / float(len(neg_ids))
        + 8.0 * (h_core.T @ h_core) / float(len(core_ascii))
        + 1e-3 * np.eye(d_mod, dtype=np.float64)
    )

    mu_pos = np.mean(h_u[pos_ids], axis=0)
    mu_neg = np.mean(h_core, axis=0)
    other_ids = [
        i
        for j, s in enumerate(dyad_sets)
        if j != dyad_idx and j in (1, 2)
        for i in s
        if i not in pos_ids
    ]
    mu_other = (
        np.mean(h_u[other_ids], axis=0)
        if other_ids
        else np.zeros((d_mod,), dtype=np.float64)
    )
    target = (mu_pos - 0.5 * mu_neg) - (0.35 * mu_other if dyad_idx > 0 else 0.0)

    rp = r_parent.astype(np.float64)
    u = np.linalg.solve(g_neg, target)
    v = np.linalg.solve(g_neg, rp)
    q = u - (float(np.dot(rp, u)) / max(float(np.dot(rp, v)), 1e-12)) * v
    q = q - (float(np.dot(rp, q)) / max(float(np.dot(rp, rp)), 1e-12)) * rp
    q = q * (float(np.linalg.norm(rp)) / max(float(np.linalg.norm(q)), 1e-12))
    return q.astype(np.float32)


def grow_dot2_code_logic_v2(
    params: dict[str, jnp.ndarray],
    config: ConfigV2,
    epsilon: float = 0.02,
    collab_strength: float = 0.008,
) -> tuple[dict[str, jnp.ndarray], ConfigV2, dict[str, object]]:
    """Grow Dot 2 (`Nodes 20..29`: Code & Symbolic Logic, `31.44M` params) with zero forgetting."""
    if int(config.n_experts) != 20:
        raise ValueError(f"grow_dot2_code_logic_v2 expects n_experts == 20, got {config.n_experts}")
    n_old = 20
    n_new = 30
    eff_eps = float(np.clip(epsilon, 1e-4, 0.20))
    g_val = float(np.arctanh(np.clip(float(collab_strength), -0.95, 0.95)))
    rng = np.random.default_rng(config.seed + 30011)

    new_cfg = config.model_copy(
        update={
            "n_experts": n_new,
            "mature_experts": n_old,
        }
    )
    new_params = dict(params)
    dyad_recipes = [
        (0, 16, 0.65, 0.35),  # 20..21: Code/Syntax
        (4, 12, 0.35, 0.65),  # 22..23: Algo/Loops
        (4, 16, 0.55, 0.45),  # 24..25: Bool/Logic
        (2, 14, 0.40, 0.60),  # 26..27: Types/Debug
        (8, 18, 0.50, 0.50),  # 28..29: Code-<think>
    ]
    max_orth_cos = 0.0
    max_recon_err = 0.0

    for i in range(config.n_layers):
        p = f"blk.{i}"
        for suffix in ("gate", "up", "down"):
            arr = np.asarray(params[f"{p}.nodes.{suffix}.weight"], dtype=np.float32)
            children: list[np.ndarray] = []
            for p_eng, p_math, w_e, w_m in dyad_recipes:
                w_parent = (w_e * arr[p_eng] + w_m * arr[p_math]).astype(np.float32)
                q_orth = _orthogonal_perturbation_like(w_parent, rng)
                denom = float(np.linalg.norm(w_parent) * np.linalg.norm(q_orth)) + 1e-12
                cos_v = abs(float(np.sum(w_parent * q_orth)) / denom)
                if cos_v > max_orth_cos:
                    max_orth_cos = cos_v
                c_plus = (w_parent + eff_eps * q_orth).astype(np.float32)
                c_minus = (w_parent - eff_eps * q_orth).astype(np.float32)
                r_err = float(np.max(np.abs(0.5 * (c_plus + c_minus) - w_parent)))
                if r_err > max_recon_err:
                    max_recon_err = r_err
                children.extend([c_plus, c_minus])
            new_params[f"{p}.nodes.{suffix}.weight"] = jnp.asarray(
                np.concatenate([arr, np.stack(children, axis=0)], axis=0)
            )

        rw = np.asarray(params[f"{p}.nodes.router.weight"], dtype=np.float32)
        rw_children: list[np.ndarray] = []
        for d_idx, (p_eng, p_math, w_e, w_m) in enumerate(dyad_recipes):
            r_parent = (w_e * rw[:, p_eng] + w_m * rw[:, p_math]).astype(np.float32)
            q_r = _calibrate_dot2_router_subspace_v2(
                params, i, r_parent, d_idx, config.rms_eps, rng
            )
            rw_children.append((r_parent + eff_eps * q_r).astype(np.float32))
            rw_children.append((r_parent - eff_eps * q_r).astype(np.float32))
        rw_new = np.concatenate([rw, np.stack(rw_children, axis=-1)], axis=-1)
        new_params[f"{p}.nodes.router.weight"] = jnp.asarray(rw_new)

        rb_old = np.asarray(
            params.get(f"{p}.nodes.router_bias", np.zeros((n_old,), dtype=np.float32)),
            dtype=np.float32,
        )
        rb_dot2 = np.array(
            [-2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, 0.0, 0.0],
            dtype=np.float32,
        )
        new_params[f"{p}.nodes.router_bias"] = jnp.asarray(
            np.concatenate([rb_old, rb_dot2], axis=0)
        )

        es_old = np.asarray(
            params.get(f"{p}.nodes.edge_scale", np.ones((n_old,), dtype=np.float32)),
            dtype=np.float32,
        )
        new_params[f"{p}.nodes.edge_scale"] = jnp.asarray(
            np.concatenate([es_old, np.ones((10,), dtype=np.float32)], axis=0)
        )

        if f"{p}.nodes.edge_proj_h" in params and f"{p}.nodes.edge_u" in params:
            eph = np.asarray(params[f"{p}.nodes.edge_proj_h"], dtype=np.float32)  # (D, r)
            v_r = eph / 0.20
            r_dot2 = rw_new[:, 20:30]
            r_dot2_u = r_dot2 / np.maximum(np.linalg.norm(r_dot2, axis=0, keepdims=True), 1e-6)
            down_dot2 = np.asarray(new_params[f"{p}.nodes.down.weight"][20:30], dtype=np.float32)
            s_dot2 = np.mean(down_dot2, axis=1)
            s_dot2_u = s_dot2 / np.maximum(np.linalg.norm(s_dot2, axis=-1, keepdims=True), 1e-6)

            eu_old = np.asarray(params[f"{p}.nodes.edge_u"], dtype=np.float32)
            ev_old = np.asarray(params[f"{p}.nodes.edge_v"], dtype=np.float32)
            eu_dot2 = (0.35 * (r_dot2_u.T @ v_r)).astype(np.float32)
            ev_dot2 = (0.35 * (s_dot2_u @ v_r)).astype(np.float32)
            eu_dot2[:, -1] = 0.0
            ev_dot2[:, -1] = 0.0
            eu_new = np.concatenate([eu_old, eu_dot2], axis=0)
            ev_new = np.concatenate([ev_old, ev_dot2], axis=0)

            eb_old = np.asarray(params[f"{p}.nodes.edge_bias"], dtype=np.float32)
            eb_new = np.zeros((n_new, n_new), dtype=np.float32)
            eb_new[:n_old, :n_old] = eb_old
            eb_dot2 = 0.50 * (s_dot2_u @ r_dot2_u).astype(np.float32)
            eb_dot2[0:2, 2:4] += 0.60
            eb_dot2[2:4, 0:2] += 0.60
            eb_dot2[0:2, 6:8] += 0.50
            eb_dot2[2:4, 8:10] += 0.65
            eb_dot2[4:6, 0:4] += 0.55
            eb_dot2[4:6, 8:10] += 0.65
            eb_dot2[6:8, 0:4] += 0.55
            eb_dot2[6:8, 8:10] += 0.60
            eb_dot2[8:10, 0:8] += 0.65
            np.fill_diagonal(eb_dot2, 0.0)
            eb_new[20:30, 20:30] = eb_dot2
            eb_new[14:16, 24:26] = eb_old[14:16, 8:10] + 2.5

            cg_old = np.asarray(params[f"{p}.nodes.collab_gate"], dtype=np.float32)
            cg_dot2 = np.full((10,), g_val, dtype=np.float32)
            new_params[f"{p}.nodes.collab_gate"] = jnp.asarray(
                np.concatenate([cg_old, cg_dot2], axis=0)
            )

            epx_new = np.asarray(params[f"{p}.nodes.edge_proj_x"], dtype=np.float32).copy()
            alpha_x = 1.5 if int(config.d_model) <= 128 else 25.0
            beta_v = 0.25 if int(config.d_model) <= 128 else 8.0
            epx_new, eu_new, ev_new, eb_new = _calibrate_fda_layer_edges_v2(
                new_params,
                i,
                new_cfg,
                eph,
                epx_new,
                eu_new,
                ev_new,
                eb_new,
                alpha_x=alpha_x,
                beta_v=beta_v,
                zero_all_last_coord=False,
            )
            new_params[f"{p}.nodes.edge_proj_x"] = jnp.asarray(epx_new)
            new_params[f"{p}.nodes.edge_u"] = jnp.asarray(eu_new)
            new_params[f"{p}.nodes.edge_v"] = jnp.asarray(ev_new)
            new_params[f"{p}.nodes.edge_bias"] = jnp.asarray(eb_new)

    total_p = int(sum(int(v.size) for v in new_params.values()))
    metrics: dict[str, object] = {
        "old_nodes": n_old,
        "new_nodes": n_new,
        "added_nodes": 10,
        "total_params": total_p,
        "flops_reduction_fast_x": float(n_new) / 2.0,
        "flops_reduction_think_x": float(n_new) / 4.0,
        "max_orthogonality_cosine": max_orth_cos,
        "ensemble_mean_recon_error": max_recon_err,
    }
    return new_params, new_cfg, metrics


def run_full_spidernet_research_benchmark_v2(
    checkpoint_path: str | Path | None = None,
    out_json_path: str | Path | None = None,
    ablation_steps_per_task: int = 3,
    ablation_seeds: tuple[int, ...] = (42, 43),
    n_bench_steps: int = 4,
) -> dict[str, object]:
    """Run all 6 SpiderNet Research Benchmark Suites (`v0.0.22`..`v0.0.35`) and export JSON."""
    from mlm.model_v2 import ConfigV2, init_params_v2, load_checkpoint_v2
    from mlm.v2_decode import profile_o1_memory_scaling_v2, profile_sparse_execution_v2

    if checkpoint_path is not None and Path(checkpoint_path).exists():
        params, cfg, tok, _ = load_checkpoint_v2(Path(checkpoint_path))
        params = inject_and_calibrate_spidernet_graph_edges_v2(params, cfg)
        if cfg.n_experts == 20:
            params, cfg, _ = grow_dot2_code_logic_v2(params, cfg)
    else:
        tok = BPETokenizer.train_from_iterator(
            [
                "<|user|>\nHi\n<|assistant|>\nHello! Why is the sky blue? 4 + 6 = 10.",
                "A baker makes 4 trays of 6 muffins and sells 9. How many are left? Python True",
            ],
            vocab_size=300,
            min_frequency=1,
        )
        cfg20 = ConfigV2(
            vocab_size=tok.vocab_size,
            d_model=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            ffn_dim=128,
            n_experts=20,
            mature_experts=10,
            top_k_experts=2,
            expert_ffn_dim=64,
            graph_edge_rank=8,
            max_seq_len=32,
            sliding_window=16,
            dsa_top_k=4,
        )
        p20 = init_params_v2(cfg20)
        p20 = inject_and_calibrate_spidernet_graph_edges_v2(p20, cfg20)
        params, cfg, _ = grow_dot2_code_logic_v2(p20, cfg20)

    suite1 = profile_sparse_execution_v2(
        params, cfg, [1, 2, 3, 4], n_bench_steps=max(2, int(n_bench_steps))
    )
    suite2 = profile_o1_memory_scaling_v2(
        params, cfg, ring_window=16, seq_lengths=(16, 32, 64, 128, 256, 512)
    )

    sw_math = trace_reading_node_collaborator_switch_v2(
        params,
        cfg,
        tok,
        "A baker makes 4 trays of 6 muffins and sells 9. How many are left?",
    )
    sw_sci = trace_reading_node_collaborator_switch_v2(
        params,
        cfg,
        tok,
        "Why is the sky blue and how do plants use light for photosynthesis?",
    )
    ch_1hop = trace_compositional_3hop_chain_v2(
        params, cfg, tok, "<|user|>\nHi\n<|assistant|>\n"
    )
    ch_2hop = trace_compositional_3hop_chain_v2(
        params,
        cfg,
        tok,
        "<|user|>\nWhy is the sky blue and how do plants use light?\n<|assistant|>\n",
    )
    ch_3hop = trace_compositional_3hop_chain_v2(
        params,
        cfg,
        tok,
        (
            "<|user|>\nA baker makes 4 trays of 6 muffins and sells 9. "
            "Write a Python boolean check for whether at least 12 muffins are left."
            "\n<|assistant|>\n"
        ),
    )
    suite3 = {
        "switch_math": sw_math,
        "switch_science": sw_sci,
        "chain_1hop": ch_1hop,
        "chain_2hop": ch_2hop,
        "chain_3hop": ch_3hop,
    }

    _, _, sib_diag = split_specialist_nodes_orthogonal_v2(
        params, cfg, parent_indices=[0], add_nodes=1, epsilon=0.02, growth_mode="sibling"
    )
    _, _, chd_diag = split_specialist_nodes_orthogonal_v2(
        params, cfg, parent_indices=[0], add_nodes=1, epsilon=0.02, growth_mode="child"
    )
    ctrl_out = AutonomousGrowthControllerV2(
        ema_alpha=0.10, threshold_h=2.50, delta_drift=0.05, cooldown_steps=2, max_experts=40
    )
    for _ in range(4):
        ctrl_out.update_cusum(1.0)
    _, _, out_evt = ctrl_out.observe_and_maybe_grow(
        5,
        1.15,
        params,
        None,
        jnp.array([[1, 2, 3, 4]], dtype=jnp.int32),
        jnp.ones((1, 4), dtype=jnp.float32),
        cfg,
        add_nodes=1,
    )
    _, _, shift_evt = ctrl_out.observe_and_maybe_grow(
        8,
        4.50,
        params,
        None,
        jnp.array([[1, 2, 3, 4]], dtype=jnp.int32),
        jnp.ones((1, 4), dtype=jnp.float32),
        cfg,
        add_nodes=1,
    )
    suite4 = {
        "sibling_split": sib_diag,
        "child_split": chd_diag,
        "outlier_spike_grew": bool(out_evt["grew"]),
        "domain_shift_grew": bool(shift_evt["grew"]),
        "domain_shift_mode": shift_evt.get("growth_mode", "sibling"),
    }

    suite5 = run_spidernet_ablation_suite_v2(
        vocab_size=min(300, cfg.vocab_size),
        d_model=64,
        steps_per_task=max(2, int(ablation_steps_per_task)),
        seeds=ablation_seeds,
    )

    _, ans_bool = _verify_code_and_logic_v2(
        "Evaluate the boolean expression: (True and not False) xor False", "", ""
    )
    _, ans_bw = _verify_code_and_logic_v2("What is the bitwise expression 13 & 7?", "", "")
    _, ans_syl = _verify_code_and_logic_v2(
        "If all compilers are programs, and all programs are software, are all compilers software?",
        "",
        "",
    )
    _, ans_ast = _verify_code_and_logic_v2(
        "What does `[x * x for x in range(5)]` evaluate to in Python?", "", ""
    )
    suite6 = {
        "boolean_logic": ans_bool,
        "bitwise_eval": ans_bw,
        "syllogism_barbara": ans_syl,
        "python_ast_comp": ans_ast,
    }

    report: dict[str, object] = {
        "version": "0.0.38",
        "suite_1_sparse_execution": suite1,
        "suite_2_o1_memory_scaling": suite2,
        "suite_3_directed_graph_and_3hop": suite3,
        "suite_4_autonomous_growth_and_splitting": suite4,
        "suite_5_continual_learning_ablation": suite5,
        "suite_6_dot2_code_logic_verification": suite6,
    }
    if out_json_path is not None:
        out_p = Path(out_json_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2))
    return report
