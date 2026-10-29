"""Explicit parameter roles, Muon orthogonalized updates, and recombinative node growth."""

import math
from collections import deque

import jax
import jax.numpy as jnp
import numpy as np
import optax

NS_COEFFS = (3.4445, -4.7750, 2.0315)

NODE_PARAMETERS = {"input_kernel", "hidden_kernel", "gru_bias", "read_kernel"}
SHARED_PARAMETERS = {
    "embedding",
    "message_kernel",
    "edge_query",
    "edge_key",
    "edge_vector",
    "norm_scale",
    "norm_bias",
    "vocab_kernel",
    "vocab_bias",
    "attention_query",
    "attention_key",
    "attention_vector",
    "mtp_kernel",
}
ROUTER_PARAMETERS = {
    "router_group_kernel",
    "router_group_bias",
    "router_child_kernel",
    "router_child_bias",
    "router_timescale_kernel",
    "router_surprise_kernel",
    "router_timescale_bias",
    "halt_kernel",
    "halt_bias",
}


MEMORY_PARAMETERS = {
    "memory_key_kernel",
    "memory_query_kernel",
    "memory_value_kernel",
    "memory_write_kernel",
    "memory_write_bias",
    "memory_decay_kernel",
    "memory_decay_bias",
    "memory_span_kernel",
    "memory_span_bias",
    "memory_rev_key_kernel",
    "memory_rev_value_kernel",
    "memory_output_kernel",
    "memory_gain",
    "episodic_compress_kernel",
    "episodic_write_kernel",
    "episodic_write_bias",
    "episodic_query_kernel",
    "episodic_pos_bias",
    "episodic_output_kernel",
    "episodic_gain",
}


ROLE_BY_PARAMETER = {
    **dict.fromkeys(NODE_PARAMETERS, "node"),
    **dict.fromkeys(SHARED_PARAMETERS, "shared"),
    **dict.fromkeys(ROUTER_PARAMETERS, "router"),
    **dict.fromkeys(MEMORY_PARAMETERS, "memory"),
}


def parameter_roles(params):
    try:
        return {name: ROLE_BY_PARAMETER[name] for name in params}
    except KeyError as exc:
        raise ValueError(f"Unknown parameter role: {exc.args[0]}") from exc


def role_scales(config):
    return {
        "shared": config.shared_lr_scale,
        "node": config.node_lr_scale,
        "router": config.router_lr_scale,
        "memory": config.memory_lr_scale,
    }


def node_activity_scales(grads, params):
    """Per-node relative gradient activity in [0, 1] for node-scoped tensors."""
    roles = parameter_roles(params)
    scales = {}
    for name, grad in grads.items():
        if roles[name] == "node" and getattr(grad, "ndim", 0) >= 2:
            axes = tuple(range(1, grad.ndim))
            norms = jnp.sqrt(jnp.sum(jnp.square(grad), axis=axes, keepdims=True))
            peak = jnp.max(norms)
            scales[name] = jnp.where(peak > 0, norms / jnp.maximum(peak, 1e-12), 0.0)
        else:
            scales[name] = jnp.asarray(1.0, dtype=grad.dtype)
    return scales


def scale_optimizer(optimizer, params, config):
    """Scale final updates, including decay, without changing moments or state shape."""
    scales = role_scales(config)
    if (
        all(value == 1 for value in scales.values())
        and not config.local_plasticity
        and config.warmup_steps == 0
    ):
        return optimizer
    roles = parameter_roles(params)
    multipliers = {name: scales[role] for name, role in roles.items()}

    def update(updates, state, params=None):
        raw_grads = updates
        updates, state = optimizer.update(updates, state, params)
        warmup = jnp.asarray(1.0, dtype=jnp.float32)
        if config.warmup_steps > 0:
            counts = [
                x
                for x in jax.tree.leaves(state)
                if getattr(x, "ndim", -1) == 0 and jnp.issubdtype(x.dtype, jnp.integer)
            ]
            if counts:
                warmup = jnp.minimum(
                    1.0, jnp.asarray(counts[0], dtype=jnp.float32) / config.warmup_steps
                )
        if config.local_plasticity:
            activity = node_activity_scales(raw_grads, params or raw_grads)
            updates = jax.tree.map(
                lambda value, scale, act: value * (scale * warmup * act),
                updates,
                multipliers,
                activity,
            )
        else:
            updates = jax.tree.map(
                lambda value, scale: value * (scale * warmup), updates, multipliers
            )
        return updates, state

    return optax.GradientTransformation(optimizer.init, update)


def newton_schulz(g: jax.Array, steps: int = 5) -> jax.Array:
    """Quintic Newton-Schulz orthogonalization over the last two axes (-2, -1)."""
    a, b, c = NS_COEFFS
    transposed = g.shape[-2] > g.shape[-1]
    x = jnp.swapaxes(g, -2, -1) if transposed else g
    norm = jnp.sqrt(jnp.sum(jnp.square(x), axis=(-2, -1), keepdims=True)) + 1e-7
    x = x / norm
    for _ in range(steps):
        gram = jnp.matmul(x, jnp.swapaxes(x, -2, -1))
        b_mat = b * gram + c * jnp.matmul(gram, gram)
        x = a * x + jnp.matmul(b_mat, x)
    return jnp.swapaxes(x, -2, -1) if transposed else x


def muon_mask(params):
    """Select internal 2D/3D weight matrices for Muon; keep embeddings and heads on AdamW."""
    return {
        name: (
            getattr(value, "ndim", 0) >= 2
            and min(value.shape[-2], value.shape[-1]) >= 2
            and name not in {"embedding", "vocab_kernel"}
            and "bias" not in name
        )
        for name, value in params.items()
    }


def make_muon_adamw(params, config, decay):
    """Hybrid Muon (internal 2D/3D matrices) + AdamW (embeddings, heads, biases, vectors)."""
    use_muon = muon_mask(params)
    adamw = optax.adamw(
        config.learning_rate,
        b1=0.9,
        b2=0.999,
        eps=1e-8,
        weight_decay=config.weight_decay,
        mask=decay,
    )
    beta = 0.95
    lr = config.learning_rate
    wd = config.weight_decay

    def init_fn(params):
        return (
            adamw.init(params),
            {name: jnp.zeros_like(val) for name, val in params.items() if use_muon[name]},
        )

    def update_fn(updates, state, params=None):
        adam_state, muon_bufs = state
        adam_updates, new_adam_state = adamw.update(updates, adam_state, params)
        new_muon_bufs = {}
        muon_updates = {}
        for name, grad in updates.items():
            if not use_muon[name]:
                muon_updates[name] = grad
                continue
            buf = muon_bufs[name]
            active = jnp.any(grad != 0, axis=(-2, -1), keepdims=True)
            next_buf = jnp.where(active, beta * buf + (1.0 - beta) * grad, buf)
            g_nest = beta * next_buf + (1.0 - beta) * grad
            rms_scale = math.sqrt(max(grad.shape[-2], grad.shape[-1]))
            orth = newton_schulz(g_nest, steps=5) * rms_scale
            decay_term = wd * params[name] if (params is not None and wd > 0) else 0.0
            step = -lr * (orth + decay_term)
            new_muon_bufs[name] = next_buf
            muon_updates[name] = jnp.where(active, step, jnp.zeros_like(step))
        out_updates = jax.tree.map(
            lambda flag, m_val, a_val: m_val if flag else a_val,
            use_muon,
            muon_updates,
            adam_updates,
        )
        return out_updates, (new_adam_state, new_muon_bufs)

    return optax.GradientTransformation(init_fn, update_fn)


def recombine_grow_nodes(
    config, params: dict, add_nodes: int, birth_gate: float = 0.05, noise_std: float = 0.01
):
    """Grow a GraphLM from N to N + add_nodes by recombining existing parent node slices."""
    if config.architecture != "graph":
        raise ValueError("Node growth requires graph architecture")
    if add_nodes < 1:
        raise ValueError("add_nodes must be positive")
    if not 0.0 <= birth_gate <= 1.0:
        raise ValueError("birth_gate must be between 0 and 1")
    n = config.nodes
    n_new = n + add_nodes
    if config.node_update == "hierarchical" and n_new % config.groups != 0:
        raise ValueError("New node count must be divisible by hierarchical groups")
    new_edges = None
    if config.edges is not None:
        from mlm.graph import topology

        default_new = topology(n_new, config.degree)
        new_edges = tuple(
            tuple(config.edges[i]) if i < n else tuple(int(x) for x in default_new[i])
            for i in range(n_new)
        )
    new_config = config.model_copy(update={"nodes": n_new, "edges": new_edges})
    norms = jnp.linalg.norm(params["read_kernel"], axis=(1, 2))
    order = jnp.argsort(-norms)
    rng = jax.random.PRNGKey(config.seed + n_new)

    def crossover_3d(arr, key=None):
        children = []
        mid = arr.shape[-1] // 2
        for j in range(add_nodes):
            p1 = arr[order[(2 * j) % n]]
            p2 = arr[order[(2 * j + 1) % n]]
            if mid > 0:
                child = jnp.concatenate([p1[..., :mid], p2[..., mid:]], axis=-1)
            else:
                child = 0.5 * (p1 + p2)
            if key is not None and noise_std > 0:
                subkey = jax.random.fold_in(key, j)
                scale = jnp.std(child) + 1e-6
                child = child + jax.random.normal(subkey, child.shape, child.dtype) * (
                    noise_std * scale
                )
            children.append(child)
        return jnp.concatenate([arr, jnp.stack(children, axis=0)], axis=0)

    new_params = dict(params)
    k_in, k_hid = jax.random.split(rng)
    new_params["input_kernel"] = crossover_3d(params["input_kernel"], k_in)
    new_params["hidden_kernel"] = crossover_3d(params["hidden_kernel"], k_hid)
    new_params["gru_bias"] = crossover_3d(params["gru_bias"], None)

    # Compensate existing read_kernel for 1/sqrt(n_new) scaling and gate newborn nodes by birth_gate
    read_expanded = crossover_3d(params["read_kernel"], None)
    parent_scale = math.sqrt(n_new / n) if config.node_readout == "sum" else 1.0
    new_params["read_kernel"] = jnp.concatenate(
        [
            read_expanded[:n] * parent_scale,
            read_expanded[n:] * birth_gate,
        ],
        axis=0,
    )

    for name in (
        "router_surprise_kernel",
        "halt_kernel",
        "memory_key_kernel",
        "memory_query_kernel",
        "memory_value_kernel",
    ):
        if name in params and params[name].ndim == 3 and params[name].shape[0] == n:
            new_params[name] = crossover_3d(params[name], None)

    for name in ("router_timescale_bias", "halt_bias"):
        if name in params and params[name].shape == (n,):
            extra = jnp.stack(
                [
                    0.5 * (params[name][order[(2 * j) % n]] + params[name][order[(2 * j + 1) % n]])
                    for j in range(add_nodes)
                ]
            )
            new_params[name] = jnp.concatenate([params[name], extra], axis=0)

    if "router_timescale_kernel" in params:
        tk = params["router_timescale_kernel"]
        extra_cols = jnp.stack(
            [
                0.5 * (tk[:, order[(2 * j) % n]] + tk[:, order[(2 * j + 1) % n]])
                for j in range(add_nodes)
            ],
            axis=-1,
        )
        new_params["router_timescale_kernel"] = jnp.concatenate([tk, extra_cols], axis=-1)

    if config.node_update == "hierarchical":
        add_per_group = add_nodes // config.groups
        ck = params["router_child_kernel"]
        cb = params["router_child_bias"]
        c_old = ck.shape[-1]
        extra_ck = jnp.stack(
            [0.5 * (ck[:, :, j % c_old] + ck[:, :, (j + 1) % c_old]) for j in range(add_per_group)],
            axis=-1,
        )
        extra_cb = jnp.stack(
            [0.5 * (cb[:, j % c_old] + cb[:, (j + 1) % c_old]) for j in range(add_per_group)],
            axis=-1,
        )
        new_params["router_child_kernel"] = jnp.concatenate([ck, extra_ck], axis=-1)
        new_params["router_child_bias"] = jnp.concatenate([cb, extra_cb], axis=-1)

    return new_config, new_params


def node_saturation(route_usage, read_kernel, min_norm: float = 0.05) -> dict:
    """Measure graph node utilization entropy and idle readout gates (mini-AGI Pool.saturation)."""
    u = np.asarray(route_usage, dtype=np.float64)
    u = np.clip(u, 0.0, None)
    total = max(float(u.sum()), 1e-12)
    u = u / total
    n = len(u)
    ideal = 1.0 / max(n, 1)
    p = np.clip(u, 1e-12, 1.0)
    ent = float(-np.sum(p * np.log(p)))
    norms = np.linalg.norm(np.asarray(read_kernel, dtype=np.float64), axis=(1, 2))
    return {
        "nodes": n,
        "idle": int(np.sum(norms < min_norm)),
        "idle_by_routing": int(np.sum(u < 0.1 * ideal)),
        "entropy_frac": ent / max(math.log(max(n, 2)), 1e-9),
        "peak_over_ideal": float(np.max(u)) / ideal,
    }


def prune_nodes(config, params: dict, protect: int = 2, min_norm: float = 0.05):
    """Prune speculative nodes (> protect) whose readout norm remained below min_norm."""
    if config.architecture != "graph":
        raise ValueError("Node pruning requires graph architecture")
    if config.node_update == "hierarchical":
        raise ValueError("Node pruning requires non-hierarchical routing")
    n = config.nodes
    protect = max(config.degree + 1, config.shared_nodes + 1, int(protect))
    norms = np.linalg.norm(np.asarray(params["read_kernel"]), axis=(1, 2))
    keep = [i for i in range(n) if i < protect or float(norms[i]) >= min_norm]
    if len(keep) == n:
        return config, params, 0
    idx = jnp.asarray(keep, dtype=jnp.int32)
    n_new = len(keep)
    new_top_k = (
        min(config.top_k_nodes, max(1, n_new - config.shared_nodes - 1))
        if config.top_k_nodes > 0
        else 0
    )
    new_config = config.model_copy(update={"nodes": n_new, "edges": None, "top_k_nodes": new_top_k})
    new_params = dict(params)
    for name in (
        "input_kernel",
        "hidden_kernel",
        "gru_bias",
        "read_kernel",
        "router_surprise_kernel",
        "halt_kernel",
        "memory_key_kernel",
        "memory_query_kernel",
        "memory_value_kernel",
        "router_timescale_bias",
        "halt_bias",
    ):
        if name in params and params[name].shape[0] == n:
            new_params[name] = params[name][idx]
    if config.node_readout == "sum":
        new_params["read_kernel"] = new_params["read_kernel"] * math.sqrt(n_new / n)
    if "router_timescale_kernel" in params:
        new_params["router_timescale_kernel"] = params["router_timescale_kernel"][:, idx]
    return new_config, new_params, n - n_new


def maybe_grow_and_prune(
    config,
    params: dict,
    route_usage,
    train_loss: float | None = None,
    val_loss: float | None = None,
    add_nodes: int = 2,
    max_nodes: int = 32,
    max_gap: float = 0.5,
    protect: int = 2,
    min_norm: float = 0.05,
    birth_gate: float = 0.05,
):
    """Autonomous node pruning and saturation-triggered growth (mini-AGI AutoGrow)."""
    pruned = 0
    if config.node_update != "hierarchical" and config.nodes > max(protect, config.degree + 1):
        config, params, pruned = prune_nodes(config, params, protect=protect, min_norm=min_norm)
    sat = node_saturation(
        np.asarray(route_usage)[: config.nodes], params["read_kernel"], min_norm=min_norm
    )
    gap = (
        (float(val_loss) - float(train_loss))
        if (val_loss is not None and train_loss is not None)
        else None
    )
    may_grow = gap is None or max_gap <= 0 or gap < max_gap
    grew = 0
    if (
        may_grow
        and config.nodes + add_nodes <= max_nodes
        and sat["idle"] == 0
        and sat["entropy_frac"] >= 0.85
    ):
        config, params = recombine_grow_nodes(
            config, params, add_nodes=add_nodes, birth_gate=birth_gate
        )
        grew = add_nodes
    return config, params, {"grew": grew, "pruned": pruned, "saturation": sat}


def edge_affinity_matrix(config, params: dict) -> np.ndarray:
    """Compute learned directed functional affinity S[i, j] from sender j to receiver i."""
    if config.architecture != "graph":
        raise ValueError("Edge affinity requires graph architecture")
    d = config.width
    read_k = np.asarray(params["read_kernel"], dtype=np.float64)
    hid_k = np.asarray(params["hidden_kernel"], dtype=np.float64)
    inp_k = np.asarray(params["input_kernel"], dtype=np.float64)
    msg_k = np.asarray(params["message_kernel"], dtype=np.float64)

    u_sub = np.concatenate([read_k, hid_k], axis=-1)
    u_norms = np.maximum(np.linalg.norm(u_sub, axis=(1, 2), keepdims=True), 1e-8)
    u_hat = u_sub / u_norms
    transmitted = np.einsum("ab,nbc->nac", msg_k, u_hat)

    rec_sub = inp_k[:, d:, :]
    rec_norms = np.maximum(np.linalg.norm(rec_sub, axis=(1, 2), keepdims=True), 1e-8)
    r_hat = rec_sub / rec_norms

    coupling = np.einsum("idr,jdc->ijrc", r_hat, transmitted)
    affinity = np.linalg.norm(coupling, axis=(2, 3))

    if config.edge_aggregation == "attention" and "edge_query" in params:
        eq = np.asarray(params["edge_query"], dtype=np.float64)
        ek = np.asarray(params["edge_key"], dtype=np.float64)
        ev = np.asarray(params["edge_vector"], dtype=np.float64)
        u_mean = u_hat.mean(axis=-1)
        q_i = (u_mean @ eq)[:, None, :]
        k_j = (u_mean @ ek)[None, :, :]
        ev_dir = ev if np.linalg.norm(ev) > 1e-8 else np.ones_like(ev) / math.sqrt(d)
        attn_score = np.tanh(q_i + k_j) @ ev_dir
        affinity = affinity + attn_score

    np.fill_diagonal(affinity, -np.inf)
    return affinity


def rewire_edges(
    config,
    params: dict,
    max_rewire_per_node: int = 1,
    min_gain: float = 1e-4,
    max_total_rewires: int | None = None,
):
    """Prune weakest incoming edges and sprout highest-affinity learned directed edges."""
    if config.architecture != "graph":
        raise ValueError("Edge rewiring requires graph architecture")
    from mlm.config import Config
    from mlm.graph import topology

    affinity = edge_affinity_matrix(config, params)
    current = (
        np.array(config.edges, dtype=np.int32)
        if config.edges is not None
        else topology(config.nodes, config.degree).copy()
    )
    rewired = 0
    n = config.nodes
    for i in range(n):
        if max_total_rewires is not None and rewired >= max_total_rewires:
            break
        row = [int(x) for x in current[i]]
        for _ in range(max_rewire_per_node):
            if max_total_rewires is not None and rewired >= max_total_rewires:
                break
            weak_pos = min(range(len(row)), key=lambda pos: affinity[i, row[pos]])
            weak_node = row[weak_pos]
            candidates = [j for j in range(n) if j != i and j not in row]
            if not candidates:
                break
            best_node = max(candidates, key=lambda j: affinity[i, j])
            if affinity[i, best_node] - affinity[i, weak_node] > min_gain:
                row[weak_pos] = best_node
                rewired += 1
            else:
                break
        current[i] = np.array(row, dtype=np.int32)
    edges_tuple = tuple(tuple(int(x) for x in row) for row in current)
    new_config = Config.model_validate({**config.model_dump(), "edges": edges_tuple})
    return new_config, {"rewired": rewired, "edges": edges_tuple, "affinity": affinity}


class PlasticityController:
    """Held-out-governed learning rate controller (mini-AGI minagi/plasticity.py)."""

    FLOOR = 0.01
    CEIL = 1.0
    LAM = 0.97
    LAM_FAST = 0.85
    EFFECT = 45.0
    MIN_EFF = 6.0
    NUDGE = 0.0008
    NUDGE_DOWN = 0.004
    T_W_DOWN = 6.0
    T_MID = 2.2
    T_W = 0.75
    MAX_MOVE = 0.02
    JUMP_SE = 4.0
    HOLD_SE = 2.0
    FLOOR_JUMP = 0.25
    UP = 2.0
    WARMUP = 100
    SPIKE = 5.0

    def __init__(self, scale: float = 1.0):
        self.scale = float(scale)
        self.slow = self._sums()
        self.fast = self._sums()
        self.i = 0.0
        self.se_hist: deque[float] = deque(maxlen=64)
        self.prev: float | None = None
        self.held: dict | None = None
        self.step = 0
        self.events: list[dict] = []

    @staticmethod
    def _sums() -> dict[str, float]:
        return {"w": 0.0, "w2": 0.0, "x": 0.0, "y": 0.0, "xx": 0.0, "xy": 0.0, "yy": 0.0}

    def factor(self) -> float:
        w = min(1.0, (self.step + 1) / self.WARMUP) if self.step < self.WARMUP else 1.0
        return self.scale * w

    def tick(self, steps: int = 1) -> None:
        self.step += int(steps)

    def _se(self) -> float:
        if not self.se_hist:
            return 0.0
        s = sorted(self.se_hist)
        n = len(s)
        return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])

    def _accumulate(self, y: float) -> None:
        self.i += 1.0
        x = self.i
        for sums, lam in ((self.slow, self.LAM), (self.fast, self.LAM_FAST)):
            for k in ("w", "x", "y", "xx", "xy", "yy"):
                sums[k] *= lam
            sums["w2"] *= lam * lam
            sums["w"] += 1.0
            sums["w2"] += 1.0
            sums["x"] += x
            sums["y"] += y
            sums["xx"] += x * x
            sums["xy"] += x * y
            sums["yy"] += y * y
        self.prev = y

    @staticmethod
    def _fit(sums: dict[str, float]) -> tuple[float, float, float]:
        if sums["w"] <= 2:
            return 0.0, 0.0, 0.0
        n_eff = (sums["w"] ** 2) / max(sums["w2"], 1e-12)
        sxx = sums["xx"] - (sums["x"] ** 2) / sums["w"]
        sxy = sums["xy"] - sums["x"] * sums["y"] / sums["w"]
        syy = sums["yy"] - (sums["y"] ** 2) / sums["w"]
        if sxx <= 0 or n_eff <= 2:
            return 0.0, n_eff, 0.0
        slope = sxy / sxx
        resid = max(syy - slope * sxy, 0.0)
        sigma = math.sqrt(resid / (n_eff - 2)) if n_eff > 2 else 0.0
        if sigma <= 0:
            return 0.0, n_eff, 0.0
        t = -slope / (sigma / math.sqrt(sxx))
        e = -slope / sigma
        return max(-50.0, min(50.0, t)), n_eff, e

    def _verdict(self) -> tuple[float, float, float]:
        t_s, n_s, e_s = self._fit(self.slow)
        _, n_f, e_f = self._fit(self.fast)
        t = min(t_s, self.EFFECT * e_s)
        if n_f >= self.MIN_EFF:
            t = min(t, self.EFFECT * e_f)
        return max(-50.0, min(50.0, t)), n_s, e_s

    @classmethod
    def nudge(cls, t: float) -> float:
        up = t >= cls.T_MID
        gain = cls.NUDGE if up else cls.NUDGE_DOWN
        width = cls.T_W if up else cls.T_W_DOWN
        f = math.exp(gain * math.tanh((t - cls.T_MID) / width))
        return max(1.0 - cls.MAX_MOVE, min(1.0 + cls.MAX_MOVE, f))

    def observe(self, val: float, se: float | None = None) -> str | None:
        if not math.isfinite(float(val)):
            return None
        y = float(val)
        if se is not None and math.isfinite(float(se)) and float(se) > 0:
            self.se_hist.append(float(se))
        s = self._se()
        h, self.held = self.held, None
        if h is not None and h.get("jump"):
            if y > h["from"] + self.HOLD_SE * max(s, 1e-9):
                before = self.scale
                self.scale = min(self.CEIL, self.scale * self.UP)
                self.events.append({"at": self.step, "kind": "regime", "scale": self.scale})
                self.slow, self.fast, self.i = self._sums(), self._sums(), 0.0
                self._accumulate(y)
                return f"regime change: rate {before:.3f} -> {self.scale:.3f}"
        if self.prev is not None and y - self.prev > max(self.JUMP_SE * s, self.FLOOR_JUMP):
            self.held = {"val": y, "from": self.prev, "jump": True}
            return None
        self._accumulate(y)
        t, n_eff, e = self._verdict()
        if n_eff >= self.MIN_EFF:
            self.scale = max(self.FLOOR, min(self.CEIL, self.scale * self.nudge(t)))
        return None

    def state(self) -> dict:
        t, n_eff, e = self._verdict()
        return {
            "scale": self.scale,
            "prev": self.prev,
            "slow": dict(self.slow),
            "fast": dict(self.fast),
            "i": self.i,
            "se": list(self.se_hist),
            "step": self.step,
            "n_eff": round(n_eff, 2),
            "t": round(t, 3),
            "e": round(e, 4),
            "events": self.events[-20:],
        }

    @classmethod
    def restore(cls, d: dict | None):
        ctrl = cls()
        if not d:
            return ctrl
        ctrl.scale = float(d.get("scale", 1.0))
        ctrl.prev = d.get("prev")
        if isinstance(d.get("slow"), dict):
            ctrl.slow.update({k: float(v) for k, v in d["slow"].items() if k in ctrl.slow})
        if isinstance(d.get("fast"), dict):
            ctrl.fast.update({k: float(v) for k, v in d["fast"].items() if k in ctrl.fast})
        ctrl.i = float(d.get("i", 0.0))
        ctrl.step = int(d.get("step", 0))
        for v in d.get("se") or []:
            ctrl.se_hist.append(float(v))
        ctrl.events = list(d.get("events") or [])
        return ctrl


class LaneSchedule:
    """Power-law marginal-worth curriculum & smooth credit scheduler (mini-AGI minagi/lanes.py)."""

    TAU = 0.15
    MIN_EFF = 4.0
    FLOOR = 0.25
    CEIL = 3.0
    STEP = 0.05
    R_STAR = 15.4

    def __init__(self, lanes: list[str], sizes: dict[str, float] | None = None, mode: str = "auto"):
        if mode not in ("auto", "equal"):
            raise ValueError("mode must be 'auto' or 'equal'")
        self.names = list(lanes)
        self.sizes = {n: float((sizes or {}).get(n, 0.0)) for n in self.names}
        self.mode = mode
        n = max(len(self.names), 1)
        self.shares = {k: 1.0 / n for k in self.names}
        self.read = {k: 0.0 for k in self.names}
        self.fits: dict[str, list[float]] = {}
        self.last: float | None = None
        self.credit = {k: 0.0 for k in self.names}

    def note_read(self, lane: str, bytes_read: float) -> None:
        if lane in self.read:
            self.read[lane] += float(bytes_read)

    def observe(self, per_subject: dict[str, float]) -> None:
        if self.mode != "auto":
            return
        total = sum(self.read.values())
        decay = (
            1.0
            if (self.last is None or self.last <= 0 or total <= 0)
            else math.exp(-max(0.0, math.log(total / self.last)) / self.TAU)
        )
        self.last = total
        for f in self.fits.values():
            for idx in range(5):
                f[idx] *= decay
        for lane in self.names:
            if self.read[lane] <= 0:
                continue
            y = per_subject.get(lane)
            if y is None or y <= 0:
                continue
            x = math.log(self.read[lane])
            ly = math.log(float(y))
            f = self.fits.setdefault(lane, [0.0] * 5)
            f[0] += 1.0
            f[1] += x
            f[2] += ly
            f[3] += x * x
            f[4] += x * ly

    def fit(self, lane: str) -> tuple[float, float] | None:
        f = self.fits.get(lane)
        if not f or f[0] < self.MIN_EFF or self.read.get(lane, 0.0) <= 0:
            return None
        w, sx, sy, sxx, sxy = f
        var = sxx / w - (sx / w) ** 2
        if var <= 1e-12:
            return None
        b = (sxy / w - (sx / w) * (sy / w)) / var
        a = sy / w - b * sx / w
        return -b, math.exp(a + b * math.log(self.read[lane]))

    def worth(self, lane: str) -> float | None:
        got = self.fit(lane)
        if got is None or self.read[lane] <= 0:
            return None
        alpha, loss_now = got
        return (abs(alpha) * loss_now / self.read[lane]) * self.reread(lane)

    def reread(self, lane: str) -> float:
        size = self.sizes.get(lane, 0.0)
        if size <= 0:
            return 1.0
        return math.exp(-max(0.0, self.read[lane] / size - 1.0) / self.R_STAR)

    def targets(self) -> dict[str, float]:
        n = len(self.names)
        if self.mode != "auto" or n == 0:
            return dict(self.shares)
        val = {k: self.worth(k) for k in self.names}
        known = sorted(v for v in val.values() if v is not None)
        if not known or known[-1] <= 0:
            base = {k: self.reread(k) for k in self.names}
            tot = max(sum(base.values()), 1e-12)
            return {k: base[k] / tot for k in self.names}
        typical = known[len(known) // 2]
        w_map: dict[str, float] = {}
        for k in self.names:
            vk = val[k]
            w_map[k] = vk if vk is not None else typical * self.reread(k)
        tot = max(sum(w_map.values()), 1e-12)
        lo, hi = self.FLOOR / n, self.CEIL / n
        clipped = {k: max(lo, min(hi, w_map[k] / tot)) for k in self.names}
        s = sum(clipped.values())
        return {k: clipped[k] / s for k in self.names}

    def step(self) -> dict[str, float]:
        if self.mode != "auto":
            return dict(self.shares)
        t = self.targets()
        new = {}
        for k in self.names:
            s = max(self.shares[k], 1e-9)
            r = math.log(max(t[k], 1e-9) / s)
            new[k] = s * math.exp(max(-self.STEP, min(self.STEP, r)))
        tot = sum(new.values())
        self.shares = {k: new[k] / tot for k in self.names}
        return dict(self.shares)

    def pick(self) -> str:
        for k in self.names:
            self.credit[k] += self.shares[k]
        chosen = max(self.names, key=lambda n: self.credit[n])
        self.credit[chosen] -= 1.0
        return chosen


def update_router_load_bias(
    router_bias,
    route_usage,
    gamma: float,
    shared_nodes: int = 0,
):
    """DeepSeek-V3 auxiliary-loss-free load-balancing bias update over specialist nodes."""
    b = jnp.asarray(router_bias, dtype=jnp.float32)
    u = jnp.asarray(route_usage, dtype=jnp.float32)
    if gamma <= 0.0 or u.shape[0] <= shared_nodes:
        return b
    spec_u = u[shared_nodes:]
    target = jnp.mean(spec_u)
    delta = gamma * jnp.sign(target - spec_u)
    spec_b = b[shared_nodes:] + delta
    spec_b = spec_b - jnp.mean(spec_b)
    if shared_nodes > 0:
        return jnp.concatenate([jnp.zeros((shared_nodes,), dtype=b.dtype), spec_b], axis=0)
    return spec_b


def grpo_objective(
    params,
    ref_params,
    apply_fn,
    batch,
    rewards,
    config,
    clip_eps: float = 0.2,
    kl_beta: float = 0.04,
    sft_alpha: float = 0.2,
):
    """DeepSeek-R1 critic-free Group Relative Policy Optimization (GRPO) objective."""
    from mlm.model import initial_state

    g_size = batch["inputs"].shape[0]
    carry = initial_state(config, g_size)
    logits, _ = apply_fn(
        {"params": params},
        batch["inputs"],
        carry,
        batch["resets"],
        valid=batch["mask"],
    )
    ref_logits, _ = apply_fn(
        {"params": jax.lax.stop_gradient(ref_params)},
        batch["inputs"],
        carry,
        batch["resets"],
        valid=batch["mask"],
    )
    valid = batch["mask"]
    target_mask = valid * batch.get("loss_mask", valid)
    counts = jnp.maximum(target_mask.sum(axis=-1), 1.0)

    token_nll = optax.softmax_cross_entropy_with_integer_labels(logits, batch["targets"])
    ref_token_nll = optax.softmax_cross_entropy_with_integer_labels(
        jax.lax.stop_gradient(ref_logits), batch["targets"]
    )
    seq_logp = -(token_nll * target_mask).sum(axis=-1) / counts
    ref_seq_logp = -(ref_token_nll * target_mask).sum(axis=-1) / counts

    r_raw = jnp.asarray(rewards, dtype=seq_logp.dtype)
    if r_raw.ndim == 2:
        r_mean_g = jnp.mean(r_raw, axis=-1, keepdims=True)
        r_std_g = jnp.sqrt(jnp.mean(jnp.square(r_raw - r_mean_g), axis=-1, keepdims=True) + 1e-8)
        advantages = jax.lax.stop_gradient(((r_raw - r_mean_g) / (r_std_g + 1e-4)).reshape(-1))
        pos_weights = jnp.where(r_raw > r_mean_g, 1.0, 0.0).reshape(-1)
        r = r_raw.reshape(-1)
        r_mean = jnp.mean(r)
    else:
        r = r_raw
        r_mean = jnp.mean(r)
        r_std = jnp.sqrt(jnp.mean(jnp.square(r - r_mean)) + 1e-8)
        advantages = jax.lax.stop_gradient((r - r_mean) / (r_std + 1e-4))
        pos_weights = jnp.where(r > r_mean, 1.0, 0.0)

    log_ratio = jnp.clip(seq_logp - ref_seq_logp, -10.0, 10.0)
    ratio = jnp.exp(log_ratio)
    clipped_ratio = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    pg_loss = -jnp.mean(jnp.minimum(ratio * advantages, clipped_ratio * advantages))

    # DeepSeek-R1 unbiased non-negative KL estimator: exp(ref - cur) - (ref - cur) - 1
    diff_ref_cur = jnp.clip(ref_seq_logp - seq_logp, -10.0, 10.0)
    kl_div = jnp.mean(jnp.exp(diff_ref_cur) - diff_ref_cur - 1.0)

    pos_denom = jnp.maximum(pos_weights.sum(), 1.0)
    sft_loss = jnp.sum((-seq_logp) * pos_weights) / pos_denom

    total = pg_loss + kl_beta * kl_div + sft_alpha * sft_loss
    metrics = {
        "loss": total,
        "pg_loss": pg_loss,
        "kl_div": kl_div,
        "sft_loss": sft_loss,
        "mean_reward": r_mean,
        "advantages": advantages,
        "seq_logp": seq_logp,
    }
    return total, metrics


def make_grpo_step(config, clip_eps: float = 0.2, kl_beta: float = 0.04, sft_alpha: float = 0.2):
    """Create a JIT-compiled critic-free GRPO update step."""

    @jax.jit
    def step(state, ref_params, batch, rewards):
        (loss, metrics), grads = jax.value_and_grad(grpo_objective, has_aux=True)(
            state.params,
            ref_params,
            state.apply_fn,
            batch,
            rewards,
            config,
            clip_eps,
            kl_beta,
            sft_alpha,
        )
        candidate = state.apply_gradients(grads=grads)
        leaves = jax.tree.leaves((candidate.params, candidate.opt_state, grads, loss))
        finite = jnp.all(jnp.stack([jnp.isfinite(x).all() for x in leaves]))
        new_state = jax.lax.cond(finite, lambda: candidate, lambda: state)
        return new_state, {**metrics, "finite": finite}

    return step


def build_grpo_question_group(question: dict | list[dict], length: int):
    """Build a verifiable GRPO candidate group batch and rewards from a QA item or group."""
    from mlm.data import BOS, EOS

    q_list = [question] if isinstance(question, dict) else list(question)
    rows_meta = []
    reward_rows = []
    for q in q_list:
        prompt = f"<|user|>\n{q['context']}\n{q['question']}\n<|assistant|>\n"
        prompt_bytes = list(prompt.encode("utf-8"))
        if "choices" in q:
            candidates = list(q["choices"])
            correct = candidates[int(q["answer_index"])]
        else:
            candidates = list(q["candidates"])
            correct = q["answer"]
        r_vec = []
        for cand in candidates:
            rows_meta.append((prompt_bytes, list(cand.encode("utf-8"))))
            r_vec.append(1.0 if cand == correct else 0.0)
        reward_rows.append(r_vec)

    total_rows = len(rows_meta)
    inputs = np.full((total_rows, length), EOS, dtype=np.int32)
    targets = np.full((total_rows, length), EOS, dtype=np.int32)
    mask = np.zeros((total_rows, length), dtype=np.float32)
    loss_mask = np.zeros((total_rows, length), dtype=np.float32)
    resets = np.zeros((total_rows, length), dtype=bool)

    for idx, (prompt_bytes, ans_bytes) in enumerate(rows_meta):
        seq = [BOS, *prompt_bytes, *ans_bytes]
        if len(seq) - 1 > length:
            seq = [BOS, *prompt_bytes[: max(1, length - len(ans_bytes) - 1)], *ans_bytes]
        n_pos = min(len(seq) - 1, length)
        inputs[idx, :n_pos] = seq[:n_pos]
        targets[idx, :n_pos] = seq[1 : n_pos + 1]
        mask[idx, :n_pos] = 1.0
        ans_start = max(0, n_pos - len(ans_bytes))
        loss_mask[idx, ans_start:n_pos] = 1.0
        resets[idx, 0] = True

    rewards_arr = (
        np.asarray(reward_rows[0], dtype=np.float32)
        if isinstance(question, dict)
        else np.asarray(reward_rows, dtype=np.float32)
    )
    batch = {
        "inputs": jnp.asarray(inputs),
        "targets": jnp.asarray(targets),
        "mask": jnp.asarray(mask),
        "loss_mask": jnp.asarray(loss_mask),
        "resets": jnp.asarray(resets),
    }
    return batch, jnp.asarray(rewards_arr)
