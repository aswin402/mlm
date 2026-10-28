"""Explicit parameter roles, Muon orthogonalized updates, and recombinative node growth."""

import math

import jax
import jax.numpy as jnp
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
    "memory_output_kernel",
    "memory_gain",
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
    if all(value == 1 for value in scales.values()) and not config.local_plasticity:
        return optimizer
    roles = parameter_roles(params)
    multipliers = {name: scales[role] for name, role in roles.items()}

    def update(updates, state, params=None):
        raw_grads = updates
        updates, state = optimizer.update(updates, state, params)
        if config.local_plasticity:
            activity = node_activity_scales(raw_grads, params or raw_grads)
            updates = jax.tree.map(
                lambda value, scale, act: value * (scale * act),
                updates,
                multipliers,
                activity,
            )
        else:
            updates = jax.tree.map(lambda value, scale: value * scale, updates, multipliers)
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
    new_config = config.model_copy(update={"nodes": n_new})
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
