"""Explicit parameter roles and effective AdamW rate multipliers."""

import jax
import jax.numpy as jnp
import optax

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
