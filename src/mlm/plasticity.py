"""Explicit parameter roles and effective AdamW rate multipliers."""

import jax
import optax

NODE_PARAMETERS = {"input_kernel", "hidden_kernel", "gru_bias", "read_kernel"}
SHARED_PARAMETERS = {
    "embedding",
    "message_kernel",
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


def parameter_roles(params):
    roles = {}
    for name in params:
        if name in NODE_PARAMETERS:
            roles[name] = "node"
        elif name in SHARED_PARAMETERS:
            roles[name] = "shared"
        elif name in ROUTER_PARAMETERS:
            roles[name] = "router"
        elif name in MEMORY_PARAMETERS:
            roles[name] = "memory"
        else:
            raise ValueError(f"Unknown parameter role: {name}")
    return roles


def role_scales(config):
    return {
        "shared": config.shared_lr_scale,
        "node": config.node_lr_scale,
        "router": config.router_lr_scale,
        "memory": config.memory_lr_scale,
    }


def scale_optimizer(optimizer, params, config):
    """Scale final updates, including decay, without changing moments or state shape."""
    scales = role_scales(config)
    if all(value == 1 for value in scales.values()):
        return optimizer
    roles = parameter_roles(params)
    multipliers = {name: scales[role] for name, role in roles.items()}

    def update(updates, state, params=None):
        updates, state = optimizer.update(updates, state, params)
        updates = jax.tree.map(lambda value, scale: value * scale, updates, multipliers)
        return updates, state

    return optax.GradientTransformation(optimizer.init, update)
