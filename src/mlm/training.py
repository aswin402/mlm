"""Masked causal prediction and clipped AdamW updates."""

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from mlm.config import Config
from mlm.model import initial_state, make_model
from mlm.plasticity import make_muon_adamw, scale_optimizer


def decay_mask(params):
    """Select weight matrices (rank >= 2); exclude scalars, 1D vectors, and biases."""
    return {
        name: getattr(value, "ndim", 0) >= 2 and "bias" not in name
        for name, value in params.items()
    }


def create_state(config: Config):
    model = make_model(config)
    tokens = jnp.zeros((1, 1), dtype=jnp.int32)
    params = model.init(
        jax.random.PRNGKey(config.seed),
        tokens,
        initial_state(config, 1),
        jnp.ones_like(tokens, dtype=bool),
    )["params"]
    decay = decay_mask(params)
    if config.optimizer == "muon":
        core_tx = make_muon_adamw(params, config, decay)
    else:
        core_tx = optax.adamw(
            config.learning_rate,
            b1=0.9,
            b2=0.999,
            eps=1e-8,
            weight_decay=config.weight_decay,
            mask=decay,
        )
    optimizer = optax.chain(optax.clip_by_global_norm(config.clip_norm), core_tx)
    optimizer = scale_optimizer(optimizer, params, config)
    return TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)


def objective(params, apply_fn, batch, config, messages=True):
    use_ponder = config.adaptive_halt and config.ponder_beta > 0
    if use_ponder:
        logits, _, ponder_kl = apply_fn(
            {"params": params},
            batch["inputs"],
            initial_state(config, batch["inputs"].shape[0]),
            batch["resets"],
            messages=messages,
            return_ponder=True,
        )
    else:
        logits, _ = apply_fn(
            {"params": params},
            batch["inputs"],
            initial_state(config, batch["inputs"].shape[0]),
            batch["resets"],
            messages=messages,
        )
    losses = optax.softmax_cross_entropy_with_integer_labels(logits, batch["targets"])
    mask = batch["mask"]
    denom = jnp.maximum(mask.sum(), 1)
    loss = (losses * mask).sum() / denom
    ponder_loss = (ponder_kl * mask).sum() / denom if use_ponder else jnp.asarray(0.0, loss.dtype)
    payload = mask * (batch["targets"] < 256)
    byte_nll = (losses * payload).sum()
    return loss + config.ponder_beta * ponder_loss, {
        "loss": loss,
        "byte_nll": byte_nll,
        "bytes": payload.sum(),
        "positions": mask.sum(),
    }


def score(state, batch, config, messages=True):
    return objective(state.params, state.apply_fn, batch, config, messages)[1]


def make_train_step(config: Config):
    @jax.jit
    def step(state, batch):
        (loss, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(
            state.params, state.apply_fn, batch, config
        )
        finite = jnp.isfinite(loss) & jnp.all(
            jnp.stack([jnp.all(jnp.isfinite(g)) for g in jax.tree.leaves(gradients)])
        )
        candidate = state.apply_gradients(grads=gradients)
        finite = finite & jnp.all(
            jnp.stack([jnp.all(jnp.isfinite(p)) for p in jax.tree.leaves(candidate.params)])
        )
        state = jax.lax.cond(finite, lambda: candidate, lambda: state)
        return state, {**metrics, "grad_norm": optax.tree.norm(gradients), "finite": finite}

    return step
