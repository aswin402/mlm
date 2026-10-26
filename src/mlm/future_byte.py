"""Optional disposable rank16 head for second-upcoming-byte supervision."""

import math

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from mlm.data import VOCAB_SIZE
from mlm.hierarchy_run import make_routed_step, routed_objective


def future_logits(params, logits):
    features = jax.nn.log_softmax(logits, axis=-1)
    return jnp.tanh(features @ params["down"]) @ params["up"] + params["bias"]


def create_head(config, seed=9042):
    a, b = jax.random.split(jax.random.PRNGKey(seed))
    params = {
        "down": jax.random.normal(a, (VOCAB_SIZE, 16)) * 0.01,
        "up": jax.random.normal(b, (16, VOCAB_SIZE)) * 0.01,
        "bias": jnp.zeros((VOCAB_SIZE,)),
    }
    tx = optax.chain(
        optax.clip_by_global_norm(config.clip_norm),
        optax.adamw(
            config.learning_rate * config.shared_lr_scale,
            weight_decay=config.weight_decay,
            mask={"down": True, "up": True, "bias": False},
        ),
    )
    return TrainState.create(apply_fn=future_logits, params=params, tx=tx)


def future_mask(batch):
    supervision = batch["mask"] * batch.get("loss_mask", batch["mask"])
    return (
        supervision[:, :-1]
        * supervision[:, 1:]
        * (batch["targets"][:, :-1] < 256)
        * (batch["targets"][:, 1:] < 256)
        * ~batch["resets"][:, 1:]
    )


def auxiliary_loss(params, logits, batch):
    predicted = future_logits(params, logits[:, :-1])
    losses = optax.softmax_cross_entropy_with_integer_labels(predicted, batch["targets"][:, 1:])
    mask = future_mask(batch)
    count = mask.sum()
    return (losses * mask).sum() / jnp.maximum(count, 1), count


def future_objective(params, head_params, apply_fn, carry, batch, config, weight):
    main, (final, metrics) = routed_objective(params, apply_fn, batch, carry, config)
    logits, _ = apply_fn(
        {"params": params}, batch["inputs"], carry, batch["resets"], valid=batch["mask"]
    )
    aux, count = auxiliary_loss(head_params, logits, batch)
    return main + weight * aux, (
        final,
        {
            **metrics,
            "main_objective": main,
            "aux_loss": aux,
            "aux_positions": count,
            "objective": main + weight * aux,
        },
    )


def make_future_step(config, weight):
    if not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("Finite auxiliary weight in [0,1] required")
    legacy = make_routed_step(config)

    @jax.jit
    def stats(source, head, carry, batch):
        return future_objective(
            source.params,
            head.params,
            source.apply_fn,
            jax.lax.stop_gradient(carry),
            batch,
            config,
            weight,
        )[1]

    @jax.jit
    def active(source, head, carry, batch):
        (_, (final, metrics)), (g, hg) = jax.value_and_grad(
            future_objective, argnums=(0, 1), has_aux=True
        )(
            source.params,
            head.params,
            source.apply_fn,
            jax.lax.stop_gradient(carry),
            batch,
            config,
            weight,
        )
        candidate = source.apply_gradients(grads=g)
        next_head = head.apply_gradients(grads=hg)
        leaves = jax.tree.leaves(
            (
                candidate.params,
                candidate.opt_state,
                next_head.params,
                next_head.opt_state,
                g,
                hg,
                final,
                metrics,
            )
        )
        finite = jnp.all(jnp.stack([jnp.isfinite(x).all() for x in leaves]))
        updated = finite & (metrics["positions"] > 0)
        return (
            jax.lax.cond(updated, lambda: candidate, lambda: source),
            jax.lax.cond(updated, lambda: next_head, lambda: head),
            jax.lax.cond(finite, lambda: final, lambda: carry),
            {**metrics, "finite": finite, "updated": updated},
        )

    def step(source, head, carry, batch):
        if weight:
            result = active(source, head, carry, batch)
            if not bool(result[3]["updated"]):
                return source, head, result[2], result[3]
            return result
        candidate, final, metrics = legacy(source, carry, batch)
        _, extra = stats(source, head, carry, batch)
        next_head = head.apply_gradients(grads=jax.tree.map(jnp.zeros_like, head.params))
        finite = bool(metrics["finite"]) and all(
            bool(jnp.isfinite(x).all())
            for x in jax.tree.leaves((extra, next_head.params, next_head.opt_state))
        )
        updated = finite and bool(metrics["updated"])
        return (
            candidate if updated else source,
            next_head if updated else head,
            final if finite else carry,
            {**extra, "finite": jnp.asarray(finite), "updated": jnp.asarray(updated)},
        )

    return step
