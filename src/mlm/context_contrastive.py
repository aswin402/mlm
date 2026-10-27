"""Differentiable same-answer context preference; no added inference parameters."""

import math
from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np

from mlm.data import BOS, EOS
from mlm.hierarchy_run import make_routed_step, routed_objective
from mlm.model import initial_state


def candidate_batch(questions, length):
    if not questions or len({q["id"] for q in questions}) != len(questions):
        raise ValueError("Unique nonempty questions required")
    pairs = defaultdict(list)
    for i, q in enumerate(questions):
        if len(q["choices"]) != 2 or len(set(q["choices"])) != 2 or q["answer_index"] not in (0, 1):
            raise ValueError("Two distinct choices and valid label required")
        pairs[q["pair"]].append(i)
    partners = np.zeros(len(questions), np.int32)
    for indices in pairs.values():
        if len(indices) != 2:
            raise ValueError("Complete context-swap pairs required")
        a, b = indices
        x, y = questions[a], questions[b]
        if (
            x["question"] != y["question"]
            or x["choices"] != y["choices"]
            or x["context"] == y["context"]
            or x["answer_index"] == y["answer_index"]
        ):
            raise ValueError("Context-swap identity/label mismatch")
        partners[a] = b
        partners[b] = a
    shape = (2 * len(questions), length)
    inputs = np.full(shape, EOS, np.int32)
    targets = inputs.copy()
    valid = np.zeros(shape, np.float32)
    mask = np.zeros(shape, np.float32)
    resets = np.zeros(shape, bool)
    resets[:, 0] = True
    for i, q in enumerate(questions):
        prompt = f"<|user|>\n{q['context']}\n{q['question']}\n<|assistant|>\n"
        prefix = [BOS, *prompt.encode()]
        for choice, answer in enumerate(q["choices"]):
            if not isinstance(answer, str) or not answer:
                raise ValueError("Nonempty answer strings required")
            tokens = np.array(prefix + list(answer.encode()), np.int32)
            count = len(tokens) - 1
            if count > length:
                raise ValueError("Candidate truncation prohibited")
            row = 2 * i + choice
            inputs[row, :count] = tokens[:-1]
            targets[row, :count] = tokens[1:]
            valid[row, :count] = 1
            mask[row, len(prefix) - 1 : count] = 1
    return {
        "inputs": inputs,
        "targets": targets,
        "mask": valid,
        "resets": resets,
        "answer_mask": mask,
        "gold": np.array([q["answer_index"] for q in questions], np.int32),
        "partners": partners,
    }


def neural_scores(params, apply_fn, batch, config):
    logits, final = apply_fn(
        {"params": params},
        batch["inputs"],
        initial_state(config, batch["inputs"].shape[0]),
        batch["resets"],
        valid=batch["mask"],
    )
    chosen = jnp.take_along_axis(jax.nn.log_softmax(logits), batch["targets"][..., None], -1)[
        ..., 0
    ]
    scores = (chosen * batch["answer_mask"]).sum(-1) / jnp.maximum(batch["answer_mask"].sum(-1), 1)
    return scores.reshape((-1, 2)), final


def context_objective(params, apply_fn, batch, config):
    scores, final = neural_scores(params, apply_fn, batch, config)
    rows = jnp.arange(scores.shape[0])
    gold = batch["gold"]
    gaps = scores[rows, gold] - scores[batch["partners"], gold]
    loss = jax.nn.softplus(-gaps).mean()
    return loss, (
        final,
        {
            "context_loss": loss,
            "context_gap": gaps.mean(),
            "context_terms": gaps.size,
            "candidate_positions": batch["answer_mask"].sum(),
        },
    )


def make_context_step(config, weight):
    if not math.isfinite(weight) or not 0 <= weight <= 100:
        raise ValueError("Finite weight in [0,100] required")
    legacy = make_routed_step(config)

    def objective(params, apply_fn, carry, batch, candidates):
        main, (final, m) = routed_objective(
            params, apply_fn, batch, jax.lax.stop_gradient(carry), config
        )
        extra, (candidate_final, cm) = context_objective(params, apply_fn, candidates, config)
        return main + weight * extra, (
            final,
            candidate_final,
            {**m, **cm, "main_objective": main, "objective": main + weight * extra},
        )

    @jax.jit
    def active(source, carry, batch, candidates):
        (_, (final, cf, m)), grad = jax.value_and_grad(objective, has_aux=True)(
            source.params, source.apply_fn, carry, batch, candidates
        )
        updated = source.apply_gradients(grads=grad)
        leaves = jax.tree.leaves((updated.params, updated.opt_state, grad, final, cf, m))
        finite = jnp.all(jnp.stack([jnp.isfinite(x).all() for x in leaves]))
        accepted = finite & (m["positions"] > 0)
        return (
            jax.lax.cond(accepted, lambda: updated, lambda: source),
            jax.lax.cond(finite, lambda: final, lambda: carry),
            {**m, "finite": finite, "updated": accepted},
        )

    def step(source, carry, batch, candidates):
        if weight:
            result = active(source, carry, batch, candidates)
            return result if bool(result[2]["updated"]) else (source, result[1], result[2])
        return legacy(source, carry, batch)

    return step


def gradient_diagnostic(source, config, batch, candidates):
    carry = initial_state(config, batch["inputs"].shape[0])

    def main(params):
        return routed_objective(params, source.apply_fn, batch, carry, config)[0]

    def extra(params):
        return context_objective(params, source.apply_fn, candidates, config)[0]

    mg = jax.jit(jax.grad(main))(source.params)
    cg = jax.jit(jax.grad(extra))(source.params)

    def norm(tree):
        return float(
            np.sqrt(
                sum(np.square(np.asarray(x, dtype=np.float64)).sum() for x in jax.tree.leaves(tree))
            )
        )

    if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves((mg, cg))):
        raise FloatingPointError("Nonfinite gradient diagnostic")
    m, c = norm(mg), norm(cg)
    return {"main_gradient_norm": m, "context_gradient_norm": c, "ratio": c / m if m else None}
