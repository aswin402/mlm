"""Frozen training-only replay probes and independent copied optimizer-step diagnostics."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from mlm.data import EOS
from mlm.hierarchy_run import routed_objective
from mlm.model import initial_state
from mlm.plasticity import parameter_roles
from mlm.stream import DocumentStream


def replay_probes(documents, batch_size, length, seed, positions):
    positions = list(positions)
    if not positions or any(type(p) is not int or p < 0 for p in positions):
        raise ValueError("Nonnegative integer probe positions required")
    if positions != sorted(set(positions)):
        raise ValueError("Strictly increasing probe positions required")
    selected = set(positions)
    stream = DocumentStream(documents, batch_size, length, seed)
    captured: list[dict[str, Any]] = []
    for position in range(positions[-1] + 1):
        if position in selected:
            before = stream.snapshot()
            prefixes = [
                np.array(stream.tokens[lane["document"]][: lane["offset"]], copy=True)
                if lane is not None
                else np.array([], dtype=np.int32)
                for lane in stream.lanes
            ]
        batch = stream.next_batch()
        if position in selected:
            captured.append(
                {
                    "position": position,
                    "batch": batch,
                    "prefixes": prefixes,
                    "metadata": {
                        "snapshot_before": before,
                        "snapshot_after": stream.snapshot(),
                        "prefix_token_counts": [len(p) for p in prefixes],
                    },
                }
            )
    padded_length = max(1, max(len(p) for row in captured for p in row["prefixes"]))
    for row in captured:
        inputs = np.full((batch_size, padded_length), EOS, np.int32)
        mask = np.zeros_like(inputs, dtype=np.float32)
        resets = np.zeros_like(inputs, dtype=bool)
        for lane, tokens in enumerate(row.pop("prefixes")):
            inputs[lane, : len(tokens)] = tokens
            mask[lane, : len(tokens)] = 1
            resets[lane, 0] = len(tokens) > 0
        row["prefix"] = {"inputs": inputs, "mask": mask, "resets": resets}
    return captured


def make_prefix_reader(config):
    @jax.jit
    def read(state, prefix):
        _, carry = state.apply_fn(
            {"params": state.params},
            prefix["inputs"],
            initial_state(config, prefix["inputs"].shape[0]),
            prefix["resets"],
            valid=prefix["mask"],
        )
        return jax.lax.stop_gradient(carry)

    return read


def _vector_metrics(names, q, e, delta, zero_delta):
    def dot(left, right):
        return float(sum(np.vdot(left[k], right[k]) for k in names))

    qnorm, enorm = np.sqrt(dot(q, q)), np.sqrt(dot(e, e))
    gradient_dot = dot(q, e)
    return {
        "parameter_count": sum(q[k].size for k in names),
        "qa_gradient_norm": float(qnorm),
        "english_gradient_norm": float(enorm),
        "gradient_dot": gradient_dot,
        "gradient_cosine": float(gradient_dot / (qnorm * enorm)) if qnorm * enorm > 0 else None,
        "update_norm": float(np.sqrt(dot(delta, delta))),
        "zero_update_norm": float(np.sqrt(dot(zero_delta, zero_delta))),
        "first_order_qa_change": dot(q, delta),
        "zero_first_order_qa_change": dot(q, zero_delta),
    }


def make_interference_probe(config):
    def qa_language(params, apply_fn, batch):
        _, (_, metrics) = routed_objective(
            params, apply_fn, batch, initial_state(config, batch["inputs"].shape[0]), config
        )
        return metrics["loss"], metrics["positions"]

    @jax.jit
    def compute(state, qa_batch, english_batch, english_carry):
        (qa_before, qa_positions), qgrad = jax.value_and_grad(qa_language, has_aux=True)(
            state.params, state.apply_fn, qa_batch
        )
        (english_before, (final_carry, english_metrics)), egrad = jax.value_and_grad(
            routed_objective, has_aux=True
        )(state.params, state.apply_fn, english_batch, jax.lax.stop_gradient(english_carry), config)
        candidate = state.apply_gradients(grads=egrad)
        zero = state.apply_gradients(grads=jax.tree.map(jnp.zeros_like, egrad))
        qa_after, _ = qa_language(candidate.params, state.apply_fn, qa_batch)
        qa_zero, _ = qa_language(zero.params, state.apply_fn, qa_batch)
        english_after, (_, after_metrics) = routed_objective(
            candidate.params,
            state.apply_fn,
            english_batch,
            jax.lax.stop_gradient(english_carry),
            config,
        )
        metrics = {
            "qa_loss_before": qa_before,
            "qa_loss_after_english": qa_after,
            "qa_loss_after_zero": qa_zero,
            "qa_supervised_positions": qa_positions,
            "english_supervised_positions": english_metrics["positions"],
            "english_objective_before": english_before,
            "english_objective_after": english_after,
            "english_language_loss_before": english_metrics["loss"],
            "english_language_loss_after": after_metrics["loss"],
        }
        leaves = jax.tree.leaves(
            (
                candidate.params,
                candidate.opt_state,
                zero.params,
                zero.opt_state,
                qgrad,
                egrad,
                metrics,
                final_carry,
                english_carry,
            )
        )
        finite = jnp.all(jnp.stack([jnp.isfinite(x).all() for x in leaves]))
        return candidate, zero, qgrad, egrad, metrics, finite

    def probe(state, qa_batch, english_batch, english_carry):
        candidate, zero, qgrad, egrad, raw, finite = compute(
            state, qa_batch, english_batch, english_carry
        )
        if not bool(finite):
            raise FloatingPointError("Nonfinite frozen optimizer diagnostic")
        result: dict[str, Any] = {name: float(value) for name, value in jax.device_get(raw).items()}
        if result["qa_supervised_positions"] <= 0 or result["english_supervised_positions"] <= 0:
            raise ValueError("Both objectives require supervised positions")
        q = {name: np.asarray(value, dtype=np.float64) for name, value in qgrad.items()}
        e = {name: np.asarray(value, dtype=np.float64) for name, value in egrad.items()}
        delta = {
            name: np.asarray(candidate.params[name], dtype=np.float64)
            - np.asarray(value, dtype=np.float64)
            for name, value in state.params.items()
        }
        zero_delta = {
            name: np.asarray(zero.params[name], dtype=np.float64)
            - np.asarray(value, dtype=np.float64)
            for name, value in state.params.items()
        }
        result.update(_vector_metrics(list(q), q, e, delta, zero_delta))
        roles = parameter_roles(state.params)
        result["roles"] = {
            role: _vector_metrics(
                [name for name in q if roles[name] == role], q, e, delta, zero_delta
            )
            for role in ["shared", "node", "router", "memory"]
        }
        result["qa_loss_change"] = result["qa_loss_after_english"] - result["qa_loss_before"]
        result["zero_qa_loss_change"] = result["qa_loss_after_zero"] - result["qa_loss_before"]
        result["english_minus_zero_qa_loss"] = (
            result["qa_loss_after_english"] - result["qa_loss_after_zero"]
        )
        result["first_order_remainder"] = result["qa_loss_change"] - result["first_order_qa_change"]
        result["optimizer_step_before"] = int(state.step)
        result["optimizer_step_after"] = int(candidate.step)
        result["zero_optimizer_step_after"] = int(zero.step)
        return candidate, zero, result

    return probe
