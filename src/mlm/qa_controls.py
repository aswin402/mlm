"""Independent copied QA steps crossing optimizer history and router regularization."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from mlm.hierarchy_run import routed_objective
from mlm.model import initial_state


def make_qa_controls(config):
    plain = config.model_copy(update={"router_balance": 0.0})

    @jax.jit
    def evaluate(state, selected, aggregate):
        _, (selected_carry, m) = routed_objective(
            state.params,
            state.apply_fn,
            selected,
            initial_state(config, selected["inputs"].shape[0]),
            config,
        )
        _, (aggregate_carry, a) = routed_objective(
            state.params,
            state.apply_fn,
            aggregate,
            initial_state(config, aggregate["inputs"].shape[0]),
            config,
        )
        weight = config.router_balance if config.node_update == "hierarchical" else 0.0
        return {
            "forward_finite": jnp.all(
                jnp.stack(
                    [
                        jnp.isfinite(x).all()
                        for x in jax.tree.leaves((selected_carry, aggregate_carry, m, a))
                    ]
                )
            ),
            "selected_language": m["loss"],
            "router_penalty": m["router_balance"],
            "configured_objective": m["loss"] + weight * m["router_balance"],
            "aggregate_language": a["loss"],
            "selected_positions": m["positions"],
            "aggregate_positions": a["positions"],
        }

    def make_compute(cfg):
        @jax.jit
        def compute(base, batch):
            (_, (carry, m)), grad = jax.value_and_grad(routed_objective, has_aux=True)(
                base.params,
                base.apply_fn,
                batch,
                initial_state(cfg, batch["inputs"].shape[0]),
                cfg,
            )
            candidate = base.apply_gradients(grads=grad)
            finite = jnp.all(
                jnp.stack(
                    [
                        jnp.isfinite(x).all()
                        for x in jax.tree.leaves(
                            (carry, m, grad, candidate.params, candidate.opt_state)
                        )
                    ]
                )
            )
            return candidate, grad, finite

        return compute

    steps = {"configured": make_compute(config), "language": make_compute(plain)}

    def probe(source, aggregate, selected):
        before = {k: float(v) for k, v in evaluate(source, selected, aggregate).items()}
        if not before["forward_finite"] or not all(np.isfinite(v) for v in before.values()):
            raise FloatingPointError("Nonfinite QA baseline")
        if min(before["selected_positions"], before["aggregate_positions"]) <= 0:
            raise ValueError("Both QA batches require supervised positions")
        states: dict[str, Any] = {}
        branches: dict[str, Any] = {}
        for history in ["loaded", "fresh"]:
            base = (
                source
                if history == "loaded"
                else source.replace(opt_state=source.tx.init(source.params))
            )
            for mode in ["configured", "language", "zero"]:
                if mode == "zero":
                    grad = jax.tree.map(jnp.zeros_like, base.params)
                    candidate = base.apply_gradients(grads=grad)
                else:
                    candidate, grad, finite = steps[mode](base, selected)
                    if not bool(finite):
                        raise FloatingPointError("Nonfinite QA forward or update")
                after = {k: float(v) for k, v in evaluate(candidate, selected, aggregate).items()}
                after["active_objective"] = (
                    after["selected_language"]
                    if mode == "language"
                    else after["configured_objective"]
                )
                leaves = jax.tree.leaves((candidate.params, candidate.opt_state, grad, after))
                if not after["forward_finite"] or not all(
                    np.isfinite(np.asarray(x)).all() for x in leaves
                ):
                    raise FloatingPointError("Nonfinite QA control")

                def norm(tree):
                    return float(
                        np.sqrt(
                            sum(
                                np.square(np.asarray(x, dtype=np.float64)).sum()
                                for x in jax.tree.leaves(tree)
                            )
                        )
                    )

                delta = jax.tree.map(
                    lambda a, b: np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64),
                    candidate.params,
                    source.params,
                )
                name = history + "_" + mode
                states[name] = candidate
                branches[name] = {
                    "after": after,
                    "selected_change": after["selected_language"] - before["selected_language"],
                    "aggregate_change": after["aggregate_language"] - before["aggregate_language"],
                    "configured_objective_change": after["configured_objective"]
                    - before["configured_objective"],
                    "active_objective_change": after["active_objective"]
                    - (
                        before["selected_language"]
                        if mode == "language"
                        else before["configured_objective"]
                    ),
                    "gradient_norm": norm(grad),
                    "update_norm": norm(delta),
                    "step_before": int(base.step),
                    "step_after": int(candidate.step),
                }
        contrasts = {}
        for metric in ["selected_change", "aggregate_change"]:
            values = {name: m[metric] for name, m in branches.items()}
            regularizer = {
                h: values[h + "_configured"] - values[h + "_language"] for h in ["loaded", "fresh"]
            }
            contrasts[metric] = {
                "regularizer_configured_minus_language": regularizer,
                "history_loaded_minus_fresh": {
                    m: values["loaded_" + m] - values["fresh_" + m]
                    for m in ["configured", "language", "zero"]
                },
                "interaction": regularizer["loaded"] - regularizer["fresh"],
                "minus_zero": {
                    h + "_" + m: values[h + "_" + m] - values[h + "_zero"]
                    for h in ["loaded", "fresh"]
                    for m in ["configured", "language"]
                },
            }
        return states, {"before": before, "branches": branches, "contrasts": contrasts}

    return probe
