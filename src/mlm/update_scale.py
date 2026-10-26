"""Counterfactual final-update scaling with loaded optimizer history; no rollout."""

import jax
import jax.numpy as jnp
import numpy as np

from mlm.hierarchy_run import make_routed_step, routed_objective
from mlm.model import initial_state

SCALES = (0.0, 0.125, 0.25, 0.5, 1.0)


def make_update_scale_probe(config):
    def finite(tree):
        return all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(tree))

    @jax.jit
    def evaluate(state, selected, aggregate):
        objective, (carry, m) = routed_objective(
            state.params,
            state.apply_fn,
            selected,
            initial_state(config, selected["inputs"].shape[0]),
            config,
        )
        _, (agg_carry, a) = routed_objective(
            state.params,
            state.apply_fn,
            aggregate,
            initial_state(config, aggregate["inputs"].shape[0]),
            config,
        )
        return {
            "selected_language": m["loss"],
            "configured_objective": objective,
            "router_penalty": m["router_balance"],
            "aggregate_language": a["loss"],
            "selected_positions": m["positions"],
            "aggregate_positions": a["positions"],
        }, (carry, agg_carry, m, a)

    train = make_routed_step(config)

    def values(state, selected, aggregate):
        m, aux = evaluate(state, selected, aggregate)
        if not finite((state.params, state.opt_state, m, aux)):
            raise FloatingPointError("Nonfinite scaled-update evaluation")
        return {k: float(v) for k, v in m.items()}

    def probe(source, aggregate, selected):
        before = values(source, selected, aggregate)
        if min(before["selected_positions"], before["aggregate_positions"]) <= 0:
            raise ValueError("Both QA batches require supervised positions")
        qa, carry, metrics = train(
            source, initial_state(config, selected["inputs"].shape[0]), selected
        )
        zero = source.apply_gradients(grads=jax.tree.map(jnp.zeros_like, source.params))
        if (
            not bool(metrics["finite"])
            or not bool(metrics["updated"])
            or not finite((qa.params, qa.opt_state, zero.params, zero.opt_state, carry, metrics))
        ):
            raise FloatingPointError("Nonfinite loaded optimizer step")
        states, branches = {}, {}
        for direction, endpoint in [("qa", qa), ("zero", zero)]:
            for i, scale in enumerate(SCALES):
                params = (
                    source.params
                    if scale == 0
                    else endpoint.params
                    if scale == 1
                    else jax.tree.map(
                        lambda a, b: a + scale * (b - a),
                        source.params,
                        endpoint.params,
                    )
                )
                candidate = endpoint.replace(params=params)
                after = values(candidate, selected, aggregate)
                delta = jax.tree.map(
                    lambda a, b: np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64),
                    params,
                    source.params,
                )
                norm = float(np.sqrt(sum(np.square(x).sum() for x in jax.tree.leaves(delta))))
                name = f"{direction}_{i}"
                states[name] = candidate
                branches[name] = {
                    "scale": scale,
                    "after": after,
                    "update_norm": norm,
                    "selected_change": after["selected_language"] - before["selected_language"],
                    "aggregate_change": after["aggregate_language"] - before["aggregate_language"],
                    "objective_change": after["configured_objective"]
                    - before["configured_objective"],
                    "step_before": int(source.step),
                    "step_after": int(candidate.step),
                }
        controls = {
            str(i): {
                key: branches[f"qa_{i}"][key] - branches[f"zero_{i}"][key]
                for key in ["selected_change", "aggregate_change", "objective_change"]
            }
            for i in range(len(SCALES))
        }
        return states, {"before": before, "branches": branches, "qa_minus_zero": controls}

    return probe
