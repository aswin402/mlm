"""Freeze inputs and evaluate bounded final-update fractions on selected QA snapshots."""

import argparse
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from mlm.checkpoint import load_checkpoint
from mlm.cli import write_json
from mlm.interference_run import SOURCES, digest, state_digest
from mlm.pair_probe import restore_next_pair
from mlm.pair_probe_run import execution_settings, json_hash, prepare, read, selected
from mlm.update_scale import SCALES, make_update_scale_probe

PRIOR = Path("docs/evidence/qa-step-controls/protocol.json")
METHOD = Path("docs/experiments/qa-update-scale/protocol.md")
BRANCHES = {f"{d}_{i}" for d in ["qa", "zero"] for i in range(len(SCALES))}


def intact(p):
    return {name: digest(name) == sha for name, sha in p["sha256"].items()}


def freeze(out):
    if out.exists():
        raise ValueError("Use fresh output directory")
    prior = read(PRIOR)
    if not all(intact(prior).values()):
        raise ValueError("Prior inputs changed")
    if {k: v["path"] for k, v in prior["checkpoints"].items()} != SOURCES:
        raise ValueError("Source selection mismatch")
    for label, d in prior["checkpoints"].items():
        state, cfg, meta = load_checkpoint(Path(d["path"]))
        batches, ids, _, docs = prepare(cfg)
        context = restore_next_pair(cfg, meta, batches, ids, docs)
        if digest(d["path"]) != d["file_sha256"] or state_digest(state) != d["state_sha256"]:
            raise ValueError("Source identity mismatch")
        if json_hash(meta) != d["metadata_sha256"] or selected(context) != d["next"]:
            raise ValueError("Next QA provenance mismatch")
    paths = [
        *map(Path, prior["sha256"]),
        PRIOR,
        METHOD,
        *Path("src/mlm").glob("*.py"),
        *Path("tests").glob("*.py"),
        *Path("docs/evidence/qa-step-controls").glob("checkpoint-*.json"),
    ]
    p = {
        "mode": "bounded_loaded_final_update",
        "scales": list(SCALES),
        "optimizer_computations": 10,
        "scaled_evaluations": 50,
        "checkpoints": prior["checkpoints"],
        "settings": execution_settings(),
        "sha256": {str(f): digest(f) for f in dict.fromkeys(paths)},
    }
    out.mkdir(parents=True)
    write_json(out / "protocol.json", p)
    print(
        json.dumps({"phase": "frozen", "inputs": len(p["sha256"]), "evaluations": 50}), flush=True
    )


def run(out):
    p = read(out / "protocol.json")
    if (out / "execution.json").exists() or list(out.glob("checkpoint-*.json")):
        raise ValueError("Preserve existing outputs")
    prior = read(PRIOR)
    if not all(intact(p).values()) or execution_settings() != p["settings"]:
        raise ValueError("Frozen inputs/settings changed")
    if (
        p["checkpoints"] != prior["checkpoints"]
        or p["scales"] != list(SCALES)
        or p["optimizer_computations"] != 10
        or p["scaled_evaluations"] != 50
    ):
        raise ValueError("Frozen design changed")
    phash = digest(out / "protocol.json")
    write_json(out / "execution.json", {"settings": execution_settings(), "protocol_sha256": phash})
    audit = {
        "sources": {k: v["path"] for k, v in p["checkpoints"].items()} == SOURCES,
        "prior_inputs": all(p["sha256"].get(k) == v for k, v in prior["sha256"].items()),
    }
    results = {}
    for label, d in p["checkpoints"].items():
        source, cfg, meta = load_checkpoint(Path(d["path"]))
        batches, ids, aggregate, docs = prepare(cfg)
        context = restore_next_pair(cfg, meta, batches, ids, docs)
        audit[label + ":source"] = (
            digest(d["path"]) == d["file_sha256"] and state_digest(source) == d["state_sha256"]
        )
        audit[label + ":metadata"] = json_hash(meta) == d["metadata_sha256"]
        audit[label + ":selection"] = selected(context) == d["next"]
        audit[label + ":budget"] = (
            int(source.step) == context["qa_updates_before"] + context["english_updates_before"]
        )
        audit[label + ":positions"] = int((aggregate["mask"] * aggregate["loss_mask"]).sum()) == 96
        if not all(audit.values()):
            raise ValueError("Source preflight failed")
        states, m = make_update_scale_probe(cfg)(
            source,
            jax.tree.map(jnp.asarray, aggregate),
            jax.tree.map(jnp.asarray, context["qa_batch"]),
        )
        audit[label + ":branches"] = set(states) == BRANCHES and set(m["branches"]) == BRANCHES
        old = read(Path("docs/evidence/qa-step-controls") / f"checkpoint-{label}.json")["metrics"]
        for direction, oldname in [("qa", "loaded_configured"), ("zero", "loaded_zero")]:
            endpoint = states[f"{direction}_4"]
            for i, scale in enumerate(SCALES):
                name = f"{direction}_{i}"
                state, b = states[name], m["branches"][name]
                after = b["after"]
                audit[label + ":" + name + ":scale"] = b["scale"] == scale
                audit[label + ":" + name + ":step"] = (
                    b["step_before"] == int(source.step) and b["step_after"] == int(source.step) + 1
                )
                audit[label + ":" + name + ":optimizer"] = serialization.to_bytes(
                    state.opt_state
                ) == serialization.to_bytes(endpoint.opt_state)
                expected = (
                    source.params
                    if scale == 0
                    else endpoint.params
                    if scale == 1
                    else jax.tree.map(
                        lambda a, z: a + scale * (z - a), source.params, endpoint.params
                    )
                )
                audit[label + ":" + name + ":displacement"] = all(
                    np.array_equal(a, z)
                    for a, z in zip(
                        jax.tree.leaves(state.params), jax.tree.leaves(expected), strict=True
                    )
                )
                audit[label + ":" + name + ":finite"] = all(
                    math.isfinite(float(x)) for x in jax.tree.leaves(b)
                )
                audit[label + ":" + name + ":objective"] = (
                    abs(
                        after["configured_objective"]
                        - after["selected_language"]
                        - cfg.router_balance * after["router_penalty"]
                    )
                    < 1e-6
                )
                for delta, key in [
                    ("selected_change", "selected_language"),
                    ("aggregate_change", "aggregate_language"),
                    ("objective_change", "configured_objective"),
                ]:
                    audit[label + ":" + name + ":" + delta] = (
                        abs(b[delta] - after[key] + m["before"][key]) < 1e-9
                    )
                if scale == 0:
                    audit[label + ":" + name + ":baseline"] = b["update_norm"] == 0 and all(
                        b[k] == 0
                        for k in ["selected_change", "aggregate_change", "objective_change"]
                    )
                if scale == 1:
                    for key in ["selected_language", "aggregate_language", "configured_objective"]:
                        audit[label + ":" + name + ":prior_" + key] = (
                            abs(after[key] - old["branches"][oldname]["after"][key]) <= 1e-6
                        )
            for i in range(len(SCALES)):
                audit[label + ":control_" + str(i)] = all(
                    abs(
                        m["qa_minus_zero"][str(i)][key]
                        - m["branches"][f"qa_{i}"][key]
                        + m["branches"][f"zero_{i}"][key]
                    )
                    < 1e-12
                    for key in ["selected_change", "aggregate_change", "objective_change"]
                )
        audit[label + ":immutable"] = (
            state_digest(source) == d["state_sha256"]
            and digest(d["path"]) == d["file_sha256"]
            and json_hash(meta) == d["metadata_sha256"]
        )
        if not all(audit.values()):
            raise ValueError("Audit failure: " + str([k for k, v in audit.items() if not v]))
        results[label] = {"source": d, "metrics": m}
        write_json(out / f"checkpoint-{label}.json", results[label])
        print(
            json.dumps(
                {
                    "checkpoint": label,
                    "qa_aggregate_changes": [
                        m["branches"][f"qa_{i}"]["aggregate_change"] for i in range(len(SCALES))
                    ],
                }
            ),
            flush=True,
        )
    audit["all_sources"] = set(results) == set(SOURCES)
    audit["protocol_immutable"] = phash == digest(out / "protocol.json")
    audit["settings_immutable"] = execution_settings() == p["settings"]
    audit.update({"input:" + k: v for k, v in intact(p).items()})
    if not all(audit.values()):
        raise ValueError("Final integrity check failed")
    write_json(
        out / "summary.json",
        {"checkpoints": results, "optimizer_computations": 10, "scaled_evaluations": 50},
    )
    write_json(out / "audit.json", audit)
    print(json.dumps({"phase": "complete", "audits": len(audit), "all_passed": True}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["freeze", "run"])
    parser.add_argument("--out", type=Path, default=Path("results/qa-update-scale"))
    args = parser.parse_args()
    (freeze if args.phase == "freeze" else run)(args.out)


if __name__ == "__main__":
    main()
