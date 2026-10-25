"""Freeze and run six independent QA controls at each prior selected snapshot."""

import argparse
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp

from mlm.checkpoint import load_checkpoint
from mlm.cli import write_json
from mlm.interference_run import SOURCES, digest, state_digest
from mlm.pair_probe import restore_next_pair
from mlm.pair_probe_run import execution_settings, json_hash, prepare, read, selected
from mlm.qa_controls import make_qa_controls

PRIOR = Path("docs/evidence/qa-replay-pair/protocol.json")
METHOD = Path("docs/experiments/qa-step-controls/protocol.md")
BRANCHES = {h + "_" + m for h in ["loaded", "fresh"] for m in ["configured", "language", "zero"]}


def check_inputs(protocol):
    return {p: digest(p) == h for p, h in protocol["sha256"].items()}


def freeze(out):
    if out.exists():
        raise ValueError("Use a fresh output directory")
    prior = read(PRIOR)
    if not all(check_inputs(prior).values()):
        raise ValueError("Prior frozen inputs changed")
    if {k: v["path"] for k, v in prior["checkpoints"].items()} != SOURCES:
        raise ValueError("Source selection mismatch")
    for label, name in SOURCES.items():
        state, cfg, meta = load_checkpoint(Path(name))
        d = prior["checkpoints"][label]
        if digest(name) != d["file_sha256"] or state_digest(state) != d["state_sha256"]:
            raise ValueError("Source identity changed")
        batches, ids, _, documents = prepare(cfg)
        if selected(restore_next_pair(cfg, meta, batches, ids, documents)) != d["next"]:
            raise ValueError("Next-input selection changed")
    paths = [
        *map(Path, prior["sha256"]),
        PRIOR,
        METHOD,
        *Path("src/mlm").glob("*.py"),
        *Path("tests").glob("*.py"),
        *Path("docs/evidence/qa-replay-pair").glob("checkpoint-*.json"),
    ]
    p = {
        "mode": "qa_optimizer_regularizer_controls",
        "copied_transitions": 30,
        "checkpoints": prior["checkpoints"],
        "settings": execution_settings(),
        "sha256": {str(p): digest(p) for p in dict.fromkeys(paths)},
    }
    out.mkdir(parents=True)
    write_json(out / "protocol.json", p)
    print(
        json.dumps(
            {"phase": "frozen", "sources": 5, "transitions": 30, "inputs": len(p["sha256"])}
        ),
        flush=True,
    )


def run(out):
    p = read(out / "protocol.json")
    if (out / "execution.json").exists() or list(out.glob("checkpoint-*.json")):
        raise ValueError("Preserve existing diagnostic outputs")
    if not all(check_inputs(p).values()) or p["settings"] != execution_settings():
        raise ValueError("Frozen inputs/settings changed")
    prior = read(PRIOR)
    if p["checkpoints"] != prior["checkpoints"] or p["copied_transitions"] != 30:
        raise ValueError("Diagnostic contract changed")
    protocol_hash = digest(out / "protocol.json")
    write_json(
        out / "execution.json", {"protocol_sha256": protocol_hash, "settings": execution_settings()}
    )
    audit = {
        "sources": {k: v["path"] for k, v in p["checkpoints"].items()} == SOURCES,
        "prior_hashes": all(p["sha256"].get(k) == v for k, v in prior["sha256"].items()),
    }
    results = {}
    for label, d in p["checkpoints"].items():
        state, cfg, meta = load_checkpoint(Path(d["path"]))
        batches, ids, aggregate, documents = prepare(cfg)
        context = restore_next_pair(cfg, meta, batches, ids, documents)
        audit[label + ":source"] = (
            digest(d["path"]) == d["file_sha256"] and state_digest(state) == d["state_sha256"]
        )
        audit[label + ":metadata"] = json_hash(meta) == d["metadata_sha256"]
        audit[label + ":selection"] = selected(context) == d["next"]
        audit[label + ":positions"] = int((aggregate["mask"] * aggregate["loss_mask"]).sum()) == 96
        if not all(audit.values()):
            raise ValueError("Source preflight failed")
        before = state_digest(state)
        states, m = make_qa_controls(cfg)(
            state,
            jax.tree.map(jnp.asarray, aggregate),
            jax.tree.map(jnp.asarray, context["qa_batch"]),
        )
        audit[label + ":branches"] = set(states) == BRANCHES and set(m["branches"]) == BRANCHES
        old = read(Path("docs/evidence/qa-replay-pair") / ("checkpoint-" + label + ".json"))[
            "metrics"
        ]
        for key, a, b in [
            (
                "prior_selected",
                m["branches"]["loaded_configured"]["after"]["selected_language"],
                old["selected_qa_loss_after"],
            ),
            (
                "prior_aggregate",
                m["branches"]["loaded_configured"]["after"]["aggregate_language"],
                old["qa_loss_after_qa"],
            ),
        ]:
            audit[label + ":" + key] = abs(a - b) <= 1e-6
        for name, b in m["branches"].items():
            after = b["after"]
            audit[label + ":" + name + ":step"] = (
                b["step_before"] == int(state.step) and b["step_after"] == int(state.step) + 1
            )
            audit[label + ":" + name + ":objective"] = (
                abs(
                    after["configured_objective"]
                    - after["selected_language"]
                    - cfg.router_balance * after["router_penalty"]
                )
                < 1e-6
            )
            audit[label + ":" + name + ":algebra"] = (
                abs(
                    b["aggregate_change"]
                    - after["aggregate_language"]
                    + m["before"]["aggregate_language"]
                )
                < 1e-9
            )
            audit[label + ":" + name + ":finite"] = all(
                math.isfinite(float(x)) for x in jax.tree.leaves(b)
            )
            if label == "initial" and name.startswith("loaded_"):
                other = states[name.replace("loaded_", "fresh_")]
                audit[label + ":" + name + ":initial_equivalence"] = state_digest(
                    states[name]
                ) == state_digest(other)
        for metric in ["selected_change", "aggregate_change"]:
            v = {name: branch[metric] for name, branch in m["branches"].items()}
            c = m["contrasts"][metric]
            reg = {h: v[h + "_configured"] - v[h + "_language"] for h in ["loaded", "fresh"]}
            audit[label + ":" + metric + ":regularizer"] = all(
                abs(c["regularizer_configured_minus_language"][h] - reg[h]) < 1e-12 for h in reg
            )
            audit[label + ":" + metric + ":history"] = all(
                abs(
                    c["history_loaded_minus_fresh"][mode] - v["loaded_" + mode] + v["fresh_" + mode]
                )
                < 1e-12
                for mode in ["configured", "language", "zero"]
            )
            audit[label + ":" + metric + ":interaction"] = (
                abs(c["interaction"] - reg["loaded"] + reg["fresh"]) < 1e-12
            )
            audit[label + ":" + metric + ":zero_control"] = all(
                abs(c["minus_zero"][h + "_" + mode] - v[h + "_" + mode] + v[h + "_zero"]) < 1e-12
                for h in reg
                for mode in ["configured", "language"]
            )
        for name, b in m["branches"].items():
            after = b["after"]
            baseline = (
                m["before"]["selected_language"]
                if name.endswith("_language")
                else m["before"]["configured_objective"]
            )
            active_after = (
                after["selected_language"]
                if name.endswith("_language")
                else after["configured_objective"]
            )
            audit[label + ":" + name + ":active_objective"] = (
                abs(b["active_objective_change"] - active_after + baseline) < 1e-9
            )
            audit[label + ":" + name + ":selected_algebra"] = (
                abs(
                    b["selected_change"]
                    - after["selected_language"]
                    + m["before"]["selected_language"]
                )
                < 1e-9
            )
            audit[label + ":" + name + ":configured_algebra"] = (
                abs(
                    b["configured_objective_change"]
                    - after["configured_objective"]
                    + m["before"]["configured_objective"]
                )
                < 1e-9
            )
        audit[label + ":loop_counts"] = (
            int(state.step) == context["qa_updates_before"] + context["english_updates_before"]
        )
        audit[label + ":immutable"] = (
            before == state_digest(state)
            and digest(d["path"]) == d["file_sha256"]
            and json_hash(meta) == d["metadata_sha256"]
        )
        if not all(audit.values()):
            raise ValueError("Control audit failed: " + str([k for k, v in audit.items() if not v]))
        results[label] = {"source": d, "metrics": m}
        write_json(out / ("checkpoint-" + label + ".json"), results[label])
        print(
            json.dumps(
                {
                    "checkpoint": label,
                    "aggregate_changes": {
                        k: v["aggregate_change"] for k, v in m["branches"].items()
                    },
                }
            ),
            flush=True,
        )
    audit["all_sources"] = set(results) == set(SOURCES)
    audit["protocol_immutable"] = protocol_hash == digest(out / "protocol.json")
    audit["settings_immutable"] = execution_settings() == p["settings"]
    audit.update({"input:" + k: v for k, v in check_inputs(p).items()})
    if not all(audit.values()):
        raise ValueError("Final input audit failed")
    write_json(out / "summary.json", {"checkpoints": results, "copied_transitions": 30})
    write_json(out / "audit.json", audit)
    print(json.dumps({"phase": "complete", "audits": len(audit), "all_passed": True}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["freeze", "run"])
    parser.add_argument("--out", type=Path, default=Path("results/qa-step-controls"))
    args = parser.parse_args()
    (freeze if args.phase == "freeze" else run)(args.out)


if __name__ == "__main__":
    main()
