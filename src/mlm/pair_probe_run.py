"""Freeze and evaluate actual-next QA/replay pairs from five existing snapshots."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from mlm.binding_run import build_batches
from mlm.checkpoint import load_checkpoint
from mlm.cli import environment, write_json
from mlm.data import read_documents
from mlm.interference_run import ENGLISH_DATA, QA_DATA, SOURCES, digest, fingerprints, state_digest
from mlm.pair_probe import make_pair_probe, restore_next_pair

PREVIOUS = Path("docs/evidence/replay-interference/protocol.json")
METHOD = Path("docs/experiments/qa-replay-pair/protocol.md")


def read(path):
    return json.loads(Path(path).read_text())


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def execution_settings():
    return {
        "environment": environment(),
        "numerical": {
            name: str(getattr(jax.config, name))
            for name in ["jax_enable_x64", "jax_default_matmul_precision", "jax_disable_jit"]
        },
        "variables": {
            name: os.environ.get(name)
            for name in [
                "JAX_ENABLE_X64",
                "JAX_DEFAULT_MATMUL_PRECISION",
                "JAX_PLATFORMS",
                "JAX_PLATFORM_NAME",
                "XLA_FLAGS",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
            ]
        },
    }


def prepare(config):
    records = [json.loads(line) for line in (QA_DATA / "train.jsonl").read_text().splitlines()]
    questions = read(QA_DATA / "train-questions.json")["questions"]
    batches, ids = build_batches(records, questions, config)
    if len(questions) != 16 or len(ids) != 4:
        raise ValueError("Expected exactly sixteen training questions/four groups")
    aggregate = {k: np.concatenate([b[k] for b in batches], axis=0) for k in batches[0]}
    return batches, ids, aggregate, read_documents(ENGLISH_DATA)


def selected(context):
    return {
        "qa_group": context["qa_group"],
        "qa_updates_before": context["qa_updates_before"],
        "english_updates_before": context["english_updates_before"],
        "sampling_seeds": context["sampling_seeds"],
        "qa_batch": fingerprints(context["qa_batch"]),
        "english_batch": fingerprints(context["english_batch"]),
        "english_carry": fingerprints({"carry": context["english_carry"]}),
        "snapshot_before_sha256": json_hash(context["english_snapshot_before"]),
        "snapshot_after_sha256": json_hash(context["english_snapshot_after"]),
    }


def freeze(out):
    if out.exists():
        raise ValueError("Use a fresh output directory")
    previous = read(PREVIOUS)
    if {k: v["path"] for k, v in previous["checkpoints"].items()} != SOURCES:
        raise ValueError("Prior snapshot selection mismatch")
    prior_checks = {p: digest(p) == h for p, h in previous["sha256"].items()}
    if not all(prior_checks.values()):
        raise ValueError("Inputs from completed frozen diagnostic changed")
    _, base, _ = load_checkpoint(Path(SOURCES["initial"]))
    batches, ids, aggregate, documents = prepare(base)
    qa_fingerprints = fingerprints(aggregate)
    qa_positions = int((aggregate["mask"] * aggregate["loss_mask"]).sum())
    if qa_fingerprints != previous["qa_batch"] or qa_positions != 96:
        raise ValueError("Aggregate QA must match prior sixteen-row/96-target probe")
    checkpoints: dict[str, Any] = {}
    for label, name in SOURCES.items():
        state, cfg, meta = load_checkpoint(Path(name))
        old = previous["checkpoints"][label]
        if digest(name) != old["file_sha256"] or state_digest(state) != old["state_sha256"]:
            raise ValueError("Original checkpoint changed")
        if cfg.model_dump(exclude={"seed"}) != base.model_dump(exclude={"seed"}):
            raise ValueError("Source configuration mismatch")
        if not meta.get("messages", True):
            raise ValueError("Graph messages required")
        context = restore_next_pair(cfg, meta, batches, ids, documents)
        if int(state.step) != context["qa_updates_before"] + context["english_updates_before"]:
            raise ValueError("Optimizer and saved loop counters differ")
        checkpoints[label] = {
            "path": name,
            "config": cfg.model_dump(),
            "step": int(state.step),
            "file_sha256": digest(name),
            "state_sha256": state_digest(state),
            "metadata_sha256": json_hash(meta),
            "next": selected(context),
        }
    paths = [
        *map(Path, SOURCES.values()),
        QA_DATA / "train.jsonl",
        QA_DATA / "train-questions.json",
        ENGLISH_DATA,
        PREVIOUS,
        Path("docs/evidence/replay-interference/checkpoint-initial.json"),
        METHOD,
        Path("pyproject.toml"),
        Path("uv.lock"),
        *Path("src/mlm").glob("*.py"),
        *Path("tests").glob("*.py"),
    ]
    paths.extend(Path(name) for name in previous["sha256"])
    protocol = {
        "mode": "copied_actual_next_pair",
        "checkpoints": checkpoints,
        "qa_groups": ids,
        "qa_evaluation": qa_fingerprints,
        "qa_supervised_positions": qa_positions,
        "settings": execution_settings(),
        "prior_inputs_unchanged": prior_checks,
        "loss_change_threshold": 1e-6,
        "copied_transitions": 25,
        "selection": "all five fixed prior snapshots; actual next QA and saved English data",
        "scope": (
            "one next pair per selected snapshot; no trajectory, ranking or generalization claim"
        ),
        "sha256": {str(p): digest(p) for p in dict.fromkeys(paths)},
    }
    out.mkdir(parents=True)
    write_json(out / "protocol.json", protocol)
    print(
        json.dumps(
            {
                "phase": "frozen",
                "pairs": 5,
                "copied_transitions": 25,
                "next_groups": {k: v["next"]["qa_group"] for k, v in checkpoints.items()},
            }
        ),
        flush=True,
    )


def run(out):
    protocol = read(out / "protocol.json")
    if (out / "execution.json").exists() or list(out.glob("checkpoint-*.json")):
        raise ValueError("Preserve existing diagnostic outputs")

    def frozen():
        return {p: digest(p) == h for p, h in protocol["sha256"].items()}

    if not all(frozen().values()):
        raise ValueError("Frozen input changed")
    settings = execution_settings()
    if settings != protocol["settings"]:
        raise ValueError("Frozen execution settings changed")
    protocol_hash = digest(out / "protocol.json")
    write_json(
        out / "execution.json",
        {"settings": settings, "protocol_sha256": protocol_hash},
    )
    _, base, _ = load_checkpoint(Path(SOURCES["initial"]))
    batches, ids, aggregate, documents = prepare(base)
    previous = read(PREVIOUS)
    audit = {
        "checkpoint_contract": {k: v["path"] for k, v in protocol["checkpoints"].items()}
        == SOURCES,
        "prior_hashes_preserved": all(
            protocol["sha256"].get(p) == h for p, h in previous["sha256"].items()
        ),
        "qa_target_contract": protocol["qa_supervised_positions"]
        == int((aggregate["mask"] * aggregate["loss_mask"]).sum())
        == 96,
        "qa_matches_prior": fingerprints(aggregate) == previous["qa_batch"],
        "qa_groups_match": ids == protocol["qa_groups"],
        "qa_evaluation_match": fingerprints(aggregate) == protocol["qa_evaluation"],
        "copied_transition_contract": protocol["copied_transitions"] == 25,
    }
    if not all(audit.values()):
        raise ValueError("Frozen diagnostic contract mismatch")
    aggregate = jax.tree.map(jnp.asarray, aggregate)
    reports = {}
    transitions = 0
    for label, descriptor in protocol["checkpoints"].items():
        state, cfg, meta = load_checkpoint(Path(descriptor["path"]))
        before, meta_before = state_digest(state), json_hash(meta)
        context = restore_next_pair(cfg, meta, batches, ids, documents)
        audit[label + ":loaded_state"] = before == descriptor["state_sha256"]
        audit[label + ":loaded_metadata"] = meta_before == descriptor["metadata_sha256"]
        audit[label + ":selection"] = selected(context) == descriptor["next"]
        if not all(audit.values()):
            raise ValueError("Snapshot reconstruction mismatch")
        incoming = context["english_carry"].copy()
        states, metrics = make_pair_probe(cfg)(
            state,
            aggregate,
            jax.tree.map(jnp.asarray, context["qa_batch"]),
            jax.tree.map(jnp.asarray, context["english_batch"]),
            jnp.asarray(incoming),
        )
        n = descriptor["step"]
        expected = {
            "qa": n + 1,
            "qa_english": n + 2,
            "qa_zero": n + 2,
            "english_only": n + 1,
            "zero_only": n + 1,
        }
        audit[label + ":branch_steps"] = {k: int(s.step) for k, s in states.items()} == expected
        transitions += len(states)
        audit[label + ":source_immutable"] = state_digest(state) == before
        audit[label + ":metadata_immutable"] = json_hash(meta) == meta_before
        audit[label + ":incoming_carry_immutable"] = np.array_equal(
            incoming, context["english_carry"]
        )
        audit[label + ":phase_algebra"] = all(
            math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-10)
            for a, b in [
                (
                    metrics["qa_step_loss_change"],
                    metrics["qa_loss_after_qa"] - metrics["qa_loss_before"],
                ),
                (
                    metrics["english_step_loss_change"],
                    metrics["qa_loss_after_pair"] - metrics["qa_loss_after_qa"],
                ),
                (
                    metrics["pair_loss_change"],
                    metrics["qa_loss_after_pair"] - metrics["qa_loss_before"],
                ),
                (
                    metrics["english_minus_zero_after_qa"],
                    metrics["qa_loss_after_pair"] - metrics["qa_loss_after_qa_zero"],
                ),
                (
                    metrics["english_only_loss_change"],
                    metrics["qa_loss_after_english_only"] - metrics["qa_loss_before"],
                ),
                (
                    metrics["preceding_qa_effect_on_english_delta"],
                    metrics["english_step_loss_change"] - metrics["english_only_loss_change"],
                ),
            ]
        )
        for branch, before_step in [("paired_english", n + 1), ("matched_english_only", n)]:
            row = metrics[branch]
            audit[label + ":" + branch + ":steps"] = (
                row["optimizer_step_before"] == before_step
                and row["optimizer_step_after"]
                == row["zero_optimizer_step_after"]
                == before_step + 1
            )
            audit[label + ":" + branch + ":qa_positions"] = (
                row["qa_supervised_positions"] == protocol["qa_supervised_positions"]
            )
            audit[label + ":" + branch + ":english_positions"] = row[
                "english_supervised_positions"
            ] == float(context["english_batch"]["mask"].sum())
            audit[label + ":" + branch + ":role_membership"] = set(row["roles"]) == {
                "shared",
                "node",
                "router",
                "memory",
            }
            audit[label + ":" + branch + ":parameter_counts"] = (
                sum(v["parameter_count"] for v in row["roles"].values())
                == row["parameter_count"]
                == sum(v.size for v in state.params.values())
            )
            audit[label + ":" + branch + ":role_accounting"] = math.isclose(
                math.fsum(v["first_order_qa_change"] for v in row["roles"].values()),
                row["first_order_qa_change"],
                rel_tol=1e-8,
                abs_tol=1e-10,
            )
        if label == "initial":
            old = read("docs/evidence/replay-interference/checkpoint-initial.json")["rows"]
            prior = next(r for r in old if r["english_seed"] == 43 and r["position"] == 0)
            audit["initial_same_english_only_input"] = (
                fingerprints(context["english_batch"]) == read(PREVIOUS)["probes"]["43"][0]["batch"]
            )
            audit["initial_reproduces_prior_english_only"] = all(
                abs(metrics["matched_english_only"][key] - prior[key]) <= 1e-6
                for key in ["qa_loss_before", "qa_loss_after_english", "qa_loss_after_zero"]
            )
        reports[label] = {"source": descriptor, "metrics": metrics}
        write_json(out / f"checkpoint-{label}.json", reports[label])
        print(
            json.dumps(
                {
                    "checkpoint": label,
                    "qa_delta": metrics["qa_step_loss_change"],
                    "english_delta": metrics["english_step_loss_change"],
                    "net_pair_delta": metrics["pair_loss_change"],
                }
            ),
            flush=True,
        )
    audit["protocol_file_unchanged"] = digest(out / "protocol.json") == protocol_hash
    audit["execution_settings_unchanged"] = execution_settings() == settings
    audit["all_five_pairs_complete"] = set(reports) == set(SOURCES)
    audit["all25_transitions_complete"] = transitions == 25
    audit.update({f"input:{p}": ok for p, ok in frozen().items()})
    write_json(out / "audit.json", audit)
    if not all(audit.values()):
        raise AssertionError([k for k, v in audit.items() if not v])
    write_json(
        out / "summary.json",
        {
            "checkpoints": reports,
            "threshold": protocol["loss_change_threshold"],
            "scope": protocol["scope"],
            "copied_transitions": transitions,
        },
    )
    print(f"All {len(audit)} audit flags pass; saved checkpoints unchanged", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["freeze", "run"])
    parser.add_argument("--out", type=Path, default=Path("results/qa-replay-pair"))
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze(args.out)
    else:
        run(args.out)


if __name__ == "__main__":
    main()
