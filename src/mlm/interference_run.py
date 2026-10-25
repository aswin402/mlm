"""Fixed training-only replay diagnostic: freeze selections first, then run copied steps."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from mlm.binding_run import build_batches
from mlm.checkpoint import load_checkpoint
from mlm.cli import environment, write_json
from mlm.data import read_documents
from mlm.interference import make_interference_probe, make_prefix_reader, replay_probes

POSITIONS = [0, 1, 2, 3, 499, 999, 1499, 1999]
ENGLISH_SEEDS = [43, 44]
SOURCES = {
    "initial": "runs/addressing-exposure-delta-g4-v1/initial.mlm",
    "q42_e43": "runs/addressing-exposure-delta-g4-v1/final.mlm",
    "q42_e44": "runs/schedule-components-q42-e44-v1/final.mlm",
    "q43_e43": "runs/schedule-components-q43-e43-v1/final.mlm",
    "q43_e44": "runs/addressing-factorial-i42-s43-v1/final.mlm",
}
QA_DATA = Path("data/addressing-v1/groups-4")
ENGLISH_DATA = Path("data/hf/stories/train.jsonl")


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_digest(state):
    return hashlib.sha256(serialization.to_bytes(state)).hexdigest()


def fingerprints(batch):
    return {
        key: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(np.asarray(value).tobytes()).hexdigest(),
        }
        for key, value in batch.items()
    }


def selection(probes):
    result = {}
    for seed, rows in probes.items():
        result[str(seed)] = [
            {
                "position": row["position"],
                "batch": fingerprints(row["batch"]),
                "prefix": fingerprints(row["prefix"]),
                "prefix_token_counts": row["metadata"]["prefix_token_counts"],
                "incoming_lanes": row["metadata"]["snapshot_before"]["lanes"],
                "snapshot_before_sha256": hashlib.sha256(
                    json.dumps(row["metadata"]["snapshot_before"], sort_keys=True).encode()
                ).hexdigest(),
                "snapshot_after_sha256": hashlib.sha256(
                    json.dumps(row["metadata"]["snapshot_after"], sort_keys=True).encode()
                ).hexdigest(),
            }
            for row in rows
        ]
    return result


def prepare(config):
    records = [json.loads(line) for line in (QA_DATA / "train.jsonl").read_text().splitlines()]
    questions = read(QA_DATA / "train-questions.json")["questions"]
    batches, ids = build_batches(records, questions, config)
    if len(questions) != 16 or len(ids) != 4:
        raise ValueError("Exactly four complete training groups required")
    qa = {key: np.concatenate([batch[key] for batch in batches], axis=0) for key in batches[0]}
    documents = read_documents(ENGLISH_DATA)
    probes = {
        seed: replay_probes(documents, config.batch_size, config.length, seed, POSITIONS)
        for seed in ENGLISH_SEEDS
    }
    return qa, ids, probes


def freeze(out):
    if out.exists():
        raise ValueError("Use a fresh output directory")
    _, config, _ = load_checkpoint(Path(SOURCES["initial"]))
    qa, ids, probes = prepare(config)
    checkpoints = {}
    for label, name in SOURCES.items():
        state, cfg, meta = load_checkpoint(Path(name))
        if cfg.model_dump(exclude={"seed"}) != config.model_dump(exclude={"seed"}):
            raise ValueError("Source configuration mismatch")
        if not meta.get("messages", True):
            raise ValueError("Graph messages must be enabled")
        expected_step = 0 if label == "initial" else 4000
        if int(state.step) != expected_step:
            raise ValueError("Unexpected optimizer step")
        checkpoints[label] = {
            "path": name,
            "step": int(state.step),
            "config": cfg.model_dump(),
            "file_sha256": digest(name),
            "state_sha256": state_digest(state),
        }
    previous = read("docs/evidence/schedule-components/protocol.json")["sha256"]
    core = [
        "model.py",
        "associative.py",
        "hierarchy_run.py",
        "training.py",
        "plasticity.py",
        "config.py",
        "checkpoint.py",
        "stream.py",
        "binding_run.py",
        "data.py",
    ]
    core_checks = {
        f"src/mlm/{name}": digest(f"src/mlm/{name}") == previous[f"src/mlm/{name}"] for name in core
    }
    if not all(core_checks.values()):
        raise ValueError("Model/optimizer/task core changed since prior study")
    paths = [
        *map(Path, SOURCES.values()),
        QA_DATA / "train.jsonl",
        QA_DATA / "train-questions.json",
        ENGLISH_DATA,
        Path("pyproject.toml"),
        Path("uv.lock"),
        Path("docs/experiments/replay-interference/protocol.md"),
        Path("docs/evidence/schedule-components/protocol.json"),
        *Path("src/mlm").glob("*.py"),
        *Path("tests").glob("*.py"),
    ]
    protocol = {
        "mode": "frozen_replay_interference",
        "checkpoints": checkpoints,
        "positions": POSITIONS,
        "english_seeds": ENGLISH_SEEDS,
        "qa_groups": ids,
        "qa_batch": fingerprints(qa),
        "qa_supervised_positions": int((qa["mask"] * qa["loss_mask"]).sum()),
        "probes": selection(probes),
        "environment": environment(),
        "core_unchanged_checks": core_checks,
        "loss_change_threshold": 1e-6,
        "selection": "fixed before computation; all five checkpoints and both streams",
        "scope": (
            "training-only local counterfactuals; frozen prefix context; "
            "no trajectory/generalization claim"
        ),
        "sha256": {str(p): digest(p) for p in dict.fromkeys(paths)},
    }
    out.mkdir(parents=True)
    write_json(out / "protocol.json", protocol)
    print(
        json.dumps(
            {
                "phase": "frozen",
                "checkpoints": len(checkpoints),
                "probes": len(checkpoints) * len(POSITIONS) * len(ENGLISH_SEEDS),
                "prefix_lengths": {
                    seed: rows[0]["prefix"]["inputs"].shape[1] for seed, rows in probes.items()
                },
            }
        ),
        flush=True,
    )


def summarize(rows, threshold):
    fields = [
        "qa_loss_change",
        "zero_qa_loss_change",
        "english_minus_zero_qa_loss",
        "first_order_qa_change",
        "first_order_remainder",
        "gradient_cosine",
    ]
    result: dict[str, Any] = {"probes": len(rows)}
    for field in fields:
        values = [row[field] for row in rows if row[field] is not None]
        result[field] = (
            {
                "mean": float(np.mean(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
            if values
            else None
        )
    for field in ["qa_loss_change", "english_minus_zero_qa_loss"]:
        result[field].update(
            {
                "harmful": sum(r[field] > threshold for r in rows),
                "helpful": sum(r[field] < -threshold for r in rows),
                "unresolved": sum(abs(r[field]) <= threshold for r in rows),
            }
        )
    result["english_objective_decreased"] = sum(
        r["english_objective_after"] < r["english_objective_before"] - threshold for r in rows
    )
    return result


def run(out):
    protocol = read(out / "protocol.json")
    if (out / "summary.json").exists() or list(out.glob("checkpoint-*.json")):
        raise ValueError("Diagnostic results already exist; preserve the fixed run")

    def frozen_hashes():
        return {name: digest(name) == expected for name, expected in protocol["sha256"].items()}

    if not all(frozen_hashes().values()):
        raise ValueError("Frozen input changed")
    _, config, _ = load_checkpoint(Path(protocol["checkpoints"]["initial"]["path"]))
    qa, ids, probes = prepare(config)
    audit = {
        "checkpoint_contract": {label: d["path"] for label, d in protocol["checkpoints"].items()}
        == SOURCES,
        "position_contract": protocol["positions"] == POSITIONS,
        "english_seed_contract": protocol["english_seeds"] == ENGLISH_SEEDS,
        "qa_groups_match": ids == protocol["qa_groups"],
        "qa_batch_match": fingerprints(qa) == protocol["qa_batch"],
        "probe_selections_match": selection(probes) == protocol["probes"],
        "environment_matches": environment() == protocol["environment"],
    }
    if not all(audit.values()):
        raise ValueError("Frozen selection/environment mismatch")
    qa = jax.tree.map(jnp.asarray, qa)
    measure, read_prefix = make_interference_probe(config), make_prefix_reader(config)
    summaries = {}
    for label, descriptor in protocol["checkpoints"].items():
        state, _, _ = load_checkpoint(Path(descriptor["path"]))
        before = state_digest(state)
        audit[label + ":loaded_state_matches"] = before == descriptor["state_sha256"]
        rows = []
        for seed in ENGLISH_SEEDS:
            for probe in probes[seed]:
                prefix = jax.tree.map(jnp.asarray, probe["prefix"])
                batch = jax.tree.map(jnp.asarray, probe["batch"])
                carry = read_prefix(state, prefix)
                candidate, zero, metrics = measure(state, qa, batch, carry)
                row = {"english_seed": seed, "position": probe["position"], **metrics}
                rows.append(row)
                key = f"{label}:e{seed}:p{probe['position']}"
                audit[key + ":independent_steps"] = (
                    int(state.step) == descriptor["step"]
                    and int(candidate.step) == int(zero.step) == descriptor["step"] + 1
                )
                audit[key + ":qa_positions"] = (
                    metrics["qa_supervised_positions"] == protocol["qa_supervised_positions"]
                )
                audit[key + ":english_positions"] = metrics[
                    "english_supervised_positions"
                ] == float(probe["batch"]["mask"].sum())
                audit[key + ":role_accounting"] = math.isclose(
                    math.fsum(r["first_order_qa_change"] for r in metrics["roles"].values()),
                    metrics["first_order_qa_change"],
                    rel_tol=1e-8,
                    abs_tol=1e-10,
                )
                audit[key + ":role_parameters"] = (
                    sum(r["parameter_count"] for r in metrics["roles"].values())
                    == metrics["parameter_count"]
                )
        audit[label + ":source_state_unchanged"] = state_digest(state) == before
        audit[label + ":complete_probes"] = len(rows) == 16 and {
            (r["english_seed"], r["position"]) for r in rows
        } == {(e, p) for e in ENGLISH_SEEDS for p in POSITIONS}
        audit[label + ":constant_qa_baseline"] = (
            max(r["qa_loss_before"] for r in rows) - min(r["qa_loss_before"] for r in rows) <= 1e-6
        )
        audit[label + ":constant_zero_control"] = (
            max(r["qa_loss_after_zero"] for r in rows) - min(r["qa_loss_after_zero"] for r in rows)
            <= 1e-6
        )
        streams = {
            str(seed): summarize(
                [r for r in rows if r["english_seed"] == seed], protocol["loss_change_threshold"]
            )
            for seed in ENGLISH_SEEDS
        }
        paired = [
            {
                "position": p,
                "english44_minus43_qa_change": next(
                    r["qa_loss_change"]
                    for r in rows
                    if r["english_seed"] == 44 and r["position"] == p
                )
                - next(
                    r["qa_loss_change"]
                    for r in rows
                    if r["english_seed"] == 43 and r["position"] == p
                ),
            }
            for p in POSITIONS
        ]
        summaries[label] = {
            "source": descriptor,
            "qa_loss_before": rows[0]["qa_loss_before"],
            "zero_qa_loss_change": rows[0]["zero_qa_loss_change"],
            "streams": streams,
            "paired": paired,
            "paired_mean_english44_minus43": float(
                np.mean([r["english44_minus43_qa_change"] for r in paired])
            ),
        }
        write_json(out / f"checkpoint-{label}.json", {"rows": rows})
        print(
            json.dumps(
                {
                    "checkpoint": label,
                    "complete": len(rows),
                    "paired_mean": summaries[label]["paired_mean_english44_minus43"],
                }
            ),
            flush=True,
        )
    audit.update({f"input:{name}": ok for name, ok in frozen_hashes().items()})
    write_json(out / "audit.json", audit)
    if not all(audit.values()):
        raise AssertionError([name for name, ok in audit.items() if not ok])
    write_json(
        out / "summary.json",
        {
            "checkpoints": summaries,
            "scope": protocol["scope"],
            "threshold": protocol["loss_change_threshold"],
        },
    )
    print(f"All {len(audit)} audits passed; no checkpoint files written", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["freeze", "run"])
    parser.add_argument("--out", type=Path, default=Path("results/replay-interference"))
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze(args.out)
    else:
        run(args.out)


if __name__ == "__main__":
    main()
