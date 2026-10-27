"""Freeze and execute two predefined sampling-seed replications of future-byte training."""

import argparse
import json
from pathlib import Path
from typing import Any

from mlm.checkpoint import load_checkpoint
from mlm.cli import write_json
from mlm.future_byte import create_head
from mlm.future_byte_run import exposure
from mlm.future_replication import aggregate_gates, seeded_schedule, train_arm
from mlm.interference_run import SOURCES, digest, fingerprints, state_digest
from mlm.pair_probe_run import execution_settings, prepare, read
from mlm.stream import DocumentStream

PRIOR = Path("docs/evidence/future-byte/continuation-protocol.json")
METHOD = Path("docs/experiments/future-byte-replication/protocol.md")
PILOT = Path("docs/evidence/future-byte/summary.json")
SEEDS = {"43": {"qa": 43, "english": 44}, "44": {"qa": 44, "english": 45}}
ARMS = {"control": 0.0, "future_byte": 0.1}
PAIRS = 128


def intact(p):
    return {name: digest(name) == value for name, value in p["sha256"].items()}


def compatible_settings(left, right):
    a = json.loads(json.dumps(left))
    b = json.loads(json.dumps(right))
    a["environment"].pop("source_sha256")
    b["environment"].pop("source_sha256")
    return a == b


def inputs(source):
    state, cfg, meta = load_checkpoint(Path(source["path"]))
    if (
        source["path"] != SOURCES["initial"]
        or digest(source["path"]) != source["file_sha256"]
        or state_digest(state) != source["state_sha256"]
        or int(state.step) != 0
        or cfg.model_dump() != source["config"]
    ):
        raise ValueError("Unchanged original initializer required")
    batches, ids, qa, docs = prepare(cfg)
    if len(docs) != 1000 or len(ids) != 4:
        raise ValueError("Fixed corpus required")
    fixed = DocumentStream(docs, cfg.batch_size, cfg.length, 43).next_batch()
    return state, cfg, batches, ids, qa, docs, fixed


def descriptions(cfg, batches, ids, docs):
    result = {}
    for label, seeds in SEEDS.items():
        order, english, end = seeded_schedule(
            cfg, docs, len(ids), seeds["qa"], seeds["english"], PAIRS
        )
        result[label] = {
            **seeds,
            "qa_trace": [ids[i] for i in order],
            "english_batches": [fingerprints(b) for b in english],
            "english_snapshot": end,
            "expected_exposure": exposure(batches, order, english),
        }
    return result


PILOT_JSON = (
    "audit.json",
    "continuation-execution.json",
    "continuation-protocol.json",
    "control/history.json",
    "control/summary.json",
    "execution.json",
    "future_byte/history.json",
    "future_byte/summary.json",
    "original-driver.json",
    "protocol.json",
    "summary.json",
    "verification.json",
)


def validated_pilot_paths(
    published=Path("docs/evidence/future-byte"), raw=Path("results/future-byte")
):
    paths = [
        root / name for root in [published, raw] for name in [*PILOT_JSON, "artifact-sha256.json"]
    ]
    paths.extend(raw / arm / name for arm in ARMS for name in ["final.mlm", "head.msgpack"])
    if not all(f.is_file() for f in paths):
        raise ValueError("Missing historical pilot artifact")
    manifest = read(published / "artifact-sha256.json")
    if set(manifest) != set(PILOT_JSON) or digest(published / "artifact-sha256.json") != digest(
        raw / "artifact-sha256.json"
    ):
        raise ValueError("Historical artifact manifest mismatch")
    if not all(
        digest(root / name) == h for root in [published, raw] for name, h in manifest.items()
    ):
        raise ValueError("Historical pilot publication changed")
    summary = read(published / "summary.json")
    if set(summary["arms"]) != set(ARMS):
        raise ValueError("Historical arms missing")
    for arm, r in summary["arms"].items():
        if (
            read(published / arm / "summary.json") != r
            or digest(raw / arm / "final.mlm") != r["final_backbone_sha256"]
            or digest(raw / arm / "head.msgpack") != r["head_sha256"]
        ):
            raise ValueError("Historical final checkpoint/head identity changed")
    for name, protocol in [
        ("execution.json", "protocol.json"),
        ("continuation-execution.json", "continuation-protocol.json"),
    ]:
        e = read(published / name)
        p = read(published / protocol)
        if e["protocol_sha256"] != digest(published / protocol) or e["settings"] != p["settings"]:
            raise ValueError("Historical execution record changed")
    return paths


def freeze(out):
    if out.exists():
        raise ValueError("Use fresh output directory")
    prior = read(PRIOR)
    if not all(intact(prior).values()):
        raise ValueError("Original pilot inputs changed")
    if not compatible_settings(execution_settings(), prior["settings"]):
        raise ValueError("Pilot numerical environment changed")
    source = prior["source"]
    state, cfg, batches, ids, qa, docs, fixed = inputs(source)
    if (
        fingerprints(qa) != prior["qa_evaluation"]
        or fingerprints(fixed) != prior["english_batches"][0]
    ):
        raise ValueError("Original evaluation probe mismatch")
    pilot_paths = validated_pilot_paths()
    paths = [
        *map(Path, prior["sha256"]),
        PRIOR,
        METHOD,
        *pilot_paths,
        *Path("src/mlm").glob("*.py"),
        *Path("tests").glob("*.py"),
    ]
    p = {
        "mode": "future_byte_sampling_seed_replication",
        "source": source,
        "pairs": PAIRS,
        "arms": ARMS,
        "seeds": SEEDS,
        "head_seed": 9042,
        "head_state_sha256": state_digest(create_head(cfg)),
        "rank": 16,
        "qa_groups": ids,
        "qa_evaluation": fingerprints(qa),
        "fixed_english": fingerprints(fixed),
        "trials": descriptions(cfg, batches, ids, docs),
        "settings": execution_settings(),
        "gates": {
            "qa_strict_delta_below": -1e-6,
            "english_delta_at_most": 1e-6,
            "required_new_comparisons": 2,
        },
        "backbone_updates": 1024,
        "sha256": {str(f): digest(f) for f in dict.fromkeys(paths)},
    }
    out.mkdir(parents=True)
    write_json(out / "protocol.json", p)
    print(
        json.dumps(
            {"phase": "frozen", "inputs": len(p["sha256"]), "new_arms": 4, "backbone_updates": 1024}
        ),
        flush=True,
    )


def run(out):
    p = read(out / "protocol.json")
    if (out / "execution.json").exists() or list(out.glob("seed-*")):
        raise ValueError("Preserve existing results")
    prior = read(PRIOR)
    validated_pilot_paths()
    if not all(intact(p).values()) or execution_settings() != p["settings"]:
        raise ValueError("Frozen inputs/settings changed")
    expected_gates = {
        "qa_strict_delta_below": -1e-6,
        "english_delta_at_most": 1e-6,
        "required_new_comparisons": 2,
    }
    if (
        p["source"] != prior["source"]
        or p["pairs"] != PAIRS
        or p["arms"] != ARMS
        or p["seeds"] != SEEDS
        or p["head_seed"] != 9042
        or p["rank"] != 16
        or p["backbone_updates"] != 1024
        or p["gates"] != expected_gates
    ):
        raise ValueError("Frozen design changed")
    state, cfg, batches, ids, qa, docs, fixed = inputs(p["source"])
    head = create_head(cfg)
    if (
        p["trials"] != descriptions(cfg, batches, ids, docs)
        or ids != p["qa_groups"]
        or fingerprints(qa) != p["qa_evaluation"]
        or fingerprints(fixed) != p["fixed_english"]
        or state_digest(head) != p["head_state_sha256"]
    ):
        raise ValueError("Frozen schedule/probe mismatch")
    phash = digest(out / "protocol.json")
    write_json(out / "execution.json", {"protocol_sha256": phash, "settings": execution_settings()})
    audit = {
        "prior_inputs": all(p["sha256"].get(f) == h for f, h in prior["sha256"].items()),
        "qa_positions": int((qa["mask"] * qa["loss_mask"]).sum()) == 96,
        "fixed_english_original": fingerprints(fixed) == prior["english_batches"][0],
    }
    if not all(audit.values()):
        raise ValueError("Preflight failed")
    all_results: dict[str, Any] = {}
    comparisons = {}
    original_state = state_digest(state)
    original_head = state_digest(head)
    reference_initial = None
    for label, seeds in SEEDS.items():
        arms = {}
        for name, weight in ARMS.items():
            final, trained, r = train_arm(
                state,
                head,
                cfg,
                batches,
                ids,
                docs,
                qa,
                fixed,
                seeds["qa"],
                seeds["english"],
                weight,
                out / f"seed-{label}" / name,
                pairs=PAIRS,
            )
            arms[name] = r
            audit.update({f"{label}:{name}:{k}": v for k, v in r["audit"].items()})
            audit[f"{label}:{name}:frozen_exposure"] = (
                r["counts"] == p["trials"][label]["expected_exposure"]
            )
            audit[f"{label}:{name}:frozen_schedule"] = (
                r["qa_trace"] == p["trials"][label]["qa_trace"]
                and r["english_snapshot"] == p["trials"][label]["english_snapshot"]
            )
            audit[f"{label}:{name}:parameters"] = (
                r["backbone_parameters"] == 384784 and r["auxiliary_parameters"] == 8514
            )
            audit[f"{label}:{name}:steps"] = int(final.step) == int(trained.step) == 256
            audit[f"{label}:{name}:checkpoint_hash"] = (
                digest(out / f"seed-{label}" / name / "final.mlm") == r["final_backbone_sha256"]
            )
            audit[f"{label}:{name}:head_hash"] = (
                digest(out / f"seed-{label}" / name / "head.msgpack") == r["head_sha256"]
            )
            if reference_initial is None:
                reference_initial = r["initial"]
            audit[f"{label}:{name}:same_initial"] = r["initial"] == reference_initial
            if not all(audit.values()):
                raise ValueError(
                    "Arm audit failure: " + str([k for k, v in audit.items() if not v])
                )
        audit[label + ":paired_exposure"] = (
            arms["control"]["counts"] == arms["future_byte"]["counts"]
        )
        audit[label + ":paired_visits"] = arms["control"]["visits"] == arms["future_byte"]["visits"]
        comparisons[label] = {
            "qa_delta": arms["future_byte"]["final"]["qa"]["loss"]
            - arms["control"]["final"]["qa"]["loss"],
            "english_delta": arms["future_byte"]["final"]["english_train_probe"]["loss"]
            - arms["control"]["final"]["english_train_probe"]["loss"],
        }
        all_results[label] = arms
        print(json.dumps({"comparison": label, **comparisons[label]}), flush=True)
    gates = aggregate_gates(comparisons)
    audit["all_trials"] = set(all_results) == set(SEEDS)
    audit["all_arms"] = all(set(v) == set(ARMS) for v in all_results.values())
    audit["source_immutable"] = (
        state_digest(state) == original_state
        and state_digest(head) == original_head
        and digest(p["source"]["path"]) == p["source"]["file_sha256"]
    )
    audit["protocol_immutable"] = phash == digest(out / "protocol.json")
    audit["settings_immutable"] = execution_settings() == p["settings"]
    audit.update({"input:" + f: v for f, v in intact(p).items()})
    if not all(audit.values()):
        raise ValueError("Final integrity failure")
    summary = {
        "trials": all_results,
        "comparisons": comparisons,
        "gates": gates,
        "new_backbone_updates": 1024,
        "historical_reference": read(PILOT)["arms"],
    }
    write_json(out / "summary.json", summary)
    write_json(out / "audit.json", audit)
    print(json.dumps({"phase": "complete", "audits": len(audit), "gates": gates}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["freeze", "run"])
    parser.add_argument("--out", type=Path, default=Path("results/future-byte-replication"))
    args = parser.parse_args()
    (freeze if args.phase == "freeze" else run)(args.out)


if __name__ == "__main__":
    main()
