"""Freeze and evaluate all completed future-byte backbones on original training answers."""

import argparse
import json
from pathlib import Path

from mlm.answer_analysis import analyze_answers, paired_changes, prepare_answers
from mlm.checkpoint import load_checkpoint
from mlm.cli import write_json
from mlm.future_replication_run import compatible_settings, validated_pilot_paths
from mlm.interference_run import QA_DATA, digest, fingerprints, state_digest
from mlm.pair_probe_run import execution_settings, json_hash, read

PRIOR = Path("docs/evidence/future-byte-replication/protocol.json")
METHOD = Path("docs/experiments/future-byte-answers/protocol.md")
PUBLIC = Path("docs/evidence/future-byte-replication")
RAW = Path("results/future-byte-replication")
COUNTS = {"first": 16, "rest": 48, "newline": 16, "eos": 16}
PAIRS = {
    "pilot42": ("pilot42_control", "pilot42_future_byte"),
    "rep43": ("rep43_control", "rep43_future_byte"),
    "rep44": ("rep44_control", "rep44_future_byte"),
}


def intact(p):
    return {f: digest(f) == h for f, h in p["sha256"].items()}


def records():
    return [json.loads(line) for line in (QA_DATA / "train.jsonl").read_text().splitlines()], read(
        QA_DATA / "train-questions.json"
    )["questions"]


def replication_paths():
    names = ["protocol.json", "execution.json", "summary.json", "audit.json", "verification.json"]
    names.extend(
        f"seed-{seed}/{arm}/{name}"
        for seed in ["43", "44"]
        for arm in ["control", "future_byte"]
        for name in ["initial.json", "summary.json", "history.json"]
    )
    paths = [root / name for root in [PUBLIC, RAW] for name in [*names, "artifact-sha256.json"]]
    paths.extend(
        RAW / f"seed-{seed}" / arm / name
        for seed in ["43", "44"]
        for arm in ["control", "future_byte"]
        for name in ["final.mlm", "head.msgpack"]
    )
    if not all(p.is_file() for p in paths):
        raise ValueError("Missing completed replication artifact")
    manifest = read(PUBLIC / "artifact-sha256.json")
    if set(manifest) != set(names) or digest(PUBLIC / "artifact-sha256.json") != digest(
        RAW / "artifact-sha256.json"
    ):
        raise ValueError("Replication artifact manifest mismatch")
    if not all(digest(root / name) == h for root in [PUBLIC, RAW] for name, h in manifest.items()):
        raise ValueError("Completed replication publication changed")
    summary = read(PUBLIC / "summary.json")
    if set(summary["trials"]) != {"43", "44"}:
        raise ValueError("Completed seeds missing")
    for seed, arms in summary["trials"].items():
        if set(arms) != {"control", "future_byte"}:
            raise ValueError("Completed arms missing")
        for arm, r in arms.items():
            if (
                read(PUBLIC / f"seed-{seed}" / arm / "summary.json") != r
                or digest(RAW / f"seed-{seed}" / arm / "final.mlm") != r["final_backbone_sha256"]
                or digest(RAW / f"seed-{seed}" / arm / "head.msgpack") != r["head_sha256"]
            ):
                raise ValueError("Completed model identity changed")
    e = read(PUBLIC / "execution.json")
    p = read(PUBLIC / "protocol.json")
    if e["protocol_sha256"] != digest(PUBLIC / "protocol.json") or e["settings"] != p["settings"]:
        raise ValueError("Completed execution record changed")
    return paths


def describe_models(prior):
    pilot = read("docs/evidence/future-byte/summary.json")["arms"]
    rep = read(PUBLIC / "summary.json")["trials"]
    source = prior["source"]
    selections = {
        "initial": (
            Path(source["path"]),
            source["file_sha256"],
            source["state_sha256"],
            pilot["control"]["initial"]["qa"]["loss"],
            0,
        )
    }
    for name, r in pilot.items():
        selections["pilot42_" + name] = (
            Path("results/future-byte") / name / "final.mlm",
            r["final_backbone_sha256"],
            r["final_state_sha256"],
            r["final"]["qa"]["loss"],
            256,
        )
    for seed, arms in rep.items():
        for name, r in arms.items():
            selections["rep" + seed + "_" + name] = (
                RAW / f"seed-{seed}" / name / "final.mlm",
                r["final_backbone_sha256"],
                r["final_state_sha256"],
                r["final"]["qa"]["loss"],
                256,
            )
    result = {}
    for label, (path, file_hash, state_hash, loss, step) in selections.items():
        state, cfg, meta = load_checkpoint(path)
        if (
            digest(path) != file_hash
            or state_digest(state) != state_hash
            or int(state.step) != step
            or cfg.model_dump() != source["config"]
            or not meta.get("messages", True)
        ):
            raise ValueError("Backbone source identity mismatch")
        result[label] = {
            "path": str(path),
            "file_sha256": file_hash,
            "state_sha256": state_hash,
            "metadata_sha256": json_hash(meta),
            "config": cfg.model_dump(),
            "step": step,
            "expected_qa_ce": loss,
        }
    if len(result) != 7:
        raise ValueError("Exactly all seven predefined models required")
    return result


def freeze(out):
    if out.exists():
        raise ValueError("Use fresh output directory")
    prior = read(PRIOR)
    if not all(intact(prior).values()) or not compatible_settings(
        execution_settings(), prior["settings"]
    ):
        raise ValueError("Original inputs/numerical settings changed")
    paths = validated_pilot_paths() + replication_paths()
    models = describe_models(prior)
    _, cfg, _ = load_checkpoint(Path(models["initial"]["path"]))
    r, q = records()
    batch, ordered, masks = prepare_answers(r, q, cfg)
    if len(q) != 16 or {k: int(v.sum()) for k, v in masks.items()} != COUNTS:
        raise ValueError("Original sixteen-question/96-target fixture required")
    paths.extend(
        [
            *map(Path, prior["sha256"]),
            PRIOR,
            METHOD,
            QA_DATA / "train.jsonl",
            QA_DATA / "train-questions.json",
            *Path("src/mlm").glob("*.py"),
            *Path("tests").glob("*.py"),
        ]
    )
    p = {
        "mode": "frozen_training_answer_analysis",
        "models": models,
        "pairs": PAIRS,
        "counts": COUNTS,
        "question_ids": [v["id"] for v in ordered],
        "batch": fingerprints(batch),
        "partitions": fingerprints(masks),
        "settings": execution_settings(),
        "candidate_teacher_tolerance": 1e-5,
        "prior_ce_tolerance": 1e-6,
        "sha256": {str(f): digest(f) for f in dict.fromkeys(paths)},
    }
    out.mkdir(parents=True)
    write_json(out / "protocol.json", p)
    print(
        json.dumps(
            {
                "phase": "frozen",
                "models": 7,
                "questions_each": 16,
                "optimizer_updates": 0,
                "inputs": len(p["sha256"]),
            }
        ),
        flush=True,
    )


def run(out):
    p = read(out / "protocol.json")
    prior = read(PRIOR)
    if (out / "execution.json").exists() or list(out.glob("model-*.json")):
        raise ValueError("Preserve existing outputs")
    if not all(intact(p).values()) or execution_settings() != p["settings"]:
        raise ValueError("Frozen inputs/settings changed")
    validated_pilot_paths()
    replication_paths()
    if (
        p["models"] != describe_models(prior)
        or p["pairs"] != {k: list(v) for k, v in PAIRS.items()}
        or p["counts"] != COUNTS
        or p["candidate_teacher_tolerance"] != 1e-5
        or p["prior_ce_tolerance"] != 1e-6
    ):
        raise ValueError("Frozen analysis design changed")
    phash = digest(out / "protocol.json")
    write_json(out / "execution.json", {"protocol_sha256": phash, "settings": execution_settings()})
    r, q = records()
    results = {}
    audit = {"prior_inputs": all(p["sha256"].get(f) == h for f, h in prior["sha256"].items())}
    for label, d in p["models"].items():
        state, cfg, meta = load_checkpoint(Path(d["path"]))
        before = state_digest(state)
        batch, ordered, masks = prepare_answers(r, q, cfg)
        audit[label + ":batch"] = (
            fingerprints(batch) == p["batch"] and fingerprints(masks) == p["partitions"]
        )
        audit[label + ":ids"] = [v["id"] for v in ordered] == p["question_ids"]
        result = analyze_answers(state, cfg, r, q)
        teacher = result["teacher"]
        rank = result["ranking"]
        audit[label + ":counts"] = (
            {k: v["count"] for k, v in teacher["parts"].items()} == COUNTS
            and teacher["total"]["count"] == 96
            and teacher["answer_only"]["count"] == 64
        )
        audit[label + ":ranking_shape"] = (
            rank["questions"] == 16
            and rank["pairs"] == 8
            and rank["overall"]["binding_groups"] == 4
        )
        audit[label + ":prior_ce"] = abs(teacher["total"]["mean_nll"] - d["expected_qa_ce"]) <= 1e-6
        audit[label + ":score_consistency"] = result["score_teacher_max_error"] <= 1e-5
        audit[label + ":partition_algebra"] = (
            abs(sum(v["sum_nll"] for v in teacher["parts"].values()) - teacher["total"]["sum_nll"])
            < 1e-9
        )
        audit[label + ":immutable"] = (
            before == state_digest(state) == d["state_sha256"]
            and digest(d["path"]) == d["file_sha256"]
            and json_hash(meta) == d["metadata_sha256"]
            and int(state.step) == d["step"]
        )
        if not all(audit.values()):
            raise ValueError(
                "Analysis audit failed: " + str([k for k, v in audit.items() if not v])
            )
        results[label] = {"source": d, "analysis": result}
        write_json(out / f"model-{label}.json", results[label])
        print(
            json.dumps(
                {
                    "model": label,
                    "correct": rank["correct_count"],
                    "groups": rank["complete_groups"],
                    "mean_margin": sum(v["gold_minus_other_margin"] for v in rank["items"]) / 16,
                }
            ),
            flush=True,
        )
    comparisons = {}
    for label, (control, active) in PAIRS.items():
        c, a = results[control]["analysis"], results[active]["analysis"]
        change = paired_changes(c, a)
        audit[label + ":delta_algebra"] = (
            abs(
                sum(v["total_ce_contribution"] for v in change["parts"].values())
                - change["qa_ce_delta"]
            )
            < 1e-9
        )
        transitions = change["ranking_transitions"]
        audit[label + ":transition_count"] = (
            sum(
                transitions[k]
                for k in ["both_correct", "control_only", "auxiliary_only", "both_wrong"]
            )
            == 16
        )
        audit[label + ":net_ranking"] = (
            transitions["net_correct_change"]
            == a["ranking"]["correct_count"] - c["ranking"]["correct_count"]
        )
        comparisons[label] = change
    audit["all_models"] = len(results) == 7
    audit["all_comparisons"] = set(comparisons) == set(PAIRS)
    audit["protocol_immutable"] = phash == digest(out / "protocol.json")
    audit["settings_immutable"] = execution_settings() == p["settings"]
    audit.update({"input:" + f: v for f, v in intact(p).items()})
    if not all(audit.values()):
        raise ValueError("Final integrity failure")
    write_json(
        out / "summary.json",
        {"models": results, "comparisons": comparisons, "optimizer_updates": 0},
    )
    write_json(out / "audit.json", audit)
    print(
        json.dumps({"phase": "complete", "audits": len(audit), "optimizer_updates": 0}), flush=True
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["freeze", "run"])
    parser.add_argument("--out", type=Path, default=Path("results/future-byte-answers"))
    args = parser.parse_args()
    (freeze if args.phase == "freeze" else run)(args.out)


if __name__ == "__main__":
    main()
