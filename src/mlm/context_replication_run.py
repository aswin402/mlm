"""Fixed independent-sampling replication of context-contrastive acquisition."""

import json
from pathlib import Path

import jax
import numpy as np

from mlm.checkpoint import load_checkpoint, save_checkpoint
from mlm.cli import write_json
from mlm.comprehension import evaluate_questions
from mlm.context_contrastive import (
    candidate_batch,
    context_objective,
    make_context_step,
)
from mlm.context_contrastive_run import gates
from mlm.future_replication import seeded_schedule
from mlm.future_replication_run import compatible_settings
from mlm.hierarchy_run import make_routed_step, routed_objective
from mlm.interference_run import QA_DATA, SOURCES, digest, fingerprints, state_digest
from mlm.model import initial_state
from mlm.pair_probe_run import execution_settings, prepare, read
from mlm.retrieval_eval import aggregate
from mlm.stream import DocumentStream


def prior_raw_jsons(
    raw=Path("results/context-contrastive"), published=Path("docs/evidence/context-contrastive")
):
    pairs = [
        (raw / name, published / name)
        for name in [
            "protocol.json",
            "gradient.json",
            "initial.json",
            "summary.json",
            "improvement-protocol.json",
        ]
    ]
    pairs += [
        (raw / f"weight-{w}/{name}.json", published / f"weight-{w}-{name}.json")
        for w in [0, 1, 100]
        for name in ["summary", "history"]
    ]
    for source, copy in pairs:
        if not source.is_file() or not copy.is_file() or digest(source) != digest(copy):
            raise ValueError(f"Previous raw/public JSON mismatch: {source}")
    return [p for pair in pairs for p in pair]


def expected_exposure(batches, candidates, order, english, weight):
    totals = dict.fromkeys(
        [
            "qa_updates",
            "english_updates",
            "qa_targets",
            "english_targets",
            "candidate_positions",
            "context_terms",
            "qa_bytes",
            "english_bytes",
        ],
        0.0,
    )
    for i, group in enumerate(order):
        for name, batch in [("qa", batches[group]), ("english", english[i])]:
            mask = np.asarray(batch["mask"]) * np.asarray(batch.get("loss_mask", batch["mask"]))
            totals[name + "_updates"] += 1
            totals[name + "_targets"] += float(mask.sum())
            totals[name + "_bytes"] += float((mask * (np.asarray(batch["targets"]) < 256)).sum())
        if weight:
            totals["candidate_positions"] += float(candidates[group]["answer_mask"].sum())
            totals["context_terms"] += len(candidates[group]["gold"])
    return totals


def check_integrity(protocol_path, protocol_hash, settings, frozen):
    if digest(protocol_path) != protocol_hash or execution_settings() != settings:
        raise ValueError("Run protocol or numerical settings changed")
    if not all(digest(p) == h for p, h in frozen.items()):
        raise ValueError("Frozen input changed")


def validate_exposure(observed, expected, visits):
    if observed != expected or not visits or set(visits.values()) != {32}:
        raise ValueError("Exposure or group-visit mismatch")


def run(out=Path("results/context-replication")):
    out = Path(out)
    if out.exists():
        raise ValueError("Fresh output directory required; preserve completed runs")
    published = read("docs/evidence/context-contrastive/artifact-sha256.json")
    if not all(digest(p) == h for p, h in published.items()):
        raise ValueError("Previous published evidence changed")
    previous = read("docs/evidence/context-contrastive/protocol.json")
    for name, expected in previous["sha256"].items():
        actual = (
            "docs/evidence/context-contrastive/executed-runner.py"
            if name == "src/mlm/context_contrastive_run.py"
            else name
        )
        if digest(actual) != expected:
            raise ValueError(f"Historical context input changed: {name}")
    if not compatible_settings(previous["settings"], execution_settings()):
        raise ValueError("Prior numerical execution environment changed")
    prior = read("docs/evidence/future-byte-answers/protocol.json")
    frozen = dict(prior["sha256"])
    paths = [
        *prior_raw_jsons(),
        *map(Path, published),
        Path("docs/evidence/context-contrastive/artifact-sha256.json"),
        *[Path(f"results/context-contrastive/weight-{w}/final.mlm") for w in [0, 1, 100]],
        *Path("src/mlm").glob("*.py"),
        *Path("tests").glob("*.py"),
        Path("docs/experiments/context-replication/protocol.md"),
        Path("docs/evidence/future-byte-answers/protocol.json"),
        Path(SOURCES["initial"]),
        QA_DATA / "train.jsonl",
        QA_DATA / "train-questions.json",
    ]
    for path in paths:
        name = str(path)
        if name in frozen and digest(path) != frozen[name]:
            raise ValueError(f"Prior frozen input changed: {name}")
        frozen[name] = digest(path)
    if not all(digest(p) == h for p, h in frozen.items()):
        raise ValueError("Prior frozen evidence changed")
    source, cfg, _ = load_checkpoint(Path(SOURCES["initial"]))
    expected_source = prior["models"]["initial"]
    for weight in [0, 1, 100]:
        summary = read(f"docs/evidence/context-contrastive/weight-{weight}-summary.json")
        if (
            digest(f"results/context-contrastive/weight-{weight}/final.mlm")
            != summary["checkpoint_sha256"]
        ):
            raise ValueError("Previous checkpoint changed")
    if (
        state_digest(source) != expected_source["state_sha256"]
        or cfg.model_dump() != expected_source["config"]
    ):
        raise ValueError("Original full source state/configuration changed")
    if int(source.step) != 0:
        raise ValueError("Original step-zero source required")
    batches, ids, qa, documents = prepare(cfg)
    questions = read(QA_DATA / "train-questions.json")["questions"]
    candidates = [
        candidate_batch([q for q in questions if q["group"] == g], cfg.length) for g in ids
    ]
    order, english, _ = seeded_schedule(cfg, documents, len(ids), 44, 45, 128)
    fixed = DocumentStream(documents, cfg.batch_size, cfg.length, 43).next_batch()
    original = state_digest(source)
    out.mkdir(parents=True)
    write_json(
        out / "protocol.json",
        {
            "sha256": frozen,
            "source_state": original,
            "settings": execution_settings(),
            "qa_seed": 44,
            "english_seed": 45,
            "pairs": 128,
            "weights": [0, 100],
            "source_config": cfg.model_dump(),
            "qa_order": [ids[i] for i in order],
            "qa_batches": [fingerprints(b) for b in batches],
            "candidates": [fingerprints(b) for b in candidates],
            "english_batches": [fingerprints(b) for b in english],
            "expected_exposure": {
                str(w): expected_exposure(batches, candidates, order, english, w) for w in [0, 100]
            },
        },
    )
    protocol_hash = digest(out / "protocol.json")
    settings = execution_settings()

    def integrity():
        check_integrity(out / "protocol.json", protocol_hash, settings, frozen)

    @jax.jit
    def ce(state, batch):
        return routed_objective(
            state.params, state.apply_fn, batch, initial_state(cfg, batch["inputs"].shape[0]), cfg
        )[1][1]["loss"]

    def measure(state):
        result = evaluate_questions(state, cfg, questions, True, "chat")
        overall = aggregate(result["items"])
        value = {
            "correct": int(round(overall["accuracy"] * 16)),
            "groups_correct": int(round(overall["binding_group_accuracy"] * 4)),
            "qa_ce": float(ce(state, qa)),
            "english_ce": float(ce(state, fixed)),
            "overall": overall,
            "items": result["items"],
            "step": int(state.step),
        }
        if not np.isfinite([value["qa_ce"], value["english_ce"]]).all():
            raise FloatingPointError("Nonfinite evaluation")
        return value

    write_json(out / "initial.json", measure(source))
    english_step = make_routed_step(cfg)

    def train(weight):
        integrity()
        arm = out / f"weight-{weight}"
        arm.mkdir()
        state = source
        carry = initial_state(cfg, cfg.batch_size)
        step = make_context_step(cfg, weight)
        history = []
        observed = {
            "qa_updates": 0.0,
            "english_updates": 0.0,
            "qa_targets": 0.0,
            "english_targets": 0.0,
            "candidate_positions": 0.0,
            "context_terms": 0.0,
            "qa_bytes": 0.0,
            "english_bytes": 0.0,
        }
        visits = dict.fromkeys(ids, 0)
        expected = read(out / "protocol.json")["expected_exposure"][str(weight)]

        for i, group in enumerate(order):
            state, _, qm = step(
                state,
                initial_state(cfg, batches[group]["inputs"].shape[0]),
                batches[group],
                candidates[group],
            )
            state, carry, em = english_step(state, carry, english[i])
            if not bool(qm["updated"]) or not bool(em["updated"]):
                raise FloatingPointError("Rejected optimizer update")
            visits[ids[group]] += 1
            for name, metrics in [("qa", qm), ("english", em)]:
                observed[name + "_updates"] += int(metrics["updated"])
                observed[name + "_targets"] += float(metrics["positions"])
                observed[name + "_bytes"] += float(metrics["bytes"])
            if weight:
                for name in ["candidate_positions", "context_terms"]:
                    observed[name] += float(qm[name])
            history.append(
                {
                    "pair": i,
                    "group": ids[group],
                    "qa_loss": float(qm["loss"]),
                    "english_loss": float(em["loss"]),
                }
            )
            if (i + 1) % 32 == 0:
                print(json.dumps({"weight": weight, "pairs": i + 1}), flush=True)
        integrity()
        validate_exposure(observed, expected, visits)
        if int(state.step) != 256 or state_digest(source) != original:
            raise ValueError("Step budget or source mutation")
        save_checkpoint(arm / "final.mlm", state, cfg, {"messages": True, "weight": weight})
        restored, _, _ = load_checkpoint(arm / "final.mlm")
        if state_digest(restored) != state_digest(state):
            raise ValueError("Checkpoint roundtrip mismatch")
        result = measure(restored)
        result["state_sha256"] = state_digest(state)
        result["checkpoint_sha256"] = digest(arm / "final.mlm")
        result["exposure"] = {"observed": observed, "expected": expected, "visits": visits}
        result["historical_acquisition_reproduced"] = (
            result["correct"] == 16 and result["groups_correct"] == 4
        )
        result["context_loss"] = float(
            context_objective(
                state.params, state.apply_fn, candidate_batch(questions, cfg.length), cfg
            )[0]
        )
        if not np.isfinite(result["context_loss"]):
            raise FloatingPointError("Nonfinite context loss")
        write_json(arm / "history.json", history)
        write_json(arm / "summary.json", result)
        return result

    control = train(0)
    active = train(100)
    outcome = {
        "weight-100": gates(active, control),
        "historical_acquisition_reproduced": active["historical_acquisition_reproduced"],
        "english_delta_vs_control": active["english_ce"] - control["english_ce"],
        "scope": "one new sampling schedule; same model initialization; training only",
        "protocol_sha256": protocol_hash,
    }
    integrity()
    write_json(out / "summary.json", outcome)
    print(json.dumps(outcome), flush=True)


if __name__ == "__main__":
    run()
