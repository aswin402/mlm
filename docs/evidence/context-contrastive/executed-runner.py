"""Bounded training-only context contrast comparison; original inference unchanged."""

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
    gradient_diagnostic,
    make_context_step,
)
from mlm.future_replication import seeded_schedule
from mlm.hierarchy_run import make_routed_step, routed_objective
from mlm.interference_run import QA_DATA, SOURCES, digest, state_digest
from mlm.model import initial_state
from mlm.pair_probe_run import execution_settings, prepare, read
from mlm.retrieval_eval import aggregate
from mlm.stream import DocumentStream


def gates(active, control):
    primary = (
        active["correct"] > max(8, control["correct"])
        and active["groups_correct"] >= 1
        and active["groups_correct"] > control["groups_correct"]
    )
    english = active["english_ce"] <= control["english_ce"] + 1e-6
    return {"primary": primary, "english": english, "joint": primary and english}


def run(out=Path("results/context-contrastive")):
    out = Path(out)
    if out.exists():
        raise ValueError("Fresh output directory required; preserve completed runs")
    prior = read("docs/evidence/future-byte-answers/protocol.json")
    frozen = dict(prior["sha256"])
    paths = [
        *Path("src/mlm").glob("*.py"),
        *Path("tests").glob("*.py"),
        Path("docs/experiments/context-contrastive/protocol.md"),
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
    if int(source.step) != 0:
        raise ValueError("Original step-zero source required")
    batches, ids, qa, documents = prepare(cfg)
    questions = read(QA_DATA / "train-questions.json")["questions"]
    candidates = [
        candidate_batch([q for q in questions if q["group"] == g], cfg.length) for g in ids
    ]
    order, english, _ = seeded_schedule(cfg, documents, len(ids), 43, 44, 128)
    fixed = DocumentStream(documents, cfg.batch_size, cfg.length, 43).next_batch()
    original = state_digest(source)
    out.mkdir(parents=True)
    write_json(
        out / "protocol.json",
        {
            "sha256": frozen,
            "source_state": original,
            "settings": execution_settings(),
            "qa_seed": 43,
            "english_seed": 44,
            "pairs": 128,
            "weights": [0, 1],
            "conditional_improvement_weight": 100,
        },
    )
    diagnostic = gradient_diagnostic(source, cfg, batches[order[0]], candidates[order[0]])
    write_json(out / "gradient.json", diagnostic)

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
        arm = out / f"weight-{weight}"
        arm.mkdir()
        state = source
        carry = initial_state(cfg, cfg.batch_size)
        step = make_context_step(cfg, weight)
        history = []
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
        if int(state.step) != 256 or state_digest(source) != original:
            raise ValueError("Step budget or source mutation")
        save_checkpoint(arm / "final.mlm", state, cfg, {"messages": True, "weight": weight})
        restored, _, _ = load_checkpoint(arm / "final.mlm")
        if state_digest(restored) != state_digest(state):
            raise ValueError("Checkpoint roundtrip mismatch")
        result = measure(restored)
        result["state_sha256"] = state_digest(state)
        result["checkpoint_sha256"] = digest(arm / "final.mlm")
        result["candidate_positions"] = 4096 if weight else 0
        result["context_terms"] = 512 if weight else 0
        result["context_loss"] = float(
            context_objective(
                state.params, state.apply_fn, candidate_batch(questions, cfg.length), cfg
            )[0]
        )
        write_json(arm / "history.json", history)
        write_json(arm / "summary.json", result)
        return result

    control = train(0)
    active = train(1)
    outcome = {"weight-1": gates(active, control), "scope": "training-only; one sampling schedule"}
    ratio = diagnostic["ratio"]
    if not outcome["weight-1"]["primary"] and ratio is not None and 0 < ratio < 0.01:
        write_json(
            out / "improvement-protocol.json",
            {
                "reason": "Predeclared small-gradient rescaling after failed acquisition",
                "weight": 100,
                "ratio": ratio,
                "source_state": original,
                "base_protocol_sha256": digest(out / "protocol.json"),
                "control_summary_sha256": digest(out / "weight-0/summary.json"),
                "weight1_summary_sha256": digest(out / "weight-1/summary.json"),
            },
        )
        improved = train(100)
        outcome["weight-100"] = gates(improved, control)
    if not all(digest(p) == h for p, h in frozen.items()):
        raise ValueError("Frozen inputs changed during run")
    write_json(out / "summary.json", outcome)
    print(json.dumps(outcome), flush=True)


if __name__ == "__main__":
    run()
