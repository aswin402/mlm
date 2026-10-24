"""Audit the crossed cells and compute descriptive contrasts over selected levels."""

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import jax
import numpy as np

from mlm.checkpoint import load_checkpoint
from mlm.retrieval_eval import aggregate
from mlm.training import create_state

OUT = Path("results/addressing-factorial")


def read(path) -> Any:
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score_evidence(items, questions):
    expected = {q["id"]: q for q in questions}
    complete = len(items) == len(questions) and {r["id"] for r in items} == set(expected)
    rows = []
    valid = complete
    for row in items:
        question = expected[row["id"]]
        scores = np.asarray(row["scores"], dtype=float)
        valid = valid and scores.shape == (2,) and bool(np.isfinite(scores).all())
        prediction = int(np.argmax(scores))
        correct = prediction == question["answer_index"]
        valid = valid and all(row.get(k) == v for k, v in question.items())
        valid = valid and row["prediction"] == prediction and row["correct"] == correct
        valid = valid and row["predicted_answer"] == question["choices"][prediction]
        rows.append({**row, "prediction": prediction, "correct": correct})
    return valid, rows


def main():
    protocol = read(OUT / "protocol.json")
    checks = {
        f"frozen:{name}": digest(name) == expected for name, expected in protocol["sha256"].items()
    }
    original, _, _ = load_checkpoint(Path(protocol["english_source"]))
    initializer_states = {
        seed: load_checkpoint(Path(path))[0]
        for seed, path in [
            (42, protocol["initializers"]["42"]),
            (43, protocol["initializers"]["43"]),
        ]
    }
    summaries = {
        int(seed): read(Path(directory) / "summary.json")
        for seed, directory in protocol["diagonal_runs"].items()
    }
    recorded_env = read(OUT / "environment-preflight.json")["current"]
    questions = read("data/addressing-v1/groups-4/train-questions.json")["questions"]
    expected_ids = {q["id"] for q in questions}
    development = read("data/addressing-v1/groups-4/validation-questions.json")["questions"]
    expected_groups = {q["group"] for q in questions}
    cells: dict[str, Any] = {}
    for cell in protocol["cells"]:
        init, schedule = cell["initialization_seed"], cell["schedule_seed"]
        label = f"i{init}-s{schedule}"
        directory = Path(cell["run"])
        summary = read(directory / "summary.json")
        own_protocol = read(directory / "protocol.json")
        initial, cfg, _ = load_checkpoint(directory / "initial.mlm")
        final_state, final_cfg, metadata = load_checkpoint(directory / "final.mlm")
        initializer = initializer_states[init]
        checks[label + ":source_graph_tensors"] = all(
            np.array_equal(np.asarray(v), np.asarray(initial.params[k]))
            for k, v in original.params.items()
        )
        checks[label + ":all_initial_tensors"] = set(initial.params) == set(
            initializer.params
        ) and all(
            np.array_equal(np.asarray(v), np.asarray(initial.params[k]))
            for k, v in initializer.params.items()
        )
        checks[label + ":zero_initial_gate"] = float(initial.params["memory_gain"]) == 0
        checks[label + ":fresh_optimizer"] = int(initial.step) == 0 and all(
            np.array_equal(np.asarray(a), np.asarray(b))
            for a, b in zip(
                jax.tree.leaves(initial.opt_state),
                jax.tree.leaves(create_state(cfg).opt_state),
                strict=True,
            )
        )
        checks[label + ":initial_fingerprint"] = (
            summary["initial_parameters_sha256"] == summaries[init]["initial_parameters_sha256"]
        )
        schedule_cfg = read(Path(protocol["diagonal_runs"][str(schedule)]) / "config.json")
        checks[label + ":schedule_config"] = cfg.model_dump() == schedule_cfg
        checks[label + ":final_config"] = final_cfg == cfg
        checks[label + ":final_step"] = int(final_state.step) == 4000
        checks[label + ":finite_parameters"] = all(
            np.isfinite(np.asarray(v)).all() for v in final_state.params.values()
        )
        checks[label + ":budget"] = (
            summary["initial_optimizer_step"] == 0
            and summary["optimizer_updates"] == 4000
            and all(
                summary["counters"][objective]["updates"] == 2000 for objective in ["qa", "english"]
            )
        )
        checks[label + ":saved_counters"] = summary["counters"] == metadata["counters"]
        checks[label + ":saved_qa_trace"] = (
            summary["update_group_ids"] == metadata["qa_group_trace"]
        )
        checks[label + ":saved_english_stream"] = (
            summary["english_stream"] == metadata["english_stream"]
        )
        checks[label + ":saved_source_hash"] = (
            own_protocol["source_sha256"] == metadata["source_sha256"]
        )
        counts = Counter(summary["update_group_ids"])
        checks[label + ":presentations"] = set(counts) == expected_groups and set(
            counts.values()
        ) == {500}
        reference = summaries[schedule]
        checks[label + ":same_qa_schedule"] = (
            summary["update_group_ids"] == reference["update_group_ids"]
        )
        checks[label + ":same_english_stream"] = (
            summary["english_stream"] == reference["english_stream"]
        )
        checks[label + ":same_byte_budget"] = summary["counters"] == reference["counters"]
        checks[label + ":recorded_environment"] = (
            read(directory / "environment.json") == recorded_env
        )
        scores = read(directory / "final-scores.json")
        train_items = scores["train"]["items"]
        checks[label + ":all_questions"] = (
            len(train_items) == 16 and {q["id"] for q in train_items} == expected_ids
        )
        train_valid, derived_train = score_evidence(train_items, questions)
        dev_valid, derived_dev = score_evidence(scores["development"]["items"], development)
        checks[label + ":question_and_score_evidence"] = train_valid and dev_valid
        checks[label + ":score_metrics"] = aggregate(derived_train) == summary["final"]["training"]
        checks[label + ":development_metrics"] = (
            aggregate(derived_dev) == summary["final"]["development"]
        )
        if cell["kind"] == "new_cross":
            execution = read(OUT / f"{label}-execution.json")
            frozen_env = read(OUT / "environment-preflight.json")
            checks[label + ":child_numerical_settings"] = (
                execution["numerical_settings"] == frozen_env["current_numerical_settings"]
            )
            checks[label + ":child_execution_environment"] = (
                execution["execution_environment"] == frozen_env["current_execution_environment"]
            )
            checks[label + ":child_recorded_environment"] = execution["environment"] == recorded_env
            checks[label + ":source_hash"] = own_protocol["source_sha256"] == digest(
                protocol["initializers"][str(init)]
            )
        final = summary["final"]
        training = final["training"]
        raw = final["fixed_training_raw_samples"]
        checks[label + ":fixed_raw_prompts"] = len(raw) == 4 and all(
            r["id"] == q["id"]
            and r["expected"] == q["choices"][q["answer_index"]]
            and r["prompt"] == f"<|user|>\n{q['context']}\n{q['question']}\n<|assistant|>\n"
            and isinstance(r["raw"], str)
            for r, q in zip(raw, questions[:4], strict=True)
        )
        cells[label] = {
            **cell,
            "final": final,
            "acquisition_pass": training["questions"] == 16
            and training["binding_groups"] == 4
            and training["accuracy"] == 1
            and training["binding_group_accuracy"] == 1,
            "retention_pass": final["english_development_loss"] <= protocol["retention_loss_max"],
            "raw_correct": sum(
                r["raw"].strip().casefold() == r["expected"].casefold() for r in raw
            ),
            "raw_count": len(raw),
            "group_presentations": dict(counts),
            "final_checkpoint_sha256": digest(directory / "final.mlm"),
            "source_sha256": own_protocol["source_sha256"],
        }
    save(OUT / "audit.json", checks)
    if not all(checks.values()):
        raise AssertionError([key for key, ok in checks.items() if not ok])
    contrasts = {}
    for metric in ["binding_group_accuracy", "accuracy"]:
        y00, y01, y10, y11 = [
            cells[k]["final"]["training"][metric]
            for k in ["i42-s42", "i42-s43", "i43-s42", "i43-s43"]
        ]
        contrasts[metric] = {
            "init43_minus42_at_schedule42": y10 - y00,
            "init43_minus42_at_schedule43": y11 - y01,
            "schedule43_minus42_at_init42": y01 - y00,
            "schedule43_minus42_at_init43": y11 - y10,
            "average_init43_minus42": ((y10 - y00) + (y11 - y01)) / 2,
            "average_schedule43_minus42": ((y01 - y00) + (y11 - y10)) / 2,
            "interaction_difference_in_differences": y11 - y10 - y01 + y00,
        }
    save(
        OUT / "summary.json",
        {
            "cells": cells,
            "descriptive_contrasts": contrasts,
            "scope": (
                "post-hoc selected2x2 with historical diagonals; one run per cell; "
                "no universal causal effect or independent replication"
            ),
        },
    )
    print(json.dumps(contrasts, indent=2))
    print(f"All {len(checks)} input/endpoint audit checks passed")


if __name__ == "__main__":
    main()
