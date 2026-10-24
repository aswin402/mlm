"""Audit fixed-initializer QA/replay crossings and descriptive selected-cell contrasts."""

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import jax
import numpy as np

from mlm.checkpoint import load_checkpoint
from mlm.config import Config
from mlm.retrieval_eval import aggregate
from mlm.training import create_state

OUT = Path("results/schedule-components")


def read(path) -> Any:
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


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
    checks = {f"frozen:{p}": digest(p) == h for p, h in protocol["sha256"].items()}
    initializer, _, _ = load_checkpoint(Path(protocol["initializer"]))
    base_cfg = Config.model_validate_json((OUT / "config.json").read_text())
    reference = {
        key: read(Path(directory) / "summary.json")
        for key, directory in protocol["references"].items()
    }
    env = read(OUT / "environment-preflight.json")
    questions = read("data/addressing-v1/groups-4/train-questions.json")["questions"]
    dev = read("data/addressing-v1/groups-4/validation-questions.json")["questions"]
    expected_groups = {q["group"] for q in questions}
    cells: dict[str, Any] = {}
    for cell in protocol["cells"]:
        qa, english = cell["qa_seed"], cell["english_seed"]
        label = f"q{qa}-e{english}"
        directory = Path(cell["run"])
        summary = read(directory / "summary.json")
        own = read(directory / "protocol.json")
        initial, cfg, _ = load_checkpoint(directory / "initial.mlm")
        final_state, final_cfg, metadata = load_checkpoint(directory / "final.mlm")
        checks[label + ":all_initial_tensors"] = set(initial.params) == set(
            initializer.params
        ) and all(
            np.array_equal(np.asarray(value), np.asarray(initial.params[name]))
            for name, value in initializer.params.items()
        )
        checks[label + ":initial_fingerprint"] = (
            summary["initial_parameters_sha256"] == reference["qa42"]["initial_parameters_sha256"]
        )
        checks[label + ":fresh_optimizer"] = int(initial.step) == 0 and all(
            np.array_equal(np.asarray(a), np.asarray(b))
            for a, b in zip(
                jax.tree.leaves(initial.opt_state),
                jax.tree.leaves(create_state(cfg).opt_state),
                strict=True,
            )
        )
        checks[label + ":geometry_and_rates"] = cfg.model_dump(
            exclude={"seed"}
        ) == base_cfg.model_dump(exclude={"seed"})
        checks[label + ":final_config"] = cfg == final_cfg
        checks[label + ":actual_steps"] = (
            int(final_state.step) == summary["optimizer_updates"] == 4000
        )
        checks[label + ":budget"] = summary["initial_optimizer_step"] == 0 and all(
            summary["counters"][k]["updates"] == 2000 for k in ["qa", "english"]
        )
        checks[label + ":finite_final_params"] = all(
            np.isfinite(np.asarray(v)).all() for v in final_state.params.values()
        )
        checks[label + ":saved_counters"] = summary["counters"] == metadata["counters"]
        checks[label + ":saved_qa_trace"] = (
            summary["update_group_ids"] == metadata["qa_group_trace"]
        )
        checks[label + ":saved_english_stream"] = (
            summary["english_stream"] == metadata["english_stream"]
        )
        checks[label + ":saved_source"] = own["source_sha256"] == metadata["source_sha256"]
        counts = Counter(summary["update_group_ids"])
        checks[label + ":exposure"] = set(counts) == expected_groups and set(counts.values()) == {
            500
        }
        checks[label + ":qa_schedule"] = (
            summary["update_group_ids"] == reference[f"qa{qa}"]["update_group_ids"]
        )
        checks[label + ":qa_byte_budget"] = (
            summary["counters"]["qa"] == reference[f"qa{qa}"]["counters"]["qa"]
        )
        checks[label + ":english_schedule"] = (
            summary["english_stream"] == reference[f"english{english}"]["english_stream"]
        )
        checks[label + ":english_byte_budget"] = (
            summary["counters"]["english"] == reference[f"english{english}"]["counters"]["english"]
        )
        actual_env = read(directory / "environment.json")
        checks[label + ":software_backend"] = {
            k: v for k, v in actual_env.items() if k != "source_sha256"
        } == {k: v for k, v in env["current"].items() if k != "source_sha256"}
        if cell["kind"] == "new_cross":
            expected_seeds = {"subset": 42, "qa": qa, "english": english}
            checks[label + ":explicit_seeds"] = (
                summary["sampling_seeds"]
                == own["sampling_seeds"]
                == metadata["sampling_seeds"]
                == expected_seeds
            )
            checks[label + ":fixed_config_seed"] = cfg.seed == 42
            checks[label + ":initializer_source"] = own["source_sha256"] == digest(
                protocol["initializer"]
            )
            checks[label + ":new_source_version"] = actual_env == env["current"]
            execution = read(OUT / f"{label}-execution.json")
            checks[label + ":child_numeric"] = (
                execution["numerical_settings"] == env["current_numerical_settings"]
            )
            checks[label + ":child_environment"] = (
                execution["execution_environment"] == env["current_execution_environment"]
                and execution["environment"] == actual_env
            )
        else:
            checks[label + ":legacy_defaults"] = cfg.seed == qa and cfg.seed + 1 == english
            checks[label + ":legacy_source_version"] = (
                actual_env["source_sha256"] == env["legacy_source_digest"]
            )
        scores = read(directory / "final-scores.json")
        train_ok, train_rows = score_evidence(scores["train"]["items"], questions)
        dev_ok, dev_rows = score_evidence(scores["development"]["items"], dev)
        checks[label + ":all_score_evidence"] = train_ok and dev_ok
        final = summary["final"]
        checks[label + ":training_metrics"] = aggregate(train_rows) == final["training"]
        checks[label + ":dev_metrics"] = aggregate(dev_rows) == final["development"]
        raw = final["fixed_training_raw_samples"]
        checks[label + ":raw_fixed_prompts"] = len(raw) == 4 and all(
            r["id"] == q["id"]
            and r["expected"] == q["choices"][q["answer_index"]]
            and r["prompt"] == f"<|user|>\n{q['context']}\n{q['question']}\n<|assistant|>\n"
            and isinstance(r["raw"], str)
            for r, q in zip(raw, questions[:4], strict=True)
        )
        training = final["training"]
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
            "counters": summary["counters"],
            "final_checkpoint_sha256": digest(directory / "final.mlm"),
        }
    save(OUT / "audit.json", checks)
    if not all(checks.values()):
        raise AssertionError([k for k, v in checks.items() if not v])
    contrasts = {}
    for metric in ["binding_group_accuracy", "accuracy"]:
        y00, y01, y10, y11 = [
            cells[k]["final"]["training"][metric]
            for k in ["q42-e43", "q42-e44", "q43-e43", "q43-e44"]
        ]
        contrasts[metric] = {
            "qa43_minus42_at_english43": y10 - y00,
            "qa43_minus42_at_english44": y11 - y01,
            "english44_minus43_at_qa42": y01 - y00,
            "english44_minus43_at_qa43": y11 - y10,
            "average_qa43_minus42": ((y10 - y00) + (y11 - y01)) / 2,
            "average_english44_minus43": ((y01 - y00) + (y11 - y10)) / 2,
            "interaction_difference_in_differences": y11 - y10 - y01 + y00,
        }
    save(
        OUT / "summary.json",
        {
            "cells": cells,
            "descriptive_contrasts": contrasts,
            "scope": (
                "selected post-hoc QA/English2x2; fixed init42; "
                "historical/current driver compatibility checked; "
                "no general causal or replication claim"
            ),
        },
    )
    print(json.dumps(contrasts, indent=2))
    print(f"All {len(checks)} input/endpoint checks passed")


if __name__ == "__main__":
    main()
