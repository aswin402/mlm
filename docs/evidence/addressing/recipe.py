"""Fixed-endpoint, training-only memory diagnostic. Run from repository root."""

import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from mlm.addressing import question_regions, read_disabled, trace_prompt
from mlm.checkpoint import load_checkpoint
from mlm.comprehension import evaluate_questions
from mlm.data import read_documents, read_training_data
from mlm.retrieval_eval import aggregate
from mlm.stream import evaluate_stream

OUT = Path("results/addressing")
SOURCE = Path("runs/scratch-english-hierarchical-v1/best.mlm")
DATA = Path("data/kv-binding-v1/diverse")


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def setup():
    records = [json.loads(s) for s in (DATA / "train.jsonl").read_text().splitlines()]
    questions = read(DATA / "train-questions.json")
    dev_records = [json.loads(s) for s in (DATA / "validation.jsonl").read_text().splitlines()]
    dev_questions = read(DATA / "validation-questions.json")
    for groups in [1, 4]:
        dest = Path(f"data/addressing-v1/groups-{groups}")
        dest.mkdir(parents=True, exist_ok=False)
        for split, rows, qs, count in [
            ("train", records, questions, groups * 4),
            ("validation", dev_records, dev_questions, 8),
        ]:
            (dest / f"{split}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows[:count])
            )
            save(dest / f"{split}-questions.json", {**qs, "questions": qs["questions"][:count]})
    paths = [
        SOURCE,
        Path("runs/associative-delta-v1/final.mlm"),
        Path("results/associative/initial-delta.mlm"),
        Path("addressing-plan.md"),
        *Path("src/mlm").glob("*.py"),
        *Path("tests").glob("*.py"),
        *Path("data/addressing-v1").rglob("*.json*"),
        Path("configs/associative-none.json"),
        Path("configs/associative-delta.json"),
        DATA / "train.jsonl",
        DATA / "train-questions.json",
        Path("data/hf/stories/train.jsonl"),
        Path("data/hf/stories/validation.jsonl"),
        Path(__file__),
        Path("pyproject.toml"),
        Path("uv.lock"),
    ]
    save(
        OUT / "protocol.json",
        {
            "sha256": {str(p): sha(p) for p in paths},
            "seed": 42,
            "qa_steps_per_probe": 500,
            "english_steps_per_probe": 500,
            "initialization": str(SOURCE),
            "groups": [1, 4],
            "memory": ["none", "delta"],
            "selection": "first 4/16 training records; fixed endpoint; no test opened",
            "trace_selection": "first eight original training prompts; first four per tiny probe",
            "scope": "training-only debugging; single seed; no generalization or promotion claim",
            "ablation": "frozen memory_gain=0; controllers/writes still run; no retraining",
        },
    )


def trace(checkpoint, questions, label):
    state, cfg, _ = load_checkpoint(Path(checkpoint))
    summaries = []
    traces = []
    for q in questions:
        prompt = f"<|user|>\n{q['context']}\n{q['question']}\n<|assistant|>\n"
        row = trace_prompt(state, cfg, prompt)
        regions = question_regions(q)
        coefficients = np.array(row["effective_read_coefficients"])
        contributions = np.array(row["conditional_write_contribution_l2"])
        region_rows = {}
        for name, (start, end) in regions.items():
            region_rows[name] = {
                "beta_mean": float(np.mean(row["betas"][start:end])),
                "coefficient_sum": float(coefficients[start:end].sum()),
                "absolute_coefficient_sum": float(np.abs(coefficients[start:end]).sum()),
                "absolute_coefficient_share": float(
                    np.abs(coefficients[start:end]).sum() / max(np.abs(coefficients).sum(), 1e-30)
                ),
                "contribution_norm_sum": float(contributions[start:end].sum()),
            }
        summary = {
            "id": q["id"],
            "group": q["group"],
            "assignment": q["assignment"],
            "query_entity": q["query_entity"],
            "regions": region_rows,
            "memory_gain": row["memory_gain"],
            "final_residual_to_base_norm": row["memory_residual_norms"][-1]
            / max(row["base_readout_norms"][-1], 1e-30),
            **{k: v for k, v in row.items() if "error" in k or "difference" in k},
        }
        summaries.append(summary)
        traces.append({"question": q, "trace": row, "regions": regions})
    swaps = []
    for i, first in enumerate(traces):
        for second in traces[i + 1 :]:
            a, b = first["question"], second["question"]
            if a["group"] == b["group"] and a["assignment"] == b["assignment"]:
                if a["context"] != b["context"] or a["query_entity"] == b["query_entity"]:
                    raise AssertionError("Invalid query swap")
                row = {"ids": [a["id"], b["id"]]}
                for name in ["queries", "last_recalled"]:
                    x = np.array(first["trace"][name])
                    y = np.array(second["trace"][name])
                    if name == "queries":
                        x, y = x[-1], y[-1]
                    row[name + "_cosine"] = float(
                        x @ y / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-30)
                    )
                    row[name + "_l2_difference"] = float(np.linalg.norm(x - y))
                swaps.append(row)
    save(OUT / f"{label}-traces.json", traces)
    save(OUT / f"{label}-trace-summary.json", {"prompts": summaries, "query_swaps": swaps})
    return {"prompts": summaries, "query_swaps": swaps}


def ablate(checkpoint, data, label):
    state, cfg, _ = load_checkpoint(Path(checkpoint))
    off = read_disabled(state)
    questions = read(Path(data) / "train-questions.json")["questions"]
    rows = {}
    docs, masks, _ = read_training_data(Path(data) / "train.jsonl")
    english = read_documents(Path("data/hf/stories/validation.jsonl"))
    for mode, model in [("on", state), ("off", off)]:
        scores = evaluate_questions(model, cfg, questions, format="chat")
        save(OUT / f"{label}-{mode}-scores.json", scores)
        rows[mode] = {
            "training": aggregate(scores["items"]),
            "answer_training_loss": evaluate_stream(model, cfg, docs, loss_masks=masks)["loss"],
            "english_development_loss": evaluate_stream(model, cfg, english)["loss"],
            "scores": scores["items"],
        }
    rows["prediction_flips"] = sum(
        a["prediction"] != b["prediction"]
        for a, b in zip(rows["on"]["scores"], rows["off"]["scores"], strict=True)
    )
    for mode in ["on", "off"]:
        rows[mode]["mean_correct_minus_wrong_margin"] = float(
            np.mean(
                [
                    x["scores"][x["answer_index"]] - x["scores"][1 - x["answer_index"]]
                    for x in rows[mode].pop("scores")
                ]
            )
        )
    save(OUT / f"{label}-ablation.json", rows)
    return rows


def probe(memory, groups):
    name = f"addressing-{memory}-g{groups}-v1"
    cmd = [
        sys.executable,
        "-m",
        "mlm.plasticity_run",
        "--data",
        f"data/addressing-v1/groups-{groups}",
        "--config",
        f"configs/associative-{memory}.json",
        "--init",
        str(SOURCE),
        "--out",
        f"runs/{name}",
        "--groups",
        str(groups),
        "--steps",
        "500",
        "--rehearsal",
    ]
    with (OUT / f"{name}.log").open("w") as log:
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
    print(f"Finished {name}", flush=True)
    return name


def main():
    setup()
    questions = read(DATA / "train-questions.json")["questions"]
    result = {
        "pilot_initial_trace": trace(
            "results/associative/initial-delta.mlm", questions[:8], "pilot-initial"
        ),
        "pilot_trained_trace": trace(
            "runs/associative-delta-v1/final.mlm", questions[:8], "pilot-trained"
        ),
        "pilot_ablation": ablate("runs/associative-delta-v1/final.mlm", DATA, "pilot"),
    }
    save(OUT / "diagnostics.json", result)
    print("Pilot traces and readout ablation finished", flush=True)
    with ThreadPoolExecutor(max_workers=2) as executor:
        names = list(
            executor.map(
                lambda args: probe(*args), [(m, g) for g in [1, 4] for m in ["none", "delta"]]
            )
        )
    result["probes"] = {}
    for name in names:
        summary = read(Path("runs") / name / "summary.json")
        result["probes"][name] = {
            k: summary[k]
            for k in [
                "groups",
                "initial",
                "final",
                "optimizer_updates",
                "counters",
                "initial_parameters_sha256",
            ]
        }
        if "-delta-" in name:
            data = Path(f"data/addressing-v1/groups-{summary['groups']}")
            qs = read(data / "train-questions.json")["questions"][:4]
            result["probes"][name]["trace"] = trace(Path("runs") / name / "final.mlm", qs, name)
            result["probes"][name]["ablation"] = ablate(
                Path("runs") / name / "final.mlm", data, name
            )
    hashes = read(OUT / "protocol.json")["sha256"]
    audit = {path: sha(path) == expected for path, expected in hashes.items()}
    save(OUT / "integrity.json", audit)
    if not all(audit.values()):
        raise AssertionError("Frozen inputs changed")
    save(OUT / "summary.json", result)
    print("Complete; frozen input hashes verified", flush=True)


if __name__ == "__main__":
    main()
