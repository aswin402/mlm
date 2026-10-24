"""Reproduce frozen supervised-reading diagnostics and raw greedy answers."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from mlm.checkpoint import load_checkpoint
from mlm.comprehension import evaluate_questions
from mlm.generate import generate


def evaluate(checkpoint, questions_file, original_file, out):
    state, config, meta = load_checkpoint(checkpoint)
    result: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "architecture": config.architecture,
        "step": int(state.step),
    }
    for name, path in (("heldout", questions_file), ("original", original_file)):
        questions = json.loads(path.read_text())["questions"]
        result[name] = evaluate_questions(
            state, config, questions, messages=meta.get("messages", True), format="chat"
        )
        result[name]["questions_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    questions = json.loads(questions_file.read_text())["questions"][:16]
    samples: list[dict[str, Any]] = []
    for question in questions:
        prompt = f"<|user|>\n{question['context']}\n{question['question']}\n<|assistant|>\n"
        raw = generate(state, config, prompt, 16, messages=meta.get("messages", True))
        expected = question["choices"][question["answer_index"]]
        samples.append(
            {
                "id": question["id"],
                "pair": question["pair"],
                "prompt": prompt,
                "raw": raw,
                "expected": expected,
                "correct": raw.strip().casefold() == expected.casefold(),
            }
        )
    result["greedy"] = {
        "max_bytes": 16,
        "selection": "first 16 held-out questions, fixed before inspection",
        "normalization": "strip surrounding whitespace and casefold; no answer extraction",
        "accuracy": sum(row["correct"] for row in samples) / len(samples),
        "samples": samples,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(out)
    print(
        json.dumps(
            {
                "heldout_accuracy": result["heldout"]["accuracy"],
                "heldout_pair_accuracy": result["heldout"]["pair_accuracy"],
                "greedy_accuracy": result["greedy"]["accuracy"],
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--questions", type=Path, default=Path("data/assistant-v2/test-questions.json")
    )
    parser.add_argument("--original", type=Path, default=Path("data/evaluation/reading-v1.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.questions, args.original, args.out)


if __name__ == "__main__":
    main()
