"""Frozen neural retrieval scoring; labels only grade predictions."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from mlm.checkpoint import load_checkpoint
from mlm.comprehension import evaluate_questions
from mlm.generate import generate


def aggregate(items):
    if not items:
        raise ValueError("Nonempty retrieval results required")
    pairs, groups = defaultdict(list), defaultdict(list)
    for item in items:
        pairs[item["pair"]].append(item)
        groups[item["group"]].append(item)
    if any(len(p) != 2 for p in pairs.values()):
        raise ValueError("Incomplete context-swap pair")
    binding = [rows for rows in groups.values() if rows[0]["distractors"]]
    query_groups = []
    for rows in binding:
        if len(rows) != 4:
            raise ValueError("Incomplete binding group")
        for assignment in [0, 1]:
            queries = [r for r in rows if r["assignment"] == assignment]
            if (
                len(queries) != 2
                or queries[0]["context"] != queries[1]["context"]
                or queries[0]["query_entity"] == queries[1]["query_entity"]
            ):
                raise ValueError("Invalid fixed-context query swap")
            query_groups.append(queries)
    return {
        "questions": len(items),
        "pairs": len(pairs),
        "binding_groups": len(binding),
        "accuracy": sum(r["correct"] for r in items) / len(items),
        "pair_accuracy": sum(all(r["correct"] for r in rows) for rows in pairs.values())
        / len(pairs),
        "binding_group_accuracy": (
            sum(all(r["correct"] for r in rows) for rows in binding) / len(binding)
        )
        if binding
        else None,
        "query_flip_fraction": (
            sum(rows[0]["prediction"] != rows[1]["prediction"] for rows in query_groups)
            / len(query_groups)
        )
        if query_groups
        else None,
        "correct_query_pair_fraction": (
            sum(all(r["correct"] for r in rows) for rows in query_groups) / len(query_groups)
        )
        if query_groups
        else None,
        "context_flip_fraction": sum(
            rows[0]["prediction"] != rows[1]["prediction"] for rows in pairs.values()
        )
        / len(pairs),
        "both_wrong_pair_fraction": sum(
            not any(r["correct"] for r in rows) for rows in pairs.values()
        )
        / len(pairs),
    }


def evaluate(checkpoint, questions_path, out, greedy=True):
    state, config, meta = load_checkpoint(checkpoint)
    questions = json.loads(questions_path.read_text())["questions"]
    result = evaluate_questions(state, config, questions, meta.get("messages", True), "chat")
    result["overall"] = aggregate(result["items"])
    cells = defaultdict(list)
    for row in result["items"]:
        cells[row["category"]].append(row)
    result["cells"] = {key: aggregate(rows) for key, rows in sorted(cells.items())}
    result["checkpoint"] = str(checkpoint)
    result["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    result["questions_sha256"] = hashlib.sha256(questions_path.read_bytes()).hexdigest()
    result["architecture"] = config.architecture
    if greedy:
        # Fixed first complete group from every cell, chosen without inspecting model outputs.
        selected = []
        for rows in cells.values():
            group = rows[0]["group"]
            selected += [row for row in rows if row["group"] == group]
        samples: list[dict[str, Any]] = []
        for row in selected:
            prompt = f"<|user|>\n{row['context']}\n{row['question']}\n<|assistant|>\n"
            raw = generate(state, config, prompt, 12, messages=meta.get("messages", True))
            expected = row["choices"][row["answer_index"]]
            samples.append(
                {
                    "id": row["id"],
                    "category": row["category"],
                    "prompt": prompt,
                    "raw": raw,
                    "expected": expected,
                    "correct": raw.strip().casefold() == expected.casefold(),
                }
            )
        result["greedy"] = {
            "max_bytes": 12,
            "selection": "first complete group per cell",
            "normalization": "strip and casefold; no extraction or repair",
            "accuracy": sum(r["correct"] for r in samples) / len(samples),
            "samples": samples,
        }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"architecture": config.architecture, **result["overall"]}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-greedy", action="store_true")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.questions, args.out, greedy=not args.no_greedy)


if __name__ == "__main__":
    main()
