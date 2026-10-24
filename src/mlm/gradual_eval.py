"""Frozen transfer evaluation plus correctness across both fact orders."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from mlm.retrieval_eval import evaluate as score_questions


def order_robust(items):
    if not items:
        raise ValueError("Nonempty scored questions required")
    cases = defaultdict(list)
    for row in items:
        names = tuple(sorted(n for n, _ in row["facts"]))
        values = tuple(sorted(row["choices"]))
        if len(names) != 2 or len(set(names)) != 2 or len(set(values)) != 2:
            raise ValueError("Expected two people and two distinct colors")
        cases[(row["category"], names, values)].append(row)
    histogram = Counter()
    for rows in cases.values():
        if len(rows) != 8:
            raise ValueError("Each case needs all eight fact-order/query swaps")
        conditions = set()
        for row in rows:
            if set(v for _, v in row["facts"]) != set(row["choices"]):
                raise ValueError("Color assignments do not match candidates")
            if row["query_entity"] not in dict(row["facts"]):
                raise ValueError("Unknown queried entity")
            conditions.add(
                (
                    tuple(sorted(tuple(fact) for fact in row["facts"])),
                    tuple(n for n, _ in row["facts"]),
                    row["query_entity"],
                )
            )
        if len(conditions) != 8:
            raise ValueError("Duplicate or missing case conditions")
        histogram[sum(row["correct"] for row in rows)] += 1
    return {
        "cases": len(cases),
        "all_correct": histogram[8],
        "accuracy": histogram[8] / len(cases),
        "correct_count_histogram": {str(k): v for k, v in sorted(histogram.items())},
        "definition": "all eight assignment/query/fact-order conditions correct",
    }


def evaluate(checkpoint, questions, out):
    result = score_questions(checkpoint, questions, out)
    result["order_robust"] = order_robust(result["items"])
    result["order_robust_pools"] = {
        category: order_robust([r for r in result["items"] if r["category"] == category])
        for category in result["cells"]
    }
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["order_robust_pools"], indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.questions, args.out)


if __name__ == "__main__":
    main()
