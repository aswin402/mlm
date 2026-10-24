"""Offline nested binding curriculum with disjoint pair and name transfer sets."""

import argparse
import hashlib
import itertools
import json
import random
from pathlib import Path

from mlm.retrieval_data import COLORS
from mlm.supervised_data import assistant_record

NAMES = ["Dan", "Ned", "Ada", "Ben", "Cal", "Eli", "Fay", "Gus"]
SIZES = [1, 2, 4, 8, 16, 32, 64]
NEW = {
    "validation": ["Abe", "Amy", "Art", "Bex", "Cid", "Dot", "Edd", "Emy"],
    "test": ["Fin", "Gia", "Ivy", "Jax", "Ken", "Liz", "Nat", "Roy"],
}
KNOWN_PAIRS = {
    "validation": [(0, 2), (1, 3), (4, 6), (5, 7)],
    "test": [(0, 3), (1, 2), (4, 7), (5, 6)],
}


def group(a, b, values, reverse, identity, category):
    records, questions = [], []
    for assignment in [0, 1]:
        facts = [(a, values[assignment]), (b, values[1 - assignment])]
        if reverse:
            facts = facts[::-1]
        context = " ".join(f"{n}'s color is {v}." for n, v in facts)
        for query in [a, b]:
            question = f"What color belongs to {query}?"
            answer = values.index(dict(facts)[query])
            pair = identity + "-" + query
            questions.append(
                {
                    "id": f"{pair}-{assignment}",
                    "pair": pair,
                    "group": identity,
                    "category": category,
                    "distractors": 1,
                    "delay_bytes": 0,
                    "assignment": assignment,
                    "facts": facts,
                    "query_entity": query,
                    "context": context,
                    "question": question,
                    "choices": list(values),
                    "answer_index": answer,
                }
            )
            record = assistant_record(
                [
                    {"role": "user", "content": context + "\n" + question},
                    {"role": "assistant", "content": values[answer]},
                ]
            )
            records.append({**record, "source": "synthetic-gradual-binding", "group": identity})
    return records, questions


def prepare(out):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use an empty data destination")
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(56291)
    reserved = {
        tuple(sorted((NAMES[a], NAMES[b]))) for pairs in KNOWN_PAIRS.values() for a, b in pairs
    }
    allowed = [
        (a, b) for a, b in itertools.combinations(NAMES, 2) if tuple(sorted((a, b))) not in reserved
    ]
    chosen = []
    seen = set()

    def add(a, b, values, reverse):
        records, questions = group(
            a, b, list(values), reverse, f"train-{len(chosen):03}", "training"
        )
        signature = frozenset(r["text"] for r in records)
        if signature not in seen:
            seen.add(signature)
            chosen.append((records, questions))

    base = [
        ("Dan", "Ned", ["gold", "blue"]),
        ("Ada", "Ben", ["blue", "gold"]),
        ("Cal", "Eli", ["lime", "pink"]),
        ("Fay", "Gus", ["gold", "pink"]),
    ]
    for a, b, values in base:
        add(a, b, values, False)
    for a, b, values in base:
        add(a, b, values, True)
    for a, b, _ in base:
        for values in [("blue", "pink"), ("gold", "lime")]:
            add(a, b, values, False)
    covered = {tuple(sorted((a, b))) for a, b, _ in base}
    for index, (a, b) in enumerate(allowed):
        if tuple(sorted((a, b))) not in covered:
            add(a, b, list(itertools.combinations(COLORS, 2))[index % 6], False)
    candidates = [
        (a, b, values, reverse)
        for a, b in allowed
        for values in itertools.combinations(COLORS, 2)
        for reverse in [False, True]
    ]
    rng.shuffle(candidates)
    for candidate in candidates:
        if len(chosen) == 64:
            break
        add(*candidate)
    assert len(chosen) == 64
    train_records = [r for records, _ in chosen for r in records]
    train_questions = [q for _, questions in chosen for q in questions]
    source = Path("results/focused-binding/single-group-questions.json")
    seed = json.loads(source.read_text())["questions"]

    def signature(qs):
        return [(q["context"], q["question"], q["choices"][q["answer_index"]]) for q in qs]

    if signature(train_questions[:4]) != signature(seed):
        raise ValueError("Seed group mismatch")
    manifest = {
        "version": 1,
        "seed": 56291,
        "sizes": SIZES,
        "training_names": NAMES,
        "new_names": NEW,
        "known_pairs": KNOWN_PAIRS,
        "rehearsal": False,
        "seed_questions_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "limits": "Names held out within this binding task, not guaranteed globally novel.",
        "splits": {},
        "sha256": {},
    }
    all_records = {"train": train_records}
    all_questions = {"train": train_questions}
    for split in ["validation", "test"]:
        records, questions = [], []
        pools = [
            ("familiar-names-new-pairs", [(NAMES[a], NAMES[b]) for a, b in KNOWN_PAIRS[split]]),
            ("new-names", list(zip(NEW[split][::2], NEW[split][1::2], strict=True))),
        ]
        for category, pairs in pools:
            for a, b in pairs:
                for values in itertools.combinations(COLORS, 2):
                    for reverse in [False, True]:
                        rs, qs = group(
                            a,
                            b,
                            list(values),
                            reverse,
                            f"{split}-{category}-{len(questions) // 4:03}",
                            category,
                        )
                        records += rs
                        questions += qs
        all_records[split] = records
        all_questions[split] = questions
    texts = {}
    for split, records in all_records.items():
        texts[split] = {r["text"] for r in records}
        if len(texts[split]) != len(records):
            raise ValueError("Duplicate documents")
        maximum = max(len(r["text"].encode()) + 1 for r in records)
        if maximum > 128:
            raise ValueError("Truncated curriculum example")
        (out / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
        (out / f"{split}-questions.json").write_text(
            json.dumps({"version": 1, "questions": all_questions[split]}, indent=2) + "\n"
        )
        manifest["splits"][split] = {
            "records": len(records),
            "groups": len(records) // 4,
            "maximum_input_bytes": maximum,
        }
    for a, b in itertools.combinations(texts, 2):
        if texts[a] & texts[b]:
            raise ValueError("Split overlap")
    for path in sorted(out.iterdir()):
        manifest["sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/gradual-v1"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.out), indent=2))


if __name__ == "__main__":
    main()
