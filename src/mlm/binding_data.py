"""Offline two-person binding groups; no dependency from neural inference."""

import argparse
import hashlib
import itertools
import json
import random
from pathlib import Path

from mlm.retrieval_data import COLORS
from mlm.retrieval_data import NAMES as RETRIEVAL_NAMES
from mlm.supervised_data import assistant_record

NAMES = {
    "train": list(RETRIEVAL_NAMES["train"]),
    "validation": ["Ira", "Jen", "Kit", "Lee", "Moe", "Neo", "Ona", "Pip"],
    "test": ["Ida", "Jay", "Kay", "Len", "Mel", "Nia", "Oli", "Pat"],
}


def make_split(split, seed):
    rng = random.Random(seed)
    pairs = list(itertools.combinations(NAMES[split], 2))
    values = list(itertools.combinations(COLORS, 2))
    rng.shuffle(pairs)
    rng.shuffle(values)
    records, questions = [], []
    for index, (a, b) in enumerate(pairs):
        choices = list(values[index % len(values)])
        rng.shuffle(choices)
        for reverse in [False, True]:
            group = f"{split}-{index:04}-{int(reverse)}"
            for assignment in [0, 1]:
                facts = [(a, choices[assignment]), (b, choices[1 - assignment])]
                if reverse:
                    facts = facts[::-1]
                context = " ".join(f"{n}'s color is {v}." for n, v in facts)
                for query in [a, b]:
                    pair = group + "-" + query
                    question = f"What color belongs to {query}?"
                    answer = choices.index(dict(facts)[query])
                    record = assistant_record(
                        [
                            {"role": "user", "content": context + "\n" + question},
                            {"role": "assistant", "content": choices[answer]},
                        ]
                    )
                    records.append({**record, "source": "synthetic-binding", "group": group})
                    questions.append(
                        {
                            "id": f"{pair}-{assignment}",
                            "pair": pair,
                            "group": group,
                            "category": "two-person-no-delay",
                            "distractors": 1,
                            "delay_bytes": 0,
                            "assignment": assignment,
                            "facts": facts,
                            "query_entity": query,
                            "context": context,
                            "question": question,
                            "choices": choices,
                            "answer_index": answer,
                        }
                    )
    return records, questions


def prepare(out):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Choose an empty dataset destination")
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": 1,
        "seed": 91827,
        "names": NAMES,
        "colors": COLORS,
        "rehearsal": False,
        "group_batch_size": 4,
        "unroll": 128,
        "limits": "Shared grammar, disjoint task names; not globally unseen names.",
        "splits": {},
        "sha256": {},
    }
    sets = {}
    for offset, split in enumerate(["train", "validation", "test"]):
        records, questions = make_split(split, 91827 + offset)
        texts = {r["text"] for r in records}
        sets[split] = texts
        if len(texts) != len(records):
            raise ValueError("Duplicate examples")
        maximum = max(len(r["text"].encode()) + 1 for r in records)
        if maximum > 128:
            raise ValueError("Example exceeds unroll")
        (out / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
        (out / f"{split}-questions.json").write_text(
            json.dumps({"version": 1, "questions": questions}, indent=2) + "\n"
        )
        manifest["splits"][split] = {
            "documents": len(records),
            "groups": len(records) // 4,
            "entity_pairs": len(NAMES[split]) * (len(NAMES[split]) - 1) // 2,
            "maximum_input_bytes": maximum,
        }
    for a, b in itertools.combinations(sets, 2):
        if sets[a] & sets[b] or set(NAMES[a]) & set(NAMES[b]):
            raise ValueError("Split overlap")
    for path in sorted(out.iterdir()):
        manifest["sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/binding-v1"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.out), indent=2))


if __name__ == "__main__":
    main()
