"""Offline controlled retrieval curriculum; no inference dependency."""

import argparse
import hashlib
import json
import random
from pathlib import Path

from mlm.data import read_documents
from mlm.supervised_data import assistant_record

NAMES = {
    "train": [
        "Ada",
        "Ben",
        "Cal",
        "Dan",
        "Eli",
        "Fay",
        "Gus",
        "Hal",
        "Ian",
        "Jan",
        "Kim",
        "Lou",
        "Max",
        "Ned",
        "Ora",
        "Pam",
    ],
    "validation": ["Ali", "Bea", "Cam", "Don", "Eva", "Flo", "Gil", "Huw"],
    "test": ["Ari", "Bob", "Cyd", "Dee", "Eve", "Fox", "Guy", "Han"],
}
COLORS = ["blue", "gold", "lime", "pink"]
FILLER = " A calm day now."


def make_split(split, seed):
    rng = random.Random(seed)
    names = NAMES[split]
    records, questions = [], []

    def add(facts, query, choices, pair, group, assignment, distractors, delay):
        filler = FILLER * (delay // len(FILLER))
        context = " ".join(f"{n}'s color is {v}." for n, v in facts) + filler
        question = f"What color belongs to {query}?"
        answer_index = choices.index(dict(facts)[query])
        item = {
            "id": f"{pair}-{assignment}",
            "pair": pair,
            "group": group,
            "category": f"d{distractors}-delay{delay}",
            "distractors": distractors,
            "delay_bytes": delay,
            "filler": filler,
            "facts": facts,
            "query_entity": query,
            "assignment": assignment,
            "context": context,
            "question": question,
            "choices": choices,
            "answer_index": answer_index,
        }
        record = assistant_record(
            [
                {"role": "user", "content": context + "\n" + question},
                {"role": "assistant", "content": choices[answer_index]},
            ]
        )
        if len(record["text"].encode()) + 1 > 384:
            raise ValueError("Retrieval example exceeds one training unroll")
        records.append({**record, "source": "synthetic-retrieval", "pair": pair})
        questions.append(item)

    for delay in [0, 64, 128]:
        for n in names:
            values = rng.sample(COLORS, 4)
            for k in [0, 2]:
                choices = values[k : k + 2]
                pair = f"{split}-d0-t{delay}-{n}-{k}"
                for assignment in [0, 1]:
                    add([(n, choices[assignment])], n, choices, pair, pair, assignment, 0, delay)
        for distractors in [1, 3]:
            for i in range(0, len(names), 2):
                a, b = names[i : i + 2]
                values = rng.sample(COLORS, 4)
                others = rng.sample([n for n in names if n not in (a, b)], 2)
                for k in [0, 2]:
                    choices = values[k : k + 2]
                    remaining = [v for v in COLORS if v not in choices]
                    for reverse in [False, True]:
                        group = f"{split}-d{distractors}-t{delay}-{a}-{b}-{k}-{int(reverse)}"
                        for assignment in [0, 1]:
                            facts = [(a, choices[assignment]), (b, choices[1 - assignment])]
                            if distractors == 3:
                                facts += list(zip(others, remaining, strict=True))
                            if reverse:
                                facts = facts[::-1]
                            for query in [a, b]:
                                add(
                                    facts,
                                    query,
                                    choices,
                                    group + "-" + query,
                                    group,
                                    assignment,
                                    distractors,
                                    delay,
                                )
    return records, questions


def prepare(out, source=Path("data/hf"), background=32):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use an empty dataset destination")
    if background < 0 or (background and source is None):
        raise ValueError("Rehearsal requires a source and nonnegative count")
    stories = (
        read_documents(Path(source) / "stories/train.jsonl")[:background] if background else []
    )
    if len(stories) != background:
        raise ValueError("Insufficient rehearsal stories")
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": 1,
        "seed": 27182,
        "names": NAMES,
        "colors": COLORS,
        "grammar": "shared across splits to isolate binding rather than wording transfer",
        "distractors": [0, 1, 3],
        "delay_bytes": [0, 64, 128],
        "unroll": 384,
        "rehearsal_stories": background,
        "source_manifest_sha256": hashlib.sha256(
            (Path(source) / "manifest.json").read_bytes()
        ).hexdigest()
        if background
        else None,
        "limits": "Synthetic task. Names disjoint within task, not necessarily in pretraining.",
        "files": {},
        "splits": {},
    }
    texts = {}
    for offset, split in enumerate(["train", "validation", "test"]):
        records, questions = make_split(split, 27182 + offset)
        texts[split] = {r["text"] for r in records}
        if len(texts[split]) != len(records):
            raise ValueError("Duplicate synthetic documents")
        basic = [
            r
            for r, q in zip(records, questions, strict=True)
            if q["distractors"] == 0 and q["delay_bytes"] == 0
        ]
        basic_questions = [q for q in questions if q["distractors"] == 0 and q["delay_bytes"] == 0]
        manifest["splits"][split] = {
            "synthetic_records": len(records),
            "basic_records": len(basic),
            "binding_groups": len({q["group"] for q in questions if q["distractors"]}),
            "maximum_synthetic_bytes_with_bos": max(len(r["text"].encode()) + 1 for r in records),
        }
        if split == "train":
            records += [{"text": s, "source": "public-story-rehearsal"} for s in stories]
        for name, content in [(f"{split}.jsonl", records), (f"basic-{split}.jsonl", basic)]:
            (out / name).write_text("".join(json.dumps(r) + "\n" for r in content))
        for name, content in [
            (f"{split}-questions.json", questions),
            (f"basic-{split}-questions.json", basic_questions),
        ]:
            (out / name).write_text(
                json.dumps({"version": 1, "split": split, "questions": content}, indent=2) + "\n"
            )
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        if texts[a] & texts[b] or set(NAMES[a]) & set(NAMES[b]):
            raise ValueError("Split overlap")
    for path in sorted(out.iterdir()):
        manifest["files"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/retrieval-v1"))
    parser.add_argument("--source", type=Path, default=Path("data/hf"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.out, args.source), indent=2))


if __name__ == "__main__":
    main()
