"""Offline supervised labels and assistant spans; never imported by inference."""

import argparse
import hashlib
import json
import random
from pathlib import Path

from mlm.data import read_documents
from mlm.hfdata import render_row

NAMES = {
    "train": ["Ava", "Ben", "Eli", "Faye", "Gus", "Hana", "Ian", "Jade"],
    "validation": ["Kira", "Leo", "Milo", "Nia"],
    "test": ["Omar", "Pia", "Remy", "Suki"],
}
WORDS = {"color": ["red", "blue", "green", "white"], "location": ["box", "bag", "desk", "shelf"]}


def assistant_record(messages):
    pieces, spans, offset = [], [], 0
    for message in messages:
        role = message["role"]
        if role not in ("system", "user", "assistant"):
            raise ValueError("Unknown role")
        header = f"<|{role}|>\n"
        content = message["content"].strip() + "\n"
        pieces.extend([header, content])
        offset += len(header.encode("utf-8"))
        start = offset
        offset += len(content.encode("utf-8"))
        if role == "assistant":
            spans.append([start, offset])
    if not spans:
        raise ValueError("Conversation has no assistant targets")
    return {
        "text": "".join(pieces),
        "loss_spans": spans,
        "score_eos": messages[-1]["role"] == "assistant",
    }


def reading_split(split, pair_count, seed):
    rng = random.Random(seed)
    questions, records, seen = [], [], set()
    while len(seen) < pair_count:
        category = "color" if len(seen) % 2 == 0 else "location"
        name, other = rng.sample(NAMES[split], 2)
        first, second = rng.sample(WORDS[category], 2)
        obj = rng.choice(
            ["hat", "coat", "scarf", "shirt"]
            if category == "color"
            else ["ball", "book", "coin", "key"]
        )
        order = rng.randrange(2)
        variant = rng.randrange(2) if split == "train" else 0
        if category == "color":
            if split == "train":
                facts = ("{n} owns a {v} {o}.", "{n} has a {v} {o}.")
                question = (
                    f"What color is {name}'s {obj}?"
                    if variant == 0
                    else f"Tell me the color of {name}'s {obj}."
                )
            elif split == "validation":
                facts = ("The {o} belonging to {n} is {v}.",)
                question = f"Which color belongs to the {obj} owned by {name}?"
            else:
                facts = ("According to the note, {n}'s {o} is {v}.",)
                question = f"From the note, identify the color of {name}'s {obj}."
        else:
            if split == "train":
                facts = ("{n} put the {o} in the {v}.", "{n} keeps the {o} in the {v}.")
                question = (
                    f"Where does {name} keep the {obj}?"
                    if variant == 0
                    else f"Where did {name} put the {obj}?"
                )
            elif split == "validation":
                facts = ("The {o} owned by {n} is inside the {v}.",)
                question = f"In which place did {name} leave the {obj}?"
            else:
                facts = ("A note says {n} left the {o} in the {v}.",)
                question = f"From the note, where can {name} find the {obj}?"
        template = facts[variant]
        contexts = []
        for a, b in ((first, second), (second, first)):
            sentences = [template.format(n=name, v=a, o=obj), template.format(n=other, v=b, o=obj)]
            contexts.append(" ".join(sentences[:: (-1 if order else 1)]))
        key = (tuple(sorted(contexts)), question)
        if key in seen:
            continue
        seen.add(key)
        pair = f"{split}-{category}-{len(seen):04}"
        for i, context in enumerate(contexts):
            choices = [first, second]
            questions.append(
                {
                    "id": f"{pair}-{i}",
                    "pair": pair,
                    "category": category,
                    "context": context,
                    "question": question,
                    "choices": choices,
                    "answer_index": i,
                    "names": [name, other],
                    "template_partition": split,
                }
            )
            record = assistant_record(
                [
                    {"role": "user", "content": context + "\n" + question},
                    {"role": "assistant", "content": choices[i]},
                ]
            )
            records.append({**record, "source": "synthetic-reading", "pair": pair})
    return records, questions


def cached_conversations(root):
    manifest = json.loads((root / "manifest.json").read_text())
    lookup = {}
    for page in manifest["sources"]["conversation"]["pages"]:
        raw = (root / page["cache"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != page["sha256"]:
            raise ValueError("Cached dataset response hash mismatch")
        for item in json.loads(raw)["rows"]:
            row = item["row"]
            lookup[render_row(row)] = {
                **assistant_record(row["messages"]),
                "source": "public-conversation",
            }
    return lookup


def prepare(out, source, pairs=800, heldout_pairs=64, repeats=8, background=64):
    out, source = Path(out), Path(source)
    if (out / "manifest.json").exists():
        raise ValueError("Prepared dataset already exists; use a new output path")
    out.mkdir(parents=True, exist_ok=True)
    lookup = cached_conversations(source)
    manifest = {
        "purpose": "assistant-only and reading supervised learning",
        "seed": 13,
        "names": NAMES,
        "qa_repeats": repeats,
        "background_per_source": background,
        "source_manifest_sha256": hashlib.sha256(
            (source / "manifest.json").read_bytes()
        ).hexdigest(),
        "original_reading_suite_sha256": hashlib.sha256(
            Path("data/evaluation/reading-v1.json").read_bytes()
        ).hexdigest(),
        "note": (
            "Names and wording are disjoint within the supervised reading task; "
            "pretraining may have encountered those names. Label rules are offline only."
        ),
        "splits": {},
    }
    all_texts = {}
    for offset, split in enumerate(("train", "validation", "test")):
        records, questions = reading_split(
            split, pairs if split == "train" else heldout_pairs, 13 + offset
        )
        all_texts[split] = {r["text"] for r in records}
        if split == "train":
            records = records * repeats
            chats = read_documents(source / "conversation/train.jsonl")[:background]
            stories = read_documents(source / "stories/train.jsonl")[:background]
            records += [lookup[text] for text in chats]
            records += [{"text": text, "source": "story-rehearsal"} for text in stories]
        path = out / f"{split}.jsonl"
        path.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )
        (out / f"{split}-questions.json").write_text(
            json.dumps(
                {
                    "version": 2,
                    "split": split,
                    "purpose": "supervised train" if split == "train" else "held-out diagnostic",
                    "questions": questions,
                },
                indent=2,
            )
            + "\n"
        )
        manifest["splits"][split] = {
            "records": len(records),
            "unique_reading_documents": len(all_texts[split]),
            "questions": len(questions),
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if all_texts[a] & all_texts[b] or set(NAMES[a]) & set(NAMES[b]):
            raise ValueError("Supervised split overlap")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/assistant-v2"))
    parser.add_argument("--source", type=Path, default=Path("data/hf"))
    args = parser.parse_args()
    prepare(args.out, args.source)


if __name__ == "__main__":
    main()
