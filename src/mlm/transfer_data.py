"""Offline balanced binding diversity/wording fixtures; never an inference solver."""

import argparse
import hashlib
import itertools
import json
import random
from pathlib import Path

from mlm.confirmation_data import edge, matching
from mlm.data import encode
from mlm.gradual_data import group
from mlm.retrieval_data import COLORS
from mlm.supervised_data import assistant_record

FACT_FORMS = [
    "{name}'s color is {value}.",
    "{name} has color {value}.",
    "{name}'s chosen color is {value}.",
]
QUESTION_FORMS = [
    "What color belongs to {name}?",
    "Which color does {name} have?",
    "What is the color of {name}?",
]


def render(a, b, values, reverse, identity, category, roles, fact_form=0, question_form=0):
    _, questions = group(a, b, values, reverse, identity, category)
    records = []
    for q in questions:
        q["context"] = " ".join(
            FACT_FORMS[fact_form].format(name=n, value=v) for n, v in q["facts"]
        )
        q["question"] = QUESTION_FORMS[question_form].format(name=q["query_entity"])
        q["wording"] = (
            "heldout" if (fact_form, question_form) == (2, 2) else f"f{fact_form}q{question_form}"
        )
        q["pair_type"] = "within" if roles[a] == roles[b] else "cross"
        record = assistant_record(
            [
                {"role": "user", "content": q["context"] + "\n" + q["question"]},
                {"role": "assistant", "content": q["choices"][q["answer_index"]]},
            ]
        )
        records.append({**record, "group": identity, "source": "synthetic-transfer"})
    return records, questions


def prepare(out, seed=904217):
    out = Path(out)
    if out.exists():
        raise ValueError("Choose a fresh data directory")
    rng = random.Random(seed)
    vocabulary = [(a + v + b).title() for a in "bcdfghjk" for v in "aeiou" for b in "lmnrst"]
    picked = rng.sample(vocabulary, 48)
    names = {"train": picked[:16], "validation": picked[16:32], "test": picked[32:]}
    roles = {n: i % 2 for pool in names.values() for i, n in enumerate(pool)}
    dev_pairs = matching(names["train"], rng, set())
    test_pairs = matching(names["train"], rng, {edge(*p) for p in dev_pairs})
    reserved = {edge(*p) for p in dev_pairs + test_pairs}
    fixed_pairs = matching(names["train"], rng, reserved)
    fixed_edges = {edge(*p) for p in fixed_pairs}
    available = [
        p
        for p in itertools.combinations(names["train"], 2)
        if edge(*p) not in reserved | fixed_edges
    ]
    extra = []
    for kind in [True, False]:
        candidates = [p for p in available if (roles[p[0]] == roles[p[1]]) == kind]
        extra += rng.sample(candidates, 12)
    train_pairs = fixed_pairs + extra
    rng.shuffle(train_pairs)
    combinations = list(itertools.combinations(COLORS, 2))
    rng.shuffle(combinations)
    styles = [(0, 0), (0, 1), (1, 0), (1, 1)] * 16
    rng.shuffle(styles)
    training = {arm: ([], []) for arm in ["fixed", "diverse", "mixed"]}
    for pi, (a, b) in enumerate(train_pairs):
        first = list(combinations[pi % 6])
        complement = [v for v in COLORS if v not in first]
        for ci, values in enumerate([first, complement]):
            style = styles[pi * 2 + ci]
            for reverse in [False, True]:
                identity = f"train-{pi * 4 + ci * 2 + int(reverse):04d}"
                canonical = render(a, b, values, reverse, identity, "training", roles)
                alternate = render(a, b, values, reverse, identity, "training", roles, *style)
                for arm, payload in [("diverse", canonical), ("mixed", alternate)]:
                    training[arm][0].extend(payload[0])
                    training[arm][1].extend(payload[1])
                if edge(a, b) in fixed_edges and ci == 0:
                    training["fixed"][0].extend(canonical[0])
                    training["fixed"][1].extend(canonical[1])
    evaluation = {}
    for split, known in [("validation", dev_pairs), ("test", test_pairs)]:
        records = []
        questions = []
        pools = [
            ("familiar-names-new-pairs", known),
            ("new-names", matching(names[split], rng, set())),
        ]
        for pool, pairs in pools:
            for wording, fi, qi in [("base", 0, 0), ("alternate", 1, 1), ("heldout", 2, 2)]:
                category = pool + "/" + wording
                for a, b in pairs:
                    for values in itertools.combinations(COLORS, 2):
                        for reverse in [False, True]:
                            identity = f"{split}-{len(questions) // 4:05d}"
                            rs, qs = render(
                                a, b, list(values), reverse, identity, category, roles, fi, qi
                            )
                            records += rs
                            questions += qs
        evaluation[split] = (records, questions)
    all_texts = {arm: {r["text"] for r in rows[0]} for arm, rows in training.items()}
    for split, (records, _) in evaluation.items():
        texts = {r["text"] for r in records}
        if len(texts) != len(records) or texts & set.union(*all_texts.values()):
            raise ValueError("Duplicate or overlapping evaluation texts")
    if {r["text"] for r in evaluation["validation"][0]} & {
        r["text"] for r in evaluation["test"][0]
    }:
        raise ValueError("Development/test text overlap")
    payloads = {}
    manifest = {
        "version": 1,
        "seed": seed,
        "names": names,
        "offline_roles": roles,
        "templates": {"facts": FACT_FORMS, "questions": QUESTION_FORMS, "heldout_indices": [2]},
        "reserved_pairs": {"validation": dev_pairs, "test": test_pairs},
        "fixed_pairs": fixed_pairs,
        "train_pairs": train_pairs,
        "arms": {},
        "sha256": {},
        "limits": (
            "Task-level name/pair/wording splits; shared colors and grammar; "
            "not global name novelty."
        ),
    }
    for arm, train in training.items():
        splits = {"train": train, **evaluation}
        manifest["arms"][arm] = {}
        for split, (records, questions) in splits.items():
            max_input = max(len(encode(r["text"])) - 1 for r in records)
            if max_input > 128:
                raise ValueError("Example exceeds full-gradient unroll")
            if len({r["text"] for r in records}) != len(records):
                raise ValueError("Duplicate training documents")
            manifest["arms"][arm][split] = {
                "records": len(records),
                "groups": len(records) // 4,
                "maximum_input_bytes": max_input,
            }
            payloads[f"{arm}/{split}.jsonl"] = "".join(json.dumps(r) + "\n" for r in records)
            payloads[f"{arm}/{split}-questions.json"] = (
                json.dumps({"questions": questions}, indent=2) + "\n"
            )
    out.mkdir(parents=True)
    for relative, text in payloads.items():
        path = out / relative
        path.parent.mkdir(exist_ok=True)
        path.write_text(text)
        manifest["sha256"][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("data/transfer-v1"))
    p.add_argument("--seed", type=int, default=904217)
    args = p.parse_args()
    manifest = prepare(args.out, args.seed)
    print(json.dumps(manifest["arms"], indent=2))


if __name__ == "__main__":
    main()
