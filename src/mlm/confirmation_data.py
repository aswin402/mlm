"""Offline role-balanced confirmation data; roles never enter model prompts."""

import argparse
import hashlib
import itertools
import json
import random
from pathlib import Path

from mlm.gradual_data import group
from mlm.retrieval_data import COLORS


def edge(a, b):
    return tuple(sorted((a, b)))


def matching(names, rng, forbidden):
    for _ in range(1000):
        a = names[::2].copy()
        b = names[1::2].copy()
        rng.shuffle(a)
        rng.shuffle(b)
        half = len(a) // 2
        pairs = list(zip(a[:half:2], a[1:half:2], strict=True))
        pairs += list(zip(b[:half:2], b[1:half:2], strict=True))
        pairs += list(zip(a[half:], b[half:], strict=True))
        if not {edge(*p) for p in pairs} & forbidden:
            return pairs
    raise ValueError("Could not construct disjoint balanced matching")


def cycle(names, rng, forbidden):
    for _ in range(1000):
        order = names.copy()
        rng.shuffle(order)
        pairs = list(zip(order, order[1:] + order[:1], strict=True))
        if not {edge(*p) for p in pairs} & forbidden:
            return pairs
    raise ValueError("Could not construct allowed cycle")


def prepare(out, seed=731029):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use an empty data destination")
    rng = random.Random(seed)
    vocabulary = [
        (a + v + b).title() for a in "bcdfghjk" for v in "aeiou" for b in "lmn rst".replace(" ", "")
    ]
    picked = rng.sample(vocabulary, 64)
    names = {"train": picked[:32], "validation": picked[32:48], "test": picked[48:]}
    roles = {n: i % 2 for pool in names.values() for i, n in enumerate(pool)}
    known_dev = matching(names["train"], rng, set())
    known_test = matching(names["train"], rng, {edge(*p) for p in known_dev})
    forbidden = {edge(*p) for p in known_dev + known_test}
    train_pairs = cycle(names["train"][::2], rng, forbidden)
    train_pairs += cycle(names["train"][1::2], rng, forbidden)
    used = set(forbidden)
    for _ in range(2):
        for attempt in range(1000):
            b = names["train"][1::2].copy()
            rng.shuffle(b)
            pairs = list(zip(names["train"][::2], b, strict=True))
            if not {edge(*p) for p in pairs} & used:
                break
        else:
            raise ValueError("Could not construct cross-role training matching")
        train_pairs += pairs
        used.update(edge(*p) for p in pairs)
    rng.shuffle(train_pairs)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": 1,
        "seed": seed,
        "names": names,
        "offline_roles": roles,
        "limits": "Artificial three-letter labels; disjoint within task, not globally novel.",
        "splits": {},
        "sha256": {},
    }
    texts = {}
    pair_sets = {}
    for split in ["train", "validation", "test"]:
        pools = (
            [("training", train_pairs)]
            if split == "train"
            else [
                ("familiar-names-new-pairs", known_dev if split == "validation" else known_test),
                ("new-names", matching(names[split], rng, set())),
            ]
        )
        records, questions = [], []
        for category, pairs in pools:
            for a, b in pairs:
                values = rng.sample(COLORS, 4)
                colors = (
                    [values[:2], values[2:]]
                    if split == "train"
                    else list(itertools.combinations(COLORS, 2))
                )
                for colors_pair in colors:
                    for reverse in [False, True]:
                        rs, qs = group(
                            a,
                            b,
                            list(colors_pair),
                            reverse,
                            f"{split}-{len(questions) // 4:04}",
                            category,
                        )
                        kind = "within" if roles[a] == roles[b] else "cross"
                        for q in qs:
                            q["pair_type"] = kind
                        records += rs
                        questions += qs
        texts[split] = {r["text"] for r in records}
        pair_sets[split] = {edge(*(n for n, _ in q["facts"])) for q in questions}
        if len(texts[split]) != len(records):
            raise ValueError("Duplicate documents")
        maximum = max(len(r["text"].encode()) + 1 for r in records)
        if maximum > 128:
            raise ValueError("Example exceeds unroll")
        (out / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
        (out / f"{split}-questions.json").write_text(
            json.dumps({"questions": questions}, indent=2) + "\n"
        )
        manifest["splits"][split] = {
            "records": len(records),
            "groups": len(records) // 4,
            "pairs": len(pair_sets[split]),
            "maximum_input_bytes": maximum,
        }
    for a, b in itertools.combinations(texts, 2):
        if texts[a] & texts[b] or pair_sets[a] & pair_sets[b]:
            raise ValueError("Split leakage")
    for path in sorted(out.iterdir()):
        manifest["sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/confirmation-v1"))
    parser.add_argument("--seed", type=int, default=731029)
    args = parser.parse_args()
    print(json.dumps(prepare(args.out, args.seed), indent=2))


if __name__ == "__main__":
    main()
