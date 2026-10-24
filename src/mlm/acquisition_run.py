"""Independent nested-set acquisition diagnostics; development-only, no test selection."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from mlm.binding_run import build_batches, warm_start
from mlm.checkpoint import load_checkpoint, save_checkpoint
from mlm.cli import environment, write_json
from mlm.comprehension import evaluate_questions
from mlm.config import Config
from mlm.data import read_documents, read_training_data
from mlm.generate import generate
from mlm.gradual_eval import order_robust
from mlm.hierarchy_run import make_routed_step
from mlm.model import initial_state
from mlm.retrieval_eval import aggregate
from mlm.stream import evaluate_stream


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nested_groups(questions, sizes, seed):
    counts = Counter(q["group"] for q in questions)
    if not counts or any(n != 4 for n in counts.values()):
        raise ValueError("Complete four-question groups required")
    if not sizes or sizes != sorted(set(sizes)) or sizes[0] < 1 or sizes[-1] > len(counts):
        raise ValueError("Strictly increasing positive subset sizes within training pool required")
    order = np.random.default_rng(seed).permutation(list(counts)).tolist()
    return {size: order[:size] for size in sizes}


def run(args):
    if args.steps < 1:
        raise ValueError("Positive update budget required")
    if args.out.exists():
        raise ValueError("Choose a fresh output directory")
    cfg = Config.model_validate_json(args.config.read_text())
    _, source_cfg, _ = load_checkpoint(args.init)
    fields = {
        "architecture",
        "node_update",
        "node_readout",
        "groups",
        "nodes",
        "width",
        "degree",
        "rounds",
    }
    if cfg.model_dump(include=fields) != source_cfg.model_dump(include=fields):
        raise ValueError("Own-architecture source required")
    records = [json.loads(line) for line in (args.data / "train.jsonl").read_text().splitlines()]
    questions = json.loads((args.data / "train-questions.json").read_text())["questions"]
    subsets = nested_groups(questions, args.sizes, cfg.seed)
    if len(records) != len(questions) or any(
        r["group"] != q["group"] for r, q in zip(records, questions, strict=True)
    ):
        raise ValueError("Aligned training groups required")
    dev_questions = json.loads((args.data / "validation-questions.json").read_text())["questions"]
    dev, dev_masks, _ = read_training_data(args.data / "validation.jsonl")
    english = read_documents(args.english_dev)
    if {r["text"] for r in records} & set(dev):
        raise ValueError("Training/development overlap")
    source_hash = digest(args.init)
    paths = [
        args.config,
        args.init,
        args.english_dev,
        Path(__file__),
        *[
            args.data / name
            for name in [
                "train.jsonl",
                "train-questions.json",
                "validation.jsonl",
                "validation-questions.json",
            ]
        ],
    ]
    protocol = {
        "source_sha256": source_hash,
        "sha256": {str(p): digest(p) for p in paths},
        "sizes": args.sizes,
        "steps_each": args.steps,
        "seed": cfg.seed,
        "subsets": subsets,
        "selection": "none; initial and fixed endpoint only; no test data opened",
        "scope": "independent task-only trials; no English rehearsal; not capacity proof",
    }
    args.out.mkdir(parents=True)
    write_json(args.out / "protocol.json", protocol)
    write_json(args.out / "config.json", cfg.model_dump())
    write_json(args.out / "environment.json", environment())
    step = make_routed_step(cfg)
    trials = []
    for size, identities in subsets.items():
        chosen = set(identities)
        pairs = [(r, q) for r, q in zip(records, questions, strict=True) if q["group"] in chosen]
        selected_records = [r for r, _ in pairs]
        selected_questions = [q for _, q in pairs]
        batches, batch_ids = build_batches(selected_records, selected_questions, cfg)
        state = warm_start(args.init, cfg)
        initial_hash = hashlib.sha256(serialization.to_bytes(state.params)).hexdigest()
        initial_step = int(state.step)
        out = args.out / f"groups-{size}"
        out.mkdir()
        names = sorted({n for q in selected_questions for n, _ in q["facts"]})
        entity_pairs = {tuple(sorted(n for n, _ in q["facts"])) for q in selected_questions}
        color_pairs = {tuple(sorted(q["choices"])) for q in selected_questions}

        def assess(label):
            train = evaluate_questions(state, cfg, selected_questions, format="chat")
            held = evaluate_questions(state, cfg, dev_questions, format="chat")
            buckets = {
                cat: order_robust([r for r in held["items"] if r["category"] == cat])
                for cat in sorted({q["category"] for q in dev_questions})
            }
            raw = []
            for q in selected_questions[:4]:
                prompt = f"<|user|>\n{q['context']}\n{q['question']}\n<|assistant|>\n"
                raw.append(
                    {
                        "id": q["id"],
                        "prompt": prompt,
                        "raw": generate(state, cfg, prompt, 12),
                        "expected": q["choices"][q["answer_index"]],
                    }
                )
            row = {
                "training": aggregate(train["items"]),
                "development": aggregate(held["items"]),
                "development_order_robust": buckets,
                "qa_development_loss": evaluate_stream(state, cfg, dev, loss_masks=dev_masks)[
                    "loss"
                ],
                "english_development_loss": evaluate_stream(state, cfg, english)["loss"],
                "fixed_training_raw_samples": raw,
            }
            write_json(out / f"{label}-scores.json", {"train": train, "development": held})
            return row

        initial = assess("initial")
        rng = np.random.default_rng(cfg.seed)
        order = rng.permutation(size)
        position = 0
        updates = []
        payload = supervised = 0
        for _ in range(args.steps):
            if position == size:
                order = rng.permutation(size)
                position = 0
            index = int(order[position])
            position += 1
            state, _, metrics = step(
                state, initial_state(cfg, 4), jax.tree.map(jnp.asarray, batches[index])
            )
            metrics = jax.device_get(metrics)
            if not bool(metrics["finite"]) or not bool(metrics["updated"]):
                raise FloatingPointError("Rejected acquisition update")
            payload += int(metrics["consumed_bytes"])
            supervised += int(metrics["bytes"])
            updates.append(batch_ids[index])
        final = assess("final")
        row = {
            "groups": size,
            "group_ids": identities,
            "unique_names": names,
            "entity_pairs": len(entity_pairs),
            "color_pairs": len(color_pairs),
            "initial_optimizer_step": initial_step,
            "initial_parameters_sha256": initial_hash,
            "optimizer_updates": int(state.step),
            "input_payload_bytes": payload,
            "supervised_bytes": supervised,
            "update_group_ids": updates,
            "trace_sha256": hashlib.sha256(json.dumps(updates).encode()).hexdigest(),
            "initial": initial,
            "final": final,
        }
        save_checkpoint(
            out / "final.mlm", state, cfg, {**protocol, "messages": True, "groups": size}
        )
        trials.append(row)
        write_json(args.out / "summary.json", {"protocol": protocol, "trials": trials})
        print(
            json.dumps(
                {
                    "groups": size,
                    "train": final["training"],
                    "development": final["development_order_robust"],
                    "english_loss": final["english_development_loss"],
                }
            ),
            flush=True,
        )
    if digest(args.init) != source_hash:
        raise RuntimeError("Source modified during study")
    return {"protocol": protocol, "trials": trials}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/scratch-binding-v1"))
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--init", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sizes", type=int, nargs="+", default=[1, 4, 16, 64])
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--english-dev", type=Path, default=Path("data/hf/stories/validation.jsonl"))
    run(p.parse_args())


if __name__ == "__main__":
    main()
