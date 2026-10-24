"""Matched gradual versus flat binding training with separate transfer diagnostics."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from mlm.binding_run import better, build_batches, warm_start
from mlm.checkpoint import save_checkpoint
from mlm.cli import environment, write_json
from mlm.comprehension import evaluate_questions
from mlm.config import Config
from mlm.data import read_documents, read_training_data
from mlm.gradual_data import NAMES, SIZES
from mlm.model import initial_state
from mlm.retrieval_eval import aggregate
from mlm.stream import evaluate_stream, make_stream_step


def run(args):
    if args.stage_steps < 1 or args.strategy not in ("curriculum", "flat"):
        raise ValueError("Positive stage budget and valid strategy required")
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Use an empty output directory; resume is not supported")
    cfg = Config.model_validate_json(args.config.read_text())
    if cfg.architecture != "graph" or cfg.node_readout != "sum":
        raise ValueError("This experiment requires the original sum-readout graph")
    train_path = args.data / "train.jsonl"
    records = [json.loads(line) for line in train_path.read_text().splitlines()]
    questions = json.loads((args.data / "train-questions.json").read_text())["questions"]
    batches, ids = build_batches(records, questions, cfg)
    if len(batches) != 64 or [q["group"] for q in questions[::4]] != ids:
        raise ValueError("Expected 64 contiguous curriculum groups")
    dev, dev_masks, _ = read_training_data(args.data / "validation.jsonl")
    dev_questions = json.loads((args.data / "validation-questions.json").read_text())["questions"]
    if {r["text"] for r in records} & set(dev):
        raise ValueError("Train/development overlap")
    if set(q["category"] for q in dev_questions) != {"familiar-names-new-pairs", "new-names"}:
        raise ValueError("Both transfer buckets required")
    stories = read_documents(args.stories) if args.stories else None
    state = warm_start(args.init, cfg)
    args.out.mkdir(parents=True, exist_ok=True)
    files = [
        train_path,
        args.data / "train-questions.json",
        args.data / "validation.jsonl",
        args.data / "validation-questions.json",
        args.config,
    ]
    provenance = {
        "mode": "gradual-binding",
        "messages": True,
        "strategy": args.strategy,
        "source": str(args.init),
        "source_sha256": hashlib.sha256(args.init.read_bytes()).hexdigest(),
        "files": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "stage_steps": args.stage_steps,
        "sizes": SIZES,
        "seen_names_scope": "groups sampled in this run; source history excluded",
        "selection": (
            "fixed development group accuracy, then target loss; endpoints from stage four groups"
        ),
        "resume": "not supported",
    }
    if args.stories:
        provenance["story_monitor_sha256"] = hashlib.sha256(args.stories.read_bytes()).hexdigest()
    write_json(args.out / "config.json", cfg.model_dump())
    write_json(args.out / "environment.json", environment())
    write_json(args.out / "protocol.json", provenance)
    rng = np.random.default_rng(cfg.seed)
    order = None
    position = 0
    stage = 0
    active = 1 if args.strategy == "curriculum" else 64
    seen = set()
    consumed = supervised = 0
    best = (-1.0, float("inf"))
    best_step = None
    logs = []

    def metadata():
        return {
            **provenance,
            "stage": stage,
            "active_groups": active,
            "seen_names": sorted(seen),
            "order": [] if order is None else order.tolist(),
            "position": position,
            "rng": rng.bit_generator.state,
        }

    def diagnose(eligible):
        ranked = evaluate_questions(state, cfg, dev_questions, format="chat")
        overall = aggregate(ranked["items"])
        pools = {
            key: aggregate([q for q in ranked["items"] if q["category"] == key])
            for key in ["familiar-names-new-pairs", "new-names"]
        }
        known = [
            q
            for q in ranked["items"]
            if q["category"] == "familiar-names-new-pairs" and {n for n, _ in q["facts"]} <= seen
        ]
        eligible_known = aggregate(known) if known else None
        loss = evaluate_stream(state, cfg, dev, loss_masks=dev_masks)["loss"]
        training = evaluate_questions(state, cfg, questions[: active * 4], format="chat")
        row = {
            "step": int(state.step),
            "stage": stage,
            "active_groups": active,
            "selection_eligible": eligible,
            "seen_names": sorted(seen),
            "development": overall,
            "pools": pools,
            "actually_seen_names_development": eligible_known,
            "target_loss": loss,
            "active_training": aggregate(training["items"]),
            "seed_group": aggregate(training["items"][:4]),
            "consumed_payload_bytes": consumed,
            "supervised_payload_bytes": supervised,
        }
        if stories:
            row["story_monitor"] = evaluate_stream(state, cfg, stories)
        return row

    row = diagnose(False)
    logs.append(row)
    print(json.dumps(row), flush=True)
    (args.out / "metrics.jsonl").write_text(json.dumps(row) + "\n")
    step = make_stream_step(cfg)
    start = time.perf_counter()
    for stage, size in enumerate(SIZES[1:], start=1):
        active = size if args.strategy == "curriculum" else 64
        if args.strategy == "curriculum" or order is None:
            order = rng.permutation(active)
            position = 0
        for _ in range(args.stage_steps):
            if position == len(order):
                order = rng.permutation(active)
                position = 0
            index = int(order[position])
            position += 1
            seen.update(n for q in questions[index * 4 : index * 4 + 4] for n, _ in q["facts"])
            batch = jax.tree.map(jnp.asarray, batches[index])
            state, _, raw = step(state, initial_state(cfg, 4), batch)
            raw = jax.device_get(raw)
            if not bool(raw["finite"]) or not bool(raw["updated"]):
                raise FloatingPointError("Rejected curriculum update")
            consumed += int(raw["consumed_bytes"])
            supervised += int(raw["bytes"])
        eligible = stage >= 2 and set(NAMES) <= seen
        row = diagnose(eligible)
        score = row["development"]["binding_group_accuracy"]
        if eligible and better(score, row["target_loss"], best):
            best = (score, row["target_loss"])
            best_step = int(state.step)
            save_checkpoint(args.out / "best.mlm", state, cfg, metadata())
        save_checkpoint(args.out / "latest.mlm", state, cfg, metadata())
        logs.append(row)
        (args.out / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in logs))
        print(json.dumps(row), flush=True)
    if best_step is None:
        raise ValueError("Budget did not produce a name-covered selection candidate")
    summary = {
        "strategy": args.strategy,
        "steps": int(state.step),
        "best_step": best_step,
        "best_development_group_accuracy": best[0],
        "best_development_target_loss": best[1],
        "consumed_payload_bytes": consumed,
        "supervised_payload_bytes": supervised,
        "training_seconds": time.perf_counter() - start,
        "parameters": sum(p.size for p in jax.tree.leaves(state.params)),
    }
    write_json(args.out / "summary.json", summary)
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/gradual-v1"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strategy", choices=["curriculum", "flat"], required=True)
    parser.add_argument("--stage-steps", type=int, default=500)
    parser.add_argument("--stories", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
