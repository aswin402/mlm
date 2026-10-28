"""Task-exposure-matched binding training with explicit optional English updates."""

import argparse
import hashlib
import json
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
from mlm.gradual_eval import order_robust
from mlm.model import initial_state
from mlm.stream import DocumentStream, evaluate_stream, make_stream_step


def choose(score, loss, english, limit, best, protect):
    return (not protect or english <= limit) and better(score, loss, best)


def run(args):
    if args.steps < 1 or args.eval_every < 1:
        raise ValueError("Positive budget/interval required")
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Use an empty output directory")
    cfg = Config.model_validate_json(args.config.read_text())
    if cfg.architecture != "graph" or cfg.node_readout != "sum":
        raise ValueError("Keep sum graph unchanged")
    train = args.data / "train.jsonl"
    records = [json.loads(line) for line in train.read_text().splitlines()]
    questions = json.loads((args.data / "train-questions.json").read_text())["questions"]
    batches, ids = build_batches(records, questions, cfg)
    dev, dev_masks, _ = read_training_data(args.data / "validation.jsonl")
    dev_questions = json.loads((args.data / "validation-questions.json").read_text())["questions"]
    if {r["text"] for r in records} & set(dev):
        raise ValueError("Task split overlap")
    english_train = read_documents(args.english_train)
    english_dev = read_documents(args.english_dev)
    if set(english_train) & set(english_dev):
        raise ValueError("English split overlap")
    state = warm_start(args.init, cfg)
    args.out.mkdir(parents=True, exist_ok=True)
    paths = [
        train,
        args.data / "train-questions.json",
        args.data / "validation.jsonl",
        args.data / "validation-questions.json",
        args.english_train,
        args.english_dev,
        args.config,
    ]
    provenance = {
        "mode": "balanced-confirmation",
        "messages": True,
        "rehearsal": args.rehearsal,
        "source": str(args.init),
        "source_sha256": hashlib.sha256(args.init.read_bytes()).hexdigest(),
        "files": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "qa_budget": args.steps,
        "eval_every_qa": args.eval_every,
        "group_ids": ids,
        "english_limit_factor": 1.25,
        "resume": "not supported",
        "selection": "macro order-robust development accuracy, then QA loss; initial eligible",
    }
    write_json(args.out / "protocol.json", provenance)
    write_json(args.out / "config.json", cfg.model_dump())
    write_json(args.out / "environment.json", environment())
    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(len(batches))
    position = 0
    stream = DocumentStream(english_train, cfg.batch_size, cfg.length, cfg.seed + 1)
    carry = initial_state(cfg, cfg.batch_size)
    counters = {
        name: {"updates": 0, "input_bytes": 0, "supervised_bytes": 0} for name in ["qa", "english"]
    }
    qa_step = 0

    def metadata():
        return {
            **provenance,
            "qa_step": qa_step,
            "counters": counters,
            "qa_order": order.tolist(),
            "qa_position": position,
            "qa_rng": rng.bit_generator.state,
            "english_stream": stream.snapshot(),
            "english_carry": np.array(carry).tolist(),
        }

    def validate():
        ranked = evaluate_questions(state, cfg, dev_questions, format="chat")
        pools = {
            name: order_robust([q for q in ranked["items"] if q["category"] == name])
            for name in ["familiar-names-new-pairs", "new-names"]
        }
        strata = {}
        for name in pools:
            for kind in ["within", "cross"]:
                subset = [
                    q for q in ranked["items"] if q["category"] == name and q["pair_type"] == kind
                ]
                strata[name + "/" + kind] = order_robust(subset)
        loss = evaluate_stream(state, cfg, dev, loss_masks=dev_masks)["loss"]
        english = evaluate_stream(state, cfg, english_dev)["loss"]
        score = sum(p["accuracy"] for p in pools.values()) / 2
        return {
            "qa_step": qa_step,
            "optimizer_step": int(state.step),
            "pools": pools,
            "strata": strata,
            "score": score,
            "qa_loss": loss,
            "english_development_loss": english,
            "counters": json.loads(json.dumps(counters)),
        }

    initial = validate()
    limit = initial["english_development_loss"] * 1.25
    best = {name: (initial["score"], initial["qa_loss"]) for name in ["best", "best-task"]}
    selected = {name: 0 for name in best}
    for name in best:
        save_checkpoint(args.out / f"{name}.mlm", state, cfg, metadata())
    initial["english_limit"] = limit
    logs = [initial]
    (args.out / "metrics.jsonl").write_text(json.dumps(initial) + "\n")
    print(json.dumps(initial), flush=True)
    step = make_stream_step(cfg)

    def update(batch, hidden, objective):
        nonlocal state
        state, hidden, raw = step(state, hidden, jax.tree.map(jnp.asarray, batch))
        raw = jax.device_get(raw)
        if not bool(raw["finite"]) or not bool(raw["updated"]):
            raise FloatingPointError("Rejected update")
        counters[objective]["updates"] += 1
        counters[objective]["input_bytes"] += int(raw["consumed_bytes"])
        counters[objective]["supervised_bytes"] += int(raw["bytes"])
        return hidden

    for qa_step in range(1, args.steps + 1):
        if position == len(order):
            order = rng.permutation(len(batches))
            position = 0
        batch = batches[int(order[position])]
        position += 1
        update(batch, initial_state(cfg, 4), "qa")
        if args.rehearsal:
            carry = update(stream.next_batch(), carry, "english")
        eligible = qa_step % args.eval_every == 0
        if eligible or qa_step == args.steps:
            row = validate()
            row["selection_eligible"] = eligible
            row["english_limit"] = limit
            for name in best:
                if eligible and choose(
                    row["score"],
                    row["qa_loss"],
                    row["english_development_loss"],
                    limit,
                    best[name],
                    name == "best",
                ):
                    best[name] = (row["score"], row["qa_loss"])
                    selected[name] = qa_step
                    save_checkpoint(args.out / f"{name}.mlm", state, cfg, metadata())
            save_checkpoint(args.out / "latest.mlm", state, cfg, metadata())
            logs.append(row)
            (args.out / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in logs))
            print(json.dumps(row), flush=True)
    summary = {
        "qa_updates": qa_step,
        "optimizer_updates": int(state.step),
        "rehearsal": args.rehearsal,
        "selected_qa_steps": selected,
        "selected_scores": {k: v[0] for k, v in best.items()},
        "english_development_limit": limit,
        "counters": counters,
        "parameters": sum(p.size for p in jax.tree.leaves(state.params)),
    }
    write_json(args.out / "summary.json", summary)
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/confirmation-v1"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--english-train", type=Path, default=Path("data/hf/stories/train.jsonl"))
    parser.add_argument(
        "--english-dev", type=Path, default=Path("data/hf/stories/validation.jsonl")
    )
    parser.add_argument("--rehearsal", action="store_true")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=500)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
