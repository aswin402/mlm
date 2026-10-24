"""Fresh-only full-document English pretraining with routed training support."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from mlm.checkpoint import save_checkpoint
from mlm.cli import environment, write_json
from mlm.config import Config
from mlm.data import corpus_hash, read_training_data
from mlm.hierarchy_run import make_routed_step
from mlm.model import initial_state
from mlm.stream import DocumentStream, evaluate_stream
from mlm.training import create_state


def run(args):
    if args.steps < 1 or args.eval_every < 1:
        raise ValueError("Positive budget/interval required")
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Use an empty output directory")
    cfg = Config.model_validate_json(args.config.read_text())
    if cfg.architecture != "graph" or cfg.node_readout != "sum":
        raise ValueError("Expected sum-readout graph")
    train, masks, _ = read_training_data(args.train)
    dev, dev_masks, _ = read_training_data(args.validation)
    if masks is not None or dev_masks is not None:
        raise ValueError("English pretraining requires full-text targets")
    if set(train) & set(dev):
        raise ValueError("Training/development overlap")
    state = create_state(cfg)
    stream = DocumentStream(train, cfg.batch_size, cfg.length, cfg.seed)
    carry = initial_state(cfg, cfg.batch_size)
    args.out.mkdir(parents=True, exist_ok=True)
    provenance = {
        "mode": "fresh-routed-pretraining",
        "messages": True,
        "initialization": "random-from-scratch",
        "seed": cfg.seed,
        "corpora": {"train": corpus_hash(train), "validation": corpus_hash(dev)},
        "sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [args.train, args.validation, args.config]
        },
        "selection": "minimum pure English development loss; initial and global intervals eligible",
        "eval_every": args.eval_every,
        "resume": "not supported by this driver",
    }
    write_json(args.out / "protocol.json", provenance)
    write_json(args.out / "config.json", cfg.model_dump())
    write_json(args.out / "environment.json", environment())

    def metadata():
        return {**provenance, "stream": stream.snapshot(), "carry": np.asarray(carry).tolist()}

    initial = evaluate_stream(state, cfg, dev)
    rows = [{"step": 0, "validation": initial, "input_payload_bytes": 0}]
    best_loss = initial["loss"]
    best_step = 0
    save_checkpoint(args.out / "initial.mlm", state, cfg, metadata())
    save_checkpoint(args.out / "best.mlm", state, cfg, metadata())
    print(json.dumps(rows[0]), flush=True)
    (args.out / "metrics.jsonl").write_text(json.dumps(rows[0]) + "\n")
    step = make_routed_step(cfg)
    payload = 0
    positions = 0.0
    usage = np.zeros(cfg.nodes)
    entropy = 0.0
    start = time.perf_counter()
    for _ in range(args.steps):
        batch = jax.tree.map(jnp.asarray, stream.next_batch())
        state, carry, raw = step(state, carry, batch)
        raw = jax.device_get(raw)
        if not bool(raw["finite"]) or not bool(raw["updated"]):
            raise FloatingPointError("Rejected pretraining update")
        payload += int(raw["consumed_bytes"])
        count = float(raw["input_positions"])
        positions += count
        usage += np.asarray(raw["route_usage"]) * count
        entropy += float(raw["route_entropy"]) * count
        eligible = int(state.step) % args.eval_every == 0
        if eligible or int(state.step) == args.steps:
            validation = evaluate_stream(state, cfg, dev)
            if eligible and validation["loss"] < best_loss:
                best_loss = validation["loss"]
                best_step = int(state.step)
                save_checkpoint(args.out / "best.mlm", state, cfg, metadata())
            save_checkpoint(args.out / "latest.mlm", state, cfg, metadata())
            row = {
                "step": int(state.step),
                "selection_eligible": eligible,
                "validation": validation,
                "input_payload_bytes": payload,
                "mean_route_usage": (usage / max(positions, 1)).tolist(),
                "mean_route_entropy": entropy / max(positions, 1),
                "epochs": stream.epochs,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
            (args.out / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = {
        "steps": int(state.step),
        "best_step": best_step,
        "best_development_loss": best_loss,
        "initial_development_loss": initial["loss"],
        "input_payload_bytes": payload,
        "training_seconds": time.perf_counter() - start,
        "parameters": sum(x.size for x in jax.tree.leaves(state.params)),
        "scope": "full-document English, truncated BPTT, routed auxiliary only in training",
    }
    write_json(args.out / "summary.json", summary)
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/hf/stories/train.jsonl"))
    parser.add_argument("--validation", type=Path, default=Path("data/hf/stories/validation.jsonl"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--eval-every", type=int, default=500)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
