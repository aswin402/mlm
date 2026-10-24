"""Fixed-budget, complete-group training; selection by development binding accuracy."""

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from mlm.checkpoint import load_checkpoint, save_checkpoint
from mlm.cli import environment, write_json
from mlm.comprehension import evaluate_questions
from mlm.config import Config
from mlm.data import encode, read_training_data
from mlm.model import initial_state
from mlm.retrieval_eval import aggregate
from mlm.stream import DocumentStream, evaluate_stream, make_stream_step
from mlm.training import create_state


def better(accuracy, loss, best):
    return accuracy > best[0] or (accuracy == best[0] and loss < best[1])


def build_batches(records, questions, config):
    if len(records) != len(questions) or config.batch_size != 4:
        raise ValueError("Aligned records/questions and batch size four required")
    grouped = defaultdict(list)
    for record, question in zip(records, questions, strict=True):
        grouped[question["group"]].append((record, question))
    batches, identities = [], []
    for identity, rows in grouped.items():
        if len(rows) != 4:
            raise ValueError("Each batch needs one complete four-question group")
        aggregate([{**q, "prediction": q["answer_index"], "correct": True} for _, q in rows])
        docs, masks = [], []
        for record, _ in rows:
            text = record["text"]
            if len(encode(text)) - 1 > config.length:
                raise ValueError("Example exceeds full-gradient unroll")
            mask = np.zeros(len(text.encode()) + 1, np.float32)
            for start, end in record["loss_spans"]:
                if not 0 <= start < end <= len(mask) - 1:
                    raise ValueError("Invalid supervision span")
                mask[start:end] = 1
            mask[-1] = bool(record["score_eos"])
            docs.append(text)
            masks.append(mask)
        stream = DocumentStream(
            docs, 4, config.length, repeat=False, shuffle=False, loss_masks=masks
        )
        batches.append(stream.next_batch())
        identities.append(identity)
        if any(lane is not None for lane in stream.lanes):
            raise ValueError("Truncated group")
    if not batches:
        raise ValueError("Nonempty training groups required")
    return batches, identities


def warm_start(source, config):
    old, old_config, meta = load_checkpoint(source)
    if not meta.get("messages", True):
        raise ValueError("This experiment requires graph messages enabled")
    fields = {"architecture", "nodes", "width", "degree", "rounds"}
    if old_config.model_dump(include=fields) != config.model_dump(include=fields):
        raise ValueError("Source architecture mismatch")
    fresh = create_state(config)
    # The conditional readout experiment may add parameters; preserve every shared tensor.
    if not set(old.params) <= set(fresh.params):
        raise ValueError("Source has parameters absent from target")
    params = dict(fresh.params)
    for key, value in old.params.items():
        if params[key].shape != value.shape:
            raise ValueError("Source parameter shape mismatch")
        params[key] = value
    return fresh.replace(params=params)


def run(args):
    if args.steps < 1 or args.eval_every < 1:
        raise ValueError("Positive budget and interval required")
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Use an empty output directory; this runner has no resume mode")
    config = Config.model_validate_json(args.config.read_text())
    train_path = args.data / "train.jsonl"
    records = [json.loads(line) for line in train_path.read_text().splitlines()]
    questions = json.loads((args.data / "train-questions.json").read_text())["questions"]
    batches, identities = build_batches(records, questions, config)
    docs, masks, mask_hash = read_training_data(train_path)
    dev, dev_masks, _ = read_training_data(args.data / "validation.jsonl")
    if set(docs) & set(dev):
        raise ValueError("Training/development overlap")
    dev_questions = json.loads((args.data / "validation-questions.json").read_text())["questions"]
    state = warm_start(args.init, config)
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.seed)
    order = rng.permutation(len(batches))
    position = 0
    epochs = 0
    provenance = {
        "mode": "grouped-binding",
        "messages": True,
        "source": str(args.init),
        "source_sha256": hashlib.sha256(args.init.read_bytes()).hexdigest(),
        "files": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                train_path,
                args.data / "train-questions.json",
                args.data / "validation.jsonl",
                args.data / "validation-questions.json",
                args.config,
            ]
        },
        "supervision_hash": mask_hash,
        "group_ids": identities,
        "selection": "maximum complete binding-group accuracy; then minimum target loss",
        "resume": "not supported; use a fresh output path",
    }
    write_json(args.out / "config.json", config.model_dump())
    write_json(args.out / "environment.json", environment())
    write_json(
        args.out / "protocol.json",
        {**provenance, "steps": args.steps, "eval_every": args.eval_every},
    )

    def metadata():
        return {
            **provenance,
            "order": order.tolist(),
            "position": position,
            "epochs": epochs,
            "rng": rng.bit_generator.state,
        }

    def validate():
        ranked = evaluate_questions(state, config, dev_questions, format="chat")
        metrics = aggregate(ranked["items"])
        loss = evaluate_stream(state, config, dev, loss_masks=dev_masks)["loss"]
        return metrics, loss

    metrics, loss = validate()
    best = (metrics["binding_group_accuracy"], loss)
    best_step = 0
    save_checkpoint(args.out / "best.mlm", state, config, metadata())
    log = [{"step": 0, "development": metrics, "target_loss": loss}]
    step = make_stream_step(config)
    consumed = supervised = 0
    start = time.perf_counter()
    for _ in range(args.steps):
        if position == len(order):
            order = rng.permutation(len(batches))
            position = 0
            epochs += 1
        batch = jax.tree.map(jnp.asarray, batches[int(order[position])])
        position += 1
        state, _, raw = step(state, initial_state(config, 4), batch)
        raw = jax.device_get(raw)
        if not bool(raw["finite"]) or not bool(raw["updated"]):
            raise FloatingPointError("Rejected grouped update")
        consumed += int(raw["consumed_bytes"])
        supervised += int(raw["bytes"])
        eligible = int(state.step) % args.eval_every == 0
        if eligible or int(state.step) == args.steps:
            metrics, loss = validate()
            if eligible and better(metrics["binding_group_accuracy"], loss, best):
                best = (metrics["binding_group_accuracy"], loss)
                best_step = int(state.step)
                save_checkpoint(args.out / "best.mlm", state, config, metadata())
            save_checkpoint(args.out / "latest.mlm", state, config, metadata())
            row = {
                "step": int(state.step),
                "selection_eligible": eligible,
                "development": metrics,
                "target_loss": loss,
                "consumed_payload_bytes": consumed,
                "supervised_payload_bytes": supervised,
                "gradient_norm": float(raw["grad_norm"]),
            }
            log.append(row)
            print(json.dumps(row), flush=True)
            (args.out / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in log))
    summary = {
        "steps": int(state.step),
        "best_step": best_step,
        "best_development_group_accuracy": best[0],
        "best_development_target_loss": best[1],
        "consumed_payload_bytes": consumed,
        "supervised_payload_bytes": supervised,
        "parameters": sum(x.size for x in jax.tree.leaves(state.params)),
        "training_seconds": time.perf_counter() - start,
        "epochs": epochs,
    }
    write_json(args.out / "summary.json", summary)
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/binding-v1"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=250)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
