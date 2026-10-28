"""Fresh-only acquisition/retention trials with explicit rate roles and English rehearsal."""

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from mlm.acquisition_run import digest, nested_groups
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
from mlm.plasticity import parameter_roles, role_scales
from mlm.retrieval_eval import aggregate
from mlm.stream import DocumentStream, evaluate_stream


def sampling_seeds(config, qa_seed=None, english_seed=None):
    """Keep initialization/subset seed fixed while allowing independent samplers."""
    seeds = {
        "subset": config.seed,
        "qa": config.seed if qa_seed is None else qa_seed,
        "english": config.seed + 1 if english_seed is None else english_seed,
    }
    for value in seeds.values():
        if type(value) is not int or not 0 <= value < 2**32:
            raise ValueError("Sampling seed must be an integer in [0, 2**32-1]")
    return seeds


def run(args):
    if args.steps < 1 or args.groups < 1:
        raise ValueError("Positive groups and update budget required")
    if args.out.exists():
        raise ValueError("Choose a fresh output directory")
    cfg = Config.model_validate_json(args.config.read_text())
    seeds = sampling_seeds(cfg, getattr(args, "qa_seed", None), getattr(args, "english_seed", None))
    if cfg.architecture != "graph" or cfg.node_readout != "sum" or cfg.batch_size != 4:
        raise ValueError("Expected batch-four sum-readout graph")
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
    identities = nested_groups(questions, [args.groups], cfg.seed)[args.groups]
    chosen = set(identities)
    if len(records) != len(questions) or any(
        r["group"] != q["group"] for r, q in zip(records, questions, strict=True)
    ):
        raise ValueError("Aligned training records required")
    pairs = [(r, q) for r, q in zip(records, questions, strict=True) if q["group"] in chosen]
    selected_records = [r for r, _ in pairs]
    selected_questions = [q for _, q in pairs]
    batches, batch_ids = build_batches(selected_records, selected_questions, cfg)
    dev_questions = json.loads((args.data / "validation-questions.json").read_text())["questions"]
    dev, dev_masks, _ = read_training_data(args.data / "validation.jsonl")
    english_dev = read_documents(args.english_dev)
    english_train = read_documents(args.english_train) if args.rehearsal else []
    if {r["text"] for r in records} & set(dev) or set(english_train) & set(english_dev):
        raise ValueError("Training/development overlap")
    state = warm_start(args.init, cfg)
    starting_params = jax.device_get(state.params)
    initial_fingerprint = hashlib.sha256(serialization.to_bytes(state.params)).hexdigest()
    initial_optimizer_step = int(state.step)
    roles = parameter_roles(state.params)
    scales = role_scales(cfg)
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
    if args.rehearsal:
        paths.append(args.english_train)
    policy = {
        role: {
            "scale": scales[role],
            "learning_rate": cfg.learning_rate * scales[role],
            "parameters": sum(
                value.size for name, value in state.params.items() if roles[name] == role
            ),
        }
        for role in scales
    }
    protocol = {
        "mode": "plasticity",
        "sampling_seeds": seeds,
        "source_sha256": source_hash,
        "messages": True,
        "sha256": {str(p): digest(p) for p in paths},
        "parameter_roles": roles,
        "rate_policy": policy,
        "qa_steps": args.steps,
        "group_ids": identities,
        "rehearsal": args.rehearsal,
        "selection": "none; initial and fixed endpoint only; no test data opened",
        "optimizer": "fresh AdamW; global clipping; final per-role update scaling includes decay",
        "resume": "not supported by this driver; checkpoint loader restores rate config",
    }
    args.out.mkdir(parents=True)
    write_json(args.out / "protocol.json", protocol)
    write_json(args.out / "config.json", cfg.model_dump())
    write_json(args.out / "environment.json", environment())
    stream = (
        DocumentStream(english_train, cfg.batch_size, cfg.length, seeds["english"])
        if args.rehearsal
        else None
    )
    carry = initial_state(cfg, cfg.batch_size)
    counters = {
        name: {"updates": 0, "input_bytes": 0, "supervised_bytes": 0} for name in ["qa", "english"]
    }
    trace = []

    def metadata():
        return {
            **protocol,
            "counters": counters,
            "qa_group_trace": trace,
            "english_stream": stream.snapshot() if stream else None,
            "english_carry": np.asarray(carry).tolist() if stream else None,
        }

    def assess(label):
        train = evaluate_questions(state, cfg, selected_questions, format="chat")
        held = evaluate_questions(state, cfg, dev_questions, format="chat")
        buckets = {
            cat: order_robust([r for r in held["items"] if r["category"] == cat])
            for cat in sorted({q["category"] for q in dev_questions})
        }
        samples = []
        for q in selected_questions[:4]:
            prompt = f"<|user|>\n{q['context']}\n{q['question']}\n<|assistant|>\n"
            samples.append(
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
            "qa_development_loss": evaluate_stream(state, cfg, dev, loss_masks=dev_masks)["loss"],
            "english_development_loss": evaluate_stream(state, cfg, english_dev)["loss"],
            "fixed_training_raw_samples": samples,
        }
        write_json(args.out / f"{label}-scores.json", {"train": train, "development": held})
        return row

    initial = assess("initial")
    save_checkpoint(args.out / "initial.mlm", state, cfg, metadata())
    rng = np.random.default_rng(seeds["qa"])
    order = rng.permutation(args.groups)
    position = 0
    step = make_routed_step(cfg)

    def update(batch, hidden, objective):
        nonlocal state
        state, hidden, metrics = step(state, hidden, jax.tree.map(jnp.asarray, batch))
        metrics = jax.device_get(metrics)
        if not bool(metrics["finite"]) or not bool(metrics["updated"]):
            raise FloatingPointError("Rejected plasticity update")
        counters[objective]["updates"] += 1
        counters[objective]["input_bytes"] += int(metrics["consumed_bytes"])
        counters[objective]["supervised_bytes"] += int(metrics["bytes"])
        return hidden

    for qa_step in range(1, args.steps + 1):
        if position == args.groups:
            order = rng.permutation(args.groups)
            position = 0
        index = int(order[position])
        position += 1
        update(batches[index], initial_state(cfg, 4), "qa")
        trace.append(batch_ids[index])
        if stream:
            carry = update(stream.next_batch(), carry, "english")
        if qa_step % 500 == 0:
            save_checkpoint(args.out / "latest.mlm", state, cfg, metadata())
            print(
                json.dumps(
                    {"qa_step": qa_step, "optimizer_step": int(state.step), "counters": counters}
                ),
                flush=True,
            )
    final = assess("final")
    final_params = jax.device_get(state.params)
    drift = {
        role: float(
            np.sqrt(
                sum(
                    np.square(
                        np.asarray(final_params[name], dtype=np.float64)
                        - np.asarray(starting_params[name], dtype=np.float64)
                    ).sum()
                    for name in roles
                    if roles[name] == role
                )
            )
        )
        for role in scales
    }
    summary = {
        "sampling_seeds": seeds,
        "initial_optimizer_step": initial_optimizer_step,
        "initial_parameters_sha256": initial_fingerprint,
        "optimizer_updates": int(state.step),
        "groups": args.groups,
        "group_ids": identities,
        "update_group_ids": trace,
        "trace_sha256": hashlib.sha256(json.dumps(trace).encode()).hexdigest(),
        "counters": counters,
        "rate_policy": policy,
        "initial": initial,
        "final": final,
        "parameter_delta_l2_by_role": drift,
        "english_stream": stream.snapshot() if stream else None,
        "unique_names": sorted({n for q in selected_questions for n, _ in q["facts"]}),
    }
    save_checkpoint(args.out / "final.mlm", state, cfg, metadata())
    write_json(args.out / "summary.json", summary)
    if digest(args.init) != source_hash:
        raise RuntimeError("Source changed during study")
    print(json.dumps({"final": final, "rate_policy": policy}), flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/scratch-binding-v1"))
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--init", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--groups", type=int, default=16)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--rehearsal", action="store_true")
    p.add_argument("--qa-seed", type=int, help="QA ordering only; default config seed")
    p.add_argument("--english-seed", type=int, help="English stream only; default config seed+1")
    p.add_argument("--english-train", type=Path, default=Path("data/hf/stories/train.jsonl"))
    p.add_argument("--english-dev", type=Path, default=Path("data/hf/stories/validation.jsonl"))
    run(p.parse_args())


if __name__ == "__main__":
    main()
