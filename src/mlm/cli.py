"""Train, evaluate, and generate with the fixed-graph research prototype."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import resource
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from mlm.checkpoint import load_checkpoint, save_checkpoint
from mlm.config import Config
from mlm.data import corpus_hash, encode, make_batch, read_training_data
from mlm.evaluate import evaluate
from mlm.generate import LiveLearner, generate
from mlm.training import create_state, make_train_step


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_history(path):
    """Recover a truncated final JSONL record, never silently ignore earlier corruption."""
    lines = path.read_text().splitlines(keepends=True) if path.exists() else []
    records = []
    for index, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not line.endswith("\n"):
                break
            raise ValueError("Corrupt metric history; branch to a new destination") from None
    return records


def environment():
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "devices": [str(d) for d in jax.devices()],
        "versions": {p: importlib.metadata.version(p) for p in ("jax", "flax", "optax")},
        "source_sha256": hashlib.sha256(
            b"".join(p.read_bytes() for p in sorted(Path(__file__).parent.glob("*.py")))
        ).hexdigest(),
    }


def run_train(args):
    if args.steps < 1 or args.eval_every < 1:
        raise ValueError("steps and eval-every must be positive")
    train, train_masks, _ = read_training_data(args.train)
    validation, validation_masks, _ = read_training_data(args.validation)
    if train_masks is not None or validation_masks is not None:
        raise ValueError("Explicit supervision requires stream-train")
    if set(train) & set(validation):
        raise ValueError("Training and validation contain identical documents")
    identity = {"train": corpus_hash(train), "validation": corpus_hash(validation)}
    rng = np.random.default_rng(42)
    if args.resume:
        state, config, previous = load_checkpoint(args.resume)
        if previous.get("corpora") != identity:
            raise ValueError("Checkpoint corpus identity does not match supplied data")
        if args.config and Config.model_validate_json(args.config.read_text()) != config:
            raise ValueError("Resume configuration must match checkpoint")
        rng.bit_generator.state = previous["sampler_rng"]
    else:
        config = Config.model_validate_json(args.config.read_text()) if args.config else Config()
        rng = np.random.default_rng(config.seed)
        state = create_state(config)
    train_prefixes = {tuple(encode(doc)[1 : config.length + 1]) for doc in train}
    val_prefixes = {tuple(encode(doc)[1 : config.length + 1]) for doc in validation}
    if train_prefixes & val_prefixes:
        raise ValueError("Training and validation contain identical scored prefixes")
    if args.out.exists() and any(args.out.iterdir()):
        if not args.resume or args.resume.resolve() != (args.out / "latest.mlm").resolve():
            raise ValueError(
                "Nonempty destination requires its own latest.mlm; branch to a new path"
            )
        history = args.out / "metrics.jsonl"
        records = read_history(history)
        if any(record["step"] > int(state.step) for record in records):
            raise ValueError("Destination history is newer than checkpoint; branch to a new path")
        old_summary = args.out / "summary.json"
        if old_summary.exists() and json.loads(old_summary.read_text())["steps"] > int(state.step):
            raise ValueError("Destination summary is newer than checkpoint; branch to a new path")
        if history.exists():
            normalized = "".join(json.dumps(record) + "\n" for record in records)
            temporary = history.with_suffix(".jsonl.tmp")
            temporary.write_text(normalized)
            temporary.replace(history)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "config.json", config.model_dump())
    write_json(args.out / "environment.json", environment())
    initial_train = evaluate(state, config, train)
    initial_validation = evaluate(state, config, validation)
    best_loss = initial_validation["loss"]
    best_step = int(state.step)
    best_path = args.out / "best.mlm"
    if args.resume and best_path.exists():
        best_state, best_config, best_metadata = load_checkpoint(best_path)
        if best_config != config or best_metadata.get("corpora") != identity:
            raise ValueError("Best checkpoint does not belong to destination run")
        best_loss = evaluate(best_state, config, validation)["loss"]
        best_step = int(best_state.step)
    else:
        save_checkpoint(
            best_path,
            state,
            config,
            {
                "corpora": identity,
                "sampler_rng": rng.bit_generator.state,
                "next_batch": int(state.step),
            },
        )
    parameters = sum(p.size for p in jax.tree.leaves(state.params))
    step_fn = make_train_step(config)
    batch = jax.tree.map(jnp.asarray, make_batch(train, [0] * config.batch_size, config.length))
    # Compile without an update; dataset RNG and optimizer are not advanced.
    started = time.perf_counter()
    compiled = step_fn.lower(state, batch).compile()
    compile_seconds = time.perf_counter() - started
    started = time.perf_counter()
    update_seconds = 0.0
    processed_bytes = 0
    latest = None
    first_step = int(state.step)
    with (args.out / "metrics.jsonl").open("a" if args.resume else "w") as log:
        for index in range(args.steps):
            indices = rng.integers(0, len(train), size=config.batch_size)
            batch = jax.tree.map(jnp.asarray, make_batch(train, indices, config.length))
            tick = time.perf_counter()
            state, raw = compiled(state, batch)
            metrics = jax.device_get(raw)
            update_seconds += time.perf_counter() - tick
            if not bool(metrics["finite"]):
                raise FloatingPointError("Nonfinite update rejected; previous checkpoint preserved")
            processed_bytes += int(metrics["bytes"])
            latest = {
                "step": int(state.step),
                "training_loss": float(metrics["loss"]),
                "gradient_norm": float(metrics["grad_norm"]),
            }
            if (index + 1) % args.eval_every == 0 or index + 1 == args.steps:
                latest["validation"] = evaluate(state, config, validation)
                metadata = {
                    "corpora": identity,
                    "sampler_rng": rng.bit_generator.state,
                    "next_batch": int(state.step),
                }
                # Commit state first; metrics must never advertise an unsaved update.
                save_checkpoint(args.out / "latest.mlm", state, config, metadata)
                if latest["validation"]["loss"] < best_loss:
                    best_loss = latest["validation"]["loss"]
                    best_step = int(state.step)
                    save_checkpoint(best_path, state, config, metadata)
                log.write(json.dumps(latest, allow_nan=False) + "\n")
                log.flush()
                print(json.dumps(latest), flush=True)
    elapsed = time.perf_counter() - started
    summary = {
        "scope": "independent prefix batches; no growth or online learning",
        "architecture": config.architecture,
        "steps": int(state.step),
        "session_steps": int(state.step) - first_step,
        "parameters": parameters,
        "best_step": best_step,
        "best_validation_loss": best_loss,
        "compile_seconds": compile_seconds,
        "session_seconds": elapsed,
        "update_seconds": update_seconds,
        "training_payload_bytes": processed_bytes,
        "update_payload_bytes_per_second": processed_bytes / max(update_seconds, 1e-9),
        "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "initial_train": initial_train,
        "train": evaluate(state, config, train),
        "initial_validation": initial_validation,
        "validation": evaluate(state, config, validation),
        "corpora": identity,
        "checkpoint_bytes": (args.out / "latest.mlm").stat().st_size,
        "truncated_train_documents": sum(len(s.encode("utf-8")) + 1 > config.length for s in train),
        "truncated_validation_documents": sum(
            len(s.encode("utf-8")) + 1 > config.length for s in validation
        ),
    }
    write_json(args.out / "summary.json", summary)
    best_state, _, _ = load_checkpoint(best_path)
    write_json(
        args.out / "samples.json",
        {
            "checkpoint": "best.mlm",
            "step": int(best_state.step),
            "decoding": "greedy",
            "max_bytes": 64,
            "utf8_errors": "replace",
            "samples": [
                {"prompt": prompt, "continuation": generate(best_state, config, prompt, 64)}
                for prompt in ("The ", train[0][:12])
            ],
        },
    )
    print(json.dumps(summary, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train on blank-line-separated documents")
    train.add_argument("--train", type=Path, required=True)
    train.add_argument("--validation", type=Path, required=True)
    train.add_argument("--config", type=Path)
    train.add_argument("--out", type=Path, required=True)
    train.add_argument("--resume", type=Path)
    train.add_argument("--steps", type=int, default=300, help="Additional optimizer updates")
    train.add_argument("--eval-every", type=int, default=50)
    stream_parser = commands.add_parser("stream-train", help="Train on complete documents")
    stream_parser.add_argument("--train", type=Path, required=True)
    stream_parser.add_argument("--validation", type=Path, required=True)
    stream_parser.add_argument("--test", type=Path)
    stream_parser.add_argument("--config", type=Path)
    stream_parser.add_argument("--out", type=Path, required=True)
    initialization = stream_parser.add_mutually_exclusive_group()
    initialization.add_argument("--resume", type=Path)
    initialization.add_argument("--init", type=Path, help="Parameters only; new optimizer and data")
    stream_parser.add_argument("--steps", type=int, default=2000)
    stream_parser.add_argument("--eval-every", type=int, default=500)
    stream_parser.add_argument("--no-messages", action="store_true")
    generate_parser = commands.add_parser("generate", help="Greedy continuation; no weight updates")
    generate_parser.add_argument("checkpoint", type=Path)
    generate_parser.add_argument("--prompt", default="")
    generate_parser.add_argument("--max-bytes", type=int, default=128)
    generate_parser.add_argument("--adapt-strength", type=float, default=0.0)
    generate_parser.add_argument("--adapt-decay", type=float, default=0.88)
    evaluation = commands.add_parser("evaluate", help="Evaluate independent document prefixes")
    evaluation.add_argument("checkpoint", type=Path)
    evaluation.add_argument("--data", type=Path, required=True)
    evaluation.add_argument("--no-messages", action="store_true", help="Graph ablation")
    evaluation.add_argument("--full-documents", action="store_true")
    chat = commands.add_parser("chat", help="Experimental role-formatted single-turn generation")
    chat.add_argument("checkpoint", type=Path)
    chat.add_argument("--message", required=True)
    chat.add_argument("--max-bytes", type=int, default=160)
    chat.add_argument("--adapt-strength", type=float, default=0.0)
    chat.add_argument("--adapt-decay", type=float, default=0.88)
    chat.add_argument("--learn", action="store_true", help="Online stream update on exchange")
    chat.add_argument("--save-out", type=Path, help="Optional path to save updated checkpoint")
    continual = commands.add_parser(
        "continual",
        help="Autonomous continual learning with LaneSchedule, controller, growth, and rewiring",
    )
    continual.add_argument("--data", type=Path, default=Path("data/confirmation-v1"))
    continual.add_argument("--config", type=Path, required=True)
    continual.add_argument("--init", type=Path)
    continual.add_argument("--out", type=Path, required=True)
    continual.add_argument(
        "--english-train", type=Path, default=Path("data/hf/stories/train.jsonl")
    )
    continual.add_argument(
        "--english-dev", type=Path, default=Path("data/hf/stories/validation.jsonl")
    )
    continual.add_argument("--steps", type=int, default=200)
    continual.add_argument("--eval-every", type=int, default=50)
    continual.add_argument("--max-records", type=int)
    continual.add_argument("--qa-seed", type=int)
    continual.add_argument("--english-seed", type=int)
    continual.add_argument("--auto-grow", action="store_true")
    continual.add_argument("--rewire", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "train":
        run_train(args)
    elif args.command == "stream-train":
        from mlm.streamrun import run

        run(args)
    elif args.command == "continual":
        from mlm.hierarchy_run import run_continual

        summary = run_continual(args)
        print(json.dumps(summary, indent=2), flush=True)
    elif args.command == "chat":
        state, config, meta = load_checkpoint(args.checkpoint)
        messages = meta.get("messages", True)
        if args.learn:
            learner = LiveLearner(
                state, config, chunk=max(2, min(16, config.length)), messages=messages
            )
            reply, _ = learner.chat_turn(
                args.message,
                max_bytes=args.max_bytes,
                adapt_strength=args.adapt_strength,
                adapt_decay=args.adapt_decay,
                learn=True,
            )
            if args.save_out:
                save_checkpoint(args.save_out, learner.state, config, meta)
        else:
            prompt = f"<|user|>\n{args.message}\n<|assistant|>\n"
            reply = generate(
                state,
                config,
                prompt,
                args.max_bytes,
                messages=messages,
                adapt_strength=args.adapt_strength,
                adapt_decay=args.adapt_decay,
            )
        print(reply.strip())
    elif args.command == "generate":
        state, config, meta = load_checkpoint(args.checkpoint)
        print(
            args.prompt
            + generate(
                state,
                config,
                args.prompt,
                args.max_bytes,
                messages=meta.get("messages", True),
                adapt_strength=args.adapt_strength,
                adapt_decay=args.adapt_decay,
            )
        )
    else:
        state, config, meta = load_checkpoint(args.checkpoint)
        from mlm.stream import evaluate_stream

        messages = not args.no_messages and meta.get("messages", True)
        if args.full_documents:
            documents, loss_masks, _ = read_training_data(args.data)
            result = evaluate_stream(state, config, documents, messages, loss_masks=loss_masks)
        else:
            documents, loss_masks, _ = read_training_data(args.data)
            if loss_masks is not None:
                raise ValueError("Explicit supervision requires --full-documents evaluation")
            result = evaluate(state, config, documents, messages=messages)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
