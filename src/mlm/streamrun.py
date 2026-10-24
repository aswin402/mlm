"""Resumable full-document training, with parameter-only warm starts."""

import hashlib
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from mlm.checkpoint import load_checkpoint, save_checkpoint
from mlm.cli import environment, read_history, write_json
from mlm.config import Config
from mlm.data import corpus_hash, read_training_data
from mlm.generate import generate
from mlm.model import initial_state
from mlm.stream import DocumentStream, evaluate_stream, make_stream_step
from mlm.training import create_state


def run(args):
    if args.steps < 1 or args.eval_every < 1:
        raise ValueError("steps and eval-every must be positive")
    paths = {"train": args.train, "validation": args.validation}
    if args.test:
        paths["test"] = args.test
    datasets = {key: read_training_data(path) for key, path in paths.items()}
    documents = {key: value[0] for key, value in datasets.items()}
    loss_masks = {key: value[1] for key, value in datasets.items()}
    supervision = {key: value[2] for key, value in datasets.items()}
    splits = list(documents)
    for i, name in enumerate(splits):
        for other in splits[i + 1 :]:
            if set(documents[name]) & set(documents[other]):
                raise ValueError(f"Duplicate documents across {name}/{other}")
    identity = {key: corpus_hash(value) for key, value in documents.items()}
    previous = None
    messages = not args.no_messages
    if args.resume:
        state, config, previous = load_checkpoint(args.resume)
        if previous.get("mode") != "stream" or previous.get("corpora") != identity:
            raise ValueError("Stream checkpoint mode/corpus mismatch")
        if previous.get("supervision", {key: None for key in paths}) != supervision:
            raise ValueError("Resume supervision masks must match")
        if previous.get("eval_every", args.eval_every) != args.eval_every:
            raise ValueError("Resume validation interval must match")
        if previous["messages"] != messages:
            raise ValueError("Resume message setting must match")
        if args.config and Config.model_validate_json(args.config.read_text()) != config:
            raise ValueError("Resume configuration must match")
    else:
        config = Config.model_validate_json(args.config.read_text()) if args.config else Config()
        state = create_state(config)
        if args.init:
            source, source_config, _ = load_checkpoint(args.init)
            architecture = {
                "architecture",
                "node_readout",
                "node_update",
                "memory",
                "memory_keys",
                "groups",
                "nodes",
                "width",
                "degree",
                "rounds",
            }
            if source_config.model_dump(include=architecture) != config.model_dump(
                include=architecture
            ):
                raise ValueError("Warm-start architecture must match")
            state = state.replace(params=source.params)
    if args.out.exists() and any(args.out.iterdir()):
        if not args.resume or args.resume.resolve() != (args.out / "latest.mlm").resolve():
            raise ValueError("Nonempty destination requires its own latest checkpoint")
        history = args.out / "metrics.jsonl"
        records = read_history(history)
        if any(record["step"] > int(state.step) for record in records):
            raise ValueError("Destination history is newer than checkpoint")
        summary = args.out / "summary.json"
        if summary.exists() and json.loads(summary.read_text())["steps"] > int(state.step):
            raise ValueError("Destination summary is newer than checkpoint")
        if history.exists():
            temporary = history.with_suffix(".tmp")
            temporary.write_text("".join(json.dumps(row) + "\n" for row in records))
            temporary.replace(history)
    args.out.mkdir(parents=True, exist_ok=True)
    stream = DocumentStream(
        documents["train"],
        config.batch_size,
        config.length,
        config.seed,
        loss_masks=loss_masks["train"],
    )
    carry = initial_state(config, config.batch_size)
    if previous:
        stream.restore(previous["stream"])
        carry = jnp.asarray(previous["carry"], dtype=jnp.float32)
        if carry.shape != initial_state(config, config.batch_size).shape:
            raise ValueError("Invalid recurrent carry shape")
    provenance = {
        "mode": "stream",
        "corpora": identity,
        "messages": messages,
        "supervision": supervision,
        "eval_every": args.eval_every,
        "initialization": str(args.init) if args.init else (previous or {}).get("initialization"),
        "initialization_sha256": hashlib.sha256(args.init.read_bytes()).hexdigest()
        if args.init
        else (previous or {}).get("initialization_sha256"),
    }

    def metadata():
        return {**provenance, "stream": stream.snapshot(), "carry": np.asarray(carry).tolist()}

    write_json(args.out / "config.json", config.model_dump())
    write_json(args.out / "environment.json", environment())
    probe = documents["train"][: min(16, len(documents["train"]))]
    probe_masks = None if loss_masks["train"] is None else loss_masks["train"][: len(probe)]
    initial_train = evaluate_stream(state, config, probe, messages, loss_masks=probe_masks)
    initial_dev = evaluate_stream(
        state, config, documents["validation"], messages, loss_masks=loss_masks["validation"]
    )
    best_path = args.out / "best.mlm"
    best_loss, best_step = initial_dev["loss"], int(state.step)
    if args.resume:
        best_source = best_path if best_path.exists() else args.resume.parent / "best.mlm"
        if not best_source.exists():
            raise ValueError("Resume requires the source best.mlm to preserve selection history")
        best, best_cfg, best_meta = load_checkpoint(best_source)
        if (
            best_cfg != config
            or best_meta.get("corpora") != identity
            or best_meta.get("supervision", {key: None for key in paths}) != supervision
            or best_meta.get("messages", True) != messages
        ):
            raise ValueError("Best checkpoint belongs to another run or objective")
        if int(best.step) > int(state.step):
            raise ValueError("Source best is newer than resumed state; use --init for a new stage")
        if (
            best_meta.get("eval_every", args.eval_every) != args.eval_every
            or int(best.step) % args.eval_every != 0
        ):
            raise ValueError("Historical best conflicts with the global selection schedule")
        saved_best_loss = evaluate_stream(
            best, config, documents["validation"], messages, loss_masks=loss_masks["validation"]
        )["loss"]
        if int(state.step) % args.eval_every == 0 and initial_dev["loss"] < saved_best_loss:
            save_checkpoint(best_path, state, config, metadata())
        else:
            best_loss, best_step = saved_best_loss, int(best.step)
            if best_source != best_path:
                save_checkpoint(best_path, best, config, best_meta)
    else:
        save_checkpoint(best_path, state, config, metadata())
    step = make_stream_step(config, messages)
    sample = DocumentStream(
        documents["train"], config.batch_size, config.length, loss_masks=loss_masks["train"]
    ).next_batch()
    tick = time.perf_counter()
    compiled = step.lower(state, carry, jax.tree.map(jnp.asarray, sample)).compile()
    compile_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    first_step = int(state.step)
    payload_bytes, update_seconds = 0, 0.0
    supervised_bytes, chunks, completed_updates = 0, 0, 0
    with (args.out / "metrics.jsonl").open("a" if args.resume else "w") as log:
        while completed_updates < args.steps:
            chunks += 1
            batch = jax.tree.map(jnp.asarray, stream.next_batch())
            update_start = time.perf_counter()
            state, carry, raw = compiled(state, carry, batch)
            metrics = jax.device_get(raw)
            update_seconds += time.perf_counter() - update_start
            if not bool(metrics["finite"]):
                raise FloatingPointError("Nonfinite stream update rejected")
            payload_bytes += int(metrics["consumed_bytes"])
            supervised_bytes += int(metrics["bytes"])
            if not bool(metrics["updated"]):
                continue
            completed_updates += 1
            selection_eligible = int(state.step) % args.eval_every == 0
            if selection_eligible or completed_updates == args.steps:
                dev = evaluate_stream(
                    state,
                    config,
                    documents["validation"],
                    messages,
                    loss_masks=loss_masks["validation"],
                )
                save_checkpoint(args.out / "latest.mlm", state, config, metadata())
                if selection_eligible and dev["loss"] < best_loss:
                    best_loss, best_step = dev["loss"], int(state.step)
                    save_checkpoint(best_path, state, config, metadata())
                record = {
                    "step": int(state.step),
                    "selection_eligible": selection_eligible,
                    "training_chunk_loss": float(metrics["loss"]),
                    "gradient_norm": float(metrics["grad_norm"]),
                    "validation": dev,
                    "epoch_restarts": stream.epochs,
                    "session_payload_bytes": payload_bytes,
                    "session_supervised_bytes": supervised_bytes,
                    "session_chunks": chunks,
                }
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(json.dumps(record), flush=True)
    session_seconds = time.perf_counter() - tick
    best, _, _ = load_checkpoint(best_path)
    summary = {
        "scope": "full-document truncated BPTT; no growth",
        "architecture": config.architecture,
        "steps": int(state.step),
        "session_steps": int(state.step) - first_step,
        "parameters": sum(p.size for p in jax.tree.leaves(state.params)),
        "messages": messages,
        "best_step": best_step,
        "best_validation_loss": best_loss,
        "selection_interval": args.eval_every,
        "selection_schedule": "global optimizer-step multiples; endpoint-only records excluded",
        "compile_seconds": compile_seconds,
        "training_loop_seconds": session_seconds,
        "update_seconds": update_seconds,
        "training_payload_bytes": payload_bytes,
        "supervised_payload_bytes": supervised_bytes,
        "chunks_processed": chunks,
        "prompt_only_chunks": chunks - completed_updates,
        "supervision": supervision,
        "update_bytes_per_second": payload_bytes / max(update_seconds, 1e-9),
        "epoch_restarts": stream.epochs,
        "initial_train_probe": initial_train,
        "train_probe": evaluate_stream(state, config, probe, messages, loss_masks=probe_masks),
        "initial_validation": initial_dev,
        "validation": evaluate_stream(
            state, config, documents["validation"], messages, loss_masks=loss_masks["validation"]
        ),
        "best_validation": evaluate_stream(
            best, config, documents["validation"], messages, loss_masks=loss_masks["validation"]
        ),
        "corpora": identity,
        "initialization": provenance,
        "checkpoint_bytes": (args.out / "latest.mlm").stat().st_size,
    }
    if "test" in documents:
        # Only after development-based checkpoint selection; never controls training.
        summary["test"] = evaluate_stream(
            best, config, documents["test"], messages, loss_masks=loss_masks["test"]
        )
    write_json(args.out / "summary.json", summary)
    prompts = ["Once upon a time", "The little girl", "<|user|>\nHi!\n<|assistant|>\n"]
    write_json(
        args.out / "samples.json",
        {
            "step": int(best.step),
            "decoding": "greedy",
            "max_bytes": 160,
            "samples": [
                {
                    "prompt": prompt,
                    "continuation": generate(best, config, prompt, 160, messages=messages),
                }
                for prompt in prompts
            ],
        },
    )
    print(json.dumps(summary, indent=2))
