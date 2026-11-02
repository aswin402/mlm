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
    chat = commands.add_parser(
        "chat",
        help="Interactive real-time chat TUI (or single-turn generation with --message)",
    )
    chat.add_argument("checkpoint", nargs="?", type=Path, default=None)
    chat.add_argument("--config", type=Path, default=None)
    chat.add_argument("--message", type=str, default=None)
    chat.add_argument("--context", type=str, default="")
    chat.add_argument("--max-bytes", type=int, default=160)
    chat.add_argument("--adapt-strength", type=float, default=0.0)
    chat.add_argument("--adapt-decay", type=float, default=0.88)
    chat.add_argument("--learn", action="store_true", help="Online stream update on exchange")
    chat.add_argument("--no-learn", action="store_true", help="Disable online stream update in TUI")
    chat.add_argument("--chunk", type=int, default=16)
    chat.add_argument("--show-thinking", action="store_true", help="Show <think> telemetry panel")
    chat.add_argument("--no-think", action="store_true", help="Hide <think> telemetry panel in TUI")
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
    continual.add_argument("--grpo", action="store_true")
    train_v2_p = commands.add_parser(
        "train-v2",
        help="Train MLM v2 (Deep Hybrid Liquid-Delta + Rust BPE + SafeTensors + GGUF)",
    )
    train_v2_p.add_argument("--train", type=Path, required=True)
    train_v2_p.add_argument("--validation", type=Path, required=True)
    train_v2_p.add_argument("--config", type=Path)
    train_v2_p.add_argument("--out", type=Path, required=True)
    train_v2_p.add_argument("--steps", type=int, default=400)
    train_v2_p.add_argument("--eval-every", type=int, default=100)
    train_v2_p.add_argument("--grow-to-layers", type=int, default=0)
    train_v2_p.add_argument("--init", type=Path, help="Continue from existing .mlm2 checkpoint")
    train_v2_p.add_argument("--add-nodes", type=int, default=0, help="Grow +N specialist nodes")
    train_v2_p.add_argument("--export-gguf", type=Path)
    grow_eng_p = commands.add_parser(
        "grow-english-v2",
        help="Staged Core+English Node Growth (4 Learn -> 6 Understand -> 8 Chat = 10M Model)",
    )
    grow_eng_p.add_argument("--data-root", type=Path, default=Path("data/curated-english-v1"))
    grow_eng_p.add_argument("--config", type=Path, default=Path("configs/mlm-v2-10m-nodes.json"))
    grow_eng_p.add_argument("--out", type=Path, required=True)
    grow_eng_p.add_argument("--s1-steps", type=int, default=80)
    grow_eng_p.add_argument("--s2-steps", type=int, default=80)
    grow_eng_p.add_argument("--s3-steps", type=int, default=120)
    grow_eng_p.add_argument("--eval-every", type=int, default=40)
    grow_eng_p.add_argument("--export-gguf", type=Path)
    export_gguf_p = commands.add_parser(
        "export-gguf",
        help="Export an MLM v2 (.mlm2) checkpoint to a llama.cpp-compatible .gguf file",
    )
    export_gguf_p.add_argument("checkpoint", type=Path)
    export_gguf_p.add_argument("--out", type=Path, required=True)
    export_gguf_p.add_argument("--f16", action="store_true")
    parquet_p = commands.add_parser(
        "stream-parquet",
        help="Stream and curate large Hugging Face Parquet shards via Rust Polars",
    )
    parquet_p.add_argument("--root", type=Path, default=Path("data/curated-english-v1"))
    parquet_p.add_argument("--max-rows", type=int, default=2000)
    cluster_p = commands.add_parser(
        "cluster-chat",
        help="Route prompts across multiple 10M MLM v2 Specialist Nodes (English/Science + Math)",
    )
    cluster_p.add_argument(
        "--english-node",
        type=Path,
        default=Path("runs/mlm-10m-colab/mlm_10m_v2.mlm2"),
    )
    cluster_p.add_argument(
        "--math-node",
        type=Path,
        default=Path("runs/mlm-10m-math-colab/mlm_10m_math_v2.mlm2"),
    )
    cluster_p.add_argument("--message", type=str, required=True)
    cluster_p.add_argument("--context", type=str, default="")
    cluster_p.add_argument("--max-tokens", type=int, default=96)
    cluster_p.add_argument("--force-node", type=str, default=None)
    cluster_p.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "train":
        run_train(args)
    elif args.command == "train-v2":
        from mlm.model_v2 import ConfigV2
        from mlm.train_v2 import run_train_v2

        cfg_v2 = (
            ConfigV2.model_validate_json(args.config.read_text())
            if args.config
            else ConfigV2()
        )
        summary = run_train_v2(
            args.train,
            args.validation,
            args.out,
            cfg_v2,
            steps=args.steps,
            eval_every=args.eval_every,
            grow_to_layers=args.grow_to_layers,
            gguf_out=args.export_gguf,
            init_checkpoint=args.init,
            add_nodes=args.add_nodes,
        )
        print(json.dumps(summary, indent=2), flush=True)
    elif args.command == "grow-english-v2":
        from mlm.model_v2 import ConfigV2
        from mlm.train_v2 import run_staged_english_growth_v2

        cfg_v2 = (
            ConfigV2.model_validate_json(args.config.read_text())
            if args.config and args.config.exists()
            else ConfigV2(n_experts=4)
        )
        summary = run_staged_english_growth_v2(
            args.data_root,
            args.out,
            cfg_v2,
            stage_steps=(args.s1_steps, args.s2_steps, args.s3_steps),
            eval_every=args.eval_every,
            gguf_out=args.export_gguf,
        )
        print(json.dumps(summary, indent=2), flush=True)
    elif args.command == "export-gguf":
        from mlm.gguf_export import export_gguf_v2, verify_gguf_with_rust
        from mlm.model_v2 import load_checkpoint_v2

        params, cfg_v2, tokenizer, _ = load_checkpoint_v2(args.checkpoint)
        info = export_gguf_v2(
            args.out, params, cfg_v2, tokenizer, use_f16=bool(args.f16)
        )
        info["rust_verification"] = verify_gguf_with_rust(args.out)
        print(json.dumps(info, indent=2), flush=True)
    elif args.command == "stream-parquet":
        from mlm.parquet_stream import build_large_parquet_corpus

        info = build_large_parquet_corpus(
            args.root,
            smoltalk_rows=args.max_rows,
            alpaca_rows=args.max_rows,
            dolly_rows=args.max_rows,
            sciq_rows=args.max_rows,
            squad_rows=args.max_rows,
            wiki_rows=args.max_rows // 2,
        )
        print(json.dumps(info, indent=2), flush=True)
    elif args.command == "cluster-chat":
        from mlm.node_cluster import MLMNodeCluster

        specs = []
        if args.english_node and args.english_node.exists():
            specs.append(("node_0_english_science", "general", args.english_node))
        if args.math_node and args.math_node.exists():
            specs.append(("node_1_mathematics", "math", args.math_node))
        if not specs:
            raise FileNotFoundError("No specialist node checkpoints found for cluster-chat")
        cluster = MLMNodeCluster.from_checkpoints(specs)
        res = cluster.generate(
            args.message.replace("\\n", "\n"),
            context=args.context,
            max_new_tokens=args.max_tokens,
            force_node=args.force_node,
        )
        if args.json:
            print(json.dumps(res, indent=2), flush=True)
        else:
            print(
                f"[{res['selected_node']} ({res['selected_domain']})]\n{res['reply']}",
                flush=True,
            )
    elif args.command == "stream-train":
        from mlm.streamrun import run

        run(args)
    elif args.command == "continual":
        from mlm.hierarchy_run import run_continual

        summary = run_continual(args)
        print(json.dumps(summary, indent=2), flush=True)
    elif args.command == "chat":
        if args.checkpoint is not None and args.checkpoint.suffix == ".mlm2":
            from mlm.model_v2 import generate_v2, load_checkpoint_v2
            from mlm.train_v2 import format_chat_prompt_v2
            from mlm.tui import run_chat_tui_v2

            if args.message is not None:
                params_v2, cfg_v2, tok_v2, _ = load_checkpoint_v2(args.checkpoint)
                prompt = format_chat_prompt_v2(args.message.replace("\\n", "\n"), args.context)
                reply = generate_v2(
                    params_v2, cfg_v2, tok_v2, prompt, max_new_tokens=max(16, args.max_bytes // 2)
                )
                print(reply.strip())
                return
            run_chat_tui_v2(
                args.checkpoint,
                max_new_tokens=max(16, args.max_bytes // 2),
                show_thinking=not args.no_think,
                initial_context=args.context,
            )
            return
        from mlm.tui import (
            format_perf_footer,
            format_thinking_panel,
            load_or_init_chat_state,
            run_chat_tui,
        )

        if args.checkpoint is not None and args.checkpoint.exists():
            state, config, meta = load_checkpoint(args.checkpoint)
            label = str(args.checkpoint)
        else:
            state, config, label = load_or_init_chat_state(args.checkpoint, args.config)
            meta = {"messages": True}
        messages = meta.get("messages", True)
        eff_chunk = max(2, min(args.chunk, config.length))

        if args.message is None:
            run_chat_tui(
                state,
                config,
                ckpt_label=label,
                max_bytes=args.max_bytes,
                adapt_strength=args.adapt_strength,
                adapt_decay=args.adapt_decay,
                learn=bool(args.learn) and not bool(args.no_learn),
                chunk=eff_chunk,
                show_thinking=not args.no_think,
                initial_context=args.context,
                save_path=args.save_out,
            )
        else:
            clean_msg = args.message.replace("\\n", "\n")
            learner = LiveLearner(state, config, chunk=eff_chunk, messages=messages)
            if args.show_thinking:
                reply, trace, steps = learner.chat_turn_with_trace(
                    clean_msg,
                    max_bytes=args.max_bytes,
                    adapt_strength=args.adapt_strength,
                    adapt_decay=args.adapt_decay,
                    learn=bool(args.learn),
                    context=args.context,
                )
                print(format_thinking_panel(trace, config, color=False))
                print(reply.strip())
                print(format_perf_footer(trace, learner, steps, color=False))
            elif args.learn:
                reply, _ = learner.chat_turn(
                    clean_msg,
                    max_bytes=args.max_bytes,
                    adapt_strength=args.adapt_strength,
                    adapt_decay=args.adapt_decay,
                    learn=True,
                    context=args.context,
                )
                print(reply.strip())
            else:
                from mlm.generate import format_chat_prompt

                prompt = format_chat_prompt(clean_msg, context=args.context)
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
            if args.save_out:
                save_checkpoint(args.save_out, learner.state, learner.config, meta)
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
