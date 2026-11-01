"""High-throughput sequence-parallel trainer for MLM v2 with Rust BPE & SafeTensors."""

from __future__ import annotations

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from mlm.gguf_export import export_gguf_v2, verify_gguf_with_rust
from mlm.model_v2 import (
    ConfigV2,
    create_state_v2,
    forward_v2,
    generate_v2,
    grow_model_v2_depth,
    grow_model_v2_nodes,
    load_checkpoint_v2,
    make_train_step_v2,
    save_checkpoint_v2,
)
from mlm.tokenizer import BPETokenizer


def load_jsonl_records(path: Path) -> list[dict]:
    """Load JSONL or plain-text training records with optional `loss_spans`."""
    path = Path(path)
    if path.suffix == ".jsonl":
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return [r for r in records if isinstance(r.get("text"), str) and r["text"].strip()]
    docs = [p.strip() for p in path.read_text(encoding="utf-8").split("\n\n") if p.strip()]
    return [{"text": d} for d in docs]


def build_tokenized_dataset(
    records: list[dict],
    tokenizer: BPETokenizer,
    seq_len: int,
) -> dict[str, np.ndarray]:
    """Encode records with Rust BPE and build padded `(N, seq_len)` `inputs, targets, mask`."""
    n = len(records)
    inputs = np.full((n, seq_len), tokenizer.pad_id, dtype=np.int32)
    targets = np.full((n, seq_len), tokenizer.pad_id, dtype=np.int32)
    masks = np.zeros((n, seq_len), dtype=np.float32)

    for i, rec in enumerate(records):
        text = rec["text"]
        spans = rec.get("loss_spans")
        score_eos = bool(rec.get("score_eos", True))
        toks, m = tokenizer.encode_with_mask(text, loss_spans=spans, score_eos=score_eos)
        length = min(seq_len, len(toks) - 1)
        inputs[i, :length] = toks[:length]
        targets[i, :length] = toks[1 : length + 1]
        masks[i, :length] = m[:length]
        if not masks[i].any() and length > 0:
            masks[i, length - 1] = 1.0
    return {"inputs": inputs, "targets": targets, "mask": masks}


def evaluate_v2(
    params: dict[str, jnp.ndarray],
    config: ConfigV2,
    dataset: dict[str, np.ndarray],
) -> dict[str, float]:
    """Evaluate average cross-entropy loss and perplexity over `dataset`."""
    n = len(dataset["inputs"])
    bsz = min(config.batch_size, n)
    total_loss = 0.0
    total_tokens = 0.0

    @jax.jit
    def _eval_batch(p, inp, tgt, msk):
        logits = forward_v2(p, inp, config)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        tok_loss = -jnp.take_along_axis(log_probs, tgt[..., None], axis=-1)[..., 0]
        return jnp.sum(tok_loss * msk), jnp.sum(msk)

    for start in range(0, n, bsz):
        end = min(start + bsz, n)
        inp = jnp.asarray(dataset["inputs"][start:end])
        tgt = jnp.asarray(dataset["targets"][start:end])
        msk = jnp.asarray(dataset["mask"][start:end])
        b_loss, b_toks = _eval_batch(params, inp, tgt, msk)
        total_loss += float(b_loss)
        total_tokens += float(b_toks)

    mean_loss = total_loss / max(total_tokens, 1.0)
    ppl = float(np.exp(min(mean_loss, 20.0)))
    return {"loss": mean_loss, "perplexity": ppl, "evaluated_tokens": total_tokens}


def format_chat_prompt_v2(user_message: str, context: str = "") -> str:
    """Format a user message into the `<|user|>...<|assistant|>` template used in v2."""
    clean_u = user_message.strip()
    if clean_u.startswith("<|user|>"):
        return clean_u
    if context.strip():
        return f"<|user|>\n{context.strip()}\n{clean_u}\n<|assistant|>\n"
    return f"<|user|>\n{clean_u}\n<|assistant|>\n"


def run_train_v2(
    train_path: Path,
    val_path: Path,
    out_dir: Path,
    config: ConfigV2,
    steps: int = 400,
    eval_every: int = 100,
    grow_to_layers: int = 0,
    gguf_out: Path | None = None,
    init_checkpoint: Path | None = None,
    add_nodes: int = 0,
    pretrained_tokenizer: BPETokenizer | None = None,
) -> dict:
    """Train MLM v2 with Rust BPE tokenization, parallel updates, and optional GGUF export."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_records = load_jsonl_records(train_path)
    val_records = load_jsonl_records(val_path)
    all_texts = [r["text"] for r in train_records + val_records]

    if init_checkpoint is not None:
        warm_params, warm_cfg, tokenizer, _ = load_checkpoint_v2(init_checkpoint)
        config = warm_cfg.model_copy(
            update={
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "max_seq_len": config.max_seq_len,
            }
        )
        if add_nodes > 0:
            warm_params, config = grow_model_v2_nodes(warm_params, config, add_nodes=add_nodes)
        state = create_state_v2(config, total_steps=steps)
        state.params = warm_params
    else:
        if pretrained_tokenizer is not None:
            tokenizer = pretrained_tokenizer
        else:
            tokenizer = BPETokenizer.train_from_iterator(
                all_texts,
                vocab_size=config.vocab_size,
                min_frequency=2,
            )
        actual_vocab = tokenizer.vocab_size
        if actual_vocab != config.vocab_size:
            config = config.model_copy(update={"vocab_size": actual_vocab})
        state = create_state_v2(config, total_steps=steps)

    train_ds = build_tokenized_dataset(train_records, tokenizer, config.max_seq_len)
    val_ds = build_tokenized_dataset(val_records, tokenizer, config.max_seq_len)
    step_fn = make_train_step_v2(config, total_steps=steps)

    rng = np.random.default_rng(config.seed)
    n_train = len(train_records)
    bsz = min(config.batch_size, n_train)

    t0 = time.perf_counter()
    processed_tokens = 0
    best_val_loss = float("inf")
    best_params = state.params
    grown = False

    for s in range(1, steps + 1):
        if grow_to_layers > config.n_layers and not grown and s == max(1, steps // 2):
            grown_params, config = grow_model_v2_depth(state.params, config, grow_to_layers)
            state = create_state_v2(config, total_steps=steps - s + 1)
            state.params = grown_params
            step_fn = make_train_step_v2(config, total_steps=steps - s + 1)
            best_params = state.params
            grown = True

        idx = rng.choice(n_train, size=bsz, replace=n_train < bsz)
        inp = jnp.asarray(train_ds["inputs"][idx])
        tgt = jnp.asarray(train_ds["targets"][idx])
        msk = jnp.asarray(train_ds["mask"][idx])

        state.params, state.opt_state, loss, grad_norm = step_fn(
            state.params, state.opt_state, inp, tgt, msk
        )
        state.step = s
        processed_tokens += int(np.sum(train_ds["mask"][idx]))

        if s % eval_every == 0 or s == steps:
            val_metrics = evaluate_v2(state.params, config, val_ds)
            if val_metrics["loss"] <= best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_params = state.params
                save_checkpoint_v2(
                    out_dir / "best.mlm2",
                    best_params,
                    config,
                    tokenizer,
                    step=s,
                    metadata={"val_loss": best_val_loss},
                )
            record = {
                "step": s,
                "train_loss": float(loss),
                "grad_norm": float(grad_norm),
                "val_loss": val_metrics["loss"],
                "val_perplexity": val_metrics["perplexity"],
            }
            print(json.dumps(record), flush=True)

    elapsed = time.perf_counter() - t0
    n_params = int(sum(int(np.prod(v.shape)) for v in jax.tree.leaves(best_params)))
    save_checkpoint_v2(
        out_dir / "latest.mlm2",
        state.params,
        config,
        tokenizer,
        step=steps,
        metadata={"val_loss": best_val_loss, "parameters": n_params},
    )

    gguf_info = None
    if gguf_out is not None:
        gguf_info = export_gguf_v2(gguf_out, state.params, config, tokenizer)
        gguf_info["rust_verification"] = verify_gguf_with_rust(gguf_out)

    sample_prompts = ["Hi", "How are you?", "What is science?"]
    samples = []
    for p_text in sample_prompts:
        formatted = format_chat_prompt_v2(p_text)
        reply = generate_v2(state.params, config, tokenizer, formatted, max_new_tokens=40)
        samples.append({"prompt": p_text, "reply": reply.strip()})

    summary = {
        "arch_version": config.arch_version,
        "parameters": n_params,
        "vocab_size": config.vocab_size,
        "n_layers": config.n_layers,
        "d_model": config.d_model,
        "n_experts": config.n_experts,
        "mature_experts": config.mature_experts,
        "steps": steps,
        "elapsed_seconds": elapsed,
        "tokens_per_second": processed_tokens / max(elapsed, 1e-6),
        "best_val_loss": best_val_loss,
        "checkpoint": str(out_dir / "best.mlm2"),
        "latest_checkpoint": str(out_dir / "latest.mlm2"),
        "gguf": gguf_info,
        "samples": samples,
    }
    (out_dir / "summary_v2.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def run_staged_english_growth_v2(
    data_root: Path,
    out_dir: Path,
    base_config: ConfigV2,
    stage_steps: tuple[int, int, int] = (80, 80, 120),
    eval_every: int = 40,
    gguf_out: Path | None = None,
) -> dict:
    """Train Core+English Seed (4 nodes) -> Understand (6 nodes) -> Chat (8 nodes = 10M model)."""
    from mlm.parquet_stream import build_staged_english_corpora

    data_root = Path(data_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    s1_dir = data_root / "stage1_english_learn"
    s2_dir = data_root / "stage2_english_understand"
    s3_dir = data_root / "stage3_english_chat"
    if not (s1_dir / "train.jsonl").exists():
        build_staged_english_corpora(data_root)

    # Train a unified Rust BPE tokenizer across all 3 English stages so vocabulary is shared
    all_texts: list[str] = []
    for d in (s1_dir, s2_dir, s3_dir):
        for r in load_jsonl_records(d / "train.jsonl"):
            all_texts.append(r["text"])
    shared_tok = BPETokenizer.train_from_iterator(
        all_texts, vocab_size=base_config.vocab_size, min_frequency=2
    )

    # Stage 1: Core Thinking + English Learn (4 specialist Graph Nodes)
    cfg_s1 = base_config.model_copy(
        update={"n_experts": max(4, base_config.n_experts), "vocab_size": shared_tok.vocab_size}
    )
    res_s1 = run_train_v2(
        s1_dir / "train.jsonl",
        s1_dir / "validation.jsonl",
        out_dir / "stage1_learn",
        cfg_s1,
        steps=stage_steps[0],
        eval_every=eval_every,
        pretrained_tokenizer=shared_tok,
    )

    # Stage 2: Grow +2 English Understand Nodes (4 -> 6 Nodes, mature_experts=4 protected)
    res_s2 = run_train_v2(
        s2_dir / "train.jsonl",
        s2_dir / "validation.jsonl",
        out_dir / "stage2_understand",
        cfg_s1,
        steps=stage_steps[1],
        eval_every=eval_every,
        init_checkpoint=Path(res_s1["latest_checkpoint"]),
        add_nodes=2,
    )

    # Stage 3: Grow +2 English Chat Nodes (6 -> 8 Nodes = 10M Model, mature_experts=6 protected)
    res_s3 = run_train_v2(
        s3_dir / "train.jsonl",
        s3_dir / "validation.jsonl",
        out_dir,
        cfg_s1,
        steps=stage_steps[2],
        eval_every=eval_every,
        init_checkpoint=Path(res_s2["latest_checkpoint"]),
        add_nodes=2,
        gguf_out=gguf_out,
    )

    pipeline_summary = {
        "pipeline": "staged_core_english_growth_4_to_6_to_8_nodes",
        "stage1_english_learn": res_s1,
        "stage2_english_understand": res_s2,
        "stage3_english_chat": res_s3,
    }
    (out_dir / "growth_pipeline_summary.json").write_text(
        json.dumps(pipeline_summary, indent=2) + "\n"
    )
    return pipeline_summary
