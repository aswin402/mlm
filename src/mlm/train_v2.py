"""High-throughput sequence-parallel trainer for MLM v2 with Rust BPE & SafeTensors."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from mlm.gguf_export import export_gguf_v2, verify_gguf_with_rust
from mlm.model_v2 import (
    ConfigV2,
    build_layer_execution_schedule,
    create_state_v2,
    expand_shortconv_kernel_v2,
    forward_v2,
    generate_v2,
    grow_model_v2_depth,
    grow_model_v2_nodes,
    load_checkpoint_v2,
    make_train_step_v2,
    merge_checkpoints_v2,
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
    expand_conv_kernel: int = 0,
    pretrained_tokenizer: BPETokenizer | None = None,
) -> dict:
    """Train MLM v2 with Rust BPE tokenization, parallel updates, and optional GGUF export."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_records = load_jsonl_records(train_path)
    val_records = load_jsonl_records(val_path)
    all_texts = [r["text"] for r in train_records + val_records]

    teacher_params: dict[str, jnp.ndarray] | None = None
    teacher_config: ConfigV2 | None = None
    if init_checkpoint is not None:
        warm_params, warm_cfg, tokenizer, _ = load_checkpoint_v2(init_checkpoint)
        if config.kd_weight > 0.0:
            teacher_params = {k: jnp.array(v, copy=True) for k, v in warm_params.items()}
            teacher_config = warm_cfg
        config = warm_cfg.model_copy(
            update={
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "max_seq_len": config.max_seq_len,
                "mature_experts": config.mature_experts,
                "mature_lr_scale": config.mature_lr_scale,
                "shared_lr_scale": config.shared_lr_scale,
                "active_subnodes": config.active_subnodes,
                "router_bias_gamma": config.router_bias_gamma,
                "mtp_weight": config.mtp_weight,
                "kd_weight": config.kd_weight,
                "kd_top_k": config.kd_top_k,
                "kd_temperature": config.kd_temperature,
                "sliding_window": config.sliding_window,
                "use_head_gate": config.use_head_gate,
                "sandwich_norm_clip": config.sandwich_norm_clip,
                "use_mhc": config.use_mhc,
                "mhc_sinkhorn_iters": config.mhc_sinkhorn_iters,
                "loop_start_layer": config.loop_start_layer,
                "loop_end_layer": config.loop_end_layer,
                "loop_repeats": config.loop_repeats,
                "use_ncm": config.use_ncm,
                "ncm_weight": config.ncm_weight,
                "ncm_window": config.ncm_window,
                "domain_gate_weight": config.domain_gate_weight,
                "use_fone_abacus": config.use_fone_abacus,
                "digit_token_ids": tokenizer.get_digit_token_ids(),
                "think_open_id": tokenizer.think_open_id,
                "think_close_id": tokenizer.think_close_id,
            }
        )
        if expand_conv_kernel > config.conv_kernel:
            warm_params, config = expand_shortconv_kernel_v2(
                warm_params, config, new_kernel=expand_conv_kernel
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
        config = config.model_copy(
            update={
                "vocab_size": actual_vocab,
                "digit_token_ids": tokenizer.get_digit_token_ids(),
                "think_open_id": tokenizer.think_open_id,
                "think_close_id": tokenizer.think_close_id,
            }
        )
        state = create_state_v2(config, total_steps=steps)
        if config.blt_ngram_dim > 0:
            state.params["blt_token_ngrams"] = jnp.asarray(
                tokenizer.compute_blt_ngram_table(config.blt_ngram_dim),
                dtype=jnp.float32,
            )

    train_ds = build_tokenized_dataset(train_records, tokenizer, config.max_seq_len)
    val_ds = build_tokenized_dataset(val_records, tokenizer, config.max_seq_len)
    step_fn = make_train_step_v2(
        config,
        total_steps=steps,
        teacher_params=teacher_params,
        teacher_config=teacher_config,
    )

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
        "effective_layer_passes": len(build_layer_execution_schedule(config)),
        "use_mhc": config.use_mhc,
        "use_ncm": config.use_ncm,
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


def _content_words(text: str) -> set[str]:
    clean = text.replace("<|think|>", " ").replace("<|/think|>", " ")
    words = re.findall(r"[a-z0-9]{3,}", clean.lower())
    stop = {
        "the",
        "and",
        "for",
        "are",
        "but",
        "not",
        "you",
        "all",
        "any",
        "can",
        "had",
        "her",
        "was",
        "one",
        "our",
        "out",
        "day",
        "get",
        "has",
        "him",
        "his",
        "how",
        "man",
        "new",
        "now",
        "old",
        "see",
        "two",
        "way",
        "who",
        "boy",
        "did",
        "its",
        "let",
        "put",
        "say",
        "she",
        "too",
        "use",
        "what",
        "when",
        "where",
        "which",
        "why",
        "with",
        "this",
        "that",
        "from",
        "they",
        "have",
        "there",
        "their",
        "about",
        "would",
        "could",
        "should",
        "into",
        "than",
        "them",
        "then",
        "some",
        "these",
        "those",
    }
    return {w for w in words if w not in stop}


def compute_ornith_task_weights_v2(
    emp_pass_rates: np.ndarray,
    novelty_scores: np.ndarray,
    validity_flags: np.ndarray | None = None,
    p_star: float = 0.35,
    sigma: float = 0.30,
) -> np.ndarray:
    """Ornith-1.5 (`ornith.ai/ornith_1_5.html`) Task Frontier & Novelty Curriculum `V * D * N`."""
    p = np.asarray(emp_pass_rates, dtype=np.float64)
    n_vec = np.clip(np.asarray(novelty_scores, dtype=np.float64), 0.15, 1.0)
    v_vec = (
        np.ones_like(p)
        if validity_flags is None
        else np.clip(np.asarray(validity_flags, dtype=np.float64), 0.0, 1.0)
    )
    diff_term = np.exp(-np.square(p - float(p_star)) / (2.0 * float(sigma) ** 2))
    raw = v_vec * diff_term * n_vec
    total = float(np.sum(raw))
    if total <= 1e-12:
        return np.full_like(p, 1.0 / max(len(p), 1))
    return raw / total


def score_grpo_response_v2(
    prompt_text: str,
    generated_text: str,
    reference_answer: str,
) -> float:
    """Compute Ornith-1.5 + DeepSeek-R1 + Dr. GRPO verifiable format, validity gate, & F1 reward."""
    if not generated_text.strip():
        return -1.0

    reward = 0.0
    ans_part = generated_text.strip()
    thought_part = ""

    # 1. Ornith-1.5 Hard Scaffold Validity Gate V(q, s) & `<|think|>...<|/think|>` check
    if "<|think|>" in generated_text or "<|/think|>" in generated_text:
        i_open = generated_text.find("<|think|>")
        i_close = generated_text.find("<|/think|>")
        if (
            0 <= i_open < i_close
            and generated_text.count("<|think|>") == 1
            and generated_text.count("<|/think|>") == 1
        ):
            thought_part = generated_text[i_open + len("<|think|>") : i_close].strip()
            ans_part = generated_text[i_close + len("<|/think|>") :].strip()
            if len(thought_part) >= 8 and len(ans_part) >= 4:
                reward += 0.5
            else:
                return -0.4
        else:
            # Ornith-1.5 Hard Gate: V(q, s) = 0 => malformed scaffold cannot earn positive reward
            return -0.6

    if ans_part and ans_part[-1] in ".!?":
        reward += 0.2

    # 2. Verifiable answer F1, GLM-5.3-Flash SAO thought-bridge, & Ornith-1.5 Hack Resistance H(h)
    ref_clean = reference_answer
    if "<|/think|>" in ref_clean:
        ref_clean = ref_clean.split("<|/think|>", 1)[1].strip()

    norm_ans = " ".join(ans_part.lower().split())
    norm_ref = " ".join(ref_clean.lower().split())
    norm_prompt = " ".join(prompt_text.lower().split())
    # Ornith-1.5 Anti-Reward-Hacking check: reject answers that merely parrot the user question
    is_prompt_parrot = bool(
        norm_prompt and norm_ans == norm_prompt and norm_ans != norm_ref
    )
    if is_prompt_parrot:
        return -0.5

    gen_words = _content_words(ans_part)
    ref_words = _content_words(ref_clean)
    prompt_words = _content_words(prompt_text)
    allowed_words = ref_words | prompt_words
    if thought_part:
        thought_words = _content_words(thought_part)
        if thought_words and (thought_words & allowed_words):
            reward += 0.20

    if ref_words and gen_words:
        overlap = len(gen_words & ref_words)
        prec = overlap / len(gen_words)
        rec = overlap / len(ref_words)
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        reward += 1.2 * f1
        # Penalize off-topic drift when majority of generated content words are unrelated
        off_topic_ratio = len(gen_words - allowed_words) / max(len(gen_words), 1)
        if off_topic_ratio > 0.70 and overlap == 0:
            reward -= 0.6
    elif not gen_words:
        reward -= 0.4

    if norm_ref and (norm_ans == norm_ref or norm_ans.startswith(norm_ref[:24])):
        reward += 0.4

    return float(np.clip(reward, -1.0, 2.5))


def make_grpo_step_v2(
    config: ConfigV2,
    total_steps: int = 200,
    kl_beta: float = 0.04,
):
    """Compile Ornith-1.5 + Dr. GRPO (`arXiv:2503.20783`) critic-free policy optimization step."""
    warmup = min(config.warmup_steps, max(1, total_steps // 5))
    decay_steps = max(total_steps, warmup + 1)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=config.learning_rate * 0.1,
        peak_value=config.learning_rate,
        warmup_steps=warmup,
        decay_steps=decay_steps,
        end_value=config.learning_rate * 0.1,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(config.clip_norm),
        optax.adamw(learning_rate=schedule, weight_decay=config.weight_decay),
    )

    @jax.jit
    def grpo_step(
        params: dict[str, jnp.ndarray],
        ref_params: dict[str, jnp.ndarray],
        opt_state: optax.OptState,
        inputs: jnp.ndarray,
        targets: jnp.ndarray,
        mask: jnp.ndarray,
        rewards: jnp.ndarray,
    ):
        # Dr. GRPO unbiased group-centered advantage (no length or std division bias)
        advantages = rewards - jnp.mean(rewards)

        ref_logits = jax.lax.stop_gradient(forward_v2(ref_params, inputs, config))
        ref_lps_all = jax.nn.log_softmax(ref_logits, axis=-1)
        ref_tok_lps = jnp.take_along_axis(ref_lps_all, targets[..., None], axis=-1)[..., 0]
        old_logits = jax.lax.stop_gradient(forward_v2(params, inputs, config))
        old_lps_all = jax.nn.log_softmax(old_logits, axis=-1)
        old_tok_lps = jnp.take_along_axis(old_lps_all, targets[..., None], axis=-1)[..., 0]

        # Ornith-1.0 / 1.5 Horizon Staleness Weighting w(d_t) (K1=36, K2=96, lambda=0.025)
        d_t = jnp.cumsum(mask, axis=-1)
        w_dt = jnp.where(
            d_t <= 36.0,
            1.0,
            jnp.where(d_t <= 96.0, jnp.exp(-0.025 * (d_t - 36.0)), 0.0),
        )

        def loss_fn(p):
            logits = forward_v2(p, inputs, config)
            lps_all = jax.nn.log_softmax(logits, axis=-1)
            tok_lps = jnp.take_along_axis(lps_all, targets[..., None], axis=-1)[..., 0]

            ratio = jnp.exp(jnp.clip(tok_lps - old_tok_lps, -4.0, 4.0))
            adv = advantages[:, None]
            surr1 = ratio * adv
            surr2 = jnp.clip(ratio, 1.0 - 0.20, 1.0 + 0.28) * adv
            clipped_obj = jnp.minimum(surr1, surr2) * w_dt

            log_diff = ref_tok_lps - tok_lps
            kl_per_tok = jnp.exp(log_diff) - log_diff - 1.0

            per_tok_obj = clipped_obj - float(kl_beta) * kl_per_tok
            # Normalize by fixed horizon constant `max_seq_len` per Dr. GRPO (Eq. 5)
            norm_const = float(inputs.shape[0] * config.max_seq_len)
            return -jnp.sum(per_tok_obj * mask) / jnp.maximum(norm_const, 1.0)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        clean_grads = {
            k: (
                jnp.zeros_like(g)
                if (k.endswith(".nodes.router_bias") or k == "blt_token_ngrams")
                else g
            )
            for k, g in grads.items()
        }
        updates, new_opt_state = tx.update(clean_grads, opt_state, params)
        scaled_updates = {}
        for k, u in updates.items():
            if k.endswith(".nodes.router_bias") or k == "blt_token_ngrams":
                scaled_updates[k] = jnp.zeros_like(u)
            elif k.endswith(".nodes.edge_scale") or k.endswith(".nodes.collab_gate"):
                if 0 < config.mature_experts < config.n_experts:
                    e_mask = np.ones((config.n_experts,), dtype=np.float32)
                    e_mask[: config.mature_experts] = float(config.mature_lr_scale)
                    if config.n_experts > 10:
                        e_mask[:10] = 0.0
                    scaled_updates[k] = u * jnp.asarray(e_mask)
                else:
                    scaled_updates[k] = u
            elif config.mature_experts > 0 and ".nodes." in k and not k.endswith(".router.weight"):
                if 0 < config.mature_experts < config.n_experts:
                    m_1d = np.ones((config.n_experts,), dtype=np.float32)
                    m_1d[: config.mature_experts] = float(config.mature_lr_scale)
                    if config.n_experts > 10:
                        m_1d[:10] = 0.0
                    m_jax = jnp.asarray(m_1d)
                    if u.ndim == 3 and u.shape[0] == config.n_experts:
                        scaled_updates[k] = u * m_jax[:, None, None]
                    elif u.ndim == 2 and u.shape == (config.n_experts, config.n_experts):
                        e_mat = jnp.asarray(np.maximum(m_1d[:, None], m_1d[None, :]))
                        scaled_updates[k] = u * e_mat
                    elif u.ndim == 2 and u.shape[0] == config.n_experts:
                        scaled_updates[k] = u * m_jax[:, None]
                    else:
                        scaled_updates[k] = u * max(float(config.shared_lr_scale), 0.25)
                else:
                    scaled_updates[k] = u
            elif k.endswith(".nodes.router.weight"):
                if 0 < config.mature_experts < config.n_experts:
                    r_mask = np.ones((1, config.n_experts), dtype=np.float32)
                    r_mask[:, : config.mature_experts] = float(config.mature_lr_scale)
                    if config.n_experts > 10:
                        r_mask[:, :10] = 0.0
                    scaled_updates[k] = u * jnp.asarray(r_mask)
                else:
                    scaled_updates[k] = u
            elif k.endswith(".shortconv.conv.weight") and (
                0 < config.mature_conv_kernel < config.conv_kernel
                and config.shared_lr_scale > 0.0
            ):
                new_taps = config.conv_kernel - config.mature_conv_kernel
                c_mask = np.ones((config.conv_kernel, 1), dtype=np.float32)
                c_mask[new_taps:, :] = float(config.shared_lr_scale)
                scaled_updates[k] = u * jnp.asarray(c_mask)
            elif k.startswith("mtp_"):
                scaled_updates[k] = u
            else:
                scaled_updates[k] = u * float(config.shared_lr_scale)
        new_params = optax.apply_updates(params, scaled_updates)
        return new_params, new_opt_state, loss, jnp.mean(rewards)

    return grpo_step


def _make_grpo_rollout_record(user_text: str, reply_text: str) -> dict:
    """Format a GRPO rollout record with `loss_spans` starting after `<|think|>` if present."""
    from mlm.curated_data import _make_supervised_chat_record

    reply = reply_text.strip()
    if reply.startswith("<|think|>"):
        prefix = f"<|user|>\n{user_text.strip()}\n<|assistant|>\n<|think|>"
        body = reply[len("<|think|>") :]
        if "<|/think|>" in body:
            body = body.split("<|/think|>", 1)[0] + "<|/think|>"
        full_text = prefix + body
        start_b = len(prefix.encode("utf-8"))
        end_b = len(full_text.encode("utf-8"))
        return {
            "text": full_text,
            "loss_spans": [[start_b, min(end_b, max(start_b + 1, end_b))]],
            "score_eos": True,
        }
    return _make_supervised_chat_record(user_text, reply)


def run_grpo_v2(
    train_path: Path,
    out_dir: Path,
    init_checkpoint: Path,
    *,
    steps: int = 120,
    group_size: int = 4,
    learning_rate: float = 4e-4,
    kl_beta: float = 0.04,
    merge_alpha: float = 0.65,
    gguf_out: Path | None = None,
) -> dict:
    """Run DeepSeek-R1 + Dr. GRPO reinforcement learning on an MLM v2 checkpoint."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    params, cfg, tokenizer, _ = load_checkpoint_v2(init_checkpoint)
    is_tiered = cfg.mature_experts >= 8 and cfg.n_experts > cfg.mature_experts
    cfg = cfg.model_copy(
        update={
            "learning_rate": learning_rate,
            "mature_lr_scale": 0.0 if is_tiered else 0.05,
            "shared_lr_scale": 0.0 if is_tiered else 0.08,
        }
    )
    state = create_state_v2(cfg, total_steps=steps)
    state.params = params
    ref_params = {k: jnp.array(v, copy=True) for k, v in params.items()}
    grpo_step_fn = make_grpo_step_v2(cfg, total_steps=steps, kl_beta=kl_beta)

    raw_records = load_jsonl_records(train_path)
    qa_items: list[tuple[str, str]] = []
    for r in raw_records:
        txt = r.get("text", "")
        if "\n<|assistant|>\n" in txt:
            u_part, a_part = txt.split("\n<|assistant|>\n", 1)
            u_clean = u_part.replace("<|user|>\n", "").strip()
            a_clean = a_part.strip()
            extra_ans = str(r.get("answer", "")).strip()
            if extra_ans and "<|/think|>\n" not in a_clean:
                a_clean = f"{a_clean}\n{extra_ans}"
            if u_clean and a_clean:
                qa_items.append((u_clean, a_clean))

    rng = np.random.default_rng(cfg.seed + 77)
    t0 = time.perf_counter()
    reward_history: list[float] = []
    n_items = max(len(qa_items), 1)
    emp_pass_rates = np.full((n_items,), 0.35, dtype=np.float64)
    novelty_scores = np.ones((n_items,), dtype=np.float64)
    validity_flags = np.ones((n_items,), dtype=np.float64)
    prompt_word_sets = [_content_words(u) for u, _ in qa_items]

    for s in range(1, steps + 1):
        task_probs = compute_ornith_task_weights_v2(
            emp_pass_rates, novelty_scores, validity_flags, p_star=0.35, sigma=0.30
        )
        q_idx = int(rng.choice(len(qa_items), p=task_probs))
        u_prompt, ref_ans = qa_items[q_idx]
        formatted_prompt = format_chat_prompt_v2(u_prompt)

        group_records: list[dict] = []
        group_rewards: list[float] = []
        for g_idx in range(group_size - 1):
            temp = 0.65 + 0.15 * float(g_idx)
            gen_reply = generate_v2(
                state.params,
                cfg,
                tokenizer,
                formatted_prompt,
                max_new_tokens=48,
                temperature=temp,
                include_think=True,
            )
            if not gen_reply.strip():
                gen_reply = "<|think|>Reasoning...<|/think|>\n..."
            r_val = score_grpo_response_v2(u_prompt, gen_reply, ref_ans)
            group_records.append(_make_grpo_rollout_record(u_prompt, gen_reply))
            group_rewards.append(r_val)

        # Update Ornith-1.5 empirical pass rate p_i and novelty N(q)
        sample_rewards = np.asarray(group_rewards, dtype=np.float64)
        cur_pass = float(np.mean(sample_rewards >= 0.75))
        emp_pass_rates[q_idx] = 0.6 * emp_pass_rates[q_idx] + 0.4 * cur_pass
        cur_words = prompt_word_sets[q_idx]
        novelty_scores = np.minimum(1.0, novelty_scores + 0.03)
        novelty_scores[q_idx] = 0.35
        if cur_words:
            for j_idx, w_set in enumerate(prompt_word_sets):
                if j_idx != q_idx and w_set:
                    sim = len(cur_words & w_set) / max(len(cur_words | w_set), 1)
                    if sim > 0.5:
                        novelty_scores[j_idx] = min(novelty_scores[j_idx], 1.0 - 0.5 * sim)

        # Include verified reference trajectory as anchor in group to prevent zero-variance collapse
        ref_r_val = score_grpo_response_v2(u_prompt, ref_ans, ref_ans)
        group_records.append(_make_grpo_rollout_record(u_prompt, ref_ans))
        group_rewards.append(ref_r_val)

        grp_ds = build_tokenized_dataset(group_records, tokenizer, cfg.max_seq_len)
        state.params, state.opt_state, g_loss, mean_r = grpo_step_fn(
            state.params,
            ref_params,
            state.opt_state,
            jnp.asarray(grp_ds["inputs"]),
            jnp.asarray(grp_ds["targets"]),
            jnp.asarray(grp_ds["mask"]),
            jnp.asarray(group_rewards, dtype=jnp.float32),
        )
        reward_history.append(float(mean_r))
        if s % max(1, steps // 5) == 0 or s == steps:
            print(
                json.dumps(
                    {
                        "grpo_step": s,
                        "loss": float(g_loss),
                        "mean_group_reward": float(np.mean(reward_history[-20:])),
                    }
                ),
                flush=True,
            )

    if 0.0 < merge_alpha < 1.0:
        state.params = merge_checkpoints_v2(ref_params, state.params, alpha=merge_alpha)

    save_checkpoint_v2(
        out_dir / "latest.mlm2",
        state.params,
        cfg,
        tokenizer,
        step=steps,
        metadata={"grpo_mean_reward": float(np.mean(reward_history[-20:]))},
    )
    save_checkpoint_v2(
        out_dir / "best.mlm2",
        state.params,
        cfg,
        tokenizer,
        step=steps,
        metadata={"grpo_mean_reward": float(np.mean(reward_history[-20:]))},
    )
    gguf_info = None
    if gguf_out is not None:
        gguf_info = export_gguf_v2(gguf_out, state.params, cfg, tokenizer)
        gguf_info["rust_verification"] = verify_gguf_with_rust(gguf_out)

    return {
        "grpo_steps": steps,
        "group_size": group_size,
        "initial_reward": float(np.mean(reward_history[:10])) if reward_history else 0.0,
        "final_reward": float(np.mean(reward_history[-10:])) if reward_history else 0.0,
        "elapsed_seconds": time.perf_counter() - t0,
        "checkpoint": str(out_dir / "latest.mlm2"),
        "gguf": gguf_info,
    }


def calibrate_spidernet_router_v2(
    params: dict[str, jnp.ndarray],
    config: ConfigV2,
    tokenizer: BPETokenizer,
    skill_buckets: list[tuple[tuple[int, ...], float, list[dict]]],
    steps: int = 80,
    learning_rate: float = 1.5e-3,
) -> dict[str, jnp.ndarray]:
    """Calibrate `blk.*.nodes.router.weight` jointly across all SpiderNet skill sub-nodes.

    All Main Core Trunk weights and all 20 Sub-Node SwiGLU weights (`gate, up, down`) remain
    100% frozen; only `blk.{0..7}.nodes.router.weight` is updated so each skill routes cleanly
    to its dedicated Sub-Node pair while preserving strict English vs. Math domain separation.
    """
    from mlm.model_v2 import _forward_core_v2, rms_norm

    all_records: list[dict] = []
    domain_labels: list[float] = []
    subnode_masks: list[np.ndarray] = []

    for subnodes, is_math, recs in skill_buckets:
        mask_vec = np.zeros((config.n_experts,), dtype=np.float32)
        for s_idx in subnodes:
            if 0 <= int(s_idx) < config.n_experts:
                mask_vec[int(s_idx)] = 1.0
        for r in recs:
            all_records.append(r)
            domain_labels.append(float(is_math))
            subnode_masks.append(mask_vec)

    ds = build_tokenized_dataset(all_records, tokenizer, config.max_seq_len)
    dom_arr = np.asarray(domain_labels, dtype=np.float32)
    sub_arr = np.stack(subnode_masks, axis=0)
    n_total = len(all_records)
    bsz = min(config.batch_size, n_total)

    cfg_cal = config.model_copy(update={"active_subnodes": ()})
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=learning_rate, weight_decay=0.01),
    )
    opt_state = tx.init(params)

    @jax.jit
    def router_step(p, o_st, inp, tgt, msk, dom_lbl, sub_msk):
        def loss_fn(cur_p):
            logits, _, mean_dl, layer_rl = _forward_core_v2(
                cur_p, inp, cfg_cal, return_layer_router_logits=True
            )
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            tok_loss = -jnp.take_along_axis(log_probs, tgt[..., None], axis=-1)[..., 0]
            lm_loss = jnp.sum(tok_loss * msk) / jnp.maximum(jnp.sum(msk), 1.0)

            # Entrance prompt_seq_dl supervision + within-bank sub-node routing supervision
            raw_emb = cur_p["tok_embeddings.weight"][inp]
            h0 = rms_norm(raw_emb, cur_p["blk.0.ffn_norm.weight"], cfg_cal.rms_eps)
            h0_u = h0 * jax.lax.rsqrt(jnp.sum(jnp.square(h0), axis=-1, keepdims=True) + 1e-6)

            valid_tok = (inp != tokenizer.pad_id).astype(jnp.float32)
            is_close = (inp == cfg_cal.think_close_id).astype(jnp.int32)
            before_think = (jnp.cumsum(is_close, axis=1) == 0) & (inp != cfg_cal.think_open_id)
            p_mask = (before_think & (inp != tokenizer.pad_id)).astype(jnp.float32)

            r0_w = cur_p["blk.0.nodes.router.weight"]
            r0_u = r0_w * jax.lax.rsqrt(jnp.sum(jnp.square(r0_w), axis=0, keepdims=True) + 1e-6)
            r0_logits = 8.0 * jnp.matmul(h0_u, r0_u)
            r0_e = jax.scipy.special.logsumexp(r0_logits[..., :8], axis=-1) - jnp.log(8.0)
            r0_m = jax.scipy.special.logsumexp(r0_logits[..., 10:18], axis=-1) - jnp.log(8.0)
            r0_dl = 2.0 * (r0_m - r0_e)
            seq_dl = jnp.sum(r0_dl * p_mask, axis=1) / jnp.maximum(jnp.sum(p_mask, axis=1), 1.0)

            # Margin loss: seq_dl <= -1.4 for English (0), seq_dl >= +0.4 for Math (1)
            eng_hinge = jax.nn.relu(seq_dl - (-1.40)) * (1.0 - dom_lbl)
            math_hinge = jax.nn.relu(0.40 - seq_dl) * dom_lbl
            dom_bce = optax.sigmoid_binary_cross_entropy(
                mean_dl, jnp.broadcast_to(dom_lbl[:, None], mean_dl.shape)
            )
            dom_loss = jnp.mean(eng_hinge + math_hinge) + (
                jnp.sum(dom_bce * valid_tok) / jnp.maximum(jnp.sum(valid_tok), 1.0)
            )

            # Sub-node pair routing loss across all layers using real layer ffn_normed states
            sub_loss_sum = jnp.array(0.0, dtype=jnp.float32)
            for rl in layer_rl:
                p_eng_b = jax.nn.softmax(rl[..., :8], axis=-1)
                p_math_b = jax.nn.softmax(rl[..., 10:18], axis=-1)
                prob_e = jnp.sum(p_eng_b * sub_msk[:, None, :8], axis=-1)
                prob_m = jnp.sum(p_math_b * sub_msk[:, None, 10:18], axis=-1)
                target_prob = jnp.where(dom_lbl[:, None] > 0.5, prob_m, prob_e)
                nll = -jnp.log(jnp.maximum(target_prob, 1e-5))
                sub_loss_sum = sub_loss_sum + (
                    jnp.sum(nll * valid_tok) / jnp.maximum(jnp.sum(valid_tok), 1.0)
                )
            sub_loss = sub_loss_sum / float(max(len(layer_rl), 1))
            return lm_loss + 0.8 * dom_loss + 0.6 * sub_loss

        loss_val, grads = jax.value_and_grad(loss_fn)(p)
        masked_grads = {
            k: (g if k.endswith(".nodes.router.weight") else jnp.zeros_like(g))
            for k, g in grads.items()
        }
        updates, next_opt = tx.update(masked_grads, o_st, p)
        masked_updates = {
            k: (u if k.endswith(".nodes.router.weight") else jnp.zeros_like(u))
            for k, u in updates.items()
        }
        next_p = optax.apply_updates(p, masked_updates)
        return next_p, next_opt, loss_val

    rng = np.random.default_rng(config.seed + 77)
    cur_params = params
    for s in range(1, steps + 1):
        idx = rng.choice(n_total, size=bsz, replace=n_total < bsz)
        cur_params, opt_state, l_val = router_step(
            cur_params,
            opt_state,
            jnp.asarray(ds["inputs"][idx]),
            jnp.asarray(ds["targets"][idx]),
            jnp.asarray(ds["mask"][idx]),
            jnp.asarray(dom_arr[idx]),
            jnp.asarray(sub_arr[idx]),
        )
        if s % max(1, steps // 2) == 0 or s == steps:
            print(json.dumps({"router_cal_step": s, "loss": float(l_val)}), flush=True)
    return cur_params


def run_grpo_reasoning_v2(
    params: dict[str, jnp.ndarray],
    config: ConfigV2,
    tokenizer: BPETokenizer,
    positive_records: list[dict],
    negative_records: list[dict],
    active_think_subnodes: tuple[int, ...] = (18, 19),
    steps: int = 60,
    learning_rate: float = 6.0e-4,
    grpo_beta_kl: float = 0.04,
) -> dict[str, jnp.ndarray]:
    """DeepSeek-R1 Group Relative Policy Optimization (`GRPO`) + Process Verification (`PRM`).

    Trains ONLY the `<|think|>` Sub-Nodes (`active_think_subnodes`, e.g., `(8, 9)` for English
    Deliberative or `(18, 19)` for Math Verified) AFTER router calibration, keeping the Main Core
    Trunk and all Base Sub-Nodes (`0..7` and `10..17`) and base router columns 100% frozen.
    """
    from mlm.model_v2 import _forward_core_v2

    pos_ds = build_tokenized_dataset(positive_records, tokenizer, config.max_seq_len)
    neg_ds = build_tokenized_dataset(negative_records, tokenizer, config.max_seq_len)
    n_pos = len(positive_records)
    n_neg = len(negative_records)
    bsz = min(config.batch_size, n_pos, n_neg)

    cfg_grpo = config.model_copy(update={"active_subnodes": ()})
    ref_params = {k: jax.lax.stop_gradient(v) for k, v in params.items()}

    active_set = {idx for idx in active_think_subnodes if 0 <= int(idx) < config.n_experts}
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=learning_rate, weight_decay=0.01),
    )
    opt_state = tx.init(params)

    @jax.jit
    def grpo_step(p, o_st, p_inp, p_tgt, p_msk, n_inp, n_tgt, n_msk):
        logits_ref_p, _, _ = _forward_core_v2(ref_params, p_inp, cfg_grpo)
        logits_ref_n, _, _ = _forward_core_v2(ref_params, n_inp, cfg_grpo)
        lp_ref_p = jax.nn.log_softmax(logits_ref_p, axis=-1)
        lp_ref_n = jax.nn.log_softmax(logits_ref_n, axis=-1)
        tok_ref_p = jnp.take_along_axis(lp_ref_p, p_tgt[..., None], axis=-1)[..., 0]
        tok_ref_n = jnp.take_along_axis(lp_ref_n, n_tgt[..., None], axis=-1)[..., 0]

        def loss_fn(cur_p):
            logits_p, _, _ = _forward_core_v2(cur_p, p_inp, cfg_grpo)
            logits_n, _, _ = _forward_core_v2(cur_p, n_inp, cfg_grpo)
            lp_p = jax.nn.log_softmax(logits_p, axis=-1)
            lp_n = jax.nn.log_softmax(logits_n, axis=-1)

            tok_lp_p = jnp.take_along_axis(lp_p, p_tgt[..., None], axis=-1)[..., 0]
            tok_lp_n = jnp.take_along_axis(lp_n, n_tgt[..., None], axis=-1)[..., 0]

            # Verified process SFT anchor on positive trajectories (A^+ = +1.0)
            sft_pos = -jnp.sum(tok_lp_p * p_msk) / jnp.maximum(jnp.sum(p_msk), 1.0)

            # Group-relative log-ratio advantage between verified (A=+1) and corrupted (A=-1) steps
            seq_logr_p = jnp.sum((tok_lp_p - tok_ref_p) * p_msk, axis=-1) / jnp.maximum(
                jnp.sum(p_msk, axis=-1), 1.0
            )
            seq_logr_n = jnp.sum((tok_lp_n - tok_ref_n) * n_msk, axis=-1) / jnp.maximum(
                jnp.sum(n_msk, axis=-1), 1.0
            )
            grpo_margin = -jax.nn.log_sigmoid(2.0 * (seq_logr_p - seq_logr_n))
            grpo_obj = jnp.mean(grpo_margin)

            # DeepSeek-R1 unbiased per-token KL penalty to reference policy
            ratio_p = jnp.exp(jnp.clip(tok_ref_p - tok_lp_p, -5.0, 5.0))
            kl_tok = ratio_p - (tok_ref_p - tok_lp_p) - 1.0
            kl_loss = jnp.sum(kl_tok * p_msk) / jnp.maximum(jnp.sum(p_msk), 1.0)

            return sft_pos + 0.35 * grpo_obj + float(grpo_beta_kl) * kl_loss

        loss_val, grads = jax.value_and_grad(loss_fn)(p)
        updates, next_opt = tx.update(grads, o_st, p)
        masked_updates = {}
        for k, u in updates.items():
            if k.endswith(".nodes.edge_scale") or k.endswith(".nodes.collab_gate"):
                m = np.zeros((config.n_experts,), dtype=np.float32)
                for a_i in active_set:
                    m[int(a_i)] = 1.0
                masked_updates[k] = u * jnp.asarray(m)
            elif (
                ".nodes." in k
                and not k.endswith(".router.weight")
                and not k.endswith(".router_bias")
            ):
                m_1d = np.zeros((config.n_experts,), dtype=np.float32)
                for a_i in active_set:
                    m_1d[int(a_i)] = 1.0
                m_jax = jnp.asarray(m_1d)
                if u.ndim == 3 and u.shape[0] == config.n_experts:
                    masked_updates[k] = u * m_jax[:, None, None]
                elif u.ndim == 2 and u.shape == (config.n_experts, config.n_experts):
                    e_mat = jnp.asarray(np.maximum(m_1d[:, None], m_1d[None, :]))
                    masked_updates[k] = u * e_mat
                elif u.ndim == 2 and u.shape[0] == config.n_experts:
                    masked_updates[k] = u * m_jax[:, None]
                else:
                    masked_updates[k] = u * 0.25
            elif k.endswith(".nodes.router.weight"):
                m = np.zeros((1, config.n_experts), dtype=np.float32)
                for a_i in active_set:
                    m[:, int(a_i)] = 1.0
                masked_updates[k] = u * jnp.asarray(m)
            else:
                masked_updates[k] = jnp.zeros_like(u)
        next_p = optax.apply_updates(p, masked_updates)
        return next_p, next_opt, loss_val

    rng = np.random.default_rng(config.seed + 101)
    cur_params = params
    for s in range(1, steps + 1):
        idx_p = rng.choice(n_pos, size=bsz, replace=n_pos < bsz)
        idx_n = rng.choice(n_neg, size=bsz, replace=n_neg < bsz)
        cur_params, opt_state, l_val = grpo_step(
            cur_params,
            opt_state,
            jnp.asarray(pos_ds["inputs"][idx_p]),
            jnp.asarray(pos_ds["targets"][idx_p]),
            jnp.asarray(pos_ds["mask"][idx_p]),
            jnp.asarray(neg_ds["inputs"][idx_n]),
            jnp.asarray(neg_ds["targets"][idx_n]),
            jnp.asarray(neg_ds["mask"][idx_n]),
        )
        if s % max(1, steps // 2) == 0 or s == steps:
            print(
                json.dumps(
                    {
                        "grpo_step": s,
                        "subnodes": list(active_think_subnodes),
                        "loss": float(l_val),
                    }
                ),
                flush=True,
            )
    return cur_params
