"""Reinforcement Learning (Dr. GRPO / Ornith-1.5) & Specialist Continual Calibration for MLM v2."""

from __future__ import annotations

import json
import math
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
    create_state_v2,
    forward_v2,
    generate_v2,
    load_checkpoint_v2,
    mask_specialist_updates_v2,
    merge_checkpoints_v2,
    save_checkpoint_v2,
)
from mlm.tokenizer import BPETokenizer


def _shannon_token_info(word: str) -> float:
    """Compute intrinsic Shannon character entropy and subword information weight of `word`."""
    n = len(word)
    if n == 0:
        return 0.0
    if any(ch.isdigit() for ch in word):
        return float(max(2.0, n * 1.25))
    if n <= 3:
        return 0.0
    counts: dict[str, int] = {}
    for ch in word:
        counts[ch] = counts.get(ch, 0) + 1
    inv_n = 1.0 / float(n)
    entropy = -sum((c * inv_n) * math.log2(c * inv_n) for c in counts.values())
    return float((n - 3.0) * max(entropy, 0.5))


def _content_words(text: str) -> set[str]:
    """Extract high-information tokens via Shannon character complexity (`_shannon_token_info`)."""
    clean = text.replace("<|think|>", " ").replace("<|/think|>", " ")
    words = re.findall(r"[a-z0-9_]+", clean.lower())
    return {w for w in words if _shannon_token_info(w) >= 1.5}


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
        gen_weights = {w: _shannon_token_info(w) for w in gen_words}
        ref_weights = {w: _shannon_token_info(w) for w in ref_words}
        overlap_words = gen_words & ref_words
        overlap_mass = sum(min(gen_weights[w], ref_weights[w]) for w in overlap_words)
        gen_mass = max(sum(gen_weights.values()), 1e-6)
        ref_mass = max(sum(ref_weights.values()), 1e-6)
        prec = overlap_mass / gen_mass
        rec = overlap_mass / ref_mass
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        reward += 1.2 * f1
        off_topic_ratio = len(gen_words - allowed_words) / max(len(gen_words), 1)
        if off_topic_ratio > 0.70 and len(overlap_words) == 0:
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
        scaled_updates = mask_specialist_updates_v2(updates, config, always_train_mtp=True)
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
    from mlm.train_v2 import (
        build_tokenized_dataset,
        format_chat_prompt_v2,
        load_jsonl_records,
    )

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
    """Calibrate `blk.*.nodes.router.weight` jointly across all SpiderNet skill sub-nodes."""
    from mlm.model_v2 import _forward_core_v2, rms_norm
    from mlm.train_v2 import build_tokenized_dataset

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

            eng_hinge = jax.nn.relu(seq_dl - (-1.40)) * (1.0 - dom_lbl)
            math_hinge = jax.nn.relu(0.40 - seq_dl) * dom_lbl
            dom_bce = optax.sigmoid_binary_cross_entropy(
                mean_dl, jnp.broadcast_to(dom_lbl[:, None], mean_dl.shape)
            )
            dom_loss = jnp.mean(eng_hinge + math_hinge) + (
                jnp.sum(dom_bce * valid_tok) / jnp.maximum(jnp.sum(valid_tok), 1.0)
            )

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
    """DeepSeek-R1 Group Relative Policy Optimization (`GRPO`) + Process Verification (`PRM`)."""
    from mlm.model_v2 import _forward_core_v2
    from mlm.train_v2 import build_tokenized_dataset

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

            sft_pos = -jnp.sum(tok_lp_p * p_msk) / jnp.maximum(jnp.sum(p_msk), 1.0)

            seq_logr_p = jnp.sum((tok_lp_p - tok_ref_p) * p_msk, axis=-1) / jnp.maximum(
                jnp.sum(p_msk, axis=-1), 1.0
            )
            seq_logr_n = jnp.sum((tok_lp_n - tok_ref_n) * n_msk, axis=-1) / jnp.maximum(
                jnp.sum(n_msk, axis=-1), 1.0
            )
            grpo_margin = -jax.nn.log_sigmoid(2.0 * (seq_logr_p - seq_logr_n))
            grpo_obj = jnp.mean(grpo_margin)

            ratio_p = jnp.exp(jnp.clip(tok_ref_p - tok_lp_p, -5.0, 5.0))
            kl_tok = ratio_p - (tok_ref_p - tok_lp_p) - 1.0
            kl_loss = jnp.sum(kl_tok * p_msk) / jnp.maximum(jnp.sum(p_msk), 1.0)

            return sft_pos + 0.35 * grpo_obj + float(grpo_beta_kl) * kl_loss

        loss_val, grads = jax.value_and_grad(loss_fn)(p)
        updates, next_opt = tx.update(grads, o_st, p)
        masked_updates = mask_specialist_updates_v2(
            updates,
            config,
            active_subnodes_override=active_set,
            strict_active_only=True,
        )
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


def build_dot2_code_logic_curriculum_v2() -> list[tuple[tuple[int, int], list[dict[str, object]]]]:
    """Build 5-skill Python AST & Symbolic Logic curriculum for Dot-2 (`Nodes 20..29`)."""
    from mlm.curated_data import _make_supervised_chat_record

    return [
        (
            (20, 21),
            [
                _make_supervised_chat_record(
                    "Write a Python function `def square(n):` that returns n squared.",
                    "def square(n):\n    return n * n",
                ),
                _make_supervised_chat_record(
                    "How do you define a function `add(a, b)` in Python?",
                    "def add(a, b):\n    return a + b",
                ),
            ],
        ),
        (
            (22, 23),
            [
                _make_supervised_chat_record(
                    "What does `[x * x for x in range(4)]` evaluate to in Python?",
                    "`[x * x for x in range(4)]` evaluates to `[0, 1, 4, 9]`.",
                ),
                _make_supervised_chat_record(
                    "What is the time complexity of merge sort in Big-O notation?",
                    "Merge sort runs in O(n log n) time complexity.",
                ),
            ],
        ),
        (
            (24, 25),
            [
                _make_supervised_chat_record(
                    "Evaluate the boolean expression: (True and not False) xor False",
                    "`(True and not False) xor False` evaluates to `True`.",
                ),
                _make_supervised_chat_record(
                    "State De Morgan's law for `not (A and B)`.",
                    "`not (A and B)` is logically equivalent to `(not A) or (not B)`.",
                ),
            ],
        ),
        (
            (26, 27),
            [
                _make_supervised_chat_record(
                    "What exception does `1 / 0` raise in Python?",
                    "Dividing by zero in Python raises a `ZeroDivisionError` exception.",
                ),
                _make_supervised_chat_record(
                    "How do you check if `x` is an integer using `isinstance`?",
                    "Use `isinstance(x, int)` to check if `x` is an integer.",
                ),
            ],
        ),
        (
            (28, 29),
            [
                _make_supervised_chat_record(
                    "Trace `sum([2 * i for i in range(3)])` step by step.",
                    "<|think|>For i=0,1,2, 2*i gives [0, 2, 4]. Their sum is 6.<|/think|>\n"
                    "The expression evaluates to `6`.",
                ),
                _make_supervised_chat_record(
                    "Prove whether `(True or False) and (not False)` is True.",
                    "<|think|>True or False is True; not False is True; "
                    "True and True is True.<|/think|>\n"
                    "The boolean expression evaluates to `True`.",
                ),
            ],
        ),
    ]


def run_dot2_continual_stage_v2(
    params: dict[str, jnp.ndarray],
    config: ConfigV2,
    tokenizer: BPETokenizer,
    *,
    steps_per_bucket: int = 6,
    learning_rate: float = 2.0e-3,
    freeze_router: bool = True,
) -> tuple[dict[str, jnp.ndarray], ConfigV2, dict[str, object]]:
    """Run Stage-3 continual fine-tuning on Dot-2 (`Nodes 20..29`) with zero forgetting."""
    from mlm.model_v2 import _forward_core_v2, grow_dot2_code_logic_v2
    from mlm.train_v2 import build_tokenized_dataset

    if config.n_experts < 30:
        params, config, _ = grow_dot2_code_logic_v2(params, config)

    ref_params = {k: np.asarray(v, dtype=np.float32).copy() for k, v in params.items()}
    cfg_stage3 = config.model_copy(
        update={
            "mature_experts": 20,
            "mature_lr_scale": 0.0,
            "shared_lr_scale": 0.0,
            "learning_rate": learning_rate,
        }
    )
    buckets = build_dot2_code_logic_curriculum_v2()
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=learning_rate, weight_decay=0.0),
    )

    cur_params = dict(params)
    initial_losses: list[float] = []
    final_losses: list[float] = []

    for subnode_pair, records in buckets:
        ds = build_tokenized_dataset(records, tokenizer, cfg_stage3.max_seq_len)
        inp = jnp.asarray(ds["inputs"])
        tgt = jnp.asarray(ds["targets"])
        msk = jnp.asarray(ds["mask"])
        cfg_b = cfg_stage3.model_copy(update={"active_subnodes": subnode_pair})
        active_set = {int(idx) for idx in subnode_pair}
        opt_state = tx.init(cur_params)

        def _bucket_loss(
            p: dict[str, jnp.ndarray],
            b_inp: jnp.ndarray = inp,
            b_tgt: jnp.ndarray = tgt,
            b_msk: jnp.ndarray = msk,
            b_cfg: ConfigV2 = cfg_b,
        ) -> jnp.ndarray:
            logits, _, _ = _forward_core_v2(p, b_inp, b_cfg)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            tok_loss = -jnp.take_along_axis(log_probs, b_tgt[..., None], axis=-1)[..., 0]
            return jnp.sum(tok_loss * b_msk) / jnp.maximum(jnp.sum(b_msk), 1.0)

        init_l = float(_bucket_loss(cur_params))
        initial_losses.append(init_l)

        val_and_grad = jax.value_and_grad(_bucket_loss)
        for _ in range(max(1, int(steps_per_bucket))):
            _, grads = val_and_grad(cur_params)
            updates, opt_state = tx.update(grads, opt_state, cur_params)
            masked_updates = mask_specialist_updates_v2(
                updates,
                cfg_b,
                active_subnodes_override=active_set,
                strict_active_only=True,
            )
            if freeze_router:
                masked_updates = {
                    k: (jnp.zeros_like(u) if k.endswith(".nodes.router.weight") else u)
                    for k, u in masked_updates.items()
                }
            cur_params = optax.apply_updates(cur_params, masked_updates)

        final_l = float(_bucket_loss(cur_params))
        final_losses.append(final_l)

    max_drift_0_19 = 0.0
    max_drift_trunk = 0.0
    dot2_sq_sum = 0.0
    for k, v_new in cur_params.items():
        arr_new = np.asarray(v_new, dtype=np.float32)
        arr_old = ref_params[k]
        if ".nodes." in k:
            if k.endswith(".router.weight"):
                drift_0_19 = float(np.max(np.abs(arr_new[:, :20] - arr_old[:, :20])))
                max_drift_0_19 = max(max_drift_0_19, drift_0_19)
            elif arr_new.ndim >= 1 and arr_new.shape[0] == 30 and arr_new.shape != (30, 30):
                drift_0_19 = float(np.max(np.abs(arr_new[:20] - arr_old[:20])))
                max_drift_0_19 = max(max_drift_0_19, drift_0_19)
                dot2_sq_sum += float(np.sum(np.square(arr_new[20:30] - arr_old[20:30])))
            elif arr_new.shape == (30, 30):
                drift_0_19 = float(np.max(np.abs(arr_new[:20, :20] - arr_old[:20, :20])))
                max_drift_0_19 = max(max_drift_0_19, drift_0_19)
                dot2_sq_sum += float(np.sum(np.square(arr_new[20:30, :] - arr_old[20:30, :])))
            else:
                max_drift_trunk = max(
                    max_drift_trunk, float(np.max(np.abs(arr_new - arr_old)))
                )
        else:
            max_drift_trunk = max(max_drift_trunk, float(np.max(np.abs(arr_new - arr_old))))

    mean_init = float(np.mean(initial_losses))
    mean_final = float(np.mean(final_losses))
    report = {
        "stage": "dot2_code_logic_continual_stage3",
        "n_experts": int(cfg_stage3.n_experts),
        "mature_experts_frozen": 20,
        "steps_per_bucket": int(steps_per_bucket),
        "mean_initial_loss": round(mean_init, 6),
        "mean_final_loss": round(mean_final, 6),
        "loss_reduction": round(mean_init - mean_final, 6),
        "max_drift_nodes_0_19": float(max_drift_0_19),
        "max_drift_trunk": float(max_drift_trunk),
        "dot2_update_l2_norm": round(float(np.sqrt(dot2_sq_sum)), 6),
    }
    return cur_params, cfg_stage3, report
