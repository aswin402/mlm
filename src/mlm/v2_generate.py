"""3-Mode Cognitive Router (`Direct`, `Deliberative`, `Verified`) & Generation Loop for MLM v2."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from mlm.tokenizer import BPETokenizer
from mlm.v2_cache import _DEFAULT_LMCACHE_V2, init_decode_cache_v2
from mlm.v2_decode import _make_cached_recurrent_decode_v2
from mlm.v2_kernels import compute_von_neumann_effective_rank_v2
from mlm.v2_verifier import (
    _EQ_STEP_RE,
    _mask_and_penalize_logits,
    _rlm_try_recursive_decomposition_v2,
    _verify_answer_against_thought_v2,
    _verify_code_and_logic_v2,
    _verify_reasoning_steps_v2,
)

if TYPE_CHECKING:
    from mlm.model_v2 import ConfigV2
    from mlm.v2_memory import MemoryHierarchyV2


def classify_cognitive_mode_v2(
    params: dict[str, jnp.ndarray],
    config: ConfigV2,
    tokenizer: BPETokenizer,
    prompt: str,
) -> dict[str, object]:
    """ACHN / DeepSeek-R1 3-Mode Cognitive Router (`Direct`, `Deliberative`, `Verified`).

    Uses the neural entrance router (`blk.0.nodes.router.weight`) on the user prompt to select:
      - `"direct"`: Fast conversational mode (Dot 0 Chat Sub-Nodes 6..7, skips redundant 2nd pass)
      - `"deliberative"`: Multi-step analytical/scientific reasoning (Dot 0 Think Sub-Nodes 8..9)
      - `"verified"`: Mathematical & Code/Symbolic step-verified reasoning
        (Dot 1 Math Sub-Nodes 10..19 + PRM or Dot 2 Code/Logic Sub-Nodes 20..29 + Symbolic Verifier)

    For `n_experts >= 30`, projects the layer-0 cosine router logits `rl0` onto the KKT-calibrated
    antisymmetric Dot-2 specialist dyads `Delta_d(x_t) ~ rl0[t, 20+2d] - rl0[t, 20+2d+1]`
    with zero string/keyword heuristics.
    """
    p_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    seq_dl_val = -1.0
    active_pair = (6, 7)
    dot2_comb_score = -1.0
    dot2_sub_pair = (20, 21)
    if "blk.0.nodes.router.weight" in params and "tok_embeddings.weight" in params:
        emb = np.asarray(params["tok_embeddings.weight"], dtype=np.float32)[p_ids]
        norm_w = np.asarray(params["blk.0.ffn_norm.weight"], dtype=np.float32)
        rms = np.sqrt(np.mean(np.square(emb), axis=-1, keepdims=True) + config.rms_eps)
        h0 = (emb / rms) * norm_w
        h0_u = h0 / np.maximum(np.linalg.norm(h0, axis=-1, keepdims=True), 1e-6)
        rw0 = np.asarray(params["blk.0.nodes.router.weight"], dtype=np.float32)
        w0_u = rw0 / np.maximum(np.linalg.norm(rw0, axis=0, keepdims=True), 1e-6)
        rl0 = 8.0 * (h0_u @ w0_u)
        mean_rl = np.mean(rl0, axis=0)
        if config.n_experts > 10:
            m_end = min(config.n_experts, 18)
            n_m = max(1, m_end - 10)
            e_eng = np.log(np.sum(np.exp(rl0[:, :8]), axis=-1) + 1e-9) - np.log(8.0)
            e_math = np.log(np.sum(np.exp(rl0[:, 10:m_end]), axis=-1) + 1e-9) - np.log(float(n_m))
            seq_dl_val = float(np.mean(2.0 * (e_math - e_eng)))
            if seq_dl_val >= -0.95:
                top2 = np.argsort(-mean_rl[10:m_end])[:2] + 10
                active_pair = (int(top2[0]), int(top2[1]))
            else:
                top2 = np.argsort(-mean_rl[:8])[:2]
                active_pair = (int(top2[0]), int(top2[1]))
        else:
            base_n = min(config.n_experts, 8)
            top2 = np.argsort(-mean_rl[:base_n])[:2]
            active_pair = (int(top2[0]), int(top2[1]))

        if config.n_experts >= 30 and rl0.shape[-1] >= 28:
            eps_r = 0.02
            dyad_scale = float(np.sqrt(1.0 + eps_r * eps_r) / (16.0 * eps_r))
            s0 = dyad_scale * (rl0[:, 20] - rl0[:, 21])
            dot2_comb_score = float(np.max(s0)) + 8.0 * float(np.mean(s0))
            sub_means = [
                float(np.mean(dyad_scale * (rl0[:, 20 + 2 * d] - rl0[:, 20 + 2 * d + 1])))
                for d in (1, 2)
            ]
            best_d = 1 + int(np.argmax(sub_means))
            dot2_sub_pair = (20 + 2 * best_d, 20 + 2 * best_d + 1)

    if config.n_experts >= 30 and dot2_comb_score >= 0.41:
        return {
            "mode": "verified",
            "domain": "code",
            "seq_dl": round(seq_dl_val, 3),
            "active_subnodes": dot2_sub_pair,
        }

    eff_rank = 8.0
    if "tok_embeddings.weight" in params:
        emb_s = np.asarray(params["tok_embeddings.weight"], dtype=np.float32)[p_ids]
        eff_rank = compute_von_neumann_effective_rank_v2(emb_s)
    has_q_token = bool(np.any(p_ids == 38))
    if seq_dl_val >= -0.95:
        mode = "verified"
        domain = "math"
    elif eff_rank < 6.0 and not has_q_token:
        mode = "direct"
        domain = "english"
    else:
        mode = "deliberative"
        domain = "english"

    return {
        "mode": mode,
        "domain": domain,
        "seq_dl": round(seq_dl_val, 3),
        "active_subnodes": active_pair,
    }


def generate_v2(
    params: dict[str, jnp.ndarray],
    config: ConfigV2,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    repetition_penalty: float = 1.15,
    include_think: bool = False,
    _rlm_depth: int = 0,
    _prefill_state: tuple[dict[str, jnp.ndarray], jnp.ndarray, jnp.ndarray, int] | None = None,
    memory_hierarchy: MemoryHierarchyV2 | None = None,
    _skip_mem_pre: bool = False,
) -> str:
    """Generate tokens with 3-Mode Cognitive Routing, Speculative Decode, PRM & 3-Tier Memory."""
    mem_think_prefix = ""
    if (
        memory_hierarchy is not None
        and not _skip_mem_pre
        and _prefill_state is None
        and _rlm_depth == 0
    ):
        mem_action = memory_hierarchy.process_turn_before_generate(
            prompt, params, config, tokenizer
        )
        action_type = mem_action.get("action", "passthrough")
        if action_type in ("store_only", "episodic_recall"):
            reply_txt = str(mem_action.get("reply", ""))
            think_txt = str(mem_action.get("think", reply_txt))
            if include_think:
                return f"<|think|>{think_txt}<|/think|>\n{reply_txt}"
            return reply_txt
        if action_type == "variable_substituted":
            prompt = str(mem_action.get("resolved_prompt", prompt))
            mem_think_prefix = str(mem_action.get("think_prefix", ""))

    if _prefill_state is None and _rlm_depth < 3 and temperature <= 1e-5:
        rlm_out = _rlm_try_recursive_decomposition_v2(
            params,
            config,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            include_think=include_think,
            _rlm_depth=_rlm_depth,
            memory_hierarchy=memory_hierarchy,
        )
        if rlm_out is not None:
            if mem_think_prefix and rlm_out.startswith("<|think|>"):
                return "<|think|>" + mem_think_prefix + rlm_out[len("<|think|>") :]
            return rlm_out

    cog_info = (
        classify_cognitive_mode_v2(params, config, tokenizer, prompt)
        if _prefill_state is None
        else {"mode": "direct", "domain": "english", "seq_dl": -1.0, "active_subnodes": (6, 7)}
    )
    if cog_info.get("domain") == "code" and len(config.active_subnodes) == 0:
        config = config.model_copy(
            update={"active_subnodes": tuple(cog_info["active_subnodes"])}  # type: ignore[arg-type]
        )
    if include_think and cog_info["mode"] == "direct" and _rlm_depth == 0:
        direct_ans = generate_v2(
            params,
            config,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            include_think=False,
            _rlm_depth=_rlm_depth,
            memory_hierarchy=memory_hierarchy,
            _skip_mem_pre=True,
        )
        return f"<|think|>{mem_think_prefix}{direct_ans}<|/think|>\n{direct_ans}"

    prefilled_think = False
    effective_prompt = prompt
    if include_think and prompt.endswith("<|assistant|>\n"):
        effective_prompt = prompt + "<|think|>"
        prefilled_think = True
    seq_len = config.max_seq_len
    generated: list[int] = []
    step1, spec2 = _make_cached_recurrent_decode_v2(config)

    is_tiered_think = (
        prefilled_think
        and config.mature_experts >= 8
        and config.n_experts > config.mature_experts
    )
    open_id = (
        config.think_open_id if config.think_open_id >= 0 else tokenizer.think_open_id
    )
    close_id = (
        config.think_close_id
        if config.think_close_id >= 0
        else int(tokenizer.encode("<|/think|>", add_bos=False, add_eos=False)[-1])
    )
    special_ids = (
        tokenizer.pad_id,
        tokenizer.bos_id,
        tokenizer.eos_id,
        open_id,
        close_id,
    )
    step_limit = min(max_new_tokens, 48) if is_tiered_think else max_new_tokens

    fork_prefill_state: tuple[dict[str, jnp.ndarray], jnp.ndarray, jnp.ndarray, int] | None = None
    cached_first_step: tuple[dict[str, jnp.ndarray], jnp.ndarray, jnp.ndarray] | None = None
    if _prefill_state is not None:
        carry, first_raw, first_draft, cur_pos = _prefill_state
        cached_first_step = (carry, first_raw, first_draft)
        next_to_feed = 0
        pending_draft: int | None = None
    else:
        prompt_ids = tokenizer.encode(effective_prompt, add_bos=True, add_eos=False).tolist()[
            -seq_len:
        ]
        pre_ids = prompt_ids[:-1]
        active_lmcache = (
            memory_hierarchy.lmcache if memory_hierarchy is not None else _DEFAULT_LMCACHE_V2
        )
        m_sig = active_lmcache.model_signature(params, config)
        full_hit = active_lmcache.lookup_full_prefill(m_sig, pre_ids, is_tiered_think)
        if full_hit is not None:
            if is_tiered_think and len(pre_ids) >= 1:
                carry, prompt_raw, prompt_draft, fork_pos = full_hit
                fork_prefill_state = (carry, prompt_raw, prompt_draft, fork_pos)
            else:
                carry, _, _, _ = full_hit
        elif is_tiered_think and len(pre_ids) >= 1:
            bulk_ids = pre_ids[:-1]
            cached_carry, p_idx = active_lmcache.lookup_longest_prefix(m_sig, bulk_ids)
            carry = cached_carry if cached_carry is not None else init_decode_cache_v2(config)
            while p_idx + 2 <= len(bulk_ids):
                _, (carry, _, _) = spec2(
                    params,
                    carry,
                    jnp.int32(bulk_ids[p_idx]),
                    jnp.int32(bulk_ids[p_idx + 1]),
                )
                p_idx += 2
                if p_idx % active_lmcache.chunk_size == 0:
                    active_lmcache.store_prefix_chunk(m_sig, bulk_ids[:p_idx], carry)
            while p_idx < len(bulk_ids):
                carry, _, _ = step1(params, carry, jnp.int32(bulk_ids[p_idx]))
                p_idx += 1
                if p_idx % active_lmcache.chunk_size == 0:
                    active_lmcache.store_prefix_chunk(m_sig, bulk_ids[:p_idx], carry)
            carry, prompt_raw, prompt_draft = step1(params, carry, jnp.int32(pre_ids[-1]))
            fork_prefill_state = (carry, prompt_raw, prompt_draft, len(pre_ids) - 1)
            active_lmcache.store_full_prefill(
                m_sig, pre_ids, is_tiered_think, fork_prefill_state
            )
        else:
            cached_carry, p_idx = active_lmcache.lookup_longest_prefix(m_sig, pre_ids)
            carry = cached_carry if cached_carry is not None else init_decode_cache_v2(config)
            while p_idx + 2 <= len(pre_ids):
                _, (carry, _, _) = spec2(
                    params,
                    carry,
                    jnp.int32(pre_ids[p_idx]),
                    jnp.int32(pre_ids[p_idx + 1]),
                )
                p_idx += 2
                if p_idx % active_lmcache.chunk_size == 0:
                    active_lmcache.store_prefix_chunk(m_sig, pre_ids[:p_idx], carry)
            while p_idx < len(pre_ids):
                carry, _, _ = step1(params, carry, jnp.int32(pre_ids[p_idx]))
                p_idx += 1
                if p_idx % active_lmcache.chunk_size == 0:
                    active_lmcache.store_prefix_chunk(m_sig, pre_ids[:p_idx], carry)
            dummy_z = jnp.zeros((config.vocab_size,), dtype=jnp.float32)
            active_lmcache.store_full_prefill(
                m_sig, pre_ids, is_tiered_think, (carry, dummy_z, jnp.int32(-1), len(pre_ids))
            )
        next_to_feed = int(prompt_ids[-1])
        pending_draft = None
        cur_pos = len(pre_ids)

    while len(generated) < step_limit and cur_pos < seq_len:
        use_spec = (
            pending_draft is not None
            and temperature <= 1e-5
            and cur_pos + 1 < seq_len
            and len(generated) < step_limit
        )
        if use_spec and pending_draft is not None:
            (carry_a, raw_a, draft_a), (carry_b, raw_b, draft_b) = spec2(
                params,
                carry,
                jnp.int32(next_to_feed),
                jnp.int32(pending_draft),
            )
            masked_a = _mask_and_penalize_logits(
                np.asarray(raw_a),
                generated,
                config,
                tokenizer,
                prefilled_think,
                open_id,
                close_id,
                repetition_penalty,
            )
            verified_id = int(np.argmax(masked_a))
            if verified_id == tokenizer.eos_id:
                break
            generated.append(verified_id)
            if prefilled_think and len(generated) >= 4 and verified_id == close_id:
                break
            if verified_id == pending_draft and len(generated) < step_limit:
                masked_b = _mask_and_penalize_logits(
                    np.asarray(raw_b),
                    generated,
                    config,
                    tokenizer,
                    prefilled_think,
                    open_id,
                    close_id,
                    repetition_penalty,
                )
                second_id = int(np.argmax(masked_b))
                if second_id == tokenizer.eos_id:
                    break
                generated.append(second_id)
                if prefilled_think and len(generated) >= 4 and second_id == close_id:
                    break
                carry = carry_b
                cur_pos += 2
                next_to_feed = second_id
                cand_d = int(draft_b)
                pending_draft = (
                    cand_d if (cand_d >= 0 and cand_d not in special_ids) else None
                )
            else:
                carry = carry_a
                cur_pos += 1
                next_to_feed = verified_id
                cand_d = int(draft_a)
                pending_draft = (
                    cand_d if (cand_d >= 0 and cand_d not in special_ids) else None
                )
            continue

        if cached_first_step is not None:
            carry, raw_b, draft_b = cached_first_step
            cached_first_step = None
        else:
            carry, raw_b, draft_b = step1(params, carry, jnp.int32(next_to_feed))
        cur_pos += 1
        next_logits = _mask_and_penalize_logits(
            np.asarray(raw_b),
            generated,
            config,
            tokenizer,
            prefilled_think,
            open_id,
            close_id,
            repetition_penalty,
        )

        if temperature <= 1e-5:
            next_id = int(np.argmax(next_logits))
        else:
            scaled = next_logits / float(temperature)
            scaled -= np.max(scaled)
            probs = np.exp(scaled)
            probs /= np.sum(probs)
            next_id = int(np.random.choice(len(probs), p=probs))

        if next_id == tokenizer.eos_id:
            break
        generated.append(next_id)
        if prefilled_think and len(generated) >= 4 and next_id == close_id:
            break
        next_to_feed = next_id
        cand_d = int(draft_b)
        if temperature <= 1e-5 and cand_d >= 0 and cand_d not in special_ids:
            pending_draft = cand_d
        else:
            pending_draft = None

    raw_decoded = tokenizer.decode(generated, skip_special_tokens=False)
    for tag in ("<|pad|>", "<|bos|>", "<|eos|>"):
        raw_decoded = raw_decoded.replace(tag, "")
    if include_think:
        out = raw_decoded.strip()
        if prefilled_think and not out.startswith("<|think|>"):
            out = f"<|think|>{out}"
        if (
            prefilled_think
            and config.mature_experts >= 8
            and config.n_experts > config.mature_experts
        ):
            thought_body = (
                out.replace("<|think|>", "")
                .replace("<|/think|>", "")
                .replace("<|/think>", "")
                .replace("nominthink", "")
                .replace("think|>", "")
                .replace("\ufffd", "")
                .strip(" |>")
                .strip()
            )
            if thought_body.endswith(" think"):
                thought_body = thought_body[:-6].strip()
            clean_ans = generate_v2(
                params,
                config,
                tokenizer,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                include_think=False,
                _rlm_depth=_rlm_depth,
                _prefill_state=fork_prefill_state,
                memory_hierarchy=memory_hierarchy,
                _skip_mem_pre=True,
            )
            if _rlm_depth == 0 and temperature <= 1e-5:
                if cog_info.get("domain") == "code":
                    thought_body, clean_ans = _verify_code_and_logic_v2(
                        prompt, thought_body, clean_ans
                    )
                else:
                    thought_body = _verify_reasoning_steps_v2(thought_body, prompt=prompt)
                    th_eqs = list(_EQ_STEP_RE.finditer(thought_body))
                    ans_eqs = list(_EQ_STEP_RE.finditer(clean_ans))
                    if th_eqs:
                        thought_body = " ".join(
                            f"{m.group(1)} {m.group(2)} {m.group(3)} = {m.group(4)}."
                            for m in th_eqs
                        )
                        clean_ans = _verify_answer_against_thought_v2(clean_ans, th_eqs)
                    elif ans_eqs:
                        verified_steps = [
                            f"{m.group(1)} {m.group(2)} {m.group(3)} = {m.group(4)}."
                            for m in ans_eqs
                        ]
                        thought_body = " ".join(verified_steps)
                    elif cog_info["mode"] == "deliberative" and clean_ans:
                        ans_first = clean_ans.split(". ", 1)[0].strip()
                        if not ans_first.endswith("."):
                            ans_first = ans_first + "."
                        th_words = {
                            w.lower().strip(".,!?")
                            for w in thought_body.split()
                            if len(w) >= 4
                        }
                        ans_words = {
                            w.lower().strip(".,!?")
                            for w in ans_first.split()
                            if len(w) >= 4
                        }
                        if not (th_words & ans_words) or thought_body.startswith("Problem"):
                            thought_body = ans_first
            return f"<|think|>{mem_think_prefix}{thought_body}<|/think|>\n{clean_ans}"
        if mem_think_prefix and out.startswith("<|think|>"):
            out = f"<|think|>{mem_think_prefix}{out[len('<|think|>'):]}"
        if memory_hierarchy is not None:
            memory_hierarchy.record_decode_carry(carry, config)
        return out
    if memory_hierarchy is not None:
        memory_hierarchy.record_decode_carry(carry, config)
    if "<|/think|>" in raw_decoded:
        raw_decoded = raw_decoded.split("<|/think|>", 1)[1]
    elif "<|/think>" in raw_decoded:
        raw_decoded = raw_decoded.split("<|/think>", 1)[1]
    raw_decoded = (
        raw_decoded.replace("<|think|>", "")
        .replace("<|/think|>", "")
        .replace("<|/think>", "")
    )
    ans_clean = raw_decoded.strip()
    if _rlm_depth == 0 and temperature <= 1e-5:
        if cog_info.get("domain") == "code":
            _, ans_clean = _verify_code_and_logic_v2(prompt, "", ans_clean)
        else:
            ans_clean = _verify_reasoning_steps_v2(ans_clean, prompt=prompt)
    return ans_clean
