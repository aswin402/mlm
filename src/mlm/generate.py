"""Greedy byte generation with explicit recurrent state and neural telemetry."""

import time
from collections.abc import Callable
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from mlm.data import BOS, EOS
from mlm.model import initial_state
from mlm.stream import make_stream_step


@lru_cache(maxsize=16)
def _make_seq_step(apply_fn, messages: bool):
    @jax.jit
    def step_seq(params, tokens_buf, init_carry, resets, valid_buf):
        logits_seq, _ = apply_fn(
            {"params": params},
            tokens_buf,
            init_carry,
            resets,
            messages=messages,
            valid=valid_buf,
        )
        return logits_seq

    return step_seq


@lru_cache(maxsize=16)
def _make_seq_step_telemetry(apply_fn, messages: bool):
    @jax.jit
    def step_seq_tel(params, tokens_buf, init_carry, resets, valid_buf):
        logits_seq, final_carry, telemetry = apply_fn(
            {"params": params},
            tokens_buf,
            init_carry,
            resets,
            messages=messages,
            valid=valid_buf,
            return_telemetry=True,
        )
        return logits_seq, final_carry, telemetry

    return step_seq_tel


def format_chat_prompt(user_message: str, context: str = "") -> str:
    """Format a user message (and optional persistent context) into `<|user|>...<|assistant|>`."""
    msg = user_message.strip()
    ctx = " ".join(context.strip().splitlines()).strip()
    if "\n" not in msg and msg.endswith("?") and ". " in msg:
        split_idx = msg.rfind(". ")
        inline_ctx = msg[: split_idx + 1].strip()
        msg = msg[split_idx + 2 :].strip()
        if inline_ctx and inline_ctx not in ctx:
            ctx = f"{ctx} {inline_ctx}".strip() if ctx else inline_ctx
    if ctx and ctx not in msg:
        body = f"{ctx}\n{msg}"
    else:
        body = msg
    if "\n" not in body and body.endswith("?") and ". " in body:
        split_idx = body.rfind(". ")
        body = f"{body[: split_idx + 1]}\n{body[split_idx + 2 :]}"
    return f"<|user|>\n{body}\n<|assistant|>\n"


def _is_word_byte(b: int) -> bool:
    if b < 33 or b > 126:
        return False
    ch = chr(b)
    return ch.isalnum() or ch in ("_", "-")


def _word_span_at(tokens_prefix: list[int], pos: int) -> tuple[str, int, int]:
    """Return `(word, start_idx, end_idx)` around token index `pos` in `tokens_prefix`."""
    n = len(tokens_prefix)
    if n <= 1:
        return "", 0, 0
    idx = int(np.clip(pos, 1, n - 1))
    if not _is_word_byte(tokens_prefix[idx]):
        if idx > 1 and _is_word_byte(tokens_prefix[idx - 1]):
            idx -= 1
        elif idx + 1 < n and _is_word_byte(tokens_prefix[idx + 1]):
            idx += 1
        else:
            b = tokens_prefix[idx]
            ch = chr(b) if 32 <= b <= 126 else f"0x{b:02x}"
            return ch, idx, idx + 1
    start = idx
    while start > 1 and _is_word_byte(tokens_prefix[start - 1]):
        start -= 1
    end = idx + 1
    while end < n and _is_word_byte(tokens_prefix[end]):
        end += 1
    word = bytes(tokens_prefix[start:end]).decode("utf-8", errors="replace")
    return word, start, end


def _all_words_in_span(
    tokens_prefix: list[int], start_idx: int, end_idx: int
) -> list[tuple[str, int, int]]:
    words: list[tuple[str, int, int]] = []
    i = max(1, start_idx)
    limit = min(len(tokens_prefix), end_idx)
    while i < limit:
        if _is_word_byte(tokens_prefix[i]):
            j = i + 1
            while j < limit and _is_word_byte(tokens_prefix[j]):
                j += 1
            w = bytes(tokens_prefix[i:j]).decode("utf-8", errors="replace")
            words.append((w, i, j))
            i = j
        else:
            i += 1
    return words


def _extract_reasoning_trace(
    tokens_prefix: list[int],
    telemetry: dict,
    final_carry,
    config,
) -> dict:
    """Decode raw neural telemetry tensors into a structured reasoning & state trace."""
    prompt_len = len(tokens_prefix)
    t_idx = max(0, prompt_len - 1)
    routes = np.asarray(telemetry["routes"][0, :prompt_len, :], dtype=np.float32)
    ponder = np.asarray(telemetry["ponder_kl"][0, :prompt_len], dtype=np.float32)
    wavelet_norm = np.asarray(telemetry["wavelet_norm"][0, :prompt_len], dtype=np.float32)

    carry_np = np.asarray(final_carry[0], dtype=np.float32)
    n_nodes = 1 if config.architecture == "gru" else config.nodes
    node_states = carry_np[:n_nodes]
    node_norms = np.linalg.norm(node_states, axis=-1).tolist()

    mem_norm = 0.0
    mem_key_norms: list[float] = []
    span_norms: list[float] = []
    ep_norm = 0.0
    ep_slot_norms: list[float] = []

    cursor = n_nodes
    if config.memory != "none":
        mem_slice = carry_np[cursor : cursor + config.memory_keys]
        mem_norm = float(np.linalg.norm(mem_slice))
        mem_key_norms = np.linalg.norm(mem_slice, axis=-1).tolist()
        cursor += config.memory_keys
        if config.symmetric_memory:
            span_slice = carry_np[cursor : cursor + 4]
            span_norms = np.linalg.norm(span_slice, axis=-1).tolist()
            cursor += 4
    if config.episodic_slots > 0:
        ep_slice = carry_np[cursor : cursor + config.episodic_slots]
        ep_norm = float(np.linalg.norm(ep_slice))
        ep_slot_norms = np.linalg.norm(ep_slice, axis=-1).tolist()

    trace: dict = {
        "routes_mean": routes.mean(axis=0).tolist() if routes.size else [],
        "routes_last": routes[t_idx].tolist() if routes.size else [],
        "ponder_mean": float(ponder.mean()) if ponder.size else 0.0,
        "ponder_max": float(ponder.max()) if ponder.size else 0.0,
        "wavelet_norm": float(wavelet_norm[t_idx]) if wavelet_norm.size else 0.0,
        "node_norms": node_norms,
        "memory_norm": mem_norm,
        "memory_key_norms": mem_key_norms,
        "span_norms": span_norms,
        "episodic_norm": ep_norm,
        "episodic_slot_norms": ep_slot_norms,
        "relational_active": False,
    }

    if "g_question" not in telemetry:
        return trace

    g_q = float(telemetry["g_question"][0, t_idx])
    g_chat = float(telemetry["g_chat"][0, t_idx])
    g_bwd = float(telemetry["g_bwd"][0, t_idx])
    ctx_mask = np.asarray(telemetry["ctx_mask"][0, t_idx, :prompt_len], dtype=np.float32)
    clause_id = np.asarray(telemetry["clause_id"][0, t_idx, :prompt_len], dtype=np.float32)
    rec_win = np.asarray(telemetry["rec_win"][0, t_idx, :prompt_len], dtype=np.float32)
    hop_attns = np.asarray(telemetry["hop_attns"][0, t_idx, :, :prompt_len], dtype=np.float32)
    hop_bridges = np.asarray(telemetry["hop_bridges"][0, t_idx, :, :prompt_len], dtype=np.float32)
    hop_gates = np.asarray(telemetry["hop_gates"][0, t_idx, :], dtype=np.float32)
    onset_attn = np.asarray(telemetry["onset_attn"][0, t_idx, :prompt_len], dtype=np.float32)
    cont_gate = float(telemetry["cont_gate"][0, t_idx])
    eos_mass = float(telemetry["eos_mass"][0, t_idx])

    clauses: dict[int, str] = {}
    ctx_indices = np.where(ctx_mask > 0.5)[0]
    if ctx_indices.size > 0:
        for cid in sorted({int(np.round(clause_id[i])) for i in ctx_indices}):
            c_pos = [int(i) for i in ctx_indices if int(np.round(clause_id[i])) == cid]
            if c_pos:
                c_bytes = bytes(tokens_prefix[c_pos[0] : c_pos[-1] + 1]).decode(
                    "utf-8", errors="replace"
                )
                clauses[cid] = c_bytes.strip()

    # Decode Hop 1..H reasoning steps from consolidated word-onset distributions
    hops_info: list[dict] = []
    num_hops = hop_attns.shape[0]
    for h_i in range(num_hops):
        attn_h = hop_attns[h_i]
        best_pos = int(np.argmax(attn_h)) if attn_h.size else 0
        best_word, best_pos, _ = _word_span_at(tokens_prefix, best_pos)
        best_mass = float(attn_h.max()) if attn_h.size else 0.0
        cid_h = int(np.round(clause_id[best_pos])) if best_pos < clause_id.size else 0

        if h_i == 0:
            hops_info.append(
                {
                    "hop": 1,
                    "active": True,
                    "gate": 1.0,
                    "bridge_word": None,
                    "target_word": best_word,
                    "target_mass": best_mass,
                    "clause_id": cid_h,
                    "clause_text": clauses.get(cid_h, ""),
                }
            )
        else:
            gate_val = float(hop_gates[h_i - 1]) if (h_i - 1) < hop_gates.size else 0.0
            br_arr = hop_bridges[h_i - 1]
            br_pos = int(np.argmax(br_arr)) if br_arr.size else 0
            br_word, _, _ = (
                _word_span_at(tokens_prefix, br_pos) if br_arr.max() > 1e-5 else ("", 0, 0)
            )
            hops_info.append(
                {
                    "hop": h_i + 1,
                    "active": gate_val > 0.25,
                    "gate": gate_val,
                    "bridge_word": br_word if gate_val > 0.25 else None,
                    "target_word": best_word,
                    "target_mass": best_mass,
                    "clause_id": cid_h,
                    "clause_text": clauses.get(cid_h, ""),
                }
            )

    # Identify query anchor word from neural q_hits or rec_win
    query_anchor = ""
    if "q_hits" in telemetry:
        q_hits = np.asarray(telemetry["q_hits"][0, t_idx, :prompt_len], dtype=np.float32) * ctx_mask
        if q_hits.size and float(q_hits.max()) > 0.5:
            hop1_cid = hops_info[0]["clause_id"] if hops_info else 0
            in_hop1 = np.exp(-10.0 * np.square(clause_id - float(hop1_cid)))
            q_Priority = q_hits * (1.0 + in_hop1)
            q_pos = int(np.argmax(q_Priority))
            query_anchor, _, _ = _word_span_at(tokens_prefix, q_pos)
    if not query_anchor and rec_win.size:
        query_anchor, _, _ = _word_span_at(tokens_prefix, int(np.argmax(rec_win)))

    onset_pos = int(np.argmax(onset_attn)) if onset_attn.size else 0
    onset_word, _, _ = _word_span_at(tokens_prefix, onset_pos)
    onset_mass = float(onset_attn.max()) if onset_attn.size else 0.0
    idx_density = (
        float(telemetry["index_share_density"][0, t_idx])
        if "index_share_density" in telemetry
        else 0.0
    )

    trace.update(
        {
            "relational_active": g_q > 0.25,
            "g_question": g_q,
            "g_chat": g_chat,
            "g_bwd": g_bwd,
            "direction": "backward (subject-query)" if g_bwd >= 0.5 else "forward (object-query)",
            "clauses": clauses,
            "query_anchor": query_anchor,
            "hops": hops_info,
            "onset_word": onset_word,
            "onset_mass": onset_mass,
            "cont_gate": cont_gate,
            "eos_mass": eos_mass,
            "index_share_density": idx_density,
        }
    )
    return trace


def _softmax_scores(scores: np.ndarray) -> np.ndarray:
    finite = np.where(np.isfinite(scores), scores, -1e9)
    shifted = finite - np.max(finite)
    exp_s = np.exp(shifted)
    return exp_s / np.maximum(np.sum(exp_s), 1e-12)


def generate_with_trace(
    state,
    config,
    prompt: str,
    max_bytes: int = 128,
    messages: bool = True,
    adapt_strength: float = 0.0,
    adapt_decay: float = 0.88,
    on_trace: Callable[[dict], None] | None = None,
    on_token: Callable[[str], None] | None = None,
) -> tuple[str, dict]:
    """Generate bytes while capturing full neural reasoning telemetry and timing."""
    if not 0 <= max_bytes <= 8192:
        raise ValueError("max_bytes must be between 0 and 8192")
    if adapt_strength < 0 or not np.isfinite(adapt_strength):
        raise ValueError("adapt_strength must be a finite non-negative float")
    if not 0.0 <= adapt_decay <= 1.0 or not np.isfinite(adapt_decay):
        raise ValueError("adapt_decay must be between 0 and 1")

    prefix = [BOS, *prompt.encode("utf-8")]
    prompt_bytes = len(prefix)
    buf_len = max(160, ((prompt_bytes + max(max_bytes, 1) + 63) // 64) * 64)
    buf = np.full((1, buf_len), EOS, dtype=np.int32)
    valid = np.zeros((1, buf_len), dtype=bool)
    resets = jnp.zeros((1, buf_len), dtype=bool).at[:, 0].set(True)
    init_carry = initial_state(config, 1)
    cur_len = prompt_bytes
    buf[0, :cur_len] = prefix
    valid[0, :cur_len] = True
    step_seq_tel = _make_seq_step_telemetry(state.apply_fn, bool(messages))

    t0 = time.perf_counter()
    logits_seq, final_carry, telemetry = step_seq_tel(
        state.params,
        jnp.asarray(buf),
        init_carry,
        resets,
        jnp.asarray(valid),
    )
    logits_np = np.asarray(logits_seq[0, cur_len - 1], dtype=np.float32)
    prompt_ms = (time.perf_counter() - t0) * 1000.0
    trace = _extract_reasoning_trace(prefix, telemetry, final_carry, config)
    trace["prompt_bytes"] = prompt_bytes
    trace["prompt_ms"] = prompt_ms
    trace["prompt_bps"] = (prompt_bytes / (prompt_ms / 1000.0)) if prompt_ms > 0 else 0.0
    if on_trace is not None:
        on_trace(trace)

    if max_bytes == 0:
        trace.update(
            {
                "eval_bytes": 0,
                "eval_ms": 0.0,
                "eval_bps": 0.0,
                "total_ms": prompt_ms,
                "token_steps": [],
            }
        )
        return "", trace

    output = bytearray()
    token_steps: list[dict] = []
    adaptation = np.zeros(258, dtype=np.float32)
    step_seq_fast = _make_seq_step(state.apply_fn, bool(messages))
    t_gen_start = time.perf_counter()
    speculative_accepted = 0

    def _record_step(scores_vec: np.ndarray, chosen_id: int) -> None:
        probs_v = _softmax_scores(scores_vec)
        top3_i = np.argsort(probs_v)[::-1][:3]
        top3_list = [
            {
                "id": int(idx),
                "label": "<EOS>"
                if int(idx) == EOS
                else (chr(int(idx)) if 32 <= int(idx) <= 126 else f"0x{int(idx):02x}"),
                "prob": float(probs_v[int(idx)]),
            }
            for idx in top3_i
        ]
        ch_lbl = (
            "<EOS>" if chosen_id == EOS else bytes([chosen_id]).decode("utf-8", errors="replace")
        )
        token_steps.append(
            {
                "byte": chosen_id,
                "char": ch_lbl,
                "prob": float(probs_v[chosen_id]),
                "top3": top3_list,
            }
        )

    # GLM-5.3-Flash Speculative Multi-Byte Span Drafting & Single-Pass Verification
    draft_word = trace.get("onset_word", "") if trace.get("relational_active") else ""
    draft_bytes = list(draft_word.encode("utf-8")) if draft_word else []
    scores0 = logits_np.copy()
    if adapt_strength > 0:
        scores0 = scores0 - adapt_strength * adaptation[: scores0.shape[0]]
    scores0[BOS] = -np.inf
    chosen0 = int(scores0.argmax())

    step_start = 0
    if (
        adapt_strength == 0.0
        and len(draft_bytes) >= 2
        and len(draft_bytes) <= max_bytes
        and chosen0 == draft_bytes[0]
    ):
        _record_step(scores0, chosen0)
        output.append(chosen0)
        if on_token is not None:
            on_token(bytes([chosen0]).decode("utf-8", errors="replace"))
        n_draft = len(draft_bytes)
        for k_d, b_d in enumerate(draft_bytes):
            buf[0, prompt_bytes + k_d] = b_d
            valid[0, prompt_bytes + k_d] = True
        logits_spec = step_seq_fast(
            state.params,
            jnp.asarray(buf),
            init_carry,
            resets,
            jnp.asarray(valid),
        )
        spec_done = False
        cur_len = prompt_bytes + 1
        step_start = 1
        for k_d in range(1, n_draft + 1):
            if step_start >= max_bytes:
                break
            sc_k = np.asarray(logits_spec[0, prompt_bytes + k_d - 1], dtype=np.float32).copy()
            sc_k[BOS] = -np.inf
            ch_k = int(sc_k.argmax())
            expected_k = draft_bytes[k_d] if k_d < n_draft else EOS
            _record_step(sc_k, ch_k)
            if ch_k == EOS:
                if expected_k == EOS:
                    speculative_accepted += 1
                spec_done = True
                break
            output.append(ch_k)
            if on_token is not None:
                on_token(bytes([ch_k]).decode("utf-8", errors="replace"))
            buf[0, cur_len] = ch_k
            valid[0, cur_len] = True
            cur_len += 1
            step_start += 1
            if ch_k == expected_k:
                speculative_accepted += 1
            else:
                buf[0, cur_len:] = EOS
                valid[0, cur_len:] = False
                break
        if spec_done:
            step_start = max_bytes

    for step_i in range(step_start, max_bytes):
        if step_i == 0:
            scores = scores0
            chosen = chosen0
        else:
            logits_seq = step_seq_fast(
                state.params,
                jnp.asarray(buf),
                init_carry,
                resets,
                jnp.asarray(valid),
            )
            scores = np.asarray(logits_seq[0, cur_len - 1], dtype=np.float32).copy()
            if adapt_strength > 0:
                scores = scores - adapt_strength * adaptation[: scores.shape[0]]
            scores[BOS] = -np.inf
            chosen = int(scores.argmax())

        _record_step(scores, chosen)
        if chosen == EOS:
            break
        if adapt_strength > 0:
            adaptation = adapt_decay * adaptation
            adaptation[chosen] += 1.0
        output.append(chosen)
        ch_str = bytes([chosen]).decode("utf-8", errors="replace")
        if on_token is not None:
            on_token(ch_str)
        buf[0, cur_len] = chosen
        valid[0, cur_len] = True
        cur_len += 1

    eval_ms = (time.perf_counter() - t_gen_start) * 1000.0
    eval_bytes = len(output)
    trace.update(
        {
            "eval_bytes": eval_bytes,
            "eval_ms": eval_ms,
            "eval_bps": (eval_bytes / (eval_ms / 1000.0)) if eval_ms > 0 else 0.0,
            "total_ms": prompt_ms + eval_ms,
            "token_steps": token_steps,
            "speculative_accepted": speculative_accepted,
        }
    )
    return output.decode("utf-8", errors="replace"), trace


def generate(
    state,
    config,
    prompt: str,
    max_bytes: int = 128,
    messages=True,
    adapt_strength: float = 0.0,
    adapt_decay: float = 0.88,
) -> str:
    if not 0 <= max_bytes <= 8192:
        raise ValueError("max_bytes must be between 0 and 8192")
    if adapt_strength < 0 or not np.isfinite(adapt_strength):
        raise ValueError("adapt_strength must be a finite non-negative float")
    if not 0.0 <= adapt_decay <= 1.0 or not np.isfinite(adapt_decay):
        raise ValueError("adapt_decay must be between 0 and 1")
    if getattr(config, "relational_hops", 0) > 0:
        if max_bytes == 0:
            return ""
        prefix = [BOS, *prompt.encode("utf-8")]
        buf_len = max(160, ((len(prefix) + max_bytes + 63) // 64) * 64)
        buf = np.full((1, buf_len), EOS, dtype=np.int32)
        valid = np.zeros((1, buf_len), dtype=bool)
        resets = jnp.zeros((1, buf_len), dtype=bool).at[:, 0].set(True)
        init_carry = initial_state(config, 1)
        cur_len = len(prefix)
        buf[0, :cur_len] = prefix
        valid[0, :cur_len] = True
        step_seq = _make_seq_step(state.apply_fn, bool(messages))

        output = bytearray()
        adaptation = np.zeros(258, dtype=np.float32)
        for _ in range(max_bytes):
            logits_seq = step_seq(
                state.params,
                jnp.asarray(buf),
                init_carry,
                resets,
                jnp.asarray(valid),
            )
            scores = np.asarray(logits_seq[0, cur_len - 1], dtype=np.float32).copy()
            if adapt_strength > 0:
                scores = scores - adapt_strength * adaptation[: scores.shape[0]]
            scores[BOS] = -np.inf
            chosen = int(scores.argmax())
            if chosen == EOS:
                break
            if adapt_strength > 0:
                adaptation = adapt_decay * adaptation
                adaptation[chosen] += 1.0
            output.append(chosen)
            buf[0, cur_len] = chosen
            valid[0, cur_len] = True
            cur_len += 1
        return output.decode("utf-8", errors="replace")

    tokens = jnp.array([[BOS, *prompt.encode("utf-8")]], dtype=jnp.int32)
    resets = jnp.zeros_like(tokens, dtype=bool).at[:, 0].set(True)
    logits, carry = state.apply_fn(
        {"params": state.params}, tokens, initial_state(config, 1), resets, messages=messages
    )

    @jax.jit
    def advance(token, hidden):
        return state.apply_fn(
            {"params": state.params},
            token,
            hidden,
            jnp.zeros((1, 1), dtype=bool),
            messages=messages,
        )

    output = bytearray()
    adaptation = np.zeros(logits.shape[-1], dtype=np.float32)
    for _ in range(max_bytes):
        # BOS cannot occur inside a document. EOS ends generation.
        scores = np.asarray(logits[0, -1], dtype=np.float32).copy()
        if adapt_strength > 0:
            scores = scores - adapt_strength * adaptation
        scores[BOS] = -np.inf
        chosen = int(scores.argmax())
        if chosen == EOS:
            break
        if adapt_strength > 0:
            adaptation = adapt_decay * adaptation
            adaptation[chosen] += 1.0
        output.append(chosen)
        logits, carry = advance(jnp.array([[chosen]], dtype=jnp.int32), carry)
    return output.decode("utf-8", errors="replace")


_STOP_WORDS = {
    "the",
    "and",
    "was",
    "are",
    "has",
    "had",
    "with",
    "under",
    "near",
    "inside",
    "from",
    "into",
    "onto",
    "that",
    "this",
    "instead",
    "where",
    "what",
    "who",
    "which",
    "when",
    "how",
    "does",
    "did",
}


def _content_words_set(text: str) -> set[str]:
    cleaned = "".join(ch if ch.isalnum() else " " for ch in text)
    return {w.lower() for w in cleaned.split() if len(w) >= 3 and w.lower() not in _STOP_WORDS}


def _extract_clauses_and_question(user_message: str) -> tuple[list[str], str]:
    msg = " ".join(user_message.strip().split())
    if not msg:
        return [], ""
    if msg.endswith("?"):
        if ". " in msg:
            split_idx = msg.rfind(". ")
            ctx_part = msg[: split_idx + 1]
            q_part = msg[split_idx + 2 :].strip()
            clauses = [c.strip() + "." for c in ctx_part.split(".") if c.strip()]
            return clauses, q_part
        return [], msg
    clauses = [c.strip() + "." for c in msg.split(".") if c.strip()]
    return clauses, ""


class LiveLearner:
    """Online continual byte-stream learner during live interaction (mini-AGI minagi/live.py)."""

    def __init__(
        self,
        state,
        config,
        chunk: int = 16,
        context: int | None = None,
        messages: bool = True,
        max_episodic_facts: int = 4,
    ):
        if chunk < 2:
            raise ValueError("chunk must be at least 2")
        self.state = state
        self.config = config
        self.chunk = int(chunk)
        self.context = int(context or config.length)
        self.messages = messages
        self.max_episodic_facts = int(max_episodic_facts)
        self.episodic_facts: list[str] = []
        self.episodic_qa: list[dict] = []
        self.compacted_facts = 0
        self.buf: list[int] = [BOS]
        self.pending = 0
        self.steps = 0
        self.carry = initial_state(config, 1)
        self.step_fn = make_stream_step(config, messages=messages)
        self.log: list[dict] = []

    @staticmethod
    def _fact_subject_relation_key(clause: str) -> tuple[str, str] | None:
        words = "".join(ch if ch.isalnum() else " " for ch in clause).lower().split()
        if len(words) < 3:
            return None
        subj = words[0]
        if subj in {"the", "a", "an"} and len(words) >= 4:
            subj = f"{words[0]}_{words[1]}"
            rel_words = words[2:]
        else:
            rel_words = words[1:]
        if any(w in rel_words for w in ("in", "under", "on", "at", "to")):
            return (subj, "location")
        if any(w in rel_words for w in ("with", "beside", "near")):
            return (subj, "companion")
        if any(w in rel_words for w in ("is", "was")) and len(rel_words) <= 2:
            return (subj, "state")
        return None

    def compact_episodic_facts(self) -> int:
        """GLM-5.3 SAO episodic compaction: evict superseded same-subject state clauses first."""
        removed = 0
        while len(self.episodic_facts) > self.max_episodic_facts:
            evict_idx = 0
            keys = [self._fact_subject_relation_key(f) for f in self.episodic_facts]
            for idx_old, k_old in enumerate(keys[:-1]):
                if k_old is not None and k_old in keys[idx_old + 1 :]:
                    evict_idx = idx_old
                    break
            self.episodic_facts.pop(evict_idx)
            removed += 1
        self.compacted_facts += removed
        return removed

    def remember_facts(self, clauses: list[str] | str) -> int:
        """Append declarative clauses to the rolling episodic working memory buffer."""
        if isinstance(clauses, str):
            extracted, _ = _extract_clauses_and_question(clauses)
            clause_list = extracted if extracted else [clauses]
        else:
            clause_list = clauses
        for cl in clause_list:
            norm_cl = " ".join(cl.strip().split())
            if not norm_cl:
                continue
            if not norm_cl.endswith("."):
                norm_cl = norm_cl + "."
            if self.episodic_facts and self.episodic_facts[-1] == norm_cl:
                continue
            self.episodic_facts.append(norm_cl)
        if len(self.episodic_facts) > self.max_episodic_facts:
            return self.compact_episodic_facts()
        return 0

    def resolve_context(self, user_message: str, explicit_context: str = "") -> str:
        """Resolve effective context by combining explicit context and episodic facts."""
        inline_clauses, q_part = _extract_clauses_and_question(user_message)
        exp_ctx = explicit_context.strip()
        if exp_ctx:
            if inline_clauses:
                self.remember_facts(inline_clauses)
            return exp_ctx
        if inline_clauses and not q_part:
            self.remember_facts(inline_clauses)
            return " ".join(self.episodic_facts)
        if not inline_clauses and q_part:
            return " ".join(self.episodic_facts)
        if inline_clauses and q_part:
            turn_words = _content_words_set(" ".join(inline_clauses) + " " + q_part)
            prior = [
                f
                for f in self.episodic_facts
                if f not in inline_clauses and bool(_content_words_set(f) & turn_words)
            ]
            self.remember_facts(inline_clauses)
            return " ".join(prior[-(self.max_episodic_facts - len(inline_clauses)) :])
        return ""

    def _record_episodic_qa(self, user_message: str, eff_context: str, reply: str) -> None:
        inline_clauses, q_part = _extract_clauses_and_question(user_message)
        if not q_part:
            return
        full_ctx = " ".join(
            part for part in [eff_context.strip(), " ".join(inline_clauses).strip()] if part
        ).strip()
        ans = reply.strip()
        if not full_ctx or not ans:
            return
        words: list[str] = []
        for token in "".join(ch if ch.isalnum() else " " for ch in full_ctx).split():
            if len(token) >= 3 and token.lower() not in _STOP_WORDS and token not in words:
                words.append(token)
        if ans not in words:
            words.insert(0, ans)
        if len(words) < 2:
            words.append("none")
        cands = [ans] + [w for w in words if w != ans][:3]
        self.episodic_qa.append(
            {
                "context": full_ctx,
                "question": q_part,
                "candidates": cands,
                "answer": ans,
            }
        )
        if len(self.episodic_qa) > 8:
            self.episodic_qa = self.episodic_qa[-8:]

    def consolidate(
        self,
        grpo_steps: int = 2,
        rewire: bool = True,
    ) -> dict:
        """Run sleep-like consolidation: flush stream buffer, GRPO on episodic QA, and rewire."""
        flushed = 0
        if self.pending > 0 and len(self.buf) >= 2:
            rec = self._step_window()
            if rec is not None:
                flushed = 1
            self.pending = 0

        grpo_done = 0
        last_grpo_loss = 0.0
        if (
            grpo_steps > 0
            and self.episodic_qa
            and getattr(self.config, "architecture", "graph") == "graph"
        ):
            from mlm.plasticity import build_grpo_question_group, make_grpo_step

            grpo_fn = make_grpo_step(self.config)
            ref_params = self.state.params
            for i in range(grpo_steps):
                qa_item = self.episodic_qa[i % len(self.episodic_qa)]
                batch, rewards = build_grpo_question_group(qa_item, min(self.config.length, 96))
                self.state, metrics = grpo_fn(self.state, ref_params, batch, rewards)
                last_grpo_loss = float(jax.device_get(metrics["loss"]))
                grpo_done += 1

        rewired = False
        if rewire and getattr(self.config, "architecture", "graph") == "graph":
            from mlm.plasticity import rewire_edges

            new_cfg, rewire_info = rewire_edges(self.config, self.state.params)
            if rewire_info["rewired"] > 0:
                self.config = new_cfg
                rewired = True

        return {
            "stream_flushed": flushed,
            "grpo_steps": grpo_done,
            "grpo_loss": last_grpo_loss,
            "rewired": bool(rewired),
            "episodic_facts": len(self.episodic_facts),
            "episodic_qa": len(self.episodic_qa),
            "step": int(self.state.step),
        }

    def _step_window(self) -> dict | None:
        if len(self.buf) < 2:
            return None
        seq = self.buf[-(self.context + 1) :]
        inp = jnp.asarray([seq[:-1]], dtype=jnp.int32)
        tgt = jnp.asarray([seq[1:]], dtype=jnp.int32)
        mask = jnp.ones_like(inp, dtype=jnp.float32)
        resets = jnp.zeros_like(inp, dtype=bool).at[:, 0].set(self.steps == 0)
        batch = {"inputs": inp, "targets": tgt, "mask": mask, "resets": resets}
        self.state, self.carry, raw = self.step_fn(self.state, self.carry, batch)
        metrics = jax.device_get(raw)
        self.steps += 1
        rec = {
            "step": self.steps,
            "loss": float(metrics["loss"]),
            "grad_norm": float(metrics["grad_norm"]),
            "bytes": int(metrics["bytes"]),
        }
        self.log.append(rec)
        return rec

    def feed(self, text: str) -> list[dict]:
        """Append text to the live stream and take one gradient step every `chunk` bytes."""
        ids = list(text.encode("utf-8"))
        records = []
        i = 0
        while i < len(ids):
            take = min(self.chunk - self.pending, len(ids) - i)
            self.buf.extend(ids[i : i + take])
            self.pending += take
            i += take
            if self.pending >= self.chunk:
                if len(self.buf) > self.context + 1:
                    del self.buf[: -(self.context + 1)]
                rec = self._step_window()
                if rec is not None:
                    records.append(rec)
                self.pending = 0
        if len(self.buf) > self.context + 1:
            del self.buf[: -(self.context + 1)]
        return records

    def chat_turn(
        self,
        user_message: str,
        max_bytes: int = 160,
        adapt_strength: float = 0.0,
        adapt_decay: float = 0.88,
        learn: bool = True,
        context: str = "",
    ) -> tuple[str, list[dict]]:
        eff_ctx = self.resolve_context(user_message, explicit_context=context)
        prompt = format_chat_prompt(user_message, context=eff_ctx)
        reply = generate(
            self.state,
            self.config,
            prompt,
            max_bytes=max_bytes,
            messages=self.messages,
            adapt_strength=adapt_strength,
            adapt_decay=adapt_decay,
        )
        self._record_episodic_qa(user_message, eff_ctx, reply)
        steps = self.feed(f"{prompt}{reply}\n") if learn else []
        return reply, steps

    def chat_turn_with_trace(
        self,
        user_message: str,
        max_bytes: int = 160,
        adapt_strength: float = 0.0,
        adapt_decay: float = 0.88,
        learn: bool = True,
        context: str = "",
        on_trace: Callable[[dict], None] | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> tuple[str, dict, list[dict]]:
        eff_ctx = self.resolve_context(user_message, explicit_context=context)
        prompt = format_chat_prompt(user_message, context=eff_ctx)
        reply, trace = generate_with_trace(
            self.state,
            self.config,
            prompt,
            max_bytes=max_bytes,
            messages=self.messages,
            adapt_strength=adapt_strength,
            adapt_decay=adapt_decay,
            on_trace=on_trace,
            on_token=on_token,
        )
        trace["episodic_facts"] = list(self.episodic_facts)
        trace["compacted_facts"] = int(self.compacted_facts)
        self._record_episodic_qa(user_message, eff_ctx, reply)
        steps = self.feed(f"{prompt}{reply}\n") if learn else []
        if learn and self.episodic_qa:
            cons = self.consolidate(grpo_steps=1, rewire=True)
            trace["auto_consolidated"] = cons
        return reply, trace, steps
