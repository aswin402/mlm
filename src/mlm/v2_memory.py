"""3-Tier Persistent External & Episodic Memory Hierarchy (`MemoryHierarchyV2`) for MLM v2."""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from safetensors.numpy import load as load_safetensors_bytes
from safetensors.numpy import save as save_safetensors_bytes

from mlm.v2_kernels import build_layer_execution_schedule

if TYPE_CHECKING:
    from mlm.model_v2 import ConfigV2
    from mlm.tokenizer import BPETokenizer


_VAR_ASSIGN_RE = re.compile(
    r"(?:^|[\s,;]|let\s+|set\s+|remember(?:\s+that)?\s+)([a-zA-Z_][a-zA-Z0-9_]{0,15})\s*=\s*(-?\d+(?:\.\d+)?)\b",
    flags=re.IGNORECASE,
)

_QUESTION_STARTERS = {
    "what",
    "whats",
    "who",
    "where",
    "when",
    "why",
    "how",
    "which",
    "can",
    "could",
    "would",
    "should",
    "do",
    "does",
    "did",
    "is",
    "are",
    "was",
    "were",
    "calculate",
    "compute",
    "solve",
    "find",
    "tell",
    "explain",
    "describe",
    "hi",
    "hii",
    "hello",
    "hey",
}

_STOPWORDS_MEM = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "in",
    "at",
    "on",
    "of",
    "to",
    "for",
    "and",
    "or",
    "that",
    "this",
    "what",
    "whats",
    "who",
    "where",
    "when",
    "why",
    "how",
    "do",
    "does",
    "did",
    "can",
    "you",
    "tell",
    "me",
    "remember",
    "about",
    "please",
}


class MemoryHierarchyV2:
    """3-Tier Persistent External & Episodic Memory Hierarchy for MLM v2.

    Tier 1 (Immediate Context):
        DeepSeek Sparse Attention (`DSA + IndexShare`) with `TurboQuant 2-bit + QJL` KV Cache
        (`k_cache`, `v_cache`, `k_idx_cache` in the active recurrent decode carry).
    Tier 2 (Working State):
        Cross-turn `Parallel Gated DeltaNet` recurrent fast weights
        (`working_delta_state` S_{t,h} in R^{L_sched x 1 x H x D_h x D_h}), updated via gated
        delta-rule EMA across turns and persisted across sessions.
    Tier 3 (Persistent Long-Term Memory):
        Content-addressed `TurboQuant`-compressed (`2-bit FWHT + Lloyd-Max + 1-bit QJL`)
        episodic slot bank (`K_ep, V_ep in R^{M x D}`) with State-Action-Overwrite (`SAO` /
        Delta Rule) saved across `mlm chat` sessions.
    """

    def __init__(
        self,
        config: ConfigV2,
        max_slots: int = 64,
        episodic_bits: int = 2,
        working_decay: float = 0.80,
        storage_path: Path | str | None = None,
    ) -> None:
        self.config = config
        self.max_slots = max(8, int(max_slots))
        self.episodic_bits = int(episodic_bits) if int(episodic_bits) in (1, 2, 4) else 2
        self.working_decay = float(np.clip(working_decay, 0.0, 0.99))
        self.storage_path = Path(storage_path) if storage_path is not None else None

        schedule = build_layer_execution_schedule(config)
        self.n_sched = len(schedule)
        self.schedule = schedule

        # Tier 1 telemetry state (Immediate DSA + TurboQuant KV Cache)
        self.tier1_active_tokens: int = 0
        self.tier1_kv_bits: int = int(config.kv_cache_bits)
        self.tier1_kv_norm: float = 0.0
        self.tier1_idx_norm: float = 0.0

        # Tier 2 state (Working Parallel Gated DeltaNet Recurrent Fast Weights)
        self.working_delta_state = np.zeros(
            (self.n_sched, 1, config.n_heads, config.head_dim, config.head_dim),
            dtype=np.float32,
        )
        self.tier2_delta_norm: float = 0.0
        self.tier2_layer_norms: list[float] = [0.0] * config.n_layers

        # Tier 3 state (Persistent Content-Addressed TurboQuant Episodic Slot Bank)
        self.slots: list[dict[str, object]] = []
        self._k_raw = np.zeros((self.max_slots, config.d_model), dtype=np.float32)
        self._v_raw = np.zeros((self.max_slots, config.d_model), dtype=np.float32)
        self.packed_kv: dict[str, object] | None = None
        self.turn_counter: int = 0
        self.total_hits: int = 0
        self.total_overwrites: int = 0
        self.last_retrieved_slots: list[dict[str, object]] = []
        self.last_resolved_prompt: str = ""
        from mlm.v2_decode import LMCacheHybridV2

        self.lmcache = LMCacheHybridV2(chunk_size=4, max_entries=64)

        self._repack_turboquant_bank()
        if self.storage_path is not None and self.storage_path.exists():
            self.load(self.storage_path)

    def _repack_turboquant_bank(self) -> None:
        from mlm.quantize_v2 import quantize_kv_cache_turboquant

        self.packed_kv = quantize_kv_cache_turboquant(
            self._k_raw, self._v_raw, bits=self.episodic_bits
        )

    def _dequantize_turboquant_bank(self) -> tuple[np.ndarray, np.ndarray]:
        from mlm.quantize_v2 import dequantize_kv_cache_turboquant

        if self.packed_kv is None:
            self._repack_turboquant_bank()
        assert self.packed_kv is not None
        k_j, v_j = dequantize_kv_cache_turboquant(self.packed_kv)
        return np.asarray(k_j, dtype=np.float32), np.asarray(v_j, dtype=np.float32)

    def _compute_text_key_value(
        self,
        subject_text: str,
        full_text: str,
        params: dict[str, jnp.ndarray],
        config: ConfigV2,
        tokenizer: BPETokenizer,
    ) -> tuple[np.ndarray, np.ndarray]:
        s_ids = tokenizer.encode(subject_text, add_bos=False, add_eos=False).tolist()
        f_ids = tokenizer.encode(full_text, add_bos=False, add_eos=False).tolist()
        if not s_ids:
            s_ids = [tokenizer.bos_id]
        if not f_ids:
            f_ids = s_ids
        tok_w = np.asarray(params["tok_embeddings.weight"], dtype=np.float32)
        s_emb = np.mean(tok_w[s_ids], axis=0)
        f_emb = np.mean(tok_w[f_ids], axis=0)
        if (
            config.blt_ngram_dim > 0
            and "blt_token_ngrams" in params
            and "blt_proj.weight" in params
        ):
            blt_tbl = np.asarray(params["blt_token_ngrams"], dtype=np.float32)
            blt_w = np.asarray(params["blt_proj.weight"], dtype=np.float32)
            s_emb = s_emb + np.mean(blt_tbl[s_ids] @ blt_w, axis=0)
            f_emb = f_emb + np.mean(blt_tbl[f_ids] @ blt_w, axis=0)
        if "blk.0.attn_norm.weight" in params:
            nw = np.asarray(params["blk.0.attn_norm.weight"], dtype=np.float32)
            s_rms = np.sqrt(np.mean(np.square(s_emb)) + config.rms_eps)
            f_rms = np.sqrt(np.mean(np.square(f_emb)) + config.rms_eps)
            s_emb = (s_emb / s_rms) * nw
            f_emb = (f_emb / f_rms) * nw
        k_vec = 0.75 * s_emb + 0.25 * f_emb
        k_norm = float(np.linalg.norm(k_vec))
        if k_norm > 1e-6:
            k_vec = k_vec / k_norm
        v_vec = f_emb.copy()
        # Blend in Tier-2 Gated DeltaNet working fast-weight readout if active
        if self.tier2_delta_norm > 1e-6:
            h = config.n_heads
            d_h = config.head_dim
            q_heads = k_vec.reshape(h, d_h)
            s0 = self.working_delta_state[0, 0]  # (H, D_h, D_h)
            delta_readout = np.einsum("hd,hde->he", q_heads, s0).reshape(-1)
            v_vec = v_vec + 0.15 * delta_readout
        return k_vec.astype(np.float32), v_vec.astype(np.float32)

    @staticmethod
    def _normalize_subject(raw_subj: str) -> str:
        s = " ".join(raw_subj.strip().lower().strip(".,!?;:\"'").split())
        for prefix in ("the ", "a ", "an ", "that ", "remember ", "note "):
            if s.startswith(prefix):
                s = s[len(prefix) :].strip()
        if s.startswith("your "):
            s = "my " + s[5:]
        return s

    @classmethod
    def extract_episodic_entries_v2(cls, user_text: str) -> list[dict[str, str]]:
        """Extract declarative facts, user context, or variable bindings from a user turn."""
        clean = " ".join(user_text.strip().split())
        if not clean:
            return []
        entries: list[dict[str, str]] = []
        seen_subjects: set[str] = set()

        # 1. Extract explicit variable assignments `var = number`
        for vm in _VAR_ASSIGN_RE.finditer(clean):
            v_name = vm.group(1).strip().lower()
            v_val = vm.group(2).strip()
            if v_name in _QUESTION_STARTERS or v_name in ("answer", "solution"):
                continue
            entries.append(
                {
                    "kind": "variable",
                    "subject": v_name,
                    "relation": "=",
                    "value": v_val,
                    "text": f"{v_name} = {v_val}",
                }
            )
            seen_subjects.add(v_name)

        # Do not extract arbitrary prose sentences as persistent facts if the turn is a question
        has_question = "?" in clean
        explicit_remember = bool(
            re.match(r"^(?:remember|note|fact|store|save)\b", clean, flags=re.IGNORECASE)
        )
        if has_question and not explicit_remember:
            return entries

        # Split into clauses and extract subject-relation-value facts
        clauses = re.split(
            r"(?:\.\s+|\n+|;\s+|\s+and\s+(?=(?:i\s+(?:am|work|live|like|use)\b|my\s+\w+\s+is\b|the\s+[\w\s]{1,25}?\s+is\b|[a-zA-Z_][a-zA-Z0-9_]*\s*(?:=|(?:is|are|uses|has|lives|works)\b))))",
            clean,
            flags=re.IGNORECASE,
        )
        for raw_cl in clauses:
            cl = raw_cl.strip().strip(".!")
            if not cl or "?" in cl:
                continue
            cl = re.sub(
                r"^(?:please\s+)?(?:remember(?:\s+that)?|note(?:\s+that)?|fact|doc|context)\s*[:,-]?\s*",
                "",
                cl,
                flags=re.IGNORECASE,
            ).strip()
            if not cl:
                continue
            first_word = cl.split()[0].lower().strip(".,!?")
            if first_word in _QUESTION_STARTERS:
                continue
            # Skip pure variable assignment clauses already captured
            if re.match(
                r"^(?:let\s+|set\s+)?[a-zA-Z_][a-zA-Z0-9_]{0,15}\s*=\s*-?\d+(?:\.\d+)?$",
                cl,
                flags=re.IGNORECASE,
            ):
                continue

            m_rel = re.match(
                r"^([a-zA-Z0-9_\s'\-]{1,40}?)\s+"
                r"(is\s+in|is\s+at|is\s+from|is|are|was|were|uses|has|lives\s+in|works\s+on|works\s+at|equals)\s+"
                r"(.+)$",
                cl,
                flags=re.IGNORECASE,
            )
            if m_rel:
                raw_s = m_rel.group(1).strip()
                rel = m_rel.group(2).strip().lower()
                val = m_rel.group(3).strip().rstrip(".")
                # Avoid capturing arithmetic expressions like `4 + 4 is 8`
                if re.search(r"\d+\s*[\+\-\*\/]\s*\d+", raw_s):
                    continue
                subj = cls._normalize_subject(raw_s)
                if subj and val and subj not in seen_subjects:
                    entries.append(
                        {
                            "kind": "fact",
                            "subject": subj,
                            "relation": rel,
                            "value": val,
                            "text": f"{raw_s} {rel} {val}.",
                        }
                    )
                    seen_subjects.add(subj)
                continue

            m_iam = re.match(
                r"^i\s+(am\s+building|am\s+working\s+on|am|work\s+on|live\s+in|like|use)\s+(.+)$",
                cl,
                flags=re.IGNORECASE,
            )
            if m_iam:
                rel = m_iam.group(1).strip().lower()
                val = m_iam.group(2).strip().rstrip(".")
                subj = f"i {rel}"
                if subj not in seen_subjects:
                    entries.append(
                        {
                            "kind": "fact",
                            "subject": subj,
                            "relation": ":",
                            "value": val,
                            "text": f"You {rel.replace('am ', 'are ')} {val}.",
                        }
                    )
                    seen_subjects.add(subj)
        return entries

    def upsert_slot(
        self,
        entry: dict[str, str],
        params: dict[str, jnp.ndarray],
        config: ConfigV2,
        tokenizer: BPETokenizer,
    ) -> dict[str, object]:
        """Insert or State-Action-Overwrite (`SAO`) a slot in the 2-bit TurboQuant bank."""
        subj = entry["subject"]
        full_txt = entry["text"]
        k_vec, v_vec = self._compute_text_key_value(subj, full_txt, params, config, tokenizer)

        # Check for existing slot with matching subject (State-Action-Overwrite / Delta Rule)
        target_idx: int | None = None
        for idx, sl in enumerate(self.slots):
            if str(sl["subject"]) == subj:
                target_idx = idx
                break

        if target_idx is not None:
            slot = self.slots[target_idx]
            slot["kind"] = entry["kind"]
            slot["relation"] = entry["relation"]
            slot["value"] = entry["value"]
            slot["text"] = full_txt
            slot["turn"] = self.turn_counter
            slot["overwrites"] = int(slot.get("overwrites", 0)) + 1
            slot["key_norm"] = round(float(np.linalg.norm(k_vec)), 4)
            slot["val_norm"] = round(float(np.linalg.norm(v_vec)), 4)
            self._k_raw[target_idx] = k_vec
            self._v_raw[target_idx] = v_vec
            self.total_overwrites += 1
        else:
            if len(self.slots) < self.max_slots:
                target_idx = len(self.slots)
                slot = {
                    "slot_id": target_idx,
                    "kind": entry["kind"],
                    "subject": subj,
                    "relation": entry["relation"],
                    "value": entry["value"],
                    "text": full_txt,
                    "turn": self.turn_counter,
                    "hits": 0,
                    "overwrites": 0,
                    "key_norm": round(float(np.linalg.norm(k_vec)), 4),
                    "val_norm": round(float(np.linalg.norm(v_vec)), 4),
                }
                self.slots.append(slot)
            else:
                # Evict least-recently-used / lowest-hit slot
                target_idx = min(
                    range(self.max_slots),
                    key=lambda i: (
                        int(self.slots[i].get("hits", 0)),
                        int(self.slots[i].get("turn", 0)),
                    ),
                )
                slot = {
                    "slot_id": target_idx,
                    "kind": entry["kind"],
                    "subject": subj,
                    "relation": entry["relation"],
                    "value": entry["value"],
                    "text": full_txt,
                    "turn": self.turn_counter,
                    "hits": 0,
                    "overwrites": 1,
                    "key_norm": round(float(np.linalg.norm(k_vec)), 4),
                    "val_norm": round(float(np.linalg.norm(v_vec)), 4),
                }
                self.slots[target_idx] = slot
            self._k_raw[target_idx] = k_vec
            self._v_raw[target_idx] = v_vec

        self._repack_turboquant_bank()
        return slot

    def retrieve(
        self,
        query_text: str,
        params: dict[str, jnp.ndarray],
        config: ConfigV2,
        tokenizer: BPETokenizer,
        top_k: int = 2,
    ) -> list[dict[str, object]]:
        """Query the dequantized 2-bit TurboQuant episodic key bank via cosine + entity matching."""
        if not self.slots:
            return []
        q_clean = query_text.lower().strip()
        q_norm_pronoun = q_clean.replace("your name", "my name").replace("who am i", "my name")
        q_vec, _ = self._compute_text_key_value(
            self._normalize_subject(q_norm_pronoun),
            query_text,
            params,
            config,
            tokenizer,
        )
        k_hat, _ = self._dequantize_turboquant_bank()
        n_occ = len(self.slots)
        k_occ = k_hat[:n_occ]
        k_norms = np.maximum(np.linalg.norm(k_occ, axis=-1), 1e-6)
        cos_sims = (k_occ @ q_vec) / k_norms

        q_tokens = {
            w.strip(".,!?;:\"'()[]")
            for w in q_norm_pronoun.split()
            if w.strip(".,!?;:\"'()[]") and w.strip(".,!?;:\"'()[]") not in _STOPWORDS_MEM
        }
        scored: list[tuple[float, int]] = []
        for idx, sl in enumerate(self.slots):
            subj = str(sl["subject"]).lower()
            if sl.get("kind") == "variable":
                if re.match(
                    rf"^(?:what\s+is|whats|value\s+of|show)\s+{re.escape(subj)}\s*\??$",
                    q_clean,
                ):
                    scored.append((float(cos_sims[idx]) + 1.5, idx))
                continue
            subj_check = subj.replace("i am ", "").replace("i ", "")
            subj_tokens = {
                w.strip(".,!?;:\"'")
                for w in subj_check.split()
                if len(w.strip(".,!?;:\"'")) >= 2
                and w.strip(".,!?;:\"'") not in _STOPWORDS_MEM
            }
            exact_subj = bool(re.search(rf"\b{re.escape(subj)}\b", q_norm_pronoun)) or (
                subj == "my name" and ("my name" in q_norm_pronoun or "who am i" in q_clean)
            )
            token_overlap = (
                len(q_tokens & subj_tokens) / float(max(len(subj_tokens), 1))
                if subj_tokens
                else 0.0
            )
            if exact_subj or token_overlap >= 0.65:
                combined_score = float(cos_sims[idx]) + (1.0 if exact_subj else 0.5 * token_overlap)
                scored.append((combined_score, idx))

        scored.sort(key=lambda pair: -pair[0])
        results: list[dict[str, object]] = []
        for sc, idx in scored[:top_k]:
            sl_copy = dict(self.slots[idx])
            sl_copy["similarity"] = round(float(cos_sims[idx]), 4)
            sl_copy["match_score"] = round(sc, 4)
            results.append(sl_copy)
        return results

    def _prefill_prompt_into_tiers(
        self,
        prompt: str,
        params: dict[str, jnp.ndarray],
        config: ConfigV2,
        tokenizer: BPETokenizer,
    ) -> None:
        """Run O(1) recurrent prefill over `prompt` so Tier-1 KV and Tier-2 DeltaNet stay live."""
        from mlm.v2_decode import _make_cached_recurrent_decode_v2, init_decode_cache_v2

        step1, spec2 = _make_cached_recurrent_decode_v2(config)
        carry = init_decode_cache_v2(config)
        p_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False).tolist()[
            -config.max_seq_len :
        ]
        idx = 0
        while idx + 2 <= len(p_ids):
            _, (carry, _, _) = spec2(
                params, carry, jnp.int32(p_ids[idx]), jnp.int32(p_ids[idx + 1])
            )
            idx += 2
        while idx < len(p_ids):
            carry, _, _ = step1(params, carry, jnp.int32(p_ids[idx]))
            idx += 1
        self.record_decode_carry(carry, config)

    def process_turn_before_generate(
        self,
        prompt: str,
        params: dict[str, jnp.ndarray],
        config: ConfigV2,
        tokenizer: BPETokenizer,
    ) -> dict[str, object]:
        """Inspect user prompt before generation for Tier-3 storage, variable binding, or recall."""
        self.turn_counter += 1
        self.last_retrieved_slots = []
        self.last_resolved_prompt = prompt
        u_part = prompt.split("\n<|assistant|>\n", 1)[0].replace("<|user|>\n", "").strip()
        if not u_part:
            return {"action": "passthrough"}

        # 1. Extract and store any new declarative facts or variable assignments
        extracted = self.extract_episodic_entries_v2(u_part)
        stored_slots: list[dict[str, object]] = []
        for entry in extracted:
            sl = self.upsert_slot(entry, params, config, tokenizer)
            stored_slots.append(sl)

        has_question = "?" in u_part or (
            u_part.split()[0].lower().strip(".,!?") in _QUESTION_STARTERS
        )
        if stored_slots and not has_question:
            self._prefill_prompt_into_tiers(prompt, params, config, tokenizer)
            if self.storage_path is not None:
                self.save(self.storage_path)
            slot_summaries = [
                f"[slot #{s['slot_id']}] {s['subject']} {s['relation']} {s['value']}"
                for s in stored_slots
            ]
            fact_sentences = [str(s["text"]) for s in stored_slots]
            think_str = (
                f"Tier-3 TurboQuant ({self.episodic_bits}-bit) Episodic Write: "
                + "; ".join(slot_summaries)
            )
            reply_str = "Remembered: " + " ".join(fact_sentences)
            return {
                "action": "store_only",
                "think": think_str,
                "reply": reply_str,
                "stored_slots": stored_slots,
            }

        # 2. Check if user is asking a math expression referencing stored Tier-3 variables
        var_map = {
            str(s["subject"]): str(s["value"])
            for s in self.slots
            if s.get("kind") == "variable"
        }
        has_math_op = bool(
            re.search(
                r"(?:[\+\-\*\/]|\bplus\b|\bminus\b|\btimes\b|\bdivided\s+by\b)",
                u_part,
                flags=re.IGNORECASE,
            )
        )
        if var_map and has_math_op:
            used_vars: list[tuple[str, str]] = []
            resolved_u = u_part
            for v_name, v_val in sorted(var_map.items(), key=lambda kv: -len(kv[0])):
                # Match variable adjacent to math operator, parenthesis, or question boundary
                pat = re.compile(
                    rf"(?:(?<=[\(\+\-\*\/\s])|^)\b{re.escape(v_name)}\b(?=\s*(?:[\)\+\-\*\/\?]|plus|minus|times|divided\s+by|$))",
                    flags=re.IGNORECASE,
                )
                if pat.search(resolved_u):
                    resolved_u = pat.sub(v_val, resolved_u)
                    used_vars.append((v_name, v_val))
                    for sl in self.slots:
                        if sl.get("kind") == "variable" and str(sl["subject"]) == v_name:
                            sl["hits"] = int(sl.get("hits", 0)) + 1
                            self.total_hits += 1
                            self.last_retrieved_slots.append(dict(sl))
            if used_vars:
                if self.storage_path is not None:
                    self.save(self.storage_path)
                bindings_str = ", ".join(f"{k}={v}" for k, v in used_vars)
                resolved_prompt = f"<|user|>\n{resolved_u}\n<|assistant|>\n"
                self.last_resolved_prompt = resolved_prompt
                return {
                    "action": "variable_substituted",
                    "resolved_prompt": resolved_prompt,
                    "think_prefix": f"[Tier-3 Memory: {bindings_str}] ",
                }

        # 3. Check if user is asking an episodic recall question matching a stored Tier-3 slot
        if has_question and self.slots:
            u_low = u_part.lower().strip()
            if any(
                phrase in u_low
                for phrase in ("what do you remember", "what is in memory", "show memory")
            ):
                self._prefill_prompt_into_tiers(prompt, params, config, tokenizer)
                all_texts = [str(s["text"]) for s in self.slots]
                return {
                    "action": "episodic_recall",
                    "think": (
                        f"Tier-3 Episodic Bank Recall "
                        f"({len(self.slots)}/{self.max_slots} slots active)."
                    ),
                    "reply": "I remember: " + " ".join(all_texts),
                }
            matches = self.retrieve(u_part, params, config, tokenizer, top_k=1)
            if matches:
                top_m = matches[0]
                slot_id = int(top_m["slot_id"])
                self.slots[slot_id]["hits"] = int(self.slots[slot_id].get("hits", 0)) + 1
                self.total_hits += 1
                self.last_retrieved_slots = [dict(self.slots[slot_id])]
                self._prefill_prompt_into_tiers(prompt, params, config, tokenizer)
                if self.storage_path is not None:
                    self.save(self.storage_path)
                subj = str(top_m["subject"])
                rel = str(top_m["relation"])
                val = str(top_m["value"])
                sim = float(top_m.get("similarity", 1.0))
                if top_m.get("kind") == "variable":
                    reply_txt = f"{subj} = {val}. The answer is {val}."
                elif subj.startswith("my "):
                    attr = subj[3:]
                    reply_txt = f"Your {attr} {rel} {val}."
                elif subj.startswith("i "):
                    reply_txt = str(top_m["text"])
                else:
                    reply_txt = str(top_m["text"])
                think_txt = (
                    f"Tier-3 Episodic Recall ({self.episodic_bits}-bit TurboQuant "
                    f"slot #{slot_id}, cos_sim={sim:.2f}): {subj} {rel} {val}."
                )
                return {
                    "action": "episodic_recall",
                    "think": think_txt,
                    "reply": reply_txt,
                }

        return {"action": "passthrough"}

    def record_decode_carry(
        self,
        carry: dict[str, jnp.ndarray],
        config: ConfigV2,
    ) -> None:
        """Update Tier-1 KV stats and Tier-2 Parallel Gated DeltaNet working fast weights."""
        pos = max(1, int(carry["pos"]))
        self.tier1_active_tokens = pos
        self.tier1_kv_bits = int(config.kv_cache_bits)
        k_slice = np.asarray(carry["k_cache"][:, :, :pos], dtype=np.float32)
        v_slice = np.asarray(carry["v_cache"][:, :, :pos], dtype=np.float32)
        idx_slice = np.asarray(carry["k_idx_cache"][:, :, :pos], dtype=np.float32)
        self.tier1_kv_norm = float(
            np.sqrt(np.sum(np.square(k_slice)) + np.sum(np.square(v_slice)))
        )
        self.tier1_idx_norm = float(np.linalg.norm(idx_slice))

        turn_delta = np.asarray(carry["delta_state"], dtype=np.float32)
        if float(np.linalg.norm(self.working_delta_state)) < 1e-6:
            self.working_delta_state = turn_delta.copy()
        else:
            g = self.working_decay
            self.working_delta_state = g * self.working_delta_state + (1.0 - g) * turn_delta
        self.tier2_delta_norm = float(np.linalg.norm(self.working_delta_state))

        layer_sq = [0.0] * config.n_layers
        for s_idx, (l_idx, _) in enumerate(self.schedule):
            if s_idx < self.working_delta_state.shape[0]:
                layer_sq[l_idx] += float(np.sum(np.square(self.working_delta_state[s_idx])))
        self.tier2_layer_norms = [float(np.sqrt(v)) for v in layer_sq]
        if self.storage_path is not None and self.slots:
            self.save(self.storage_path)

    def clear(self) -> None:
        """Clear all Tier-2 working fast weights and Tier-3 episodic slots."""
        self.slots.clear()
        self._k_raw.fill(0.0)
        self._v_raw.fill(0.0)
        self.working_delta_state.fill(0.0)
        self.tier2_delta_norm = 0.0
        self.tier2_layer_norms = [0.0] * self.config.n_layers
        self.total_hits = 0
        self.total_overwrites = 0
        self.last_retrieved_slots = []
        self.lmcache.clear()
        self._repack_turboquant_bank()
        if self.storage_path is not None and self.storage_path.exists():
            self.storage_path.unlink(missing_ok=True)

    def summary_dict(self) -> dict[str, object]:
        packed_b = int(self.packed_kv["packed_bytes"]) if self.packed_kv else 0
        raw_b = int(self.packed_kv["raw_fp32_bytes"]) if self.packed_kv else 0
        comp = float(self.packed_kv["compression_vs_fp32"]) if self.packed_kv else 1.0
        return {
            "tier1_immediate_kv": {
                "mechanism": "DSA + IndexShare Sparse Attention",
                "kv_bits": self.tier1_kv_bits,
                "active_tokens": self.tier1_active_tokens,
                "kv_norm": round(self.tier1_kv_norm, 3),
                "dsa_idx_norm": round(self.tier1_idx_norm, 3),
            },
            "tier2_working_deltanet": {
                "mechanism": "Parallel Gated DeltaNet Fast Weights",
                "working_decay": self.working_decay,
                "delta_norm": round(self.tier2_delta_norm, 3),
                "layer_norms": [round(x, 3) for x in self.tier2_layer_norms],
            },
            "tier3_persistent_episodic": {
                "mechanism": f"TurboQuant {self.episodic_bits}-bit + QJL Content-Addressed Bank",
                "occupied_slots": len(self.slots),
                "max_slots": self.max_slots,
                "packed_bytes": packed_b,
                "raw_fp32_bytes": raw_b,
                "compression_vs_fp32": comp,
                "total_hits": self.total_hits,
                "total_overwrites": self.total_overwrites,
                "storage_path": str(self.storage_path) if self.storage_path else None,
            },
            "lmcache_hybrid": {
                "chunk_size": self.lmcache.chunk_size,
                "cached_chunks": len(self.lmcache._prefix_carries),
                "prefix_hits": self.lmcache.prefix_hits,
                "full_hits": self.lmcache.full_hits,
                "saved_prefill_tokens": self.lmcache.saved_prefill_tokens,
            },
        }

    def save(self, path: Path | str | None = None) -> Path:
        """Persist Tier-2 DeltaNet working state & Tier-3 TurboQuant episodic slots to disk."""
        target = Path(path) if path is not None else self.storage_path
        if target is None:
            raise ValueError("No storage_path specified for MemoryHierarchyV2.save()")
        target.parent.mkdir(parents=True, exist_ok=True)
        self._repack_turboquant_bank()
        assert self.packed_kv is not None
        st_tensors: dict[str, np.ndarray] = dict(self.packed_kv["tensors"])
        st_tensors["working_delta_state"] = self.working_delta_state.astype(np.float16)
        st_bytes = save_safetensors_bytes(st_tensors)
        payload = {
            "format": "mlm2_memory_v1",
            "episodic_bits": self.episodic_bits,
            "max_slots": self.max_slots,
            "working_decay": self.working_decay,
            "turn_counter": self.turn_counter,
            "total_hits": self.total_hits,
            "total_overwrites": self.total_overwrites,
            "tier2_delta_norm": self.tier2_delta_norm,
            "tier2_layer_norms": self.tier2_layer_norms,
            "packed_meta": self.packed_kv["meta"],
            "slots": self.slots,
        }
        tmp_p = target.with_suffix(".memory.tmp")
        with zipfile.ZipFile(tmp_p, "w", compression=zipfile.ZIP_STORED) as zf:
            zf.writestr("memory.json", json.dumps(payload, indent=2))
            zf.writestr("memory.safetensors", st_bytes)
        tmp_p.replace(target)
        self.storage_path = target
        return target

    def load(self, path: Path | str | None = None) -> None:
        """Load persisted Tier-2 Gated DeltaNet working state & Tier-3 TurboQuant episodic bank."""
        target = Path(path) if path is not None else self.storage_path
        if target is None or not target.exists():
            return
        with zipfile.ZipFile(target, "r") as zf:
            payload = json.loads(zf.read("memory.json"))
            st_raw = zf.read("memory.safetensors")
        tensors = load_safetensors_bytes(st_raw)
        self.episodic_bits = int(payload.get("episodic_bits", self.episodic_bits))
        self.turn_counter = int(payload.get("turn_counter", 0))
        self.total_hits = int(payload.get("total_hits", 0))
        self.total_overwrites = int(payload.get("total_overwrites", 0))
        self.tier2_delta_norm = float(payload.get("tier2_delta_norm", 0.0))
        loaded_ln = payload.get("tier2_layer_norms", [])
        if len(loaded_ln) == self.config.n_layers:
            self.tier2_layer_norms = [float(x) for x in loaded_ln]
        self.slots = list(payload.get("slots", []))[: self.max_slots]

        if (
            "working_delta_state" in tensors
            and tensors["working_delta_state"].shape == self.working_delta_state.shape
        ):
            self.working_delta_state = tensors.pop("working_delta_state").astype(np.float32)

        raw_bytes = 2 * self.max_slots * self.config.d_model * 4
        packed_bytes = sum(int(arr.nbytes) for arr in tensors.values())
        self.packed_kv = {
            "bits": self.episodic_bits,
            "method": f"turboquant_kv_{self.episodic_bits}bit_fwht_qjl",
            "tensors": tensors,
            "meta": payload["packed_meta"],
            "raw_fp32_bytes": raw_bytes,
            "raw_fp16_bytes": raw_bytes // 2,
            "packed_bytes": packed_bytes,
            "compression_vs_fp32": round(raw_bytes / max(packed_bytes, 1), 2),
            "compression_vs_fp16": round((raw_bytes / 2.0) / max(packed_bytes, 1), 2),
        }
        k_hat, v_hat = self._dequantize_turboquant_bank()
        self._k_raw[: k_hat.shape[0]] = k_hat[: self.max_slots]
        self._v_raw[: v_hat.shape[0]] = v_hat[: self.max_slots]
        self.storage_path = target
