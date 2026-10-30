"""Greedy byte generation with explicit recurrent state."""

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


class LiveLearner:
    """Online continual byte-stream learner during live interaction (mini-AGI minagi/live.py)."""

    def __init__(
        self,
        state,
        config,
        chunk: int = 16,
        context: int | None = None,
        messages: bool = True,
    ):
        if chunk < 2:
            raise ValueError("chunk must be at least 2")
        self.state = state
        self.config = config
        self.chunk = int(chunk)
        self.context = int(context or config.length)
        self.messages = messages
        self.buf: list[int] = [BOS]
        self.pending = 0
        self.steps = 0
        self.carry = initial_state(config, 1)
        self.step_fn = make_stream_step(config, messages=messages)
        self.log: list[dict] = []

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
    ) -> tuple[str, list[dict]]:
        prompt = f"<|user|>\n{user_message}\n<|assistant|>\n"
        reply = generate(
            self.state,
            self.config,
            prompt,
            max_bytes=max_bytes,
            messages=self.messages,
            adapt_strength=adapt_strength,
            adapt_decay=adapt_decay,
        )
        steps = self.feed(f"{prompt}{reply}\n") if learn else []
        return reply, steps
