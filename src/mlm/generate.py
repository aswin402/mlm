"""Greedy byte generation with explicit recurrent state."""

import jax
import jax.numpy as jnp
import numpy as np

from mlm.data import BOS, EOS
from mlm.model import initial_state


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
