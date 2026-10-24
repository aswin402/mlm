"""Greedy byte generation with explicit recurrent state."""

import jax
import jax.numpy as jnp
import numpy as np

from mlm.data import BOS, EOS
from mlm.model import initial_state


def generate(state, config, prompt: str, max_bytes: int = 128, messages=True) -> str:
    if not 0 <= max_bytes <= 8192:
        raise ValueError("max_bytes must be between 0 and 8192")
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
    for _ in range(max_bytes):
        # BOS cannot occur inside a document. EOS ends generation.
        scores = np.asarray(logits[0, -1]).copy()
        scores[BOS] = -np.inf
        chosen = int(scores.argmax())
        if chosen == EOS:
            break
        output.append(chosen)
        logits, carry = advance(jnp.array([[chosen]], dtype=jnp.int32), carry)
    return output.decode("utf-8", errors="replace")
