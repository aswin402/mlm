"""Exploratory frozen context sensitivity, not a semantic-memory benchmark."""

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from mlm.checkpoint import load_checkpoint
from mlm.comprehension import score_candidates
from mlm.data import BOS
from mlm.model import initial_state


def probe(checkpoint):
    state, config, meta = load_checkpoint(checkpoint)
    results = []
    for repeats in [0, 2, 8, 16]:
        filler = " It is a calm day." * repeats
        prompts = [
            f"<|user|>\nAva owns a {color} hat.{filler}\nWhat color is Ava's hat?\n<|assistant|>\n"
            for color in ["green", "white"]
        ]
        ids = jnp.asarray([[BOS, *p.encode()] for p in prompts], dtype=jnp.int32)
        resets = jnp.zeros_like(ids, dtype=bool).at[:, 0].set(True)
        _, carry = jax.jit(state.apply_fn, static_argnames=("messages",))(
            {"params": state.params},
            ids,
            initial_state(config, 2),
            resets,
            messages=meta.get("messages", True),
        )
        carry = np.asarray(carry)
        relative = np.linalg.norm(carry[0] - carry[1]) / max(
            np.linalg.norm(carry[0]), np.linalg.norm(carry[1]), 1e-12
        )
        scores = score_candidates(
            state,
            config,
            [p for p in prompts for _ in range(2)],
            ["green", "white"] * 2,
            messages=meta.get("messages", True),
        )
        margin = np.asarray(scores).reshape(2, 2)
        effect = (margin[0, 0] - margin[0, 1]) - (margin[1, 0] - margin[1, 1])
        results.append(
            {
                "filler_bytes": len(filler.encode()),
                "relative_final_state_difference": float(relative),
                "context_effect_nats_per_answer_byte": float(effect),
                "scores": margin.tolist(),
            }
        )
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "scope": "Equal-length green/white substitution; no training or selection",
        "limits": (
            "State differences and tiny score changes do not establish retrievable "
            "semantic memory or prove a cause of failure."
        ),
        "probes": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = probe(args.checkpoint)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
