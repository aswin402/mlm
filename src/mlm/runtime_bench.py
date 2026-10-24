"""Warmed CPU forward timing and explicit tensor sizes; not generation or sparse benchmarks."""

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from mlm.checkpoint import load_checkpoint
from mlm.cli import environment
from mlm.data import BOS
from mlm.model import initial_state


def benchmark(checkpoint, out, lengths=(32, 128, 512), repeats=30, warmups=5):
    if out.exists():
        raise ValueError("Choose a fresh benchmark output")
    if repeats < 1 or warmups < 1 or any(n < 1 for n in lengths):
        raise ValueError("Positive benchmark dimensions required")
    state, cfg, _ = load_checkpoint(checkpoint)

    def tensor_bytes(tree):
        return sum(np.asarray(v).nbytes for v in jax.tree.leaves(tree))

    compiled = jax.jit(state.apply_fn)
    rng = np.random.default_rng(42)
    results = []
    for length in lengths:
        tokens = jnp.array(rng.integers(0, 256, size=(1, length), dtype=np.int32)).at[:, 0].set(BOS)
        resets = jnp.zeros_like(tokens, dtype=bool).at[:, 0].set(True)
        hidden = initial_state(cfg, 1)
        tick = time.perf_counter()
        jax.block_until_ready(compiled({"params": state.params}, tokens, hidden, resets))
        compilation = time.perf_counter() - tick
        for _ in range(warmups):
            jax.block_until_ready(compiled({"params": state.params}, tokens, hidden, resets))
        times = []
        for _ in range(repeats):
            tick = time.perf_counter_ns()
            jax.block_until_ready(compiled({"params": state.params}, tokens, hidden, resets))
            times.append((time.perf_counter_ns() - tick) / 1e6)
        results.append(
            {
                "length": length,
                "input_sha256": hashlib.sha256(np.asarray(tokens).tobytes()).hexdigest(),
                "compile_and_first_forward_seconds": compilation,
                "p50_ms": float(np.median(times)),
                "p95_ms": float(np.percentile(times, 95)),
                "samples_ms": times,
            }
        )
    result = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "node_update": cfg.node_update,
        "batch_size": 1,
        "repeats": repeats,
        "warmups": warmups,
        "scope": "warmed full forward; same fixed inputs; excludes text I/O and generation loop",
        "memory_limits": (
            "peak RSS includes checkpoint optimizer loading, JAX and compilation; "
            "not isolated model inference RAM"
        ),
        "parameter_bytes": tensor_bytes(state.params),
        "optimizer_tensor_bytes": tensor_bytes(state.opt_state),
        "persistent_state_bytes_per_sequence": tensor_bytes(initial_state(cfg, 1)),
        "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "environment": environment(),
        "results": results,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "node_update",
                    "parameter_bytes",
                    "persistent_state_bytes_per_sequence",
                    "peak_process_rss_kib",
                ]
            },
            indent=2,
        )
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    benchmark(args.checkpoint, args.out)


if __name__ == "__main__":
    main()
