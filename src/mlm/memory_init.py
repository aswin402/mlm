"""Offline chrono-inspired retention-bias intervention; no inference rules."""

import argparse
import hashlib
import json

import jax.numpy as jnp
import numpy as np

from mlm.checkpoint import load_checkpoint, save_checkpoint
from mlm.training import create_state


def retention_parameters(params, config, maximum=128.0):
    if not np.isfinite(maximum) or maximum < 2:
        raise ValueError("Maximum nominal timescale must be finite and at least two")
    count = config.nodes if config.architecture == "graph" else config.width
    times = np.geomspace(2.0, maximum, count)
    offsets = jnp.asarray(np.log(times - 1), dtype=params["gru_bias"].dtype)
    if config.architecture == "graph":
        offsets = offsets[:, None]
    updated = dict(params)
    updated["gru_bias"] = params["gru_bias"].at[..., : config.width].add(offsets)
    return updated, times


def prepare(source, out, maximum=128.0):
    if out.exists() or out.with_suffix(".json").exists():
        raise ValueError("Choose a fresh initialization checkpoint path")
    state, config, meta = load_checkpoint(source)
    params, times = retention_parameters(state.params, config, maximum)
    fresh = create_state(config).replace(params=params)
    provenance = {
        "intervention": "add log(tau-1) to pretrained retention-gate bias only",
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_step": int(state.step),
        "messages": meta.get("messages", True),
        "nominal_timescales": times.tolist(),
        "optimizer": "fresh; step zero; no source stream state",
        "reference": "https://arxiv.org/abs/1804.11188",
    }
    save_checkpoint(out, fresh, config, provenance)
    out.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def main():
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--maximum", type=float, default=128.0)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.out, args.maximum), indent=2))


if __name__ == "__main__":
    main()
