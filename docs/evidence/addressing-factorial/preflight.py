"""Verify saved initializers and unchanged warm_start before crossing schedules."""

import hashlib
import json
from pathlib import Path

import jax
import numpy as np
from flax import serialization

from mlm.binding_run import warm_start
from mlm.checkpoint import load_checkpoint
from mlm.config import Config
from mlm.training import create_state

OUT = Path("results/addressing-factorial")


def main():
    commands = json.loads((OUT / "commands.json").read_text())
    english, _, _ = load_checkpoint(Path("runs/scratch-english-hierarchical-v1/best.mlm"))
    checks = {}
    sources = {}
    fingerprints = {}
    for trial in commands:
        init, schedule = trial["initialization_seed"], trial["schedule_seed"]
        label = f"i{init}-s{schedule}"
        source, source_cfg, _ = load_checkpoint(Path(trial["source"]))
        cfg = Config.model_validate_json(Path(trial["config"]).read_text())
        checks[label + ":source_step_zero"] = int(source.step) == 0
        checks[label + ":source_seed"] = source_cfg.seed == init
        checks[label + ":target_seed"] = cfg.seed == schedule
        checks[label + ":zero_gate"] = float(source.params["memory_gain"]) == 0
        checks[label + ":config_except_seed"] = source_cfg.model_dump(
            exclude={"seed"}
        ) == cfg.model_dump(exclude={"seed"})
        checks[label + ":english_tensors"] = all(
            np.array_equal(np.asarray(v), np.asarray(source.params[k]))
            for k, v in english.params.items()
        )
        fresh = warm_start(Path(trial["source"]), cfg)
        checks[label + ":all_initializer_tensors_preserved"] = set(fresh.params) == set(
            source.params
        ) and all(
            np.array_equal(np.asarray(v), np.asarray(fresh.params[k]))
            for k, v in source.params.items()
        )
        checks[label + ":fresh_optimizer"] = int(fresh.step) == 0 and all(
            np.array_equal(np.asarray(a), np.asarray(b))
            for a, b in zip(
                jax.tree.leaves(fresh.opt_state),
                jax.tree.leaves(create_state(cfg).opt_state),
                strict=True,
            )
        )
        sources[init] = source
        fingerprints[label] = hashlib.sha256(serialization.to_bytes(fresh.params)).hexdigest()
    checks["same_parameter_names"] = set(sources[42].params) == set(sources[43].params)
    memory_names = set(sources[42].params) - set(english.params)
    checks["only_memory_parameters_added"] = memory_names == {
        "memory_key_kernel",
        "memory_query_kernel",
        "memory_value_kernel",
        "memory_write_kernel",
        "memory_write_bias",
        "memory_output_kernel",
        "memory_gain",
    }
    checks["memory_initializations_differ"] = any(
        not np.array_equal(np.asarray(sources[42].params[k]), np.asarray(sources[43].params[k]))
        for k in memory_names
    )
    result = {"checks": checks, "initial_parameters_sha256": fingerprints}
    (OUT / "preflight.json").write_text(json.dumps(result, indent=2) + "\n")
    if not all(checks.values()):
        raise AssertionError([k for k, ok in checks.items() if not ok])
    print(f"All {len(checks)} initializer/optimizer checks passed")


if __name__ == "__main__":
    main()
