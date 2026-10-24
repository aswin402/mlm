"""Compare the preserved old driver and current default on the actual task/source."""

import contextlib
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np

from mlm.checkpoint import load_checkpoint
from mlm.plasticity_run import run

OUT = Path("results/schedule-components")


def main():
    legacy_path = OUT / "legacy/src/mlm/plasticity_run.py"
    spec = importlib.util.spec_from_file_location("legacy_plasticity", legacy_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load preserved legacy driver")
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    common = {
        "data": Path("data/addressing-v1/groups-4"),
        "config": OUT / "config.json",
        "init": Path("runs/addressing-exposure-delta-g4-v1/initial.mlm"),
        "groups": 4,
        "steps": 8,
        "rehearsal": True,
        "english_train": Path("data/hf/stories/train.jsonl"),
        "english_dev": Path("data/hf/stories/validation.jsonl"),
    }
    summaries = {}
    states = {}
    metadata = {}
    with (OUT / "compatibility.log").open("x") as log, contextlib.redirect_stdout(log):
        for label, driver in [("legacy", legacy.run), ("modern", run)]:
            args = SimpleNamespace(**common, out=OUT / "compatibility" / label)
            summaries[label] = driver(args)
            states[label], _, metadata[label] = load_checkpoint(args.out / "final.mlm")
    old, new = summaries["legacy"], summaries["modern"]
    checks = {
        key: old[key] == new[key]
        for key in [
            "initial_parameters_sha256",
            "initial_optimizer_step",
            "group_ids",
            "update_group_ids",
            "counters",
            "final",
            "initial",
            "english_stream",
            "parameter_delta_l2_by_role",
        ]
    }
    checks["all_params_and_moments_bitwise_equal"] = all(
        np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(
            jax.tree.leaves((states["legacy"].params, states["legacy"].opt_state)),
            jax.tree.leaves((states["modern"].params, states["modern"].opt_state)),
            strict=True,
        )
    )
    checks["english_carry_equal"] = (
        metadata["legacy"]["english_carry"] == metadata["modern"]["english_carry"]
    )
    checks["actual_final_step"] = int(states["legacy"].step) == int(states["modern"].step) == 16
    checks["new_seed_policy"] = new["sampling_seeds"] == {"subset": 42, "qa": 42, "english": 43}
    source = common["init"]
    result = {
        "checks": checks,
        "budget_per_driver": {"qa": 8, "english": 8},
        "legacy_driver_sha256": hashlib.sha256(legacy_path.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "limit": (
            "Eight QA/replay pairs verify default compatibility, not full endpoint replication."
        ),
    }
    (OUT / "compatibility.json").write_text(json.dumps(result, indent=2) + "\n")
    if not all(checks.values()):
        raise AssertionError([k for k, v in checks.items() if not v])
    print(f"All {len(checks)} legacy-default compatibility checks passed")


if __name__ == "__main__":
    main()
