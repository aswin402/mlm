"""Audit every predefined endpoint; preserve the selected seed42 reference separately."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from mlm.checkpoint import load_checkpoint
from mlm.config import Config

OUT = Path("results/addressing-replication")


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    protocol = read(OUT / "protocol.json")
    checks = {f"input:{p}": digest(p) == h for p, h in protocol["sha256"].items()}
    checks.update(
        {f"runner:{p}": digest(p) == h for p, h in read(OUT / "runner-sha256.json").items()}
    )
    source_path = Path("runs/scratch-english-hierarchical-v1/best.mlm")
    source, _, _ = load_checkpoint(source_path)
    expected_groups = {
        q["group"] for q in read("data/addressing-v1/groups-4/train-questions.json")["questions"]
    }
    results = {}
    for seed in protocol["seeds"]:
        pair = {}
        for memory in ["none", "delta"]:
            label = f"{memory}-s{seed}"
            directory = Path(f"runs/addressing-replication-{label}-v1")
            summary = read(directory / "summary.json")
            own_protocol = read(directory / "protocol.json")
            cfg = read(directory / "config.json")
            template = read(f"configs/associative-{memory}.json")
            expected = Config.model_validate({**template, "seed": seed}).model_dump()
            checks[f"{label}:config"] = cfg == expected
            checks[f"{label}:source"] = own_protocol["source_sha256"] == digest(source_path)
            counts = Counter(summary["update_group_ids"])
            checks[f"{label}:presentations"] = len(counts) == 4 and set(counts.values()) == {500}
            checks[f"{label}:group_identity"] = (
                set(counts) == set(summary["group_ids"]) == expected_groups
            )
            checks[f"{label}:fresh_optimizer"] = summary["initial_optimizer_step"] == 0
            checks[f"{label}:budget"] = summary["optimizer_updates"] == 4000 and all(
                summary["counters"][x]["updates"] == 2000 for x in ["qa", "english"]
            )
            initial, _, _ = load_checkpoint(directory / "initial.mlm")
            checks[f"{label}:source_parameters"] = all(
                np.array_equal(np.asarray(value), np.asarray(initial.params[key]))
                for key, value in source.params.items()
            )
            if memory == "delta":
                checks[f"{label}:zero_initial_gain"] = float(initial.params["memory_gain"]) == 0
            endpoint, final_cfg, metadata = load_checkpoint(directory / "final.mlm")
            checks[f"{label}:checkpoint_step"] = int(endpoint.step) == 4000
            checks[f"{label}:checkpoint_config"] = final_cfg.model_dump() == expected
            checks[f"{label}:checkpoint_counters"] = metadata["counters"] == summary["counters"]
            final = summary["final"]
            checks[f"{label}:score_coverage"] = (
                final["training"]["questions"] == 16 and final["training"]["binding_groups"] == 4
            )
            raw = final["fixed_training_raw_samples"]
            ranking_success = (
                final["training"]["accuracy"] == 1.0
                and final["training"]["binding_group_accuracy"] == 1.0
                and final["training"]["questions"] == 16
                and final["training"]["binding_groups"] == 4
            )
            retention = final["english_development_loss"] <= protocol["retention_loss_max"]
            pair[memory] = summary
            results[label] = {
                "seed": seed,
                "memory": memory,
                "final": final,
                "acquisition_pass": ranking_success,
                "retention_pass": retention,
                "joint_pass": ranking_success and retention,
                "fixed_raw_correct": sum(
                    r["raw"].strip().casefold() == r["expected"].casefold() for r in raw
                ),
                "fixed_raw_count": len(raw),
                "group_presentations": dict(counts),
                "counters": summary["counters"],
                "source_sha256": own_protocol["source_sha256"],
                "initial_parameters_sha256": summary["initial_parameters_sha256"],
                "final_checkpoint_sha256": digest(directory / "final.mlm"),
            }
        a, b = pair["none"], pair["delta"]
        checks[f"s{seed}:qa_order"] = a["update_group_ids"] == b["update_group_ids"]
        checks[f"s{seed}:byte_budgets"] = a["counters"] == b["counters"]
        checks[f"s{seed}:english_stream"] = a["english_stream"] == b["english_stream"]
        checks[f"s{seed}:same_groups"] = a["group_ids"] == b["group_ids"]
    save(OUT / "audit.json", checks)
    if not all(checks.values()):
        raise AssertionError([key for key, ok in checks.items() if not ok])
    reference = read("results/addressing-exposure/summary.json")
    aggregate = {
        memory: {
            "acquisition_successes": sum(
                results[f"{memory}-s{seed}"]["acquisition_pass"] for seed in protocol["seeds"]
            ),
            "retention_successes": sum(
                results[f"{memory}-s{seed}"]["retention_pass"] for seed in protocol["seeds"]
            ),
            "joint_successes": sum(
                results[f"{memory}-s{seed}"]["joint_pass"] for seed in protocol["seeds"]
            ),
            "seeds": len(protocol["seeds"]),
        }
        for memory in ["none", "delta"]
    }
    result = {
        "new_trials": results,
        "new_seed_aggregate": aggregate,
        "predefined_delta_replication_gate_pass": aggregate["delta"]["acquisition_successes"] == 2,
        "selected_seed42_reference": {m: v["four_group_2000"] for m, v in reference.items()},
        "scope": "fixed-source adaptation-seed replication; no new test or architecture change",
    }
    save(OUT / "summary.json", result)
    print(json.dumps(aggregate, indent=2))
    print("All frozen input and endpoint audits passed")


if __name__ == "__main__":
    main()
