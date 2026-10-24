"""Audit fixed endpoints and compare with the prior frozen short probes."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from mlm.checkpoint import load_checkpoint

OUT = Path("results/addressing-exposure")


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    protocol = read(OUT / "protocol.json")
    checks = {f"input:{p}": digest(p) == h for p, h in protocol["sha256"].items()}
    summaries, initials, results = {}, {}, {}
    for memory in ["none", "delta"]:
        name = f"addressing-exposure-{memory}-g4-v1"
        directory = Path("runs") / name
        summary = read(directory / "summary.json")
        old = read(f"runs/addressing-{memory}-g4-v1/summary.json")
        single = read(f"runs/addressing-{memory}-g1-v1/summary.json")
        initial, _, _ = load_checkpoint(directory / "initial.mlm")
        initials[memory] = initial
        summaries[memory] = summary
        counts = Counter(summary["update_group_ids"])
        checks[f"{memory}:exact_exposure"] = len(counts) == 4 and set(counts.values()) == {500}
        checks[f"{memory}:fresh_optimizer"] = summary["initial_optimizer_step"] == 0
        checks[f"{memory}:exact_budget"] = summary["optimizer_updates"] == 4000 and all(
            summary["counters"][x]["updates"] == 2000 for x in ["qa", "english"]
        )
        checks[f"{memory}:same_initial_as_short"] = (
            summary["initial_parameters_sha256"] == old["initial_parameters_sha256"]
        )
        checks[f"{memory}:same_sampler_prefix"] = (
            summary["update_group_ids"][:500] == old["update_group_ids"]
        )
        checks[f"{memory}:same_rate_policy"] = summary["rate_policy"] == old["rate_policy"]
        checks[f"{memory}:same_initial_assessment"] = summary["initial"] == old["initial"]
        results[memory] = {
            "prior_one_group_500": single["final"],
            "prior_four_group_500": old["final"],
            "four_group_2000": summary["final"],
            "group_presentations": dict(counts),
            "counters": summary["counters"],
            "final_checkpoint_sha256": digest(directory / "final.mlm"),
        }
    checks["shared_initial_params_equal"] = all(
        np.array_equal(np.asarray(value), np.asarray(initials["delta"].params[key]))
        for key, value in initials["none"].params.items()
    )
    checks["delta_initial_gain_zero"] = float(initials["delta"].params["memory_gain"]) == 0
    checks["matched_qa_order"] = (
        summaries["none"]["update_group_ids"] == summaries["delta"]["update_group_ids"]
    )
    checks["matched_byte_budgets"] = summaries["none"]["counters"] == summaries["delta"]["counters"]
    checks["matched_english_stream"] = (
        summaries["none"]["english_stream"] == summaries["delta"]["english_stream"]
    )
    save(OUT / "audit.json", checks)
    if not all(checks.values()):
        raise AssertionError([key for key, ok in checks.items() if not ok])
    save(OUT / "summary.json", results)
    print(json.dumps({k: v["four_group_2000"] for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
