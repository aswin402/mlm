"""Frozen two-arm training-only feasibility pilot for a disposable future-byte head."""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from mlm.checkpoint import load_checkpoint, save_checkpoint
from mlm.cli import write_json
from mlm.future_byte import create_head, future_mask, future_objective, make_future_step
from mlm.interference_run import SOURCES, digest, fingerprints, state_digest
from mlm.model import initial_state
from mlm.pair_probe_run import execution_settings, prepare, read
from mlm.stream import DocumentStream

PRIOR = Path("docs/evidence/qa-update-scale/protocol.json")
METHOD = Path("docs/experiments/future-byte/protocol.md")
PAIRS = 128
ARMS = {"control": 0.0, "future_byte": 0.1}


def intact(p):
    return {name: digest(name) == value for name, value in p["sha256"].items()}


def schedule(cfg, documents, groups):
    rng = np.random.default_rng(42)
    order = []
    for _ in range(PAIRS // groups):
        order.extend(map(int, rng.permutation(groups)))
    stream = DocumentStream(documents, cfg.batch_size, cfg.length, 43)
    english = [stream.next_batch() for _ in range(PAIRS)]
    return order, english, stream.snapshot()


def expected_counts(batch):
    valid = np.asarray(batch["mask"])
    supervised = valid * np.asarray(batch.get("loss_mask", batch["mask"]))
    targets = np.asarray(batch["targets"])
    auxiliary = (
        supervised[:, :-1]
        * supervised[:, 1:]
        * (targets[:, :-1] < 256)
        * (targets[:, 1:] < 256)
        * (~np.asarray(batch["resets"])[:, 1:])
    )
    return {
        "updates": 1,
        "positions": int(supervised.sum()),
        "aux_positions": int(auxiliary.sum()),
        "input_bytes": int((valid * (targets < 256)).sum()),
        "supervised_bytes": int((supervised * (targets < 256)).sum()),
    }


def exposure(batches, order, english):
    totals = {}
    for phase, sequence in [("qa", [batches[i] for i in order]), ("english", english)]:
        counts = [expected_counts(b) for b in sequence]
        totals[phase] = {key: sum(c[key] for c in counts) for key in counts[0]}
    return totals


def freeze(out):
    if out.exists():
        raise ValueError("Use fresh output directory")
    prior = read(PRIOR)
    if not all(intact(prior).values()):
        raise ValueError("Prior inputs changed")
    d = prior["checkpoints"]["initial"]
    state, cfg, meta = load_checkpoint(Path(d["path"]))
    if (
        d["path"] != SOURCES["initial"]
        or int(state.step) != 0
        or state_digest(state) != d["state_sha256"]
        or digest(d["path"]) != d["file_sha256"]
    ):
        raise ValueError("Expected unchanged common step0 source")
    batches, ids, qa, documents = prepare(cfg)
    if len(documents) != 1000 or len(ids) != 4:
        raise ValueError("Fixed training corpus expected")
    order, english, snapshot = schedule(cfg, documents, len(ids))
    paths = [
        *map(Path, prior["sha256"]),
        PRIOR,
        METHOD,
        *Path("src/mlm").glob("*.py"),
        *Path("tests").glob("*.py"),
    ]
    p = {
        "mode": "future_byte_feasibility",
        "source": d,
        "pairs": PAIRS,
        "arms": ARMS,
        "head_seed": 9042,
        "expected_exposure": exposure(batches, order, english),
        "rank": 16,
        "qa_seed": 42,
        "english_seed": 43,
        "qa_groups": ids,
        "qa_trace": [ids[i] for i in order],
        "qa_evaluation": fingerprints(qa),
        "english_batches": [fingerprints(b) for b in english],
        "english_snapshot": snapshot,
        "settings": execution_settings(),
        "sha256": {str(f): digest(f) for f in dict.fromkeys(paths)},
    }
    out.mkdir(parents=True)
    write_json(out / "protocol.json", p)
    print(
        json.dumps({"phase": "frozen", "inputs": len(p["sha256"]), "backbone_updates": 512}),
        flush=True,
    )


def serialized_values_equal(left, right):
    if jax.tree.structure(left) != jax.tree.structure(right):
        return False
    return all(
        np.asarray(a).dtype == np.asarray(b).dtype and np.array_equal(a, b)
        for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)
    )


def amend(out):
    original = read(out / "protocol.json")
    driver = "src/mlm/future_byte_run.py"
    if (out / "continuation-protocol.json").exists() or (out / "future_byte").exists():
        raise ValueError("Preserve pilot outputs")
    if digest(out / "original-future_byte_run.py") != original["sha256"][driver]:
        raise ValueError("Original driver not preserved")
    if not all(digest(f) == h for f, h in original["sha256"].items() if f != driver):
        raise ValueError("Non-audit input changed")
    now = execution_settings()
    old_settings = json.loads(json.dumps(original["settings"]))
    new_settings = json.loads(json.dumps(now))
    old_settings["environment"].pop("source_sha256")
    new_settings["environment"].pop("source_sha256")
    if old_settings != new_settings:
        raise ValueError("Numerical environment changed")
    completed = [
        out / "control" / f for f in ["summary.json", "history.json", "final.mlm", "head.msgpack"]
    ]
    if not all(f.exists() for f in completed):
        raise ValueError("Completed control required")
    paths = [
        *map(Path, original["sha256"]),
        out / "protocol.json",
        out / "execution.json",
        out / "original-future_byte_run.py",
        Path("tests/test_future_byte_audit.py"),
        *completed,
    ]
    revised = {
        **original,
        "settings": now,
        "sha256": {str(f): digest(f) for f in paths},
        "continuation": {
            "reason": "Audit-only head mapping order; helper and schedule unchanged",
            "original_protocol_sha256": digest(out / "protocol.json"),
            "old_driver_sha256": original["sha256"][driver],
            "new_driver_sha256": digest(driver),
            "completed_control": {str(f): digest(f) for f in completed},
        },
    }
    write_json(out / "continuation-protocol.json", revised)
    print(
        json.dumps(
            {
                "phase": "audit_amendment_frozen",
                "inputs": len(revised["sha256"]),
                "control_retrained": False,
            }
        ),
        flush=True,
    )


def completed_control(out, p, source, head0, scores):
    for f, h in p["continuation"]["completed_control"].items():
        if digest(f) != h:
            raise ValueError("Completed control changed")
    directory = out / "control"
    r = read(directory / "summary.json")
    loaded, cfg, meta = load_checkpoint(directory / "final.mlm")
    data = (directory / "head.msgpack").read_bytes()
    head = serialization.from_bytes(head0, data)
    history = read(directory / "history.json")

    def same_metrics(a, b):
        return set(a) == set(b) and all(
            set(a[k]) == set(b[k]) and all(abs(a[k][j] - b[k][j]) <= 1e-6 for j in a[k]) for k in a
        )

    valid = (
        r["weight"] == 0
        and r["counts"] == p["expected_exposure"]
        and r["visits"] == {k: 32 for k in p["qa_groups"]}
        and int(loaded.step) == int(head.step) == 256
        and cfg.model_dump() == p["source"]["config"]
        and r["final_backbone_sha256"] == digest(directory / "final.mlm")
        and r["final_state_sha256"] == state_digest(loaded)
        and r["head_sha256"] == digest(directory / "head.msgpack")
        and meta["counts"] == r["counts"]
        and meta["qa_trace"] == p["qa_trace"]
        and meta["english_stream"] == p["english_snapshot"]
        and meta["auxiliary_weight"] == 0
        and meta["joint_resume_supported"] is False
        and meta["auxiliary_state_sha256"] == r["head_sha256"]
        and np.asarray(meta["english_carry"]).shape == initial_state(cfg, cfg.batch_size).shape
        and np.isfinite(meta["english_carry"]).all()
        and len(history) == 256
        and [(v["pair"], v["phase"]) for v in history]
        == [(i, phase) for i in range(PAIRS) for phase in ["qa", "english"]]
        and all(np.isfinite(v[k]) for v in history for k in ["loss", "aux_loss", "objective"])
        and r["initial"] == scores(source, head0)
        and same_metrics(r["final"], scores(loaded, head))
        and r["backbone_parameters"] == sum(x.size for x in jax.tree.leaves(loaded.params))
        and r["auxiliary_parameters"] == sum(x.size for x in jax.tree.leaves(head.params))
        and serialized_values_equal(
            serialization.msgpack_restore(data), serialization.to_state_dict(head)
        )
    )
    if not valid:
        raise ValueError("Completed control validation failed")
    return r


def run(out, continuing=False):
    protocol_path = out / ("continuation-protocol.json" if continuing else "protocol.json")
    execution_path = out / ("continuation-execution.json" if continuing else "execution.json")
    p = read(protocol_path)
    if (
        execution_path.exists()
        or (out / "future_byte").exists()
        or (out / "summary.json").exists()
        or (out / "audit.json").exists()
        or (not continuing and (out / "control").exists())
    ):
        raise ValueError("Preserve existing outputs")
    if (
        continuing
        and digest(out / "protocol.json") != p["continuation"]["original_protocol_sha256"]
    ):
        raise ValueError("Original protocol changed")
    if continuing:
        original = read(out / "protocol.json")
        driver = "src/mlm/future_byte_run.py"
        allowed = {
            "protocol.json",
            "execution.json",
            "original-future_byte_run.py",
            "control",
            "continuation-protocol.json",
        }
        if {f.name for f in out.iterdir()} - allowed:
            raise ValueError("Unexpected partial output")
        if {k: v for k, v in p.items() if k not in {"settings", "sha256", "continuation"}} != {
            k: v for k, v in original.items() if k not in {"settings", "sha256"}
        }:
            raise ValueError("Original experimental design changed")
        if not all(
            p["sha256"].get(f) == h and digest(f) == h
            for f, h in original["sha256"].items()
            if f != driver
        ):
            raise ValueError("Non-audit original input changed")
        c = p["continuation"]
        if (
            c["old_driver_sha256"] != original["sha256"][driver]
            or c["new_driver_sha256"] != digest(driver)
            or digest(out / "original-future_byte_run.py") != original["sha256"][driver]
        ):
            raise ValueError("Audit-only driver exception mismatch")
        oe = read(out / "execution.json")
        if (
            oe["protocol_sha256"] != digest(out / "protocol.json")
            or oe["settings"] != original["settings"]
        ):
            raise ValueError("Original execution changed")
        old = json.loads(json.dumps(original["settings"]))
        new = json.loads(json.dumps(p["settings"]))
        old["environment"].pop("source_sha256")
        new["environment"].pop("source_sha256")
        if old != new:
            raise ValueError("Original numerical environment changed")
    prior = read(PRIOR)
    if not all(intact(p).values()) or p["settings"] != execution_settings():
        raise ValueError("Frozen inputs/settings changed")
    if (
        p["source"] != prior["checkpoints"]["initial"]
        or p["arms"] != ARMS
        or p["pairs"] != PAIRS
        or p["rank"] != 16
        or p["head_seed"] != 9042
        or p["qa_seed"] != 42
        or p["english_seed"] != 43
    ):
        raise ValueError("Frozen design changed")
    phash = digest(protocol_path)
    write_json(execution_path, {"protocol_sha256": phash, "settings": execution_settings()})
    d = p["source"]
    source, cfg, meta = load_checkpoint(Path(d["path"]))
    batches, ids, qa, documents = prepare(cfg)
    order, english, snapshot = schedule(cfg, documents, len(ids))
    qa = jax.tree.map(jnp.asarray, qa)
    fixed_english = jax.tree.map(jnp.asarray, english[0])
    head0 = create_head(cfg)
    initial_backbone = state_digest(source)
    initial_head = state_digest(head0)
    audit = {
        "source_identity": initial_backbone == d["state_sha256"]
        and digest(d["path"]) == d["file_sha256"],
        "prior_inputs": all(p["sha256"].get(k) == v for k, v in prior["sha256"].items()),
        "qa_evaluation": fingerprints(qa) == p["qa_evaluation"],
        "qa_selection": ids == p["qa_groups"] and [ids[i] for i in order] == p["qa_trace"],
        "english_selection": [fingerprints(b) for b in english] == p["english_batches"]
        and snapshot == p["english_snapshot"],
        "qa_positions": int((qa["mask"] * qa["loss_mask"]).sum()) == 96,
        "source_step": int(source.step) == 0,
        "expected_exposure": p["expected_exposure"] == exposure(batches, order, english),
    }
    if not all(audit.values()):
        raise ValueError("Pilot preflight failed")

    @jax.jit
    def evaluate(state, head, batch):
        _, (_, m) = future_objective(
            state.params,
            head.params,
            state.apply_fn,
            initial_state(cfg, batch["inputs"].shape[0]),
            batch,
            cfg,
            0.0,
        )
        return {
            k: m[k] for k in ["loss", "main_objective", "aux_loss", "positions", "aux_positions"]
        }

    def scores(state, head):
        values = {
            "qa": evaluate(state, head, qa),
            "english_train_probe": evaluate(state, head, fixed_english),
        }
        if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(values)):
            raise FloatingPointError("Nonfinite pilot score")
        return {name: {k: float(v) for k, v in m.items()} for name, m in values.items()}

    results: dict[str, Any] = {}
    for name, weight in ARMS.items():
        if continuing and name == "control":
            results[name] = completed_control(out, p, source, head0, scores)
            audit["control:reused_completed_without_retraining"] = True
            print(
                json.dumps(
                    {
                        "arm": "control",
                        "status": "validated_completed_artifacts",
                        "retrained": False,
                    }
                ),
                flush=True,
            )
            continue
        state, head = source, head0
        step = make_future_step(cfg, weight)
        before = scores(state, head)
        carry = initial_state(cfg, cfg.batch_size)
        stream = DocumentStream(documents, cfg.batch_size, cfg.length, 43)
        visits = {identity: 0 for identity in ids}
        counts = {
            phase: {
                "updates": 0,
                "positions": 0,
                "aux_positions": 0,
                "input_bytes": 0,
                "supervised_bytes": 0,
            }
            for phase in ["qa", "english"]
        }
        history = []
        started = time.perf_counter()
        for index, group in enumerate(order):
            eb = stream.next_batch()
            if fingerprints(eb) != p["english_batches"][index]:
                raise ValueError("Replay batch changed")
            for phase, batch in [("qa", batches[group]), ("english", eb)]:
                batch = jax.tree.map(jnp.asarray, batch)
                incoming = initial_state(cfg, cfg.batch_size) if phase == "qa" else carry
                state, head, final, m = step(state, head, incoming, batch)
                if not bool(m["finite"]) or not bool(m["updated"]):
                    raise FloatingPointError("Rejected pilot update")
                expected_aux = float(np.asarray(future_mask(batch)).sum())
                if float(m["aux_positions"]) != expected_aux:
                    raise ValueError("Auxiliary mask count mismatch")
                if phase == "english":
                    carry = final
                c = counts[phase]
                c["updates"] += 1
                c["positions"] += int(m["positions"])
                c["aux_positions"] += int(m["aux_positions"])
                c["input_bytes"] += int(m["consumed_bytes"])
                c["supervised_bytes"] += int(m["bytes"])
                history.append(
                    {
                        "pair": index,
                        "phase": phase,
                        "loss": float(m["loss"]),
                        "aux_loss": float(m["aux_loss"]),
                        "objective": float(m["objective"]),
                    }
                )
            visits[ids[group]] += 1
            if (index + 1) % 32 == 0:
                print(
                    json.dumps({"arm": name, "pairs": index + 1, "step": int(state.step)}),
                    flush=True,
                )
        elapsed = time.perf_counter() - started
        after = scores(state, head)
        directory = out / name
        directory.mkdir()
        head_bytes = serialization.to_bytes(head)
        (directory / "head.msgpack").write_bytes(head_bytes)
        metadata = {
            "mode": "future_byte_backbone_only",
            "auxiliary_weight": weight,
            "auxiliary_state_sha256": digest(directory / "head.msgpack"),
            "joint_resume_supported": False,
            "counts": counts,
            "qa_trace": p["qa_trace"],
            "english_stream": stream.snapshot(),
            "english_carry": np.asarray(carry).tolist(),
            "messages": True,
        }
        save_checkpoint(directory / "final.mlm", state, cfg, metadata)
        loaded, loaded_cfg, loaded_meta = load_checkpoint(directory / "final.mlm")
        audit[name + ":checkpoint_roundtrip"] = (
            state_digest(loaded) == state_digest(state)
            and loaded_cfg == cfg
            and loaded_meta == metadata
        )
        audit[name + ":head_roundtrip"] = serialized_values_equal(
            serialization.msgpack_restore(head_bytes),
            serialization.to_state_dict(serialization.from_bytes(head0, head_bytes)),
        )
        audit[name + ":budget"] = int(state.step) == int(head.step) == 2 * PAIRS and all(
            c["updates"] == PAIRS for c in counts.values()
        )
        audit[name + ":visits"] = (
            all(v == 32 for v in visits.values()) and sum(visits.values()) == PAIRS
        )
        audit[name + ":stream"] = stream.snapshot() == p["english_snapshot"]
        audit[name + ":exposure"] = counts == p["expected_exposure"]
        audit[name + ":source_immutable"] = (
            state_digest(source) == initial_backbone and state_digest(head0) == initial_head
        )
        results[name] = {
            "weight": weight,
            "initial": before,
            "final": after,
            "counts": counts,
            "visits": visits,
            "backbone_parameters": sum(x.size for x in jax.tree.leaves(state.params)),
            "auxiliary_parameters": sum(x.size for x in jax.tree.leaves(head.params)),
            "elapsed_training_seconds_including_compile": elapsed,
            "final_backbone_sha256": digest(directory / "final.mlm"),
            "final_state_sha256": state_digest(state),
            "head_sha256": digest(directory / "head.msgpack"),
        }
        write_json(directory / "history.json", history)
        write_json(directory / "summary.json", results[name])
        if not all(audit.values()):
            raise ValueError("Arm audit failed")
    audit["matched_initial"] = results["control"]["initial"] == results["future_byte"]["initial"]
    audit["matched_exposure"] = results["control"]["counts"] == results["future_byte"]["counts"]
    audit["matched_visits"] = results["control"]["visits"] == results["future_byte"]["visits"]
    audit["all_arms"] = set(results) == set(ARMS)
    audit["source_file_immutable"] = digest(d["path"]) == d["file_sha256"]
    audit["protocol_immutable"] = phash == digest(protocol_path)
    audit["settings_immutable"] = execution_settings() == p["settings"]
    audit.update({"input:" + k: v for k, v in intact(p).items()})
    if not all(audit.values()):
        raise ValueError("Final audit failed: " + str([k for k, v in audit.items() if not v]))
    write_json(out / "summary.json", {"arms": results, "backbone_updates": 512})
    write_json(out / "audit.json", audit)
    print(
        json.dumps(
            {
                "phase": "complete",
                "audits": len(audit),
                "final_main_losses": {
                    n: {k: v["loss"] for k, v in r["final"].items()} for n, r in results.items()
                },
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["freeze", "run", "amend", "continue"])
    parser.add_argument("--out", type=Path, default=Path("results/future-byte"))
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze(args.out)
    elif args.phase == "amend":
        amend(args.out)
    else:
        run(args.out, continuing=args.phase == "continue")


if __name__ == "__main__":
    main()
