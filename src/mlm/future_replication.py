"""Explicit sampling schedules and isolated training arms for future-byte replication."""

import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from mlm.checkpoint import load_checkpoint, save_checkpoint
from mlm.cli import write_json
from mlm.future_byte import future_objective, make_future_step
from mlm.future_byte_run import expected_counts, exposure, serialized_values_equal
from mlm.interference_run import digest, fingerprints, state_digest
from mlm.model import initial_state
from mlm.stream import DocumentStream


def seeded_schedule(config, documents, groups, qa_seed, english_seed, pairs):
    if any(type(s) is not int or not 0 <= s < 2**32 for s in [qa_seed, english_seed]):
        raise ValueError("Integer sampler seeds in [0,2**32) required")
    if (
        type(groups) is not int
        or groups < 1
        or type(pairs) is not int
        or pairs < 1
        or pairs % groups
    ):
        raise ValueError("Positive budget divisible by groups required")
    rng = np.random.default_rng(qa_seed)
    order = []
    for _ in range(pairs // groups):
        order.extend(map(int, rng.permutation(groups)))
    stream = DocumentStream(documents, config.batch_size, config.length, english_seed)
    english = [stream.next_batch() for _ in range(pairs)]
    if any(b is None for b in english):
        raise ValueError("Nonempty repeatable English corpus required")
    return order, english, stream.snapshot()


def aggregate_gates(comparisons):
    if set(comparisons) != {"43", "44"}:
        raise ValueError("Exactly new comparison seeds43 and44 required")
    if not all(
        math.isfinite(v[k]) for v in comparisons.values() for k in ["qa_delta", "english_delta"]
    ):
        raise ValueError("Finite comparisons required")
    q = sum(v["qa_delta"] < -1e-6 for v in comparisons.values())
    e = sum(v["english_delta"] <= 1e-6 for v in comparisons.values())
    return {
        "primary_qa": q == 2,
        "secondary_english": e == 2,
        "joint": q == 2 and e == 2,
        "qa_passes": q,
        "english_passes": e,
        "new_comparisons": 2,
    }


def make_scores(config):
    @jax.jit
    def evaluate(state, head, batch):
        _, (_, m) = future_objective(
            state.params,
            head.params,
            state.apply_fn,
            initial_state(config, batch["inputs"].shape[0]),
            batch,
            config,
            0.0,
        )
        return {
            k: m[k] for k in ["loss", "main_objective", "aux_loss", "positions", "aux_positions"]
        }

    def scores(state, head, qa, english):
        result = {
            "qa": evaluate(state, head, jax.tree.map(jnp.asarray, qa)),
            "english_train_probe": evaluate(state, head, jax.tree.map(jnp.asarray, english)),
        }
        if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(result)):
            raise FloatingPointError("Nonfinite replication evaluation")
        return {name: {k: float(v) for k, v in values.items()} for name, values in result.items()}

    return scores


def train_arm(
    source,
    head0,
    config,
    batches,
    ids,
    documents,
    qa,
    fixed_english,
    qa_seed,
    english_seed,
    weight,
    out,
    pairs=128,
):
    out = Path(out)
    if out.exists():
        raise ValueError("Use fresh arm directory")
    if len(batches) != len(ids) or len(ids) != len(set(ids)) or not ids:
        raise ValueError("Unique aligned QA groups required")
    order, english, end = seeded_schedule(config, documents, len(ids), qa_seed, english_seed, pairs)
    expected = exposure(batches, order, english)
    scores = make_scores(config)
    state, head = source, head0
    before = scores(source, head0, qa, fixed_english)
    source_digest, head_digest = state_digest(source), state_digest(head0)
    step = make_future_step(config, weight)
    carry = initial_state(config, config.batch_size)
    stream = DocumentStream(documents, config.batch_size, config.length, english_seed)
    counts = {phase: {k: 0 for k in expected[phase]} for phase in expected}
    visits = {identity: 0 for identity in ids}
    trace = []
    history = []
    out.mkdir(parents=True)
    write_json(out / "initial.json", before)
    start = time.perf_counter()
    for index, group in enumerate(order):
        replay = stream.next_batch()
        if fingerprints(replay) != fingerprints(english[index]):
            raise ValueError("Replay schedule changed")
        for phase, batch in [("qa", batches[group]), ("english", replay)]:
            expected_batch = expected_counts(batch)
            batch = jax.tree.map(jnp.asarray, batch)
            incoming = initial_state(config, config.batch_size) if phase == "qa" else carry
            state, head, final, m = step(state, head, incoming, batch)
            if not bool(m["finite"]) or not bool(m["updated"]):
                raise FloatingPointError("Rejected replication step")
            observed = {
                "updates": 1,
                "positions": int(m["positions"]),
                "aux_positions": int(m["aux_positions"]),
                "input_bytes": int(m["consumed_bytes"]),
                "supervised_bytes": int(m["bytes"]),
            }
            if observed != expected_batch:
                raise ValueError("Independent target/byte count mismatch")
            for k, v in observed.items():
                counts[phase][k] += v
            if phase == "english":
                carry = final
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
        trace.append(ids[group])
        if (index + 1) % 32 == 0:
            print(
                f"qa{qa_seed}/english{english_seed} weight{weight}: {index + 1}/{pairs} pairs",
                flush=True,
            )
    elapsed = time.perf_counter() - start
    after = scores(state, head, qa, fixed_english)
    payload = serialization.to_bytes(head)
    (out / "head.msgpack").write_bytes(payload)
    meta = {
        "mode": "future_byte_replication_backbone_only",
        "qa_seed": qa_seed,
        "english_seed": english_seed,
        "auxiliary_weight": weight,
        "joint_resume_supported": False,
        "messages": True,
        "counts": counts,
        "qa_trace": trace,
        "english_stream": stream.snapshot(),
        "english_carry": np.asarray(carry).tolist(),
        "auxiliary_state_sha256": digest(out / "head.msgpack"),
    }
    save_checkpoint(out / "final.mlm", state, config, meta)
    loaded, cfg, saved = load_checkpoint(out / "final.mlm")
    audit = {
        "budget": int(state.step) == int(source.step) + 2 * pairs
        and int(head.step) == int(head0.step) + 2 * pairs,
        "exposure": counts == expected,
        "visits": all(v == pairs // len(ids) for v in visits.values()),
        "trace": trace == [ids[i] for i in order],
        "stream": stream.snapshot() == end,
        "source_immutable": state_digest(source) == source_digest
        and state_digest(head0) == head_digest,
        "backbone_roundtrip": state_digest(loaded) == state_digest(state)
        and cfg == config
        and saved == meta,
        "head_roundtrip": serialized_values_equal(
            serialization.msgpack_restore(payload),
            serialization.to_state_dict(serialization.from_bytes(head0, payload)),
        ),
    }
    if not all(audit.values()):
        raise ValueError("Arm audit failed: " + str([k for k, v in audit.items() if not v]))
    result = {
        "weight": weight,
        "qa_seed": qa_seed,
        "english_seed": english_seed,
        "initial": before,
        "final": after,
        "counts": counts,
        "visits": visits,
        "qa_trace": trace,
        "english_snapshot": end,
        "english_carry": np.asarray(carry).tolist(),
        "backbone_parameters": sum(x.size for x in jax.tree.leaves(state.params)),
        "auxiliary_parameters": sum(x.size for x in jax.tree.leaves(head.params)),
        "elapsed_training_seconds_including_compile": elapsed,
        "audit": audit,
        "final_backbone_sha256": digest(out / "final.mlm"),
        "final_state_sha256": state_digest(state),
        "head_sha256": digest(out / "head.msgpack"),
    }
    write_json(out / "history.json", history)
    write_json(out / "summary.json", result)
    return state, head, result
