"""Restore actual next training inputs and compare independent copied QA/replay branches."""

import copy
from typing import Any

import jax
import numpy as np

from mlm.hierarchy_run import make_routed_step, routed_objective
from mlm.interference import make_interference_probe
from mlm.model import initial_state
from mlm.plasticity_run import sampling_seeds
from mlm.stream import DocumentStream


def restore_next_pair(config, metadata, qa_batches, group_ids, english_documents) -> dict[str, Any]:
    if not group_ids or len(set(group_ids)) != len(group_ids) or len(qa_batches) != len(group_ids):
        raise ValueError("Unique aligned QA groups required")
    if set(metadata["group_ids"]) != set(group_ids):
        raise ValueError("Saved QA membership mismatch")
    saved_seeds = metadata.get("sampling_seeds")
    if saved_seeds is None:
        seeds = sampling_seeds(config)
    else:
        if set(saved_seeds) != {"subset", "qa", "english"}:
            raise ValueError("Incomplete saved seed policy")
        seeds = sampling_seeds(config, saved_seeds["qa"], saved_seeds["english"])
        if seeds != saved_seeds:
            raise ValueError("Saved seed policy mismatch")
    qa_count = metadata["counters"]["qa"]["updates"]
    english_count = metadata["counters"]["english"]["updates"]
    if any(type(n) is not int or n < 0 for n in [qa_count, english_count]):
        raise ValueError("Nonnegative update counts required")
    if qa_count != english_count:
        raise ValueError("Checkpoint must follow a complete paired update")
    history = metadata["qa_group_trace"]
    if len(history) != qa_count:
        raise ValueError("Saved QA history length mismatch")
    rng = np.random.default_rng(seeds["qa"])
    order = rng.permutation(len(group_ids))
    position = 0
    for actual in history:
        if position == len(group_ids):
            order = rng.permutation(len(group_ids))
            position = 0
        if actual != group_ids[int(order[position])]:
            raise ValueError("Saved QA history differs from seed/order")
        position += 1
    if position == len(group_ids):
        order = rng.permutation(len(group_ids))
        position = 0
    index = int(order[position])
    reference = DocumentStream(
        english_documents, config.batch_size, config.length, seeds["english"]
    )
    for _ in range(english_count):
        reference.next_batch()
    if reference.snapshot() != metadata["english_stream"]:
        raise ValueError("Saved English stream differs from seed/count")
    stream = DocumentStream(english_documents, config.batch_size, config.length, seeds["english"])
    stream.restore(metadata["english_stream"])
    batch = stream.next_batch()
    expected = reference.next_batch()
    if (
        batch is None
        or expected is None
        or any(not np.array_equal(batch[key], expected[key]) for key in expected)
        or stream.snapshot() != reference.snapshot()
    ):
        raise ValueError("Restored English next batch mismatch")
    carry = np.array(metadata["english_carry"], dtype=np.float32, copy=True)
    if (
        carry.shape != initial_state(config, config.batch_size).shape
        or not np.isfinite(carry).all()
    ):
        raise ValueError("Saved English carry shape/values invalid")
    if english_count == 0 and np.any(carry):
        raise ValueError("Initial English carry must be zero")
    return {
        "qa_batch": qa_batches[index],
        "qa_group": group_ids[index],
        "qa_updates_before": qa_count,
        "english_updates_before": english_count,
        "english_batch": batch,
        "english_carry": carry,
        "sampling_seeds": seeds,
        "english_snapshot_before": copy.deepcopy(metadata["english_stream"]),
        "english_snapshot_after": stream.snapshot(),
    }


def make_pair_probe(config):
    train = make_routed_step(config)
    english_probe = make_interference_probe(config)

    @jax.jit
    def group_loss(state, batch):
        return routed_objective(
            state.params,
            state.apply_fn,
            batch,
            initial_state(config, batch["inputs"].shape[0]),
            config,
        )[1][1]["loss"]

    def probe(source, qa_evaluation, next_qa, next_english, english_carry):
        after_qa, _, qa_metrics = train(
            source, initial_state(config, next_qa["inputs"].shape[0]), next_qa
        )
        if not bool(qa_metrics["finite"]):
            raise FloatingPointError("Nonfinite copied QA step")
        if not bool(qa_metrics["updated"]):
            raise ValueError("QA step requires supervised positions")
        after_pair, after_qa_zero, paired = english_probe(
            after_qa, qa_evaluation, next_english, english_carry
        )
        english_only, source_zero, matched = english_probe(
            source, qa_evaluation, next_english, english_carry
        )
        before = matched["qa_loss_before"]
        after = paired["qa_loss_before"]
        states = {
            "qa": after_qa,
            "qa_english": after_pair,
            "qa_zero": after_qa_zero,
            "english_only": english_only,
            "zero_only": source_zero,
        }
        result: dict[str, Any] = {
            "qa_loss_before": before,
            "qa_loss_after_qa": after,
            "qa_loss_after_pair": paired["qa_loss_after_english"],
            "qa_loss_after_qa_zero": paired["qa_loss_after_zero"],
            "qa_loss_after_english_only": matched["qa_loss_after_english"],
            "qa_loss_after_source_zero": matched["qa_loss_after_zero"],
            "qa_step_loss_change": after - before,
            "english_step_loss_change": paired["qa_loss_change"],
            "pair_loss_change": paired["qa_loss_after_english"] - before,
            "zero_pair_loss_change": paired["qa_loss_after_zero"] - before,
            "english_minus_zero_after_qa": paired["english_minus_zero_qa_loss"],
            "english_only_loss_change": matched["qa_loss_change"],
            "english_only_minus_zero": matched["english_minus_zero_qa_loss"],
            "preceding_qa_effect_on_english_delta": paired["qa_loss_change"]
            - matched["qa_loss_change"],
            "selected_qa_loss_before": float(qa_metrics["loss"]),
            "selected_qa_loss_after": float(group_loss(after_qa, next_qa)),
            "paired_english": paired,
            "matched_english_only": matched,
            "steps": {
                "source": int(source.step),
                **{name: int(state.step) for name, state in states.items()},
            },
        }
        return states, result

    return probe
