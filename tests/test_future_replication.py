import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import serialization

from mlm.config import Config
from mlm.future_byte import create_head
from mlm.future_byte_run import schedule as original_schedule
from mlm.future_replication import aggregate_gates, seeded_schedule, train_arm
from mlm.hierarchy_run import make_routed_step
from mlm.model import initial_state
from mlm.stream import DocumentStream
from mlm.training import create_state


def test_seeded_schedule_matches_legacy_and_independent_stream():
    cfg = Config(nodes=4, width=8, groups=2, node_update="hierarchical", batch_size=4, length=16)
    docs = [f"Story {i}. " * 30 for i in range(12)]
    old, english, snapshot = original_schedule(cfg, docs, 4)
    order, actual, end = seeded_schedule(cfg, docs, 4, 42, 43, 128)
    assert order == old and end == snapshot
    for a, b in zip(actual, english, strict=True):
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])
    for q, e in [(43, 44), (44, 45), (0, 0)]:
        order, batches, saved = seeded_schedule(cfg, docs, 4, q, e, 8)
        rng = np.random.default_rng(q)
        assert order == list(rng.permutation(4)) + list(rng.permutation(4))
        stream = DocumentStream(docs, 4, 16, e)
        for actual in batches:
            expected = stream.next_batch()
            for k in expected:
                np.testing.assert_array_equal(actual[k], expected[k])
        assert saved == stream.snapshot()
    for args in [(True, 43, 8), (-1, 43, 8), (42, 2**32, 8), (42, 43, 7)]:
        with pytest.raises(ValueError):
            seeded_schedule(cfg, docs, 4, *args)


def test_gates_require_both_new_seeds_and_correct_signs():
    def rows(q, e):
        return {str(seed): {"qa_delta": q, "english_delta": e} for seed in [43, 44]}

    assert aggregate_gates(rows(-0.01, 0))["joint"]
    assert not aggregate_gates(rows(-1e-6, 0))["primary_qa"]
    assert aggregate_gates(rows(-0.01, 1e-6))["secondary_english"]
    assert not aggregate_gates(rows(-0.01, 1.0001e-6))["joint"]
    r = rows(-0.01, 0)
    r["44"]["qa_delta"] = 0
    assert not aggregate_gates(r)["primary_qa"]
    with pytest.raises(ValueError):
        aggregate_gates({"42": r["43"], "43": r["43"]})
    with pytest.raises(ValueError):
        aggregate_gates(rows(float("nan"), 0))


def test_small_real_arm_matches_direct_steps_and_preserves_source(tmp_path):
    cfg = Config(nodes=4, width=8, groups=2, node_update="hierarchical", batch_size=4, length=8)
    batch = {
        "inputs": np.array([[256, 65, 66, 67, 68, 69, 70, 71]] * 4, np.int32),
        "targets": np.array([[65, 66, 67, 68, 69, 70, 71, 257]] * 4, np.int32),
        "mask": np.ones((4, 8), np.float32),
        "loss_mask": np.array([[0, 0, 0, 0, 1, 1, 1, 1]] * 4, np.float32),
        "resets": np.array([[True] + [False] * 7] * 4),
    }
    docs = [
        "Small story. " * 10,
        "Another story. " * 10,
        "Red blue green. " * 10,
        "English words. " * 10,
    ]
    other = {
        **batch,
        "inputs": np.where(batch["inputs"] < 256, batch["inputs"] + 10, batch["inputs"]),
        "targets": np.where(batch["targets"] < 256, batch["targets"] + 10, batch["targets"]),
    }
    batches = [batch, other]
    source = create_state(cfg)
    head = create_head(cfg)
    before = serialization.to_bytes(source)
    order, english, snapshot = seeded_schedule(cfg, docs, 2, 43, 44, 2)
    expected, eh = source, head
    carry = initial_state(cfg, 4)
    step = make_routed_step(cfg)
    # Independent chronological direct loop.
    expected, eh = source, head
    for index, i in enumerate(order):
        expected, _, _ = step(
            expected, initial_state(cfg, 4), jax.tree.map(jnp.asarray, batches[i])
        )
        eh = eh.apply_gradients(grads=jax.tree.map(jnp.zeros_like, eh.params))
        expected, carry, _ = step(expected, carry, jax.tree.map(jnp.asarray, english[index]))
        eh = eh.apply_gradients(grads=jax.tree.map(jnp.zeros_like, eh.params))
    state, trained, result = train_arm(
        source,
        head,
        cfg,
        batches,
        ["a", "b"],
        docs,
        batch,
        english[0],
        43,
        44,
        0.0,
        tmp_path / "arm",
        pairs=2,
    )
    assert serialization.to_bytes(state) == serialization.to_bytes(expected)
    assert serialization.to_bytes(trained) == serialization.to_bytes(eh)
    assert result["english_snapshot"] == snapshot and result["visits"] == {"a": 1, "b": 1}
    np.testing.assert_array_equal(np.asarray(result["english_carry"]), carry)
    assert result["counts"]["qa"]["updates"] == 2 and int(state.step) == 4
    assert serialization.to_bytes(source) == before
    with pytest.raises(ValueError):
        train_arm(
            source,
            head,
            cfg,
            batches,
            ["a", "b"],
            docs,
            batch,
            english[0],
            43,
            44,
            0.0,
            tmp_path / "arm",
            pairs=2,
        )
