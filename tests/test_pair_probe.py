import copy
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import serialization

from mlm.binding_run import build_batches
from mlm.config import Config
from mlm.gradual_data import group
from mlm.hierarchy_run import make_routed_step, routed_objective
from mlm.interference import make_prefix_reader, replay_probes
from mlm.model import initial_state
from mlm.pair_probe import make_pair_probe, restore_next_pair
from mlm.plasticity import parameter_roles
from mlm.stream import DocumentStream
from mlm.training import create_state


def test_saved_sampler_carry_and_copied_pair_match_real_trainer():
    cfg = Config(
        nodes=4,
        width=8,
        groups=2,
        node_update="hierarchical",
        memory="delta",
        memory_keys=4,
        batch_size=4,
        length=128,
        shared_lr_scale=0.1,
    )
    records, questions = [], []
    for index, reverse in enumerate([False, True]):
        r, q = group("Dan", "Ned", ["gold", "blue"], reverse, f"g{index}", "training")
        records.extend(r)
        questions.extend(q)
    batches, ids = build_batches(records, questions, cfg)
    qa_seed, english_seed = 7, 11
    rng = np.random.default_rng(qa_seed)
    order = np.concatenate([rng.permutation(len(ids)) for _ in range(3)])
    documents = [f"Story {i}. " * 70 for i in range(8)]
    stream = DocumentStream(documents, 4, 128, english_seed)
    english_batches, snapshots = [], [stream.snapshot()]
    for _ in range(4):
        english_batches.append(stream.next_batch())
        snapshots.append(stream.snapshot())
    state = create_state(cfg)
    carry = initial_state(cfg, 4)
    trainer = make_routed_step(cfg)
    saved: dict[int, Any] = {}
    for count in range(4):
        if count in [0, 2, 3]:
            meta: dict[str, Any] = {
                "sampling_seeds": {"subset": cfg.seed, "qa": qa_seed, "english": english_seed},
                "group_ids": list(reversed(ids)),
                "qa_group_trace": [ids[int(i)] for i in order[:count]],
                "counters": {"qa": {"updates": count}, "english": {"updates": count}},
                "english_stream": snapshots[count],
                "english_carry": np.asarray(carry).tolist(),
            }
            saved[count] = (state, copy.deepcopy(meta), np.asarray(carry).copy())
        if count == 3:
            break
        state, _, _ = trainer(
            state, initial_state(cfg, 4), jax.tree.map(jnp.asarray, batches[int(order[count])])
        )
        state, carry, _ = trainer(state, carry, jax.tree.map(jnp.asarray, english_batches[count]))
    aggregate = {k: np.concatenate([b[k] for b in batches], axis=0) for k in batches[0]}
    aggregate = jax.tree.map(jnp.asarray, aggregate)
    probe = make_pair_probe(cfg)
    for count, (source, meta, actual_carry) in saved.items():
        original_state = serialization.to_bytes(source)
        original_meta = copy.deepcopy(meta)
        restored = restore_next_pair(cfg, meta, batches, ids, documents)
        assert restored["qa_group"] == ids[int(order[count])]
        assert restored["qa_updates_before"] == count
        assert restored["sampling_seeds"] == meta["sampling_seeds"]
        for k, v in restored["qa_batch"].items():
            np.testing.assert_array_equal(v, batches[int(order[count])][k])
        for k, v in restored["english_batch"].items():
            np.testing.assert_array_equal(v, english_batches[count][k])
        np.testing.assert_array_equal(restored["english_carry"], actual_carry)
        assert meta == original_meta
        qa = jax.tree.map(jnp.asarray, restored["qa_batch"])
        english = jax.tree.map(jnp.asarray, restored["english_batch"])
        incoming = jnp.asarray(restored["english_carry"])
        states, result = probe(source, aggregate, qa, english, incoming)
        manual_qa, _, _ = trainer(source, initial_state(cfg, 4), qa)
        manual_pair, _, _ = trainer(manual_qa, incoming, english)
        manual_only, _, _ = trainer(source, incoming, english)
        manual_zero = manual_qa.apply_gradients(
            grads=jax.tree.map(jnp.zeros_like, manual_qa.params)
        )
        source_zero = source.apply_gradients(grads=jax.tree.map(jnp.zeros_like, source.params))
        expected = {
            "qa": manual_qa,
            "qa_english": manual_pair,
            "qa_zero": manual_zero,
            "english_only": manual_only,
            "zero_only": source_zero,
        }
        for name, state_expected in expected.items():
            assert int(states[name].step) == int(state_expected.step)
            for a, b in zip(
                jax.tree.leaves((states[name].params, states[name].opt_state)),
                jax.tree.leaves((state_expected.params, state_expected.opt_state)),
                strict=True,
            ):
                np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-7)

        def loss(s):
            return float(
                routed_objective(s.params, s.apply_fn, aggregate, initial_state(cfg, 8), cfg)[1][1][
                    "loss"
                ]
            )

        before, after_qa, after_pair, after_zero = map(
            loss, [source, manual_qa, manual_pair, manual_zero]
        )
        after_only, after_source_zero = map(loss, [manual_only, source_zero])
        assert result["qa_loss_after_qa_zero"] == pytest.approx(after_zero, abs=1e-6)
        assert result["qa_loss_after_english_only"] == pytest.approx(after_only, abs=1e-6)
        assert result["qa_loss_after_source_zero"] == pytest.approx(after_source_zero, abs=1e-6)
        assert result["zero_pair_loss_change"] == pytest.approx(after_zero - before, abs=1e-6)
        assert result["english_only_loss_change"] == pytest.approx(after_only - before, abs=1e-6)
        assert result["english_only_minus_zero"] == pytest.approx(
            after_only - after_source_zero, abs=1e-6
        )
        assert result["preceding_qa_effect_on_english_delta"] == pytest.approx(
            (after_pair - after_qa) - (after_only - before), abs=1e-6
        )
        selected_before = routed_objective(
            source.params, source.apply_fn, qa, initial_state(cfg, 4), cfg
        )[1][1]["loss"]
        selected_after = routed_objective(
            manual_qa.params, manual_qa.apply_fn, qa, initial_state(cfg, 4), cfg
        )[1][1]["loss"]
        assert result["selected_qa_loss_before"] == pytest.approx(float(selected_before), abs=1e-6)
        assert result["selected_qa_loss_after"] == pytest.approx(float(selected_after), abs=1e-6)
        for name, start, end in [
            ("paired_english", manual_qa, manual_pair),
            ("matched_english_only", source, manual_only),
        ]:
            before_objective, (_, before_metrics) = routed_objective(
                start.params, start.apply_fn, english, incoming, cfg
            )
            after_objective, (_, after_metrics) = routed_objective(
                end.params, end.apply_fn, english, incoming, cfg
            )
            measured = result[name]
            assert measured["english_objective_before"] == pytest.approx(
                float(before_objective), abs=1e-6
            )
            assert measured["english_objective_after"] == pytest.approx(
                float(after_objective), abs=1e-6
            )
            assert measured["english_language_loss_before"] == pytest.approx(
                float(before_metrics["loss"]), abs=1e-6
            )
            assert measured["english_language_loss_after"] == pytest.approx(
                float(after_metrics["loss"]), abs=1e-6
            )
            if count == 3:
                qgrad = jax.grad(
                    lambda params: routed_objective(
                        params, start.apply_fn, aggregate, initial_state(cfg, 8), cfg
                    )[1][1]["loss"]
                )(start.params)
                egrad = jax.grad(
                    lambda params: routed_objective(params, start.apply_fn, english, incoming, cfg)[
                        0
                    ]
                )(start.params)
                roles = parameter_roles(start.params)
                for role, stats in measured["roles"].items():
                    names = [key for key in start.params if roles[key] == role]
                    qnorm = np.sqrt(
                        sum(np.square(np.asarray(qgrad[k], dtype=np.float64)).sum() for k in names)
                    )
                    enorm = np.sqrt(
                        sum(np.square(np.asarray(egrad[k], dtype=np.float64)).sum() for k in names)
                    )
                    dot = sum(
                        np.vdot(
                            np.asarray(qgrad[k], dtype=np.float64),
                            np.asarray(end.params[k], dtype=np.float64)
                            - np.asarray(start.params[k], dtype=np.float64),
                        )
                        for k in names
                    )
                    assert stats["qa_gradient_norm"] == pytest.approx(
                        float(qnorm), rel=1e-5, abs=1e-6
                    )
                    assert stats["english_gradient_norm"] == pytest.approx(
                        float(enorm), rel=1e-5, abs=1e-6
                    )
                    assert stats["first_order_qa_change"] == pytest.approx(
                        float(dot), rel=1e-5, abs=2e-6
                    )
        assert result["qa_loss_before"] == pytest.approx(before, abs=1e-6)
        assert result["qa_loss_after_qa"] == pytest.approx(after_qa, abs=1e-6)
        assert result["qa_loss_after_pair"] == pytest.approx(after_pair, abs=1e-6)
        assert result["qa_step_loss_change"] == pytest.approx(after_qa - before, abs=1e-6)
        assert result["english_step_loss_change"] == pytest.approx(after_pair - after_qa, abs=1e-6)
        assert result["pair_loss_change"] == pytest.approx(after_pair - before, abs=1e-6)
        assert result["english_minus_zero_after_qa"] == pytest.approx(
            after_pair - after_zero, abs=1e-6
        )
        assert serialization.to_bytes(source) == original_state
    source, meta, actual_carry = saved[3]
    bad_history = copy.deepcopy(meta)
    bad_history["qa_group_trace"][:2] = list(reversed(bad_history["qa_group_trace"][:2]))
    with pytest.raises(ValueError, match="QA history"):
        restore_next_pair(cfg, bad_history, batches, ids, documents)
    reference_next = DocumentStream(documents, 4, 128, english_seed)
    reference_next.restore(meta["english_stream"])
    expected_next = reference_next.next_batch()
    for mutation in ["rng", "order", "position"]:
        corrupted = copy.deepcopy(meta)
        snapshot = corrupted["english_stream"]
        if mutation == "rng":
            snapshot["rng"]["state"]["state"] ^= 1
        elif mutation == "order":
            snapshot["order"][4], snapshot["order"][5] = snapshot["order"][5], snapshot["order"][4]
        else:
            snapshot["position"] += 1
        altered_stream = DocumentStream(documents, 4, 128, english_seed)
        altered_stream.restore(snapshot)
        next_batch = altered_stream.next_batch()
        for key in expected_next:
            np.testing.assert_array_equal(next_batch[key], expected_next[key])
        with pytest.raises(ValueError, match="English stream"):
            restore_next_pair(cfg, corrupted, batches, ids, documents)
    frozen = replay_probes(documents, 4, 128, english_seed, [3])[0]["prefix"]
    reconstructed = make_prefix_reader(cfg)(source, jax.tree.map(jnp.asarray, frozen))
    assert np.linalg.norm(actual_carry - np.asarray(reconstructed)) > 1e-6
    empty = {**batches[0], "loss_mask": np.zeros_like(batches[0]["loss_mask"])}
    with pytest.raises(ValueError, match="supervised"):
        probe(
            source,
            aggregate,
            jax.tree.map(jnp.asarray, empty),
            jax.tree.map(jnp.asarray, english_batches[3]),
            jnp.asarray(actual_carry),
        )


def test_legacy_seed_restoration_and_corrupt_metadata_rejection():
    cfg = Config(nodes=4, width=8, batch_size=4, length=8)
    docs = [f"Story {i} continues." for i in range(8)]
    ids = ["a", "b", "c"]
    batches = [{"inputs": np.full((4, 8), i, np.int32)} for i in range(3)]
    stream = DocumentStream(docs, 4, 8, cfg.seed + 1)
    meta: dict[str, Any] = {
        "group_ids": list(reversed(ids)),
        "qa_group_trace": [],
        "counters": {"qa": {"updates": 0}, "english": {"updates": 0}},
        "english_stream": stream.snapshot(),
        "english_carry": np.asarray(initial_state(cfg, 4)).tolist(),
    }
    result = restore_next_pair(cfg, meta, batches, ids, docs)
    assert result["sampling_seeds"] == {"subset": 42, "qa": 42, "english": 43}
    assert result["qa_group"] == ids[int(np.random.default_rng(42).permutation(3)[0])]
    for k, v in stream.next_batch().items():
        np.testing.assert_array_equal(result["english_batch"][k], v)
    mutations = []
    bad = copy.deepcopy(meta)
    bad["counters"]["qa"]["updates"] = 1
    mutations.append(bad)
    bad = copy.deepcopy(meta)
    bad["qa_group_trace"] = ["wrong"]
    mutations.append(bad)
    bad = copy.deepcopy(meta)
    bad["english_stream"]["corpus"] = "wrong"
    mutations.append(bad)
    bad = copy.deepcopy(meta)
    bad["english_carry"] = [0.0]
    mutations.append(bad)
    bad = copy.deepcopy(meta)
    bad["english_carry"][0][0][0] = float("nan")
    mutations.append(bad)
    bad = copy.deepcopy(meta)
    bad["sampling_seeds"] = {"subset": 99, "qa": 42, "english": 43}
    mutations.append(bad)
    for corrupted in mutations:
        with pytest.raises(ValueError):
            restore_next_pair(cfg, corrupted, batches, ids, docs)
