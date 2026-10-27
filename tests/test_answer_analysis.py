import copy

import jax.numpy as jnp
import numpy as np
import pytest
from flax import serialization

from mlm.answer_analysis import (
    analyze_answers,
    paired_changes,
    prepare_answers,
    ranking_transitions,
    summarize_nll,
)
from mlm.comprehension import evaluate_questions
from mlm.config import Config
from mlm.gradual_data import group
from mlm.hierarchy_run import routed_objective
from mlm.model import initial_state
from mlm.training import create_state


def fixture():
    cfg = Config(
        nodes=4,
        width=8,
        groups=2,
        node_update="hierarchical",
        memory="delta",
        memory_keys=4,
        batch_size=4,
        length=128,
    )
    records, questions = group("Bon", "Bom", ["gold", "lime"], False, "g0", "training")
    return cfg, records, questions


def test_partition_alignment_and_corruption_rejection():
    cfg, records, questions = fixture()
    batch, ordered, masks = prepare_answers(records, questions, cfg)
    assert [q["id"] for q in ordered] == [q["id"] for q in questions]
    assert {k: int(v.sum()) for k, v in masks.items()} == {
        "first": 4,
        "rest": 12,
        "newline": 4,
        "eos": 4,
    }
    np.testing.assert_array_equal(sum(masks.values()), batch["mask"] * batch["loss_mask"])
    nll = np.arange(batch["targets"].size).reshape(batch["targets"].shape) / 10
    result = summarize_nll(nll, masks, ordered)
    assert result["total"]["count"] == 24 and result["answer_only"]["count"] == 16
    assert sum(v["sum_nll"] for v in result["parts"].values()) == pytest.approx(
        result["total"]["sum_nll"]
    )
    assert result["total"]["mean_nll"] == pytest.approx(
        sum(float((nll * m).sum()) for m in masks.values()) / 24
    )
    for mode in ["text", "span", "label", "group"]:
        r, q = copy.deepcopy(records), copy.deepcopy(questions)
        if mode == "text":
            r[0]["text"] += "!"
        elif mode == "span":
            r[0]["loss_spans"][0][0] += 1
        elif mode == "label":
            q[0]["answer_index"] = 1 - q[0]["answer_index"]
        else:
            r[0]["group"] = "different"
        with pytest.raises(ValueError):
            prepare_answers(r, q, cfg)


def test_ranking_transition_counts_and_identity_checks():
    def item(i, correct, prediction):
        return {
            "id": str(i),
            "group": "g",
            "choices": ["a", "b"],
            "answer_index": 0,
            "correct": correct,
            "prediction": prediction,
        }

    control = [item(0, True, 0), item(1, True, 0), item(2, False, 1), item(3, False, 1)]
    active = [item(0, True, 0), item(1, False, 1), item(2, True, 0), item(3, False, 1)]
    assert ranking_transitions(control, active) == {
        "both_correct": 1,
        "control_only": 1,
        "auxiliary_only": 1,
        "both_wrong": 1,
        "prediction_flips": 2,
        "net_correct_change": 0,
    }
    bad = copy.deepcopy(active)
    bad[0]["id"] = "missing"
    with pytest.raises(ValueError):
        ranking_transitions(control, bad)


def test_neural_scorer_teacher_consistency_and_no_state_change():
    cfg, records, questions = fixture()
    state = create_state(cfg)
    before = serialization.to_bytes(state)
    result = analyze_answers(state, cfg, records, questions)
    legacy = evaluate_questions(state, cfg, questions, format="chat")
    assert [r["prediction"] for r in result["ranking"]["items"]] == [
        r["prediction"] for r in legacy["items"]
    ]
    assert result["score_teacher_max_error"] < 1e-5
    batch, _, _ = prepare_answers(records, questions, cfg)
    _, (_, m) = routed_objective(
        state.params,
        state.apply_fn,
        {k: jnp.asarray(v) for k, v in batch.items()},
        initial_state(cfg, 4),
        cfg,
    )
    assert result["teacher"]["total"]["mean_nll"] == pytest.approx(float(m["loss"]), abs=1e-6)
    for row, rank in zip(result["teacher"]["items"], result["ranking"]["items"], strict=True):
        assert row["id"] == rank["id"]
        assert row["gold_answer_mean_nll"] == pytest.approx(
            -rank["scores"][rank["answer_index"]], abs=1e-5
        )
    assert serialization.to_bytes(state) == before


def test_paired_position_contributions_reconstruct_total_change():
    cfg, records, questions = fixture()
    batch, ordered, masks = prepare_answers(records, questions, cfg)
    first = summarize_nll(np.ones_like(batch["mask"]), masks, ordered)
    second = summarize_nll(
        np.ones_like(batch["mask"]) + 0.6 * masks["first"] - 0.2 * masks["newline"], masks, ordered
    )

    def ranked(teacher, flip):
        rows = []
        for i, (q, item) in enumerate(zip(ordered, teacher["items"], strict=True)):
            gold = q["answer_index"]
            pred = 1 - gold if flip and i == 0 else gold
            rows.append(
                {
                    **q,
                    "prediction": pred,
                    "correct": pred == gold,
                    "gold_minus_other_margin": -1.0 if flip and i == 0 else 1.0,
                }
            )
        return {"teacher": teacher, "ranking": {"items": rows}}

    result = paired_changes(ranked(first, False), ranked(second, True))
    assert result["qa_ce_delta"] == pytest.approx((4 * 0.6 - 4 * 0.2) / 24, abs=1e-7)
    assert sum(v["total_ce_contribution"] for v in result["parts"].values()) == pytest.approx(
        result["qa_ce_delta"], abs=1e-12
    )
    assert result["ranking_transitions"]["net_correct_change"] == -1
    assert result["ranking_transitions"]["control_only"] == 1
    assert result["margin_changes"][0]["margin_delta"] == -2
