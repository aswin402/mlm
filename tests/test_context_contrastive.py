import copy

import jax
import numpy as np
import pytest
from flax import serialization

from mlm.binding_run import build_batches
from mlm.comprehension import score_candidates
from mlm.config import Config
from mlm.context_contrastive import (
    candidate_batch,
    context_objective,
    make_context_step,
    neural_scores,
)
from mlm.context_contrastive_run import gates
from mlm.gradual_data import group
from mlm.hierarchy_run import make_routed_step
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


def test_candidate_scores_match_canonical_and_labels_are_metadata():
    cfg, _, questions = fixture()
    state = create_state(cfg)
    batch = candidate_batch(questions, cfg.length)
    assert batch["answer_mask"].sum() == 32
    scores, _ = neural_scores(state.params, state.apply_fn, batch, cfg)
    prompts = [
        f"<|user|>\n{q['context']}\n{q['question']}\n<|assistant|>\n"
        for q in questions
        for _ in q["choices"]
    ]
    answers = [a for q in questions for a in q["choices"]]
    expected = score_candidates(state, cfg, prompts, answers)
    np.testing.assert_allclose(np.asarray(scores).ravel(), expected, atol=1e-6)
    changed = dict(batch, gold=1 - batch["gold"])
    other, _ = neural_scores(state.params, state.apply_fn, changed, cfg)
    np.testing.assert_array_equal(scores, other)
    bad = copy.deepcopy(questions)
    bad[0]["context"] = bad[2]["context"]
    with pytest.raises(ValueError):
        candidate_batch(bad, cfg.length)


def test_zero_weight_exact_legacy_and_active_gradient():
    cfg, records, questions = fixture()
    state = create_state(cfg)
    batch = build_batches(records, questions, cfg)[0][0]
    candidates = candidate_batch(questions, cfg.length)
    carry = initial_state(cfg, 4)
    old = make_routed_step(cfg)(state, carry, batch)
    zero = make_context_step(cfg, 0)(state, carry, batch, candidates)
    assert serialization.to_bytes(old[0]) == serialization.to_bytes(zero[0])
    np.testing.assert_array_equal(old[1], zero[1])
    grad = jax.grad(lambda p: context_objective(p, state.apply_fn, candidates, cfg)[0])(
        state.params
    )
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(grad))
    assert sum(float(np.square(x).sum()) for x in jax.tree.leaves(grad)) > 0
    active = make_context_step(cfg, 1)(state, carry, batch, candidates)
    assert bool(active[2]["updated"]) and int(active[0].step) == 1
    assert serialization.to_bytes(active[0]) != serialization.to_bytes(zero[0])


def test_gates_require_binding_and_retention():
    control = dict(correct=8, groups_correct=0, english_ce=1.0)
    assert not gates(control, control)["primary"]
    improved = dict(correct=10, groups_correct=1, english_ce=1.0)
    assert gates(improved, control)["joint"]
    improved["english_ce"] = 1.1
    assert not gates(improved, control)["joint"]
