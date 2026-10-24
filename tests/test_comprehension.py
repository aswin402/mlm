from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from mlm.comprehension import evaluate_questions, score_candidates
from mlm.config import Config


def fake_model():
    def apply(variables, tokens, carry, resets, messages=True, valid=None):
        logits = jnp.zeros((*tokens.shape, 258)).at[:, :, ord("a")].set(2.0)
        return logits, carry

    return SimpleNamespace(params={}, apply_fn=apply)


def test_likelihood_masks_prompt_and_scores_only_candidate_bytes():
    cfg = Config(width=8)
    scores = score_candidates(
        fake_model(),
        cfg,
        ["prefix", "a much longer prefix", "prefix", "prefix"],
        ["a", "a", "b", "aa"],
    )
    np.testing.assert_allclose(scores[0], scores[1], atol=1e-6)
    assert scores[0] > scores[2]
    expected = 2 - np.log(np.exp(2) + 257)
    np.testing.assert_allclose(scores[0], expected, atol=1e-6)
    np.testing.assert_allclose(scores[0], scores[3], atol=1e-6)


def test_fixed_answer_preference_cannot_pass_counterfactual_pairs():
    questions = [
        {
            "id": "one",
            "pair": "pair",
            "context": "It is a.",
            "question": "What is it?",
            "choices": ["a", "b"],
            "answer_index": 0,
        },
        {
            "id": "two",
            "pair": "pair",
            "context": "It is b.",
            "question": "What is it?",
            "choices": ["a", "b"],
            "answer_index": 1,
        },
    ]
    result = evaluate_questions(fake_model(), Config(width=8), questions)
    assert result["accuracy"] == 0.5
    assert result["pair_accuracy"] == 0
    assert result["changed_prediction_fraction"] == 0


def test_prompt_separator_matches_plain_and_chat_formats(monkeypatch):
    import mlm.comprehension as module

    seen = []

    def scorer(state, config, prompts, answers, messages):
        seen.append((prompts, answers))
        return [0.0] * len(answers)

    monkeypatch.setattr(module, "score_candidates", scorer)
    questions = [
        {
            "id": "a",
            "pair": "p",
            "context": "It is a.",
            "question": "What?",
            "choices": ["a", "b"],
            "answer_index": 0,
        },
        {
            "id": "b",
            "pair": "p",
            "context": "It is b.",
            "question": "What?",
            "choices": ["a", "b"],
            "answer_index": 1,
        },
    ]
    module.evaluate_questions(fake_model(), Config(width=8), questions, format="plain")
    module.evaluate_questions(fake_model(), Config(width=8), questions, format="chat")
    assert all(prompt.endswith("Answer: ") for prompt in seen[0][0])
    assert all(prompt.endswith("<|assistant|>\n") for prompt in seen[1][0])
    assert seen[0][1] == ["a", "b", "a", "b"]
    assert seen[1][1] == seen[0][1]
