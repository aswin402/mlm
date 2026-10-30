"""Evaluation-only paired reading probes ranked by frozen neural likelihoods."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from mlm.checkpoint import load_checkpoint
from mlm.data import BOS, EOS
from mlm.model import initial_state


def score_candidates(state, config, prompts, answers, messages=True):
    """Mean answer-byte log probability; the complete prompt includes any separator."""
    if not prompts or len(prompts) != len(answers):
        raise ValueError("Prompts and answers must have equal nonzero length")
    rows = []
    for prompt, answer in zip(prompts, answers, strict=True):
        if not isinstance(answer, str) or not answer:
            raise ValueError("Candidate answers must be nonempty strings")
        prefix = [BOS, *prompt.encode("utf-8")]
        suffix = list(answer.encode("utf-8"))
        ids = prefix + suffix
        if len(ids) - 1 > 1024:
            raise ValueError("Probe exceeds the 1024-byte evaluation limit")
        rows.append((ids, len(prefix) - 1))
    length = ((max(len(ids) - 1 for ids, _ in rows) + 31) // 32) * 32
    max_bs = 2 if (config.width * config.nodes >= 512 or length >= 256) else 8
    batch_size = min(max_bs, len(rows))
    outputs = []

    @jax.jit
    def compute(params, tokens, targets, valid, answer_mask):
        resets = jnp.zeros_like(tokens, dtype=bool).at[:, 0].set(True)
        logits, _ = state.apply_fn(
            {"params": params},
            tokens,
            initial_state(config, batch_size),
            resets,
            messages=messages,
            valid=valid,
        )
        chosen = jnp.take_along_axis(jax.nn.log_softmax(logits), targets[..., None], -1)[..., 0]
        return (chosen * answer_mask).sum(-1) / jnp.maximum(answer_mask.sum(-1), 1)

    for start in range(0, len(rows), batch_size):
        inputs = np.full((batch_size, length), EOS, np.int32)
        targets = inputs.copy()
        valid = np.zeros_like(inputs, dtype=bool)
        answer_mask = np.zeros_like(inputs, dtype=np.float32)
        for i, (ids, answer_start) in enumerate(rows[start : start + batch_size]):
            count = len(ids) - 1
            inputs[i, :count], targets[i, :count] = ids[:-1], ids[1:]
            valid[i, :count] = True
            answer_mask[i, answer_start:count] = 1
        scores = np.asarray(
            compute(
                state.params,
                jnp.asarray(inputs),
                jnp.asarray(targets),
                jnp.asarray(valid),
                jnp.asarray(answer_mask),
            )
        )
        outputs.extend(float(value) for value in scores[: min(batch_size, len(rows) - start)])
    if not np.isfinite(outputs).all():
        raise FloatingPointError("Nonfinite answer likelihood")
    return outputs


def evaluate_questions(state, config, questions, messages=True, format="plain"):
    if format not in ("plain", "chat") or not questions:
        raise ValueError("Nonempty questions and a valid format required")
    ids, prompts, answers, offsets = set(), [], [], []
    pairs = defaultdict(list)
    for question in questions:
        if question["id"] in ids:
            raise ValueError("Duplicate probe ID")
        ids.add(question["id"])
        choices = question["choices"]
        if len(choices) != 2 or len(set(choices)) != 2 or question["answer_index"] not in (0, 1):
            raise ValueError("Each probe needs two distinct choices and a valid answer index")
        if format == "plain":
            prompt = f"{question['context']}\nQuestion: {question['question']}\nAnswer: "
        else:
            prompt = f"<|user|>\n{question['context']}\n{question['question']}\n<|assistant|>\n"
        offsets.append(len(answers))
        for choice in choices:
            prompts.append(prompt)
            answers.append(choice)
        pairs[question["pair"]].append(question)
    for members in pairs.values():
        if (
            len(members) != 2
            or members[0]["choices"] != members[1]["choices"]
            or members[0]["question"] != members[1]["question"]
            or members[0]["answer_index"] == members[1]["answer_index"]
            or members[0]["context"] == members[1]["context"]
        ):
            raise ValueError(
                "Each pair must change context and correct answer, keeping question/choices"
            )
    # Expected indices are never passed into the neural scorer.
    likelihoods = score_candidates(state, config, prompts, answers, messages)
    results = []
    for question, offset in zip(questions, offsets, strict=True):
        scores = likelihoods[offset : offset + 2]
        prediction = int(np.argmax(scores))
        results.append(
            {
                **question,
                "scores": scores,
                "prediction": prediction,
                "predicted_answer": question["choices"][prediction],
                "correct": prediction == question["answer_index"],
            }
        )
    by_pair = defaultdict(list)
    for row in results:
        by_pair[row["pair"]].append(row)
    return {
        "questions": len(results),
        "pairs": len(by_pair),
        "accuracy": sum(row["correct"] for row in results) / len(results),
        "pair_accuracy": sum(all(row["correct"] for row in group) for group in by_pair.values())
        / len(by_pair),
        "changed_prediction_fraction": sum(
            group[0]["prediction"] != group[1]["prediction"] for group in by_pair.values()
        )
        / len(by_pair),
        "balanced_fixed_choice_accuracy": 0.5,
        "format": format,
        "score_rule": (
            "mean conditional log probability of answer bytes; separator belongs to prompt"
        ),
        "scorer_version": 2,
        "items": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--questions", type=Path, default=Path("data/evaluation/reading-v1.json"))
    parser.add_argument("--format", choices=("plain", "chat"), default="plain")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    state, config, meta = load_checkpoint(args.checkpoint)
    questions = json.loads(args.questions.read_text())["questions"]
    result = evaluate_questions(state, config, questions, meta.get("messages", True), args.format)
    result["checkpoint"] = str(args.checkpoint)
    result["checkpoint_sha256"] = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    result["questions_sha256"] = hashlib.sha256(args.questions.read_bytes()).hexdigest()
    result["architecture"] = config.architecture
    result["messages"] = meta.get("messages", True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.out)
    print(json.dumps({key: value for key, value in result.items() if key != "items"}, indent=2))


if __name__ == "__main__":
    main()
