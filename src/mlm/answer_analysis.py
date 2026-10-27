"""Read-only answer rankings and explicitly aligned teacher-target NLL partitions."""

from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np
import optax

from mlm.binding_run import build_batches
from mlm.comprehension import evaluate_questions
from mlm.data import encode
from mlm.model import initial_state
from mlm.retrieval_eval import aggregate


def prepare_answers(records, questions, config):
    if len(records) != len(questions) or len({q["id"] for q in questions}) != len(questions):
        raise ValueError("Unique aligned question records required")
    for record, q in zip(records, questions, strict=True):
        prompt = f"<|user|>\n{q['context']}\n{q['question']}\n<|assistant|>\n"
        answer = q["choices"][q["answer_index"]]
        start = len(prompt.encode())
        end = start + len(answer.encode())
        if (
            not answer
            or record["text"] != prompt + answer + "\n"
            or record.get("group") != q["group"]
        ):
            raise ValueError("Record/gold question text mismatch")
        if record["loss_spans"] != [[start, end + 1]] or record.get("score_eos") is not True:
            raise ValueError("Exact answer/newline/EOS supervision required")
    batches, ids = build_batches(records, questions, config)
    batch = {k: np.concatenate([b[k] for b in batches], axis=0) for k in batches[0]}
    ordered = []
    aligned = []
    for identity in ids:
        for record, q in zip(records, questions, strict=True):
            if q["group"] == identity:
                ordered.append(q)
                aligned.append(record)
    masks = {k: np.zeros_like(batch["mask"]) for k in ["first", "rest", "newline", "eos"]}
    for row, (record, q) in enumerate(zip(aligned, ordered, strict=True)):
        tokens = encode(record["text"])
        count = len(tokens) - 1
        if (
            not np.array_equal(batch["inputs"][row, :count], tokens[:-1])
            or not np.array_equal(batch["targets"][row, :count], tokens[1:])
            or not np.all(batch["mask"][row, :count] == 1)
            or np.any(batch["mask"][row, count:])
        ):
            raise ValueError("Teacher batch token/valid-mask alignment mismatch")
        start, end = record["loss_spans"][0]
        masks["first"][row, start] = 1
        masks["rest"][row, start + 1 : end - 1] = 1
        masks["newline"][row, end - 1] = 1
        masks["eos"][row, end] = 1
    if not np.array_equal(sum(masks.values()), batch["mask"] * batch["loss_mask"]):
        raise ValueError("Teacher supervised partition mismatch")
    return batch, ordered, masks


def summarize_nll(nll, masks, questions):
    nll = np.asarray(nll, dtype=np.float64)
    if (
        not np.isfinite(nll).all()
        or nll.shape != next(iter(masks.values())).shape
        or nll.shape[0] != len(questions)
    ):
        raise ValueError("Finite aligned target NLLs required")

    def stats(mask):
        count = int(mask.sum())
        total = float((nll * mask).sum())
        return {"count": count, "sum_nll": total, "mean_nll": total / count if count else None}

    parts = {k: stats(m) for k, m in masks.items()}
    answer = masks["first"] + masks["rest"]
    total = sum(masks.values())
    items = []
    for row, q in enumerate(questions):
        entries = []
        for role, mask in masks.items():
            entries.extend(
                {"position": int(p), "role": role, "nll": float(nll[row, p])}
                for p in np.flatnonzero(mask[row])
            )
        entries.sort(key=lambda x: x["position"])
        count = int(answer[row].sum())
        items.append(
            {
                "id": q["id"],
                "group": q["group"],
                "targets": entries,
                "gold_answer_mean_nll": float((nll[row] * answer[row]).sum() / count),
            }
        )
    return {"parts": parts, "answer_only": stats(answer), "total": stats(total), "items": items}


def ranking_transitions(control, active):
    if not control or len(control) != len(active):
        raise ValueError("Aligned nonempty ranking items required")
    result = {
        "both_correct": 0,
        "control_only": 0,
        "auxiliary_only": 0,
        "both_wrong": 0,
        "prediction_flips": 0,
        "net_correct_change": 0,
    }
    for c, a in zip(control, active, strict=True):
        if any(c[k] != a[k] for k in ["id", "group", "choices", "answer_index"]):
            raise ValueError("Ranking identity/label mismatch")
        name = (
            "both_correct"
            if c["correct"] and a["correct"]
            else "control_only"
            if c["correct"]
            else "auxiliary_only"
            if a["correct"]
            else "both_wrong"
        )
        result[name] += 1
        result["prediction_flips"] += int(c["prediction"] != a["prediction"])
        result["net_correct_change"] += int(a["correct"]) - int(c["correct"])
    return result


def analyze_answers(state, config, records, questions):
    batch, ordered, masks = prepare_answers(records, questions, config)

    @jax.jit
    def compute(params):
        b = {k: jnp.asarray(v) for k, v in batch.items()}
        logits, carry = state.apply_fn(
            {"params": params},
            b["inputs"],
            initial_state(config, b["inputs"].shape[0]),
            b["resets"],
            valid=b["mask"],
        )
        return optax.softmax_cross_entropy_with_integer_labels(logits, b["targets"]), carry

    nll, carry = compute(state.params)
    if not np.isfinite(np.asarray(carry)).all():
        raise FloatingPointError("Nonfinite teacher carry")
    teacher = summarize_nll(nll, masks, ordered)
    ranked = evaluate_questions(state, config, ordered, format="chat")
    ranked["overall"] = aggregate(ranked["items"])
    groups = defaultdict(list)
    for row in ranked["items"]:
        row["gold_minus_other_margin"] = (
            row["scores"][row["answer_index"]] - row["scores"][1 - row["answer_index"]]
        )
        row["exact_tie"] = row["scores"][0] == row["scores"][1]
        row["near_tie"] = abs(row["gold_minus_other_margin"]) <= 1e-6
        groups[row["group"]].append(row)
    ranked["correct_count"] = sum(r["correct"] for r in ranked["items"])
    ranked["complete_groups"] = sum(all(r["correct"] for r in rows) for rows in groups.values())
    ranked["exact_ties"] = sum(r["exact_tie"] for r in ranked["items"])
    ranked["near_ties"] = sum(r["near_tie"] for r in ranked["items"])
    errors = [
        abs(t["gold_answer_mean_nll"] + r["scores"][r["answer_index"]])
        for t, r in zip(teacher["items"], ranked["items"], strict=True)
    ]
    maximum = max(errors)
    if maximum > 1e-5:
        raise ValueError("Teacher/gold candidate score mismatch")
    return {"ranking": ranked, "teacher": teacher, "score_teacher_max_error": maximum}


def paired_changes(control, active):
    c, a = control["teacher"], active["teacher"]
    if c["total"]["count"] != a["total"]["count"] or set(c["parts"]) != set(a["parts"]):
        raise ValueError("Comparable target partitions required")
    parts = {}
    for key, v in c["parts"].items():
        w = a["parts"][key]
        if v["count"] != w["count"]:
            raise ValueError("Partition counts differ")
        parts[key] = {
            "mean_nll_delta": w["mean_nll"] - v["mean_nll"] if v["count"] else None,
            "total_ce_contribution": (w["sum_nll"] - v["sum_nll"]) / c["total"]["count"],
        }
    return {
        "ranking_transitions": ranking_transitions(
            control["ranking"]["items"], active["ranking"]["items"]
        ),
        "qa_ce_delta": a["total"]["mean_nll"] - c["total"]["mean_nll"],
        "answer_ce_delta": a["answer_only"]["mean_nll"] - c["answer_only"]["mean_nll"],
        "parts": parts,
        "margin_changes": [
            {
                "id": x["id"],
                "margin_delta": y["gold_minus_other_margin"] - x["gold_minus_other_margin"],
            }
            for x, y in zip(control["ranking"]["items"], active["ranking"]["items"], strict=True)
        ],
    }
