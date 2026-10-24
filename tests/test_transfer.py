import hashlib
import json
from collections import Counter

import pytest

from mlm.data import encode
from mlm.gradual_eval import order_robust
from mlm.retrieval_eval import aggregate
from mlm.transfer_data import prepare


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def validate_rendered_record(record, question):
    fact_forms = [
        "{name}'s color is {value}.",
        "{name} has color {value}.",
        "{name}'s chosen color is {value}.",
    ]
    query_forms = [
        "What color belongs to {name}?",
        "Which color does {name} have?",
        "What is the color of {name}?",
    ]
    styles = {"f0q0": (0, 0), "f0q1": (0, 1), "f1q0": (1, 0), "f1q1": (1, 1), "heldout": (2, 2)}
    fi, qi = styles[question["wording"]]
    facts = dict(question["facts"])
    assert len(facts) == 2 and question["query_entity"] in facts
    assert set(facts.values()) == set(question["choices"])
    answer = facts[question["query_entity"]]
    assert question["choices"][question["answer_index"]] == answer
    context = " ".join(fact_forms[fi].format(name=n, value=v) for n, v in question["facts"])
    query = query_forms[qi].format(name=question["query_entity"])
    assert question["context"] == context
    assert question["question"] == query
    prompt = f"<|user|>\n{context}\n{query}\n<|assistant|>\n"
    expected = prompt + answer + "\n"
    assert record["text"] == expected
    assert record["loss_spans"] == [[len(prompt.encode()), len(expected.encode())]]
    assert record["score_eos"] is True


def test_transfer_splits_renderers_and_paired_episodes(tmp_path):
    out = tmp_path / "data"
    manifest = prepare(out, 904217)
    again = tmp_path / "again"
    assert prepare(again, 904217) == manifest
    pools = manifest["names"]
    assert all(len(v) == 16 for v in pools.values())
    assert not set(pools["train"]) & set(pools["validation"])
    assert not set(pools["train"]) & set(pools["test"])
    assert not set(pools["validation"]) & set(pools["test"])
    train_pairs = {}
    texts = {}
    for arm, groups in [("fixed", 16), ("diverse", 128), ("mixed", 128)]:
        records = read(out / arm / "train.jsonl")
        qs = json.loads((out / arm / "train-questions.json").read_text())["questions"]
        assert len(records) == len(qs) == groups * 4
        assert {n for q in qs for n, _ in q["facts"]} == set(pools["train"])
        assert all(q["wording"] != "heldout" for q in qs)
        assert max(len(encode(r["text"])) - 1 for r in records) <= 128
        assert set(Counter(q["group"] for q in qs).values()) == {4}
        for r, q in zip(records, qs, strict=True):
            validate_rendered_record(r, q)
            assert q["choices"][q["answer_index"]] == dict(q["facts"])[q["query_entity"]]
            assert r["group"] == q["group"]
        perfect = [{**q, "prediction": q["answer_index"], "correct": True} for q in qs]
        assert aggregate(perfect)["binding_group_accuracy"] == 1
        assert order_robust(perfect)["all_correct"] == groups // 2
        assert all(
            "chosen color" not in r["text"] and "What is the color of" not in r["text"]
            for r in records
        )
        assert {v for q in qs for v in q["choices"]} == {"blue", "gold", "lime", "pink"}
        train_pairs[arm] = {tuple(sorted(n for n, _ in q["facts"])) for q in qs}
        texts[arm] = {r["text"] for r in records}
    assert len(train_pairs["fixed"]) == 8 and len(train_pairs["diverse"]) == 32
    assert train_pairs["fixed"] <= train_pairs["diverse"] == train_pairs["mixed"]
    assert texts["fixed"] <= texts["diverse"]
    canonical = json.loads((out / "diverse/train-questions.json").read_text())["questions"]
    mixed = json.loads((out / "mixed/train-questions.json").read_text())["questions"]
    for a, b in zip(canonical, mixed, strict=True):
        for key in [
            "id",
            "pair",
            "group",
            "facts",
            "query_entity",
            "choices",
            "answer_index",
            "pair_type",
        ]:
            assert a[key] == b[key]
    assert Counter(q["wording"] for q in mixed) == {
        "f0q0": 128,
        "f0q1": 128,
        "f1q0": 128,
        "f1q1": 128,
    }
    assert Counter(q["pair_type"] for q in canonical) == {"within": 256, "cross": 256}
    evaluation = {}
    for split in ["validation", "test"]:
        qs = json.loads((out / "fixed" / f"{split}-questions.json").read_text())["questions"]
        evaluation[split] = qs
        records = read(out / "fixed" / f"{split}.jsonl")
        assert len(qs) == len(records) == 2304
        for r, q in zip(records, qs, strict=True):
            validate_rendered_record(r, q)
            expected_style = {"base": "f0q0", "alternate": "f1q1", "heldout": "heldout"}[
                q["category"].rsplit("/", 1)[-1]
            ]
            assert q["wording"] == expected_style
        assert max(len(encode(r["text"])) - 1 for r in records) <= 128
        assert len({q["id"] for q in qs}) == len(qs)
        assert not {r["text"] for r in records} & set.union(*texts.values())
        for cat, count in Counter(q["category"] for q in qs).items():
            assert count == 384
            actual_names = {n for q in qs if q["category"] == cat for n, _ in q["facts"]}
            expected_names = set(pools["train"] if cat.startswith("familiar") else pools[split])
            assert actual_names == expected_names
            subset = [{**q, "correct": True} for q in qs if q["category"] == cat]
            assert order_robust(subset)["all_correct"] == 48
            assert Counter(q["pair_type"] for q in subset) == {"within": 192, "cross": 192}
        familiar = {
            tuple(sorted(n for n, _ in q["facts"]))
            for q in qs
            if q["category"].startswith("familiar")
        }
        assert not familiar & train_pairs["diverse"]
        for arm in ["diverse", "mixed"]:
            assert (out / arm / f"{split}-questions.json").read_bytes() == (
                out / "fixed" / f"{split}-questions.json"
            ).read_bytes()
    known = [
        {
            tuple(sorted(n for n, _ in q["facts"]))
            for q in evaluation[s]
            if q["category"].startswith("familiar")
        }
        for s in ["validation", "test"]
    ]
    assert not known[0] & known[1]
    for path, expected in manifest["sha256"].items():
        assert hashlib.sha256((out / path).read_bytes()).hexdigest() == expected
    with pytest.raises(ValueError, match="fresh"):
        prepare(out, 904217)
