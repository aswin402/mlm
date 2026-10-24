import json
from collections import defaultdict

import numpy as np
import pytest

from mlm.data import read_training_data
from mlm.retrieval_data import NAMES, make_split, prepare
from mlm.retrieval_eval import aggregate


def test_controlled_data_labels_balance_and_bounds():
    texts = {}
    for split in ["train", "validation", "test"]:
        records, questions = make_split(split, 17)
        texts[split] = {r["text"] for r in records}
        assert len(texts[split]) == len(records)
        assert (records, questions) == make_split(split, 17)
        pairs, groups = defaultdict(list), defaultdict(list)
        cells = defaultdict(list)
        for record, q in zip(records, questions, strict=True):
            assert len(record["text"].encode()) + 1 <= 384
            assert q["query_entity"] in NAMES[split]
            assert len(q["facts"]) == q["distractors"] + 1
            assert len({v for _, v in q["facts"]}) == len(q["facts"])
            assert len(q["filler"].encode()) == q["delay_bytes"]
            answer = dict(q["facts"])[q["query_entity"]]
            assert q["choices"][q["answer_index"]] == answer
            start, end = record["loss_spans"][0]
            assert record["text"].encode()[start:end].decode() == answer + "\n"
            pairs[q["pair"]].append(q)
            groups[q["group"]].append(q)
            cells[q["category"]].append(q)
        assert len(cells) == 9
        for rows in pairs.values():
            assert len(rows) == 2
            assert rows[0]["question"] == rows[1]["question"]
            assert rows[0]["context"] != rows[1]["context"]
            assert {r["answer_index"] for r in rows} == {0, 1}
        for rows in groups.values():
            if rows[0]["distractors"]:
                assert len(rows) == 4
                for assignment in [0, 1]:
                    same = [r for r in rows if r["assignment"] == assignment]
                    assert len(same) == 2 and same[0]["context"] == same[1]["context"]
                    assert same[0]["question"] != same[1]["question"]
                    assert same[0]["answer_index"] != same[1]["answer_index"]
        for rows in cells.values():
            positions = [
                next(i for i, (n, _) in enumerate(r["facts"]) if n == r["query_entity"])
                for r in rows
            ]
            if rows[0]["distractors"]:
                assert positions.count(0) == positions.count(rows[0]["distractors"])
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        assert not texts[a] & texts[b]
        assert not set(NAMES[a]) & set(NAMES[b])


def test_retrieval_preparation_and_target_masks(tmp_path):
    out = tmp_path / "data"
    prepare(out, source=None, background=0)
    for split in ["train", "validation", "test"]:
        docs, masks, digest = read_training_data(out / f"{split}.jsonl")
        assert digest and len(docs) == len(masks)
        assert all(int(np.sum(m)) == 6 for m in masks)
        basic, _, _ = read_training_data(out / f"basic-{split}.jsonl")
        assert set(basic) <= set(docs)
    with pytest.raises(ValueError, match="empty"):
        prepare(out, source=None, background=0)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["rehearsal_stories"] == 0


def test_binding_aggregation_exposes_wrong_query_flips():
    _, questions = make_split("validation", 19)
    perfect = [{**q, "prediction": q["answer_index"], "correct": True} for q in questions]
    result = aggregate(perfect)
    assert result["accuracy"] == 1 and result["binding_group_accuracy"] == 1
    assert result["query_flip_fraction"] == 1
    wrong = [{**r, "prediction": 1 - r["prediction"], "correct": False} for r in perfect]
    result = aggregate(wrong)
    assert result["binding_group_accuracy"] == 0
    assert result["query_flip_fraction"] == 1
    fixed = [{**r, "prediction": 0, "correct": r["answer_index"] == 0} for r in perfect]
    result = aggregate(fixed)
    assert result["accuracy"] == 0.5 and result["pair_accuracy"] == 0
    assert result["binding_group_accuracy"] == 0 and result["query_flip_fraction"] == 0
    with pytest.raises(ValueError):
        aggregate(perfect[:-1])
