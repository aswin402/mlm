import itertools
from collections import defaultdict

import numpy as np
import pytest

from mlm.binding_data import NAMES, make_split
from mlm.binding_run import better, build_batches
from mlm.config import Config
from mlm.retrieval_eval import aggregate


def test_binding_all_pairings_groups_and_split_names():
    texts = {}
    for split in ["train", "validation", "test"]:
        records, questions = make_split(split, 81)
        assert (records, questions) == make_split(split, 81)
        texts[split] = {r["text"] for r in records}
        assert len(texts[split]) == len(records)
        groups = defaultdict(list)
        entity_pairs = set()
        for record, q in zip(records, questions, strict=True):
            assert q["distractors"] == 1 and q["delay_bytes"] == 0
            assert len(record["text"].encode()) + 1 <= 128
            assert q["choices"][q["answer_index"]] == dict(q["facts"])[q["query_entity"]]
            groups[q["group"]].append(q)
            entity_pairs.add(tuple(sorted(n for n, _ in q["facts"])))
        assert entity_pairs == set(itertools.combinations(sorted(NAMES[split]), 2))
        for rows in groups.values():
            assert len(rows) == 4
            assert sorted(r["answer_index"] for r in rows) == [0, 0, 1, 1]
            for a in [0, 1]:
                same = [r for r in rows if r["assignment"] == a]
                assert same[0]["context"] == same[1]["context"]
                assert same[0]["query_entity"] != same[1]["query_entity"]
        cfg = Config(nodes=4, width=8, batch_size=4, length=128)
        batches, identities = build_batches(records, questions, cfg)
        assert len(batches) == len(groups) == len(identities)
        for batch in batches:
            assert batch["resets"][:, 0].all()
            np.testing.assert_array_equal(batch["loss_mask"].sum(axis=1), [6] * 4)
        perfect = [{**q, "prediction": q["answer_index"], "correct": True} for q in questions]
        assert aggregate(perfect)["binding_group_accuracy"] == 1
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        assert not texts[a] & texts[b]
        assert not set(NAMES[a]) & set(NAMES[b])


def test_binding_selection_prefers_complete_correctness():
    assert better(0.5, 8.0, (0.25, 0.001))
    assert not better(0.25, 0.001, (0.5, 8.0))
    assert better(0.5, 0.1, (0.5, 0.2))
    assert not better(0.5, 0.2, (0.5, 0.2))
    records, questions = make_split("validation", 82)
    with pytest.raises(ValueError):
        build_batches(records[:-1], questions[:-1], Config(batch_size=4, length=128))
    with pytest.raises(ValueError):
        build_batches(records, questions, Config(batch_size=2, length=128))
    with pytest.raises(ValueError):
        build_batches(records, questions, Config(batch_size=4, length=16))


def test_grouped_runner_saves_selected_state_without_mutating_source(tmp_path):
    import hashlib
    import json
    from types import SimpleNamespace

    from mlm.binding_data import prepare
    from mlm.binding_run import run
    from mlm.checkpoint import load_checkpoint, save_checkpoint
    from mlm.training import create_state

    data = tmp_path / "data"
    prepare(data)
    cfg = Config(nodes=4, width=8, batch_size=4, length=128)
    config = tmp_path / "config.json"
    config.write_text(cfg.model_dump_json())
    source = tmp_path / "source.mlm"
    save_checkpoint(source, create_state(cfg).replace(step=12), cfg, {})
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    out = tmp_path / "run"
    args = SimpleNamespace(data=data, config=config, init=source, out=out, steps=2, eval_every=2)
    run(args)
    state, _, meta = load_checkpoint(out / "latest.mlm")
    assert int(state.step) == 2 and meta["position"] == 2
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    rows = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    selected = min(
        rows, key=lambda r: (-r["development"]["binding_group_accuracy"], r["target_loss"])
    )
    best, _, _ = load_checkpoint(out / "best.mlm")
    assert int(best.step) == selected["step"]
    assert json.loads((out / "summary.json").read_text())["supervised_payload_bytes"] == 40
    with pytest.raises(ValueError, match="empty"):
        run(args)
