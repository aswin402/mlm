import json
from collections import Counter

import pytest

from mlm.confirmation_data import prepare
from mlm.gradual_eval import order_robust


def test_balanced_confirmation_data(tmp_path):
    out = tmp_path / "data"
    manifest = prepare(out)
    sets = {}
    pairs = {}
    train_names = set(manifest["names"]["train"])
    new_sets = []
    for split in ["train", "validation", "test"]:
        records = [json.loads(line) for line in (out / f"{split}.jsonl").read_text().splitlines()]
        qs = json.loads((out / f"{split}-questions.json").read_text())["questions"]
        sets[split] = {r["text"] for r in records}
        assert len(sets[split]) == len(records) == len(qs)
        assert max(len(r["text"].encode()) + 1 for r in records) <= 128
        assert order_robust([{**q, "correct": True} for q in qs])["accuracy"] == 1
        pairs[split] = {tuple(sorted(n for n, _ in q["facts"])) for q in qs}
        for record, q in zip(records, qs, strict=True):
            expected = q["choices"][q["answer_index"]]
            assert dict(q["facts"])[q["query_entity"]] == expected
            a, b = record["loss_spans"][0]
            assert record["text"].encode()[a:b].decode() == expected + "\n"
        for category in {q["category"] for q in qs}:
            unique = {
                tuple(sorted(n for n, _ in q["facts"])): q["pair_type"]
                for q in qs
                if q["category"] == category
            }
            count = Counter(unique.values())
            assert count["within"] == count["cross"]
        if split != "train":
            known = {
                n for q in qs if q["category"] == "familiar-names-new-pairs" for n, _ in q["facts"]
            }
            assert known == train_names
            new = {n for q in qs if q["category"] == "new-names" for n, _ in q["facts"]}
            assert not new & train_names and len(new) == 16
            new_sets.append(new)
    assert not new_sets[0] & new_sets[1]
    degree = Counter(n for pair in pairs["train"] for n in pair)
    assert set(degree) == train_names and set(degree.values()) == {4}
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        assert not sets[a] & sets[b] and not pairs[a] & pairs[b]
    assert len(sets["train"]) == 1024
    with pytest.raises(ValueError, match="empty"):
        prepare(out)


def test_retention_selection_rejects_binding_gain_with_english_regression():
    from mlm.confirmation_run import choose

    assert choose(0.8, 0.2, 1.1, 1.25, (0.5, 0.1), True)
    assert not choose(0.8, 0.2, 1.3, 1.25, (0.5, 0.1), True)
    assert choose(0.8, 0.2, 1.3, 1.25, (0.5, 0.1), False)
    assert not choose(0.5, 0.1, 1.1, 1.25, (0.5, 0.1), True)


def test_confirmation_objectives_and_source_integrity(tmp_path):
    import hashlib
    from types import SimpleNamespace

    from mlm.checkpoint import load_checkpoint, save_checkpoint
    from mlm.config import Config
    from mlm.confirmation_run import run
    from mlm.training import create_state

    data = tmp_path / "data"
    prepare(data)
    cfg = Config(nodes=4, width=8, batch_size=4, length=128, learning_rate=0.001)
    config = tmp_path / "config.json"
    config.write_text(cfg.model_dump_json())
    source = tmp_path / "source.mlm"
    save_checkpoint(source, create_state(cfg), cfg, {})
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    english_train = tmp_path / "english-train.txt"
    english_train.write_text("A story.\n\nAnother story.")
    english_dev = tmp_path / "english-dev.txt"
    english_dev.write_text("Held out English.")
    out = tmp_path / "run"
    args = SimpleNamespace(
        data=data,
        config=config,
        init=source,
        out=out,
        steps=1,
        eval_every=1,
        english_train=english_train,
        english_dev=english_dev,
        rehearsal=True,
    )
    run(args)
    latest, _, meta = load_checkpoint(out / "latest.mlm")
    assert int(latest.step) == 2 and meta["qa_step"] == 1
    assert meta["counters"]["qa"]["updates"] == meta["counters"]["english"]["updates"] == 1
    assert meta["counters"]["qa"]["supervised_bytes"] == 20
    assert meta["counters"]["english"]["input_bytes"] > 0
    rows = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert rows[0]["counters"]["qa"]["updates"] == 0
    assert rows[-1]["counters"]["qa"]["updates"] == 1
    best, _, bm = load_checkpoint(out / "best.mlm")
    chosen = next(row for row in rows if row["qa_step"] == bm["qa_step"])
    assert chosen["english_development_loss"] <= rows[0]["english_limit"]
    assert int(best.step) in [0, 2]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    with pytest.raises(ValueError, match="empty"):
        run(args)
