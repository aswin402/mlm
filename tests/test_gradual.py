import itertools
import json
from collections import defaultdict

import pytest

from mlm.binding_run import build_batches
from mlm.config import Config
from mlm.gradual_data import NAMES, SIZES, prepare


def test_nested_curriculum_and_heldout_pair_partitions(tmp_path):
    manifest = prepare(tmp_path / "data")
    root = tmp_path / "data"
    records = [json.loads(line) for line in (root / "train.jsonl").read_text().splitlines()]
    questions = json.loads((root / "train-questions.json").read_text())["questions"]
    batches, ids = build_batches(records, questions, Config(batch_size=4, length=128))
    assert len(batches) == 64
    assert SIZES == [1, 2, 4, 8, 16, 32, 64]
    for count in SIZES:
        assert len({q["group"] for q in questions[: count * 4]}) == count
    source = json.loads(open("results/focused-binding/single-group-questions.json").read())[
        "questions"
    ]
    assert [
        (q["context"], q["question"], q["choices"][q["answer_index"]]) for q in questions[:4]
    ] == [(q["context"], q["question"], q["choices"][q["answer_index"]]) for q in source]
    assert {n for q in questions[:16] for n, _ in q["facts"]} == set(NAMES)
    train_pairs = {tuple(sorted(n for n, _ in q["facts"])) for q in questions}
    assert len(train_pairs) == 20
    sets = {"train": {r["text"] for r in records}}
    heldout = []
    for split in ["validation", "test"]:
        qs = json.loads((root / f"{split}-questions.json").read_text())["questions"]
        rows = [json.loads(line) for line in (root / f"{split}.jsonl").read_text().splitlines()]
        sets[split] = {r["text"] for r in rows}
        assert len(sets[split]) == len(rows) == 384
        assert len({q["group"] for q in qs}) == 96
        pools = defaultdict(list)
        for q in qs:
            pools[q["category"]].append(q)
        assert {len(v) for v in pools.values()} == {192}
        known = {tuple(sorted(n for n, _ in q["facts"])) for q in pools["familiar-names-new-pairs"]}
        assert not train_pairs & known
        assert set(n for pair in known for n in pair) == set(NAMES)
        heldout.append(known)
        new = {n for q in pools["new-names"] for n, _ in q["facts"]}
        assert not new & set(NAMES)
    assert not heldout[0] & heldout[1]
    for a, b in itertools.combinations(sets, 2):
        assert not sets[a] & sets[b]
    assert len(sets["train"]) == 256
    assert manifest["rehearsal"] is False
    with pytest.raises(ValueError, match="empty"):
        prepare(root)


def test_gradual_runner_preserves_optimizer_steps_and_selection(tmp_path):
    import hashlib
    from types import SimpleNamespace

    from mlm.checkpoint import load_checkpoint, save_checkpoint
    from mlm.gradual_run import run
    from mlm.training import create_state

    data = tmp_path / "data"
    prepare(data)
    cfg = Config(nodes=4, width=8, batch_size=4, length=128, learning_rate=0.003)
    config = tmp_path / "config.json"
    config.write_text(cfg.model_dump_json())
    source = tmp_path / "source.mlm"
    save_checkpoint(source, create_state(cfg).replace(step=17), cfg, {})
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    out = tmp_path / "run"
    args = SimpleNamespace(
        data=data,
        config=config,
        init=source,
        out=out,
        strategy="curriculum",
        stage_steps=4,
        stories=None,
    )
    run(args)
    latest, _, meta = load_checkpoint(out / "latest.mlm")
    assert int(latest.step) == 24 and len(meta["seen_names"]) == 8
    rows = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert rows[0]["seen_names"] == []
    assert [r["step"] for r in rows] == [0, 4, 8, 12, 16, 20, 24]
    assert [r["active_groups"] for r in rows] == SIZES
    eligible = [r for r in rows if r["selection_eligible"]]
    assert min(r["step"] for r in eligible) >= 8
    chosen = min(
        eligible, key=lambda r: (-r["development"]["binding_group_accuracy"], r["target_loss"])
    )
    best, _, _ = load_checkpoint(out / "best.mlm")
    assert int(best.step) == chosen["step"]
    assert json.loads((out / "summary.json").read_text())["supervised_payload_bytes"] == 480
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    with pytest.raises(ValueError, match="empty"):
        run(args)


def test_order_robust_evaluation_rejects_position_luck(tmp_path):
    from mlm.gradual_eval import order_robust

    data = tmp_path / "data"
    prepare(data)
    questions = json.loads((data / "validation-questions.json").read_text())["questions"]
    perfect = [{**q, "correct": True} for q in questions]
    assert order_robust(perfect)["accuracy"] == 1
    assert order_robust(perfect)["cases"] == 48
    # A whole group correct in one fact order and wholly wrong in the reverse.
    positional = [{**q, "correct": q["facts"][0][0] < q["facts"][1][0]} for q in questions]
    assert sum(q["correct"] for q in positional) / len(positional) == 0.5
    assert order_robust(positional)["accuracy"] == 0
    with pytest.raises(ValueError):
        order_robust(perfect[:-1])
    duplicate = perfect[:-1] + [perfect[0]]
    with pytest.raises(ValueError):
        order_robust(duplicate)
