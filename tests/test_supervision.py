import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mlm.config import Config
from mlm.data import read_training_data
from mlm.model import initial_state
from mlm.stream import DocumentStream, make_stream_step, stream_objective
from mlm.training import create_state


def test_byte_spans_unicode_and_eos(tmp_path):
    path = tmp_path / "data.jsonl"
    # Span offsets are UTF-8 bytes, not character indices.
    text = "user é: answer"
    start = len("user é: ".encode())
    path.write_text(
        json.dumps({"text": text, "loss_spans": [[start, len(text.encode())]], "score_eos": True})
        + "\n"
    )
    docs, masks, identity = read_training_data(path)
    assert docs == [text]
    assert masks[0].tolist() == [0] * start + [1] * 6 + [1]
    assert identity
    before = identity
    path.write_text(
        json.dumps({"text": text, "loss_spans": [[0, len(text.encode())]], "score_eos": True})
        + "\n"
    )
    assert read_training_data(path)[2] != before
    path.write_text(json.dumps({"text": "a", "loss_spans": [[1, 3]]}) + "\n")
    with pytest.raises(ValueError):
        read_training_data(path)


def test_prompt_only_chunk_advances_memory_not_optimizer():
    cfg = Config(nodes=4, width=8, degree=2, batch_size=1, length=4)
    state = create_state(cfg)
    mask = np.array([0] * 8 + [1] * 5, dtype=np.float32)
    stream = DocumentStream(["abcdefghijkl"], 1, 4, loss_masks=[mask])
    carry = initial_state(cfg, 1)
    step = make_stream_step(cfg)
    first = jax.tree.map(jnp.asarray, stream.next_batch())
    updated, hidden, metrics = step(state, carry, first)
    assert not bool(metrics["updated"])
    assert int(updated.step) == 0
    assert not np.allclose(hidden, carry)
    for a, b in zip(jax.tree.leaves(state), jax.tree.leaves(updated), strict=True):
        np.testing.assert_array_equal(a, b)
    step(updated, hidden, jax.tree.map(jnp.asarray, stream.next_batch()))
    third = jax.tree.map(jnp.asarray, stream.next_batch())
    trained, _, metrics = step(updated, hidden, third)
    assert bool(metrics["updated"])
    assert int(trained.step) == 1
    third = {**third, "loss_mask": third["loss_mask"].at[0, 0].set(0)}
    changed = {**third, "targets": third["targets"].at[0, 0].set(99)}
    mixed_x = stream_objective(state.params, state.apply_fn, third, carry)[0]
    mixed_y = stream_objective(state.params, state.apply_fn, changed, carry)[0]
    np.testing.assert_array_equal(mixed_x, mixed_y)
    # A masked target must have no loss contribution.
    first_changed = first.copy()
    first_changed["targets"] = first["targets"].at[0, 0].set(99)
    x = stream_objective(state.params, state.apply_fn, first, carry)[0]
    y = stream_objective(state.params, state.apply_fn, first_changed, carry)[0]
    np.testing.assert_array_equal(x, y)
    other = DocumentStream(["abcdefghijkl"], 1, 4, loss_masks=[np.ones(13)])
    with pytest.raises(ValueError, match="loss"):
        other.restore(stream.snapshot())


def test_public_role_renderer_masks_only_assistant_bytes():
    from mlm.supervised_data import assistant_record

    record = assistant_record(
        [{"role": "user", "content": "é?"}, {"role": "assistant", "content": "Yes."}]
    )
    start, end = record["loss_spans"][0]
    assert record["text"].encode()[start:end] == b"Yes.\n"
    assert record["score_eos"]
    assert record["text"].encode()[:start].endswith(b"<|assistant|>\n")


def test_reading_splits_are_balanced_and_disjoint():
    from mlm.supervised_data import NAMES, reading_split

    sets = {}
    for i, split in enumerate(("train", "validation", "test")):
        pairs = 800 if split == "train" else 64
        records, questions = reading_split(split, pairs, 13 + i)
        assert len(records) == len(questions) == 2 * pairs
        assert len({record["text"] for record in records}) == len(records)
        assert sum(q["answer_index"] for q in questions) == pairs
        assert all(q["template_partition"] == split for q in questions)
        for record, q in zip(records, questions, strict=True):
            start, end = record["loss_spans"][0]
            assert (
                record["text"].encode()[start:end].decode().strip()
                == q["choices"][q["answer_index"]]
            )
        sets[split] = {r["text"] for r in records}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert not sets[a] & sets[b]
        assert not set(NAMES[a]) & set(NAMES[b])


def test_assistant_stream_resume_and_mask_change_guard(tmp_path):
    from mlm.checkpoint import load_checkpoint
    from mlm.cli import main

    train, dev = tmp_path / "train.jsonl", tmp_path / "dev.jsonl"
    train.write_text(
        json.dumps({"text": "abcdefghijkl", "loss_spans": [[8, 12]], "score_eos": True}) + "\n"
    )
    dev.write_text(
        json.dumps({"text": "different prompt!", "loss_spans": [[13, 17]], "score_eos": True})
        + "\n"
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"nodes": 4, "width": 8, "degree": 2, "batch_size": 1, "length": 4})
    )
    common = [
        "stream-train",
        "--train",
        str(train),
        "--validation",
        str(dev),
        "--config",
        str(config),
        "--eval-every",
        "2",
    ]
    run = tmp_path / "run"
    main([*common, "--out", str(run), "--steps", "3"])
    summary = json.loads((run / "summary.json").read_text())
    assert summary["prompt_only_chunks"] > 0
    assert summary["validation"]["payload_bytes"] == 4
    assert summary["validation"]["consumed_payload_bytes"] == 17
    main([*common, "--out", str(run), "--resume", str(run / "latest.mlm"), "--steps", "2"])
    other = tmp_path / "other"
    main([*common, "--out", str(other), "--steps", "5"])
    a, _, meta_a = load_checkpoint(run / "latest.mlm")
    b, _, meta_b = load_checkpoint(other / "latest.mlm")
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_array_equal(x, y)
    np.testing.assert_array_equal(meta_a["carry"], meta_b["carry"])
    assert meta_a["stream"] == meta_b["stream"]
    train.write_text(
        json.dumps({"text": "abcdefghijkl", "loss_spans": [[0, 12]], "score_eos": True}) + "\n"
    )
    with pytest.raises(ValueError, match="supervision"):
        main([*common, "--out", str(run), "--resume", str(run / "latest.mlm"), "--steps", "1"])


def test_supervised_data_rejects_prefix_evaluation(tmp_path):
    from mlm.checkpoint import save_checkpoint
    from mlm.cli import main

    cfg = Config(nodes=4, width=8, degree=2)
    checkpoint = tmp_path / "model.mlm"
    save_checkpoint(checkpoint, create_state(cfg), cfg, {})
    data = tmp_path / "data.jsonl"
    data.write_text(
        json.dumps({"text": "prompt answer", "loss_spans": [[7, 13]], "score_eos": True}) + "\n"
    )
    with pytest.raises(ValueError, match="full-documents"):
        main(["evaluate", str(checkpoint), "--data", str(data)])


def test_global_selection_schedule_and_relocated_best(tmp_path, monkeypatch):
    from argparse import Namespace

    import mlm.streamrun as runner
    from mlm.checkpoint import load_checkpoint

    train, dev = tmp_path / "train.txt", tmp_path / "dev.txt"
    train.write_text("training text")
    dev.write_text("development text")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"nodes": 4, "width": 8, "degree": 2, "length": 4, "batch_size": 1})
    )
    monkeypatch.setattr(runner, "generate", lambda *args, **kwargs: "")

    def score(model, *args, **kwargs):
        return {"loss": 10.0 - int(model.step)}

    monkeypatch.setattr(runner, "evaluate_stream", score)

    def run(out, steps, interval=2, resume=None):
        runner.run(
            Namespace(
                train=train,
                validation=dev,
                test=None,
                config=config,
                out=out,
                steps=steps,
                eval_every=interval,
                resume=resume,
                init=None,
                no_messages=False,
            )
        )

    out = tmp_path / "schedule"
    run(out, 3)
    best, _, _ = load_checkpoint(out / "best.mlm")
    assert int(best.step) == 2  # endpoint 3 is logged, not a selection candidate
    run(out, 3, resume=out / "latest.mlm")
    records = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in records if r["selection_eligible"]] == [2, 4, 6]
    # A separate decreasing-quality run must keep its historical best after relocation.
    monkeypatch.setattr(
        runner, "evaluate_stream", lambda model, *a, **kw: {"loss": 1.0 + int(model.step)}
    )
    source = tmp_path / "source"
    run(source, 1, interval=1)
    destination = tmp_path / "relocated"
    run(destination, 1, interval=1, resume=source / "latest.mlm")
    best, _, _ = load_checkpoint(destination / "best.mlm")
    assert int(best.step) == 0

    # A legacy endpoint selected at step 3 cannot silently enter an interval-2 study.
    legacy = tmp_path / "legacy"
    monkeypatch.setattr(runner, "evaluate_stream", score)
    run(legacy, 3)
    latest, cfg, meta = load_checkpoint(legacy / "latest.mlm")
    meta.pop("eval_every", None)
    from mlm.checkpoint import save_checkpoint

    save_checkpoint(legacy / "best.mlm", latest, cfg, meta)
    with pytest.raises(ValueError, match="selection"):
        run(legacy, 1, resume=legacy / "latest.mlm")
