import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mlm.config import Config
from mlm.data import encode, read_documents
from mlm.model import initial_state
from mlm.stream import DocumentStream, evaluate_stream, make_stream_step
from mlm.training import create_state


def test_jsonl_keeps_paragraphs_and_conversations(tmp_path):
    path = tmp_path / "docs.jsonl"
    path.write_text(json.dumps({"text": "hello\n\nworld"}) + "\n")
    assert read_documents(path) == ["hello\n\nworld"]


def test_stream_covers_every_target_and_restores_cursor():
    docs = ["abcdefghijk", "é", "xyz"]
    stream = DocumentStream(docs, 1, 3, seed=7, repeat=False, shuffle=False)
    targets, resets = [], 0
    while (batch := stream.next_batch()) is not None:
        targets.extend(batch["targets"][batch["mask"].astype(bool)].tolist())
        resets += int(batch["resets"].sum())
    assert targets == [v for doc in docs for v in encode(doc)[1:].tolist()]
    assert resets == 3
    stream = DocumentStream(docs, 2, 3, seed=9)
    stream.next_batch()
    restored = DocumentStream(docs, 2, 3, seed=9)
    restored.restore(stream.snapshot())
    for _ in range(20):
        a, b = stream.next_batch(), restored.next_batch()
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])


def test_full_evaluation_and_padding_state():
    cfg = Config(nodes=4, width=8, degree=2, batch_size=2, length=4)
    state = create_state(cfg)
    docs = ["a", "abcdefghijklmnop"]
    result = evaluate_stream(state, cfg, docs)
    assert result["payload_bytes"] == sum(len(d.encode()) for d in docs)
    assert result["positions"] == result["payload_bytes"] + len(docs)
    # Padding must not alter a lane's hidden state.
    h = initial_state(cfg, 2) + 0.3
    tokens = jnp.ones((2, 4), dtype=jnp.int32)
    _, frozen = state.apply_fn(
        {"params": state.params},
        tokens,
        h,
        jnp.zeros_like(tokens, dtype=bool),
        valid=jnp.zeros_like(tokens),
    )
    np.testing.assert_array_equal(h, frozen)


def test_stream_update_resumes_carry_and_optimizer(tmp_path):
    from mlm.checkpoint import load_checkpoint, save_checkpoint

    cfg = Config(nodes=4, width=8, degree=2, batch_size=2, length=4)
    state = create_state(cfg)
    stream = DocumentStream(["abcdefghijklmnop", "different text"], 2, 4, seed=4)
    carry = initial_state(cfg, 2)
    step = make_stream_step(cfg)
    batch = jax.tree.map(jnp.asarray, stream.next_batch())
    state, carry, metrics = step(state, carry, batch)
    assert bool(metrics["finite"])
    path = tmp_path / "state.mlm"
    save_checkpoint(
        path, state, cfg, {"stream": stream.snapshot(), "carry": np.asarray(carry).tolist()}
    )
    restored, _, meta = load_checkpoint(path)
    other = DocumentStream(["abcdefghijklmnop", "different text"], 2, 4, seed=4)
    other.restore(meta["stream"])
    a = step(state, carry, jax.tree.map(jnp.asarray, stream.next_batch()))
    b = step(restored, jnp.asarray(meta["carry"]), jax.tree.map(jnp.asarray, other.next_batch()))
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("memory", ["none", "delta"])
def test_stream_cli_exact_resume_and_separate_test(tmp_path, memory):
    from mlm.checkpoint import load_checkpoint
    from mlm.cli import main

    train, validation, test = [
        tmp_path / (name + ".jsonl") for name in ("train", "validation", "test")
    ]
    for path, documents in (
        (train, ["abcdefghijklmnopqrstuv", "another full document"]),
        (validation, ["held out development text"]),
        (test, ["untouched test document"]),
    ):
        path.write_text("".join(json.dumps({"text": doc}) + "\n" for doc in documents))
    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps(
            {
                "nodes": 4,
                "width": 8,
                "degree": 2,
                "length": 8,
                "batch_size": 2,
                "memory": memory,
                "memory_keys": 4,
            }
        )
    )
    common = [
        "stream-train",
        "--train",
        str(train),
        "--validation",
        str(validation),
        "--test",
        str(test),
        "--config",
        str(cfg),
        "--eval-every",
        "2",
    ]
    run = tmp_path / "run"
    main([*common, "--out", str(run), "--steps", "3"])
    main([*common, "--out", str(run), "--steps", "2", "--resume", str(run / "latest.mlm")])
    state, _, meta = load_checkpoint(run / "latest.mlm")
    other = tmp_path / "other"
    main([*common, "--out", str(other), "--steps", "5"])
    expected, _, expected_meta = load_checkpoint(other / "latest.mlm")
    for a, b in zip(jax.tree.leaves(state), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(meta["carry"], expected_meta["carry"])
    assert meta["stream"] == expected_meta["stream"]
    summary = json.loads((run / "summary.json").read_text())
    assert summary["test"]["payload_bytes"] == len("untouched test document".encode())
    assert summary["validation"]["coverage"] == "complete documents"


def test_generate_cli_honors_checkpoint_message_mode(monkeypatch, capsys):
    import mlm.cli as cli

    seen = []
    monkeypatch.setattr(cli, "load_checkpoint", lambda path: (None, None, {"messages": False}))

    def fake_generate(*args, **kwargs):
        seen.append(kwargs.get("messages", True))
        return "sample"

    monkeypatch.setattr(cli, "generate", fake_generate)
    cli.main(["generate", "unused.mlm"])
    assert seen == [False]


def test_resume_preserves_better_latest_after_interruption(tmp_path, monkeypatch):
    from argparse import Namespace

    import mlm.streamrun as runner
    from mlm.checkpoint import load_checkpoint, save_checkpoint
    from mlm.data import corpus_hash

    cfg = Config(nodes=4, width=8, degree=2, batch_size=1, length=4)
    train, validation = tmp_path / "train.txt", tmp_path / "validation.txt"
    train.write_text("abcdefghijk")
    validation.write_text("held out")
    out = tmp_path / "run"
    out.mkdir()
    state = create_state(cfg)
    stream = DocumentStream(["abcdefghijk"], 1, 4, cfg.seed)
    carry = initial_state(cfg, 1)

    def metadata():
        return {
            "mode": "stream",
            "messages": True,
            "corpora": {
                "train": corpus_hash(["abcdefghijk"]),
                "validation": corpus_hash(["held out"]),
            },
            "stream": stream.snapshot(),
            "carry": np.asarray(carry).tolist(),
        }

    save_checkpoint(out / "best.mlm", state, cfg, metadata())
    state, carry, _ = make_stream_step(cfg)(
        state, carry, jax.tree.map(jnp.asarray, stream.next_batch())
    )
    save_checkpoint(out / "latest.mlm", state, cfg, metadata())

    def evaluation(model, *args, **kwargs):
        return {"loss": 1.0 if int(model.step) == 1 else 5.0}

    monkeypatch.setattr(runner, "evaluate_stream", evaluation)
    monkeypatch.setattr(runner, "generate", lambda *args, **kwargs: "")
    runner.run(
        Namespace(
            train=train,
            validation=validation,
            test=None,
            resume=out / "latest.mlm",
            init=None,
            config=None,
            out=out,
            steps=1,
            eval_every=1,
            no_messages=False,
        )
    )
    best, _, _ = load_checkpoint(out / "best.mlm")
    assert int(best.step) == 1
