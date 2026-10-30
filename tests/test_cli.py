import json

import numpy as np
import pytest

from mlm.checkpoint import load_checkpoint
from mlm.cli import main
from mlm.generate import generate


def test_training_resume_and_generation(tmp_path):
    train = tmp_path / "train.txt"
    validation = tmp_path / "validation.txt"
    train.write_text("red bird\n\nblue bird")
    validation.write_text("green bird")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"nodes": 4, "width": 8, "degree": 2, "length": 12, "batch_size": 2})
    )
    run = tmp_path / "run"
    args = [
        "train",
        "--train",
        str(train),
        "--validation",
        str(validation),
        "--config",
        str(config),
        "--out",
        str(run),
        "--steps",
        "4",
    ]
    main(args)
    summary = json.loads((run / "summary.json").read_text())
    assert summary["steps"] == 4
    best, _, _ = load_checkpoint(run / "best.mlm")
    assert int(best.step) <= 4
    assert summary["best_validation_loss"] <= summary["initial_validation"]["loss"]
    samples = json.loads((run / "samples.json").read_text())
    assert samples["decoding"] == "greedy"
    assert samples["samples"]
    assert summary["parameters"] > 0
    assert summary["validation"]["payload_bytes"] > 0
    state, cfg, _ = load_checkpoint(run / "latest.mlm")
    first = generate(state, cfg, "red", 8)
    assert first == generate(state, cfg, "red", 8)
    main([*args[:-1], "2", "--resume", str(run / "latest.mlm")])
    resumed, _, _ = load_checkpoint(run / "latest.mlm")
    assert int(resumed.step) == 6
    # Compare uninterrupted training with the resumed continuation.
    other = tmp_path / "uninterrupted"
    main([*args[:-3], str(other), "--steps", "6"])
    expected, _, _ = load_checkpoint(other / "latest.mlm")
    import jax

    for a, b in zip(jax.tree.leaves(resumed), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(a, b)
    train.write_text("changed corpus")
    with pytest.raises(ValueError, match="corpus"):
        main([*args, "--resume", str(run / "latest.mlm")])


def test_rejects_scored_prefix_overlap(tmp_path):
    train = tmp_path / "train.txt"
    val = tmp_path / "val.txt"
    train.write_text("a" * 40 + "train suffix")
    val.write_text("a" * 40 + "validation suffix")
    with pytest.raises(ValueError, match="prefix"):
        main(
            [
                "train",
                "--train",
                str(train),
                "--validation",
                str(val),
                "--out",
                str(tmp_path / "run"),
                "--steps",
                "1",
            ]
        )


def test_resume_cannot_overwrite_unrelated_directory(tmp_path):
    train = tmp_path / "train.txt"
    val = tmp_path / "val.txt"
    train.write_text("red bird")
    val.write_text("green bird")
    run = tmp_path / "first"
    main(
        [
            "train",
            "--train",
            str(train),
            "--validation",
            str(val),
            "--out",
            str(run),
            "--steps",
            "1",
        ]
    )
    unrelated = tmp_path / "other"
    unrelated.mkdir()
    marker = unrelated / "summary.json"
    marker.write_text("do not overwrite")
    with pytest.raises(ValueError, match="destination"):
        main(
            [
                "train",
                "--train",
                str(train),
                "--validation",
                str(val),
                "--out",
                str(unrelated),
                "--steps",
                "1",
                "--resume",
                str(run / "latest.mlm"),
            ]
        )
    assert marker.read_text() == "do not overwrite"


def test_resume_after_checkpoint_interrupt_and_partial_log(tmp_path, monkeypatch):
    import mlm.cli as cli

    train = tmp_path / "train.txt"
    val = tmp_path / "val.txt"
    train.write_text("red bird")
    val.write_text("green bird")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"nodes": 4, "width": 8, "degree": 2, "length": 12, "batch_size": 2})
    )
    run = tmp_path / "run"
    args = [
        "train",
        "--train",
        str(train),
        "--validation",
        str(val),
        "--config",
        str(config),
        "--out",
        str(run),
        "--steps",
        "1",
    ]
    original = cli.save_checkpoint

    def interrupted(path, *values):
        if path.name == "latest.mlm":
            committed = [
                json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()
            ]
            assert all(row["step"] < int(values[0].step) for row in committed)
        original(path, *values)
        if path.name == "latest.mlm":
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr(cli, "save_checkpoint", interrupted)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        main(args)
    # No metrics may claim an update newer than the committed checkpoint.
    state, _, _ = load_checkpoint(run / "latest.mlm")
    records = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert all(record["step"] <= int(state.step) for record in records)
    monkeypatch.setattr(cli, "save_checkpoint", original)
    # A power interruption can leave an incomplete final JSON record.
    with (run / "metrics.jsonl").open("a") as log:
        log.write('{"step":')
    main([*args, "--resume", str(run / "latest.mlm")])
    restored, _, _ = load_checkpoint(run / "latest.mlm")
    assert int(restored.step) == 2
    for line in (run / "metrics.jsonl").read_text().splitlines():
        assert json.loads(line)["step"] <= 2


def test_chat_tui_and_thinking_telemetry(tmp_path, capsys):
    import io

    from mlm.config import Config
    from mlm.generate import format_chat_prompt, generate_with_trace
    from mlm.training import create_state
    from mlm.tui import chat_main, run_chat_tui

    cfg = Config(
        node_update="timescale",
        nodes=4,
        shared_nodes=1,
        top_k_nodes=2,
        width=16,
        degree=2,
        rounds=1,
        memory="relational",
        memory_keys=4,
        memory_decay=True,
        symmetric_memory=True,
        episodic_slots=4,
        relational_hops=3,
    )
    state = create_state(cfg)

    prompt = format_chat_prompt(
        "Where is the key?",
        context="Mia is in the attic. Dan is with Mia. The key is with Dan.",
    )
    reply, trace = generate_with_trace(state, cfg, prompt, max_bytes=16)
    assert reply == "attic"
    assert trace["relational_active"] is True
    assert trace["query_anchor"] == "key"
    assert len(trace["hops"]) == 3
    assert trace["hops"][0]["target_word"] == "Dan"
    assert trace["hops"][1]["target_word"] == "Mia"
    assert trace["hops"][2]["target_word"] == "attic"
    assert trace["onset_word"] == "attic"
    assert trace["speculative_accepted"] >= 4
    assert trace["index_share_density"] > 0.0

    # Autonomous command-free TUI session with natural multi-turn facts & multi-hop QA
    in_buf = io.StringIO(
        "Mia is in the attic. Dan is with Mia. The key is with Dan.\n"
        "Where is the key?\n"
        "Where is Dan?\n"
        "Where is Mia?\n"
        "Tom was in the yard.\n"
        "Tom is in the hall.\n"
        "Where is Tom?\n"
        "quit\n"
    )
    out_buf = io.StringIO()
    saved_ckpt = tmp_path / "tui_saved.mlm"
    new_state, new_cfg = run_chat_tui(
        state,
        cfg,
        ckpt_label="test-init",
        max_bytes=16,
        learn=False,
        show_thinking=True,
        save_path=saved_ckpt,
        input_stream=in_buf,
        output_stream=out_buf,
        color=False,
    )
    output = out_buf.getvalue()
    assert "MLM Interactive Neural Console" in output
    assert "Autonomous Chat: Type facts or questions naturally" in output
    assert "Model Thinking & Telemetry (<think>)" in output
    assert "Hop 1 [Clause C2]: anchor 'key'" in output
    assert "Hop 2 [Clause C1]: bridge 'Dan'" in output
    assert "Hop 3 [Clause C0]: bridge 'Mia'" in output
    assert "spec_accepted=" in output
    assert "idx_share=" in output
    assert "MLM ❯ attic" in output
    assert "MLM ❯ hall" in output
    assert "SAO Episodic Bank:" in output
    assert new_cfg.nodes == 4
    assert saved_ckpt.exists()

    # Verify SAO episodic state-overwrite compaction evicts superseded same-subject location facts
    from mlm.generate import LiveLearner

    learner = LiveLearner(state, cfg, max_episodic_facts=3)
    learner.remember_facts("Tom is in the yard.")
    learner.remember_facts("Mia is in the attic.")
    learner.remember_facts("Dan is in the lab.")
    compacted = learner.remember_facts("Tom is in the hall.")
    assert compacted >= 1
    assert len(learner.episodic_facts) <= 3
    assert "Tom is in the hall." in learner.episodic_facts
    assert "Tom is in the yard." not in learner.episodic_facts

    # Single-turn CLI `mlm chat` and `mlm-chat` with --show-thinking
    chat_main(
        [
            str(saved_ckpt),
            "--context",
            "Bob's pet is cat. cat's toy is orb.",
            "--message",
            "What toy belongs to Bob?",
            "--show-thinking",
        ]
    )
    captured = capsys.readouterr().out
    assert "Model Thinking & Telemetry (<think>)" in captured
    assert "orb" in captured
    assert "prompt eval:" in captured
