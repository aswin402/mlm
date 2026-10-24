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
