import json
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from mlm.checkpoint import load_checkpoint
from mlm.config import Config
from mlm.pretrain_run import run


def test_fresh_pretraining_preserves_steps_and_full_stream(tmp_path):
    train = tmp_path / "train.txt"
    train.write_text("A small training story.\n\nAnother training tale.")
    dev = tmp_path / "dev.txt"
    dev.write_text("Held out English.")
    shared = {}
    for mode in ["dense", "hierarchical"]:
        cfg = Config(nodes=4, width=8, groups=2, node_update=mode, batch_size=2, length=16)
        config = tmp_path / f"{mode}.json"
        config.write_text(cfg.model_dump_json())
        out = tmp_path / mode
        args = SimpleNamespace(
            train=train, validation=dev, config=config, out=out, steps=2, eval_every=1
        )
        run(args)
        initial, _, meta = load_checkpoint(out / "initial.mlm")
        latest, _, last = load_checkpoint(out / "latest.mlm")
        assert int(initial.step) == 0 and int(latest.step) == 2
        assert meta["initialization"] == "random-from-scratch"
        assert last["stream"]["position"] > 0
        summary = json.loads((out / "summary.json").read_text())
        assert summary["input_payload_bytes"] > 0
        rows = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
        best, _, _ = load_checkpoint(out / "best.mlm")
        assert int(best.step) == min(rows, key=lambda r: r["validation"]["loss"])["step"]
        shared[mode] = initial.params
        with pytest.raises(ValueError, match="empty"):
            run(args)
    for key in shared["dense"]:
        np.testing.assert_array_equal(shared["dense"][key], shared["hierarchical"][key])
    assert len(jax.tree.leaves(shared["hierarchical"])) > len(jax.tree.leaves(shared["dense"]))


def test_runtime_probe_records_scope_and_tensor_sizes(tmp_path):
    from mlm.checkpoint import save_checkpoint
    from mlm.runtime_bench import benchmark
    from mlm.training import create_state

    cfg = Config(nodes=4, width=8, groups=2, node_update="hierarchical")
    state = create_state(cfg)
    path = tmp_path / "model.mlm"
    save_checkpoint(path, state, cfg, {})
    out = tmp_path / "timing.json"
    result = benchmark(path, out, lengths=(8,), repeats=2, warmups=1)
    assert result["parameter_bytes"] == sum(
        np.asarray(v).nbytes for v in jax.tree.leaves(state.params)
    )
    assert result["persistent_state_bytes_per_sequence"] == 4 * 8 * 4
    assert result["results"][0]["p50_ms"] >= 0
    assert result["results"][0]["p95_ms"] >= 0
    assert len(result["results"][0]["samples_ms"]) == 2
    with pytest.raises(ValueError, match="fresh"):
        benchmark(path, out)
