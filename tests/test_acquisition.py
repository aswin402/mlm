import hashlib
from types import SimpleNamespace

import pytest

from mlm.acquisition_run import nested_groups, run
from mlm.checkpoint import load_checkpoint, save_checkpoint
from mlm.config import Config
from mlm.confirmation_data import prepare
from mlm.training import create_state


def test_nested_subsets_keep_complete_groups_and_reject_bad_sizes():
    questions = [{"group": f"g{g}", "id": f"{g}-{i}"} for g in range(8) for i in range(4)]
    subsets = nested_groups(questions, [1, 4, 8], 42)
    assert subsets == nested_groups(questions, [1, 4, 8], 42)
    assert len(subsets[4]) == 4 and set(subsets[1]) <= set(subsets[4]) <= set(subsets[8])
    with pytest.raises(ValueError):
        nested_groups(questions[:-1], [1], 42)
    for sizes in [[0], [9], [4, 1], [1, 1]]:
        with pytest.raises(ValueError):
            nested_groups(questions, sizes, 42)


def test_trials_reset_source_optimizer_and_preserve_budget(tmp_path):
    data = tmp_path / "data"
    prepare(data, 631204)
    stories = tmp_path / "stories.txt"
    stories.write_text("An English development story.")
    cfg = Config(nodes=4, width=8, groups=2, batch_size=4, length=128, learning_rate=0.003)
    config = tmp_path / "config.json"
    config.write_text(cfg.model_dump_json())
    source = tmp_path / "source.mlm"
    save_checkpoint(source, create_state(cfg).replace(step=17), cfg, {"messages": True})
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    args = SimpleNamespace(
        data=data,
        config=config,
        init=source,
        out=tmp_path / "run",
        sizes=[1, 4],
        steps=2,
        english_dev=stories,
    )
    result = run(args)
    assert len(result["trials"]) == 2
    initial_hashes = []
    for row in result["trials"]:
        assert row["optimizer_updates"] == 2 and row["initial_optimizer_step"] == 0
        assert len(row["update_group_ids"]) == 2
        assert row["supervised_bytes"] == 40
        state, _, meta = load_checkpoint(args.out / f"groups-{row['groups']}" / "final.mlm")
        assert int(state.step) == 2 and meta["source_sha256"] == digest
        initial_hashes.append(row["initial_parameters_sha256"])
    assert initial_hashes[0] == initial_hashes[1]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    with pytest.raises(ValueError, match="fresh"):
        run(args)
    args.out = tmp_path / "bad"
    args.steps = 0
    with pytest.raises(ValueError):
        run(args)
