import jax
import numpy as np
import pytest

from mlm.config import Config
from mlm.memory_init import retention_parameters
from mlm.training import create_state


@pytest.mark.parametrize("architecture", ["graph", "gru"])
def test_retention_intervention_only_changes_z_bias(architecture):
    config = Config(architecture=architecture, nodes=4, width=8)
    state = create_state(config)
    before = jax.tree.map(lambda x: np.array(x, copy=True), state.params)
    changed, times = retention_parameters(state.params, config, 128)
    assert times[0] == pytest.approx(2)
    assert times[-1] == pytest.approx(128)
    expected = np.broadcast_to(
        np.log(times - 1)[:, None] if architecture == "graph" else np.log(times - 1),
        changed["gru_bias"][..., : config.width].shape,
    )
    np.testing.assert_allclose(
        np.asarray(changed["gru_bias"])[..., : config.width]
        - before["gru_bias"][..., : config.width],
        expected,
        atol=1e-6,
    )
    np.testing.assert_array_equal(
        changed["gru_bias"][..., config.width :], before["gru_bias"][..., config.width :]
    )
    for key in before:
        np.testing.assert_array_equal(state.params[key], before[key])
        if key != "gru_bias":
            np.testing.assert_array_equal(changed[key], before[key])
    assert sum(x.size for x in jax.tree.leaves(changed)) == sum(
        x.size for x in jax.tree.leaves(before)
    )
    for invalid in [1, float("nan"), float("inf")]:
        with pytest.raises(ValueError):
            retention_parameters(state.params, config, invalid)


def test_prepared_checkpoint_has_fresh_optimizer_and_provenance(tmp_path):
    import hashlib

    from mlm.checkpoint import load_checkpoint, save_checkpoint
    from mlm.memory_init import prepare

    config = Config(nodes=4, width=8)
    source = tmp_path / "english.mlm"
    state = create_state(config).replace(step=37)
    save_checkpoint(source, state, config, {"messages": True})
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    out = tmp_path / "slow.mlm"
    metadata = prepare(source, out)
    restored, restored_config, restored_meta = load_checkpoint(out)
    assert restored_config == config
    assert int(restored.step) == 0
    assert restored_meta == metadata
    assert metadata["source_sha256"] == digest
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    expected, _ = retention_parameters(state.params, config)
    for a, b in zip(jax.tree.leaves(restored.params), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(a, b)
    fresh = create_state(config)
    for a, b in zip(
        jax.tree.leaves(restored.opt_state), jax.tree.leaves(fresh.opt_state), strict=True
    ):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="fresh"):
        prepare(source, out)

    other = tmp_path / "occupied.mlm"
    other.with_suffix(".json").write_text("keep metadata")
    with pytest.raises(ValueError, match="fresh"):
        prepare(source, other)
    assert other.with_suffix(".json").read_text() == "keep metadata"
    assert not other.exists()
