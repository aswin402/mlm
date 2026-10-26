import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import serialization

from mlm.config import Config
from mlm.future_byte import (
    auxiliary_loss,
    create_head,
    future_logits,
    future_mask,
    future_objective,
    make_future_step,
)
from mlm.hierarchy_run import make_routed_step
from mlm.model import initial_state
from mlm.training import create_state


def fixture():
    cfg = Config(
        nodes=4,
        width=8,
        groups=2,
        node_update="hierarchical",
        memory="delta",
        memory_keys=4,
        batch_size=2,
        length=8,
    )
    batch = {
        "inputs": jnp.array([[256, 65, 66, 67, 68, 69, 70, 71]] * 2),
        "targets": jnp.array([[65, 66, 67, 68, 69, 70, 71, 257]] * 2),
        "mask": jnp.ones((2, 8)),
        "loss_mask": jnp.array([[0, 0, 0, 0, 1, 1, 1, 1]] * 2),
        "resets": jnp.array([[True] + [False] * 7] * 2),
    }
    return cfg, batch


def test_mask_targets_and_no_target_gradients():
    cfg, b = fixture()
    np.testing.assert_array_equal(future_mask(b), np.array([[0, 0, 0, 0, 1, 1, 0]] * 2))
    altered = {**b, "resets": b["resets"].at[0, 5].set(True), "mask": b["mask"].at[1, 6].set(0)}
    np.testing.assert_array_equal(
        future_mask(altered), np.array([[0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 0, 0]])
    )
    logits = jnp.arange(2 * 8 * 258, dtype=jnp.float32).reshape(2, 8, 258) * 1e-4
    h = create_head(cfg)
    loss, count = auxiliary_loss(h.params, logits, b)
    assert int(count) == 4 and np.isfinite(float(loss))
    direct = optax.softmax_cross_entropy_with_integer_labels(
        future_logits(h.params, logits[:, :-1]), b["targets"][:, 1:]
    )
    assert float(loss) == pytest.approx(
        float((direct * jnp.array([[0, 0, 0, 0, 1, 1, 0]] * 2)).sum() / 4), abs=1e-6
    )
    empty = {**b, "loss_mask": jnp.zeros((2, 8))}
    (loss, count), g = jax.value_and_grad(auxiliary_loss, has_aux=True)(h.params, logits, empty)
    assert float(loss) == 0 and int(count) == 0
    assert all(np.all(np.asarray(x) == 0) for x in jax.tree.leaves(g))
    assert serialization.to_bytes(
        serialization.from_bytes(h, serialization.to_bytes(h))
    ) == serialization.to_bytes(h)


def test_zero_weight_legacy_identity_and_active_backbone_gradient():
    cfg, b = fixture()
    source = create_state(cfg)
    head = create_head(cfg)
    carry = initial_state(cfg, 2)
    before = serialization.to_bytes(source)
    legacy, c, m = make_routed_step(cfg)(source, carry, b)
    zero, zh, zc, zm = make_future_step(cfg, 0.0)(source, head, carry, b)
    assert serialization.to_bytes(legacy) == serialization.to_bytes(zero)
    np.testing.assert_array_equal(c, zc)
    active, ah, ac, am = make_future_step(cfg, 0.1)(source, head, carry, b)
    assert bool(am["finite"]) and bool(am["updated"])
    assert serialization.to_bytes(active.params) != serialization.to_bytes(zero.params)
    assert serialization.to_bytes(ah.params) != serialization.to_bytes(zh.params)
    assert int(am["aux_positions"]) == 4 and int(active.step) == 1 and int(ah.step) == 1
    (_, _), (g, hg) = jax.value_and_grad(future_objective, argnums=(0, 1), has_aux=True)(
        source.params, head.params, source.apply_fn, carry, b, cfg, 0.1
    )
    expected = source.apply_gradients(grads=g)
    expected_head = head.apply_gradients(grads=hg)
    for actual, wanted in [(active, expected), (ah, expected_head)]:
        for x, y in zip(jax.tree.leaves(actual), jax.tree.leaves(wanted), strict=True):
            np.testing.assert_allclose(x, y, atol=1e-7, rtol=1e-5)
    assert serialization.to_bytes(source) == before

    # Main predictor remains causal: later inputs/labels do not affect earlier logits.
    def output(batch):
        return source.apply_fn(
            {"params": source.params}, batch["inputs"], carry, batch["resets"], valid=batch["mask"]
        )[0]

    modified = {
        **b,
        "inputs": b["inputs"].at[:, 6:].set(99),
        "targets": b["targets"].at[:, 6:].set(100),
    }
    np.testing.assert_allclose(output(b)[:, :6], output(modified)[:, :6], atol=1e-7)
    np.testing.assert_allclose(
        future_logits(head.params, output(b))[:, :6],
        future_logits(head.params, output(modified))[:, :6],
        atol=1e-7,
    )
    legacy2, c2, _ = make_routed_step(cfg)(legacy, c, b)
    zero2, _, zc2, _ = make_future_step(cfg, 0.0)(zero, zh, zc, b)
    assert serialization.to_bytes(legacy2) == serialization.to_bytes(zero2)
    np.testing.assert_array_equal(c2, zc2)
    bad = {**b, "loss_mask": jnp.zeros((2, 8))}
    unchanged, _, _, metrics = make_future_step(cfg, 0.1)(source, head, carry, bad)
    assert not bool(metrics["updated"]) and serialization.to_bytes(unchanged) == before
    corrupt = source.replace(
        params=jax.tree.map(lambda x: jnp.full_like(x, jnp.nan), source.params)
    )
    _, _, _, metrics = make_future_step(cfg, 0.1)(corrupt, head, carry, b)
    assert not bool(metrics["finite"]) and not bool(metrics["updated"])
    with pytest.raises(ValueError):
        make_future_step(cfg, float("nan"))
