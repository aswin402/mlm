import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.training.train_state import TrainState

from mlm.config import Config
from mlm.hierarchy_run import make_routed_step, routed_objective
from mlm.interference import make_interference_probe, make_prefix_reader, replay_probes
from mlm.model import initial_state
from mlm.plasticity import scale_optimizer
from mlm.stream import DocumentStream
from mlm.training import create_state


def _known_state(cfg):
    def apply(variables, tokens, carry, resets, valid=None, return_routes=False):
        logits = jnp.broadcast_to(variables["params"]["vocab_bias"], (*tokens.shape, 2))
        routes = jnp.broadcast_to(
            jax.nn.softmax(variables["params"]["router_group_bias"]),
            (*tokens.shape, cfg.nodes),
        )
        return (logits, carry, routes) if return_routes else (logits, carry)

    params = {
        "vocab_bias": jnp.array([0.2, -0.2]),
        "router_group_bias": jnp.array([0.3, -0.2, 0.1, 0.0]),
    }
    tx = optax.chain(optax.clip_by_global_norm(0.25), optax.adamw(0.02, weight_decay=0.1))
    tx = scale_optimizer(tx, params, cfg)
    state = TrainState.create(apply_fn=apply, params=params, tx=tx)
    return state.apply_gradients(
        grads={
            "vocab_bias": jnp.array([0.3, -0.4]),
            "router_group_bias": jnp.array([0.05, -0.1, 0.04, 0.02]),
        }
    )


def test_copied_adam_step_matches_trainer_and_closed_form_masked_losses():
    cfg = Config(
        nodes=4,
        width=8,
        groups=2,
        node_update="hierarchical",
        shared_lr_scale=0.5,
        router_lr_scale=0.25,
        router_balance=0.2,
    )
    state = _known_state(cfg)
    original = jax.tree.map(lambda x: np.asarray(x).copy(), (state.params, state.opt_state))
    qa = {
        "inputs": jnp.zeros((1, 3), jnp.int32),
        "targets": jnp.array([[1, 0, 1]]),
        "mask": jnp.array([[1.0, 1.0, 0.0]]),
        "loss_mask": jnp.array([[0.0, 1.0, 1.0]]),
        "resets": jnp.array([[True, False, False]]),
    }
    english = {
        "inputs": jnp.zeros((1, 3), jnp.int32),
        "targets": jnp.ones((1, 3), jnp.int32),
        "mask": jnp.array([[1.0, 1.0, 0.0]]),
        "resets": jnp.array([[True, False, False]]),
    }
    carry = initial_state(cfg, 1)
    candidate, zero, result = make_interference_probe(cfg)(state, qa, english, carry)
    expected, _, raw = make_routed_step(cfg)(state, carry, english)
    assert bool(raw["updated"])
    for a, b in zip(
        jax.tree.leaves((candidate.params, candidate.opt_state)),
        jax.tree.leaves((expected.params, expected.opt_state)),
        strict=True,
    ):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)
    manual_zero = state.apply_gradients(grads=jax.tree.map(jnp.zeros_like, state.params))
    for a, b in zip(
        jax.tree.leaves((zero.params, zero.opt_state)),
        jax.tree.leaves((manual_zero.params, manual_zero.opt_state)),
        strict=True,
    ):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)

    def loss(params):
        v = np.asarray(params["vocab_bias"], dtype=np.float64)
        return np.logaddexp(v[0], v[1]) - v[0]

    before, after, control = loss(state.params), loss(candidate.params), loss(zero.params)
    assert result["qa_loss_before"] == pytest.approx(before, abs=1e-6)
    assert result["qa_loss_after_english"] == pytest.approx(after, abs=1e-6)
    assert result["qa_loss_after_zero"] == pytest.approx(control, abs=1e-6)
    assert result["qa_loss_change"] == pytest.approx(after - before, abs=1e-6)
    assert result["english_minus_zero_qa_loss"] == pytest.approx(after - control, abs=1e-6)
    logits = np.asarray(state.params["vocab_bias"], dtype=np.float64)
    probabilities = np.exp(logits) / np.exp(logits).sum()
    gq, ge = probabilities - np.array([1, 0]), probabilities - np.array([0, 1])
    route_logits = np.asarray(state.params["router_group_bias"], dtype=np.float64)
    routes = np.exp(route_logits) / np.exp(route_logits).sum()
    balance_gradient = cfg.router_balance * 2 * cfg.nodes * routes * (routes - (routes**2).sum())
    english_gradient_norm = np.sqrt(ge @ ge + balance_gradient @ balance_gradient)
    assert result["roles"]["router"]["qa_gradient_norm"] == 0
    assert result["roles"]["router"]["english_gradient_norm"] == pytest.approx(
        np.linalg.norm(balance_gradient), abs=1e-6
    )
    assert result["roles"]["router"]["english_gradient_norm"] > 1e-5
    expected_english_ce = np.logaddexp(logits[0], logits[1]) - logits[1]
    expected_balance = cfg.router_balance * (cfg.nodes * (routes**2).sum() - 1)
    assert result["english_objective_before"] == pytest.approx(
        expected_english_ce + expected_balance, abs=1e-6
    )
    delta = np.asarray(candidate.params["vocab_bias"]) - np.asarray(state.params["vocab_bias"])
    assert result["gradient_cosine"] == pytest.approx(
        float(gq @ ge / np.linalg.norm(gq) / english_gradient_norm), abs=1e-6
    )
    assert result["first_order_qa_change"] == pytest.approx(float(gq @ delta), abs=1e-7)
    assert sum(row["first_order_qa_change"] for row in result["roles"].values()) == pytest.approx(
        result["first_order_qa_change"], abs=1e-12
    )
    assert result["roles"]["memory"]["gradient_cosine"] is None
    assert result["qa_supervised_positions"] == 1
    assert result["english_supervised_positions"] == 2
    assert result["qa_loss_change"] > 0
    assert int(candidate.step) == int(zero.step) == int(state.step) + 1
    assert not np.array_equal(zero.params["vocab_bias"], state.params["vocab_bias"])
    for a, b in zip(
        jax.tree.leaves((state.params, state.opt_state)), jax.tree.leaves(original), strict=True
    ):
        np.testing.assert_array_equal(a, b)
    empty = {**english, "mask": jnp.zeros_like(english["mask"])}
    with pytest.raises(ValueError, match="supervised"):
        make_interference_probe(cfg)(state, qa, empty, carry)
    bad = state.replace(params={**state.params, "vocab_bias": jnp.array([jnp.nan, 0.0])})
    with pytest.raises(FloatingPointError):
        make_interference_probe(cfg)(bad, qa, english, carry)


def test_frozen_prefix_reconstruction_matches_sequential_reading():
    cfg = Config(
        nodes=4, width=8, groups=2, node_update="hierarchical", memory="delta", memory_keys=4
    )
    state = create_state(cfg)
    docs = ["a", "A longer document continues across many chunks.", "Short.", "Different text " * 4]
    selected = [0, 1, 3, 7, 11]
    probes = replay_probes(docs, 2, 8, 43, selected)
    stream = DocumentStream(docs, 2, 8, 43)
    carry = initial_state(cfg, 2)
    reader = make_prefix_reader(cfg)
    by_position = {p["position"]: p for p in probes}
    for position in range(max(selected) + 1):
        batch = stream.next_batch()
        if position in by_position:
            probe = by_position[position]
            for key in batch:
                np.testing.assert_array_equal(probe["batch"][key], batch[key])
            reconstructed = reader(state, jax.tree.map(jnp.asarray, probe["prefix"]))
            # A new-document reset makes the old row carry irrelevant.
            live = np.asarray(batch["resets"][:, 0]) == 0
            np.testing.assert_allclose(
                np.asarray(reconstructed)[live], np.asarray(carry)[live], atol=1e-6
            )
            _, a = state.apply_fn(
                {"params": state.params},
                jnp.asarray(batch["inputs"]),
                reconstructed,
                jnp.asarray(batch["resets"]),
                valid=jnp.asarray(batch["mask"]),
            )
            _, b = state.apply_fn(
                {"params": state.params},
                jnp.asarray(batch["inputs"]),
                carry,
                jnp.asarray(batch["resets"]),
                valid=jnp.asarray(batch["mask"]),
            )
            np.testing.assert_allclose(a, b, atol=1e-6)
        _, carry = state.apply_fn(
            {"params": state.params},
            jnp.asarray(batch["inputs"]),
            carry,
            jnp.asarray(batch["resets"]),
            valid=jnp.asarray(batch["mask"]),
        )
    continuation = next(p for p in probes if p["prefix"]["mask"].any())
    prefix = jax.tree.map(jnp.asarray, continuation["prefix"])
    batch = jax.tree.map(jnp.asarray, continuation["batch"])

    def raw_prefix(params):
        return state.apply_fn(
            {"params": params},
            prefix["inputs"],
            initial_state(cfg, 2),
            prefix["resets"],
            valid=prefix["mask"],
        )[1]

    raw_gradient = jax.grad(lambda params: jnp.sum(raw_prefix(params)))(state.params)
    detached_gradient = jax.grad(
        lambda params: jnp.sum(reader(state.replace(params=params), prefix))
    )(state.params)
    assert float(optax.tree.norm(raw_gradient)) > 1e-6
    assert float(optax.tree.norm(detached_gradient)) == 0
    frozen_carry = reader(state, prefix)
    held_gradient = jax.grad(
        lambda params: routed_objective(params, state.apply_fn, batch, frozen_carry, cfg)[0]
    )(state.params)
    full_gradient = jax.grad(
        lambda params: routed_objective(params, state.apply_fn, batch, raw_prefix(params), cfg)[0]
    )(state.params)
    assert (
        float(optax.tree.norm(jax.tree.map(lambda a, b: a - b, held_gradient, full_gradient)))
        > 1e-6
    )
    actual, _, metrics = make_interference_probe(cfg)(state, batch, batch, frozen_carry)
    expected = state.apply_gradients(grads=held_gradient)
    assert metrics["english_gradient_norm"] == pytest.approx(
        float(optax.tree.norm(held_gradient)), rel=1e-5
    )
    for a, b in zip(
        jax.tree.leaves((actual.params, actual.opt_state)),
        jax.tree.leaves((expected.params, expected.opt_state)),
        strict=True,
    ):
        np.testing.assert_allclose(a, b, atol=1e-7, rtol=1e-5)
    again = replay_probes(docs, 2, 8, 43, selected)
    for a, b in zip(probes, again, strict=True):
        for key in a["prefix"]:
            np.testing.assert_array_equal(a["prefix"][key], b["prefix"][key])
    for invalid in [[], [-1], [1, 1], [2, 1]]:
        with pytest.raises(ValueError):
            replay_probes(docs, 2, 8, 43, invalid)


def test_aggregate_qa_is_token_weighted_with_newline_and_eos():
    from mlm.data import EOS

    cfg = Config(nodes=4, width=8)

    def apply(variables, tokens, carry, resets, valid=None, return_routes=False):
        logits = jnp.broadcast_to(variables["params"]["vocab_bias"], (*tokens.shape, 258))
        routes = jnp.full((*tokens.shape, cfg.nodes), 1 / cfg.nodes)
        return (logits, carry, routes) if return_routes else (logits, carry)

    state = TrainState.create(
        apply_fn=apply, params={"vocab_bias": jnp.linspace(-1, 1, 258)}, tx=optax.sgd(0.002)
    )
    qa = {
        "inputs": jnp.zeros((2, 5), jnp.int32),
        "targets": jnp.array([[103, 10, EOS, 77, 99], [71, 72, 73, 10, EOS]]),
        "mask": jnp.array([[1.0, 1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]]),
        "loss_mask": jnp.array([[0.0, 1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]]),
        "resets": jnp.array([[True, False, False, False, False]] * 2),
    }
    english = {
        "inputs": jnp.zeros((1, 5), jnp.int32),
        "targets": jnp.array([[71, 72, 73, 10, EOS]]),
        "mask": jnp.ones((1, 5)),
        "resets": jnp.array([[True, False, False, False, False]]),
    }
    candidate, _, result = make_interference_probe(cfg)(state, qa, english, initial_state(cfg, 1))
    targets = np.array([10, EOS, 71, 72, 73, 10, EOS])
    values = np.asarray(state.params["vocab_bias"], dtype=np.float64)
    probabilities = np.exp(values) / np.exp(values).sum()
    expected_gradient = probabilities - np.bincount(targets, minlength=258) / len(targets)
    expected_loss = np.log(np.exp(values).sum()) - values[targets].mean()
    assert result["qa_supervised_positions"] == 7
    assert result["qa_loss_before"] == pytest.approx(expected_loss, abs=1e-6)
    assert result["qa_gradient_norm"] == pytest.approx(np.linalg.norm(expected_gradient), abs=1e-6)
    delta = np.asarray(candidate.params["vocab_bias"], dtype=np.float64) - values
    assert result["first_order_qa_change"] == pytest.approx(
        float(expected_gradient @ delta), abs=1e-8
    )
