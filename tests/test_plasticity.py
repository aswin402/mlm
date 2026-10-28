import hashlib
import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from pydantic import ValidationError

from mlm.checkpoint import load_checkpoint, save_checkpoint
from mlm.config import Config
from mlm.confirmation_data import prepare
from mlm.generate import generate
from mlm.plasticity import newton_schulz, parameter_roles, recombine_grow_nodes
from mlm.plasticity_run import run
from mlm.training import create_state, decay_mask, make_train_step


def test_role_rates_scale_adamw_updates_and_roundtrip(tmp_path):
    cfg = Config(
        nodes=4,
        width=8,
        groups=2,
        node_update="hierarchical",
        learning_rate=0.003,
        shared_lr_scale=0.1,
        node_lr_scale=1,
        router_lr_scale=0.25,
        memory="delta",
        memory_keys=4,
        memory_lr_scale=0.5,
    )
    state = create_state(cfg)
    baseline_cfg = cfg.model_copy(
        update={"shared_lr_scale": 1.0, "router_lr_scale": 1.0, "memory_lr_scale": 1.0}
    )
    baseline = create_state(baseline_cfg)
    # Nonzero router tensors make weight decay observable in every role.
    params = jax.tree.map(lambda p: p + 0.2, state.params)
    state = state.replace(params=params)
    baseline = baseline.replace(params=params)
    grads = jax.tree.map(lambda p: jnp.ones_like(p) * 0.1, params)
    roles = parameter_roles(params)
    expected = {"shared": 0.1, "node": 1.0, "router": 0.25, "memory": 0.5}
    update, _ = state.tx.update(grads, state.opt_state, params)
    ordinary, _ = baseline.tx.update(grads, baseline.opt_state, params)
    for name in params:
        np.testing.assert_allclose(
            update[name], ordinary[name] * expected[roles[name]], rtol=1e-6, atol=1e-9
        )
    state = state.apply_gradients(grads=grads)
    path = tmp_path / "split.mlm"
    save_checkpoint(path, state, cfg, {"mode": "plasticity", "messages": True})
    loaded, loaded_cfg, _ = load_checkpoint(path)
    assert loaded_cfg == cfg
    a = state.apply_gradients(grads=grads)
    b = loaded.apply_gradients(grads=grads)
    for x, y in zip(jax.tree.leaves(a.params), jax.tree.leaves(b.params), strict=True):
        np.testing.assert_array_equal(x, y)
    with pytest.raises(ValueError, match="Unknown"):
        parameter_roles({"unregistered_weight": jnp.ones(1)})
    for field in ["shared_lr_scale", "node_lr_scale", "router_lr_scale", "memory_lr_scale"]:
        for bad in [0, -1, 2, float("nan"), float("inf")]:
            with pytest.raises(ValidationError):
                Config.model_validate({field: bad})


def test_default_optimizer_matches_legacy():
    cfg = Config(nodes=4, width=8)
    state = create_state(cfg)
    legacy = optax.chain(
        optax.clip_by_global_norm(cfg.clip_norm),
        optax.adamw(
            cfg.learning_rate,
            b1=0.9,
            b2=0.999,
            eps=1e-8,
            weight_decay=cfg.weight_decay,
            mask=decay_mask(state.params),
        ),
    )
    gradients = jax.tree.map(jnp.ones_like, state.params)
    a, sa = state.tx.update(gradients, state.opt_state, state.params)
    b, sb = legacy.update(gradients, legacy.init(state.params), state.params)
    for x, y in zip(jax.tree.leaves((a, sa)), jax.tree.leaves((b, sb)), strict=True):
        np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize(
    "rehearsal,qa_seed,english_seed",
    [
        pytest.param(False, None, None, id="legacy_task_only"),
        pytest.param(True, None, None, id="legacy_rehearsal"),
        pytest.param(True, 43, None, id="qa_override"),
        pytest.param(True, None, 44, id="english_override"),
        pytest.param(True, 0, 0, id="zero_seeds"),
    ],
)
def test_plasticity_runner_budget_source_and_stream(
    tmp_path, rehearsal, qa_seed, english_seed, monkeypatch
):
    data = tmp_path / "data"
    prepare(data, 631204)
    # One whole eight-condition development case keeps this integration test small.
    dev = json.loads((data / "validation-questions.json").read_text())["questions"][:8]
    (data / "validation-questions.json").write_text(json.dumps({"questions": dev}))
    records = (data / "validation.jsonl").read_text().splitlines()[:8]
    (data / "validation.jsonl").write_text("\n".join(records) + "\n")
    english = tmp_path / "english.jsonl"
    english.write_text(
        "".join(
            json.dumps({"text": f"Story {i}: " + "An English training story continues. " * 8})
            + "\n"
            for i in range(12)
        )
    )
    english_dev = tmp_path / "english-dev.txt"
    english_dev.write_text("A separate development story.")
    cfg = Config(
        nodes=4,
        width=8,
        groups=2,
        node_update="hierarchical",
        batch_size=4,
        length=128,
        shared_lr_scale=0.1,
        learning_rate=0.003,
    )
    config = tmp_path / "config.json"
    config.write_text(cfg.model_dump_json())
    source = tmp_path / "source.mlm"
    source_cfg = cfg.model_copy(update={"shared_lr_scale": 1.0})
    source_state = create_state(source_cfg)
    gradients = jax.tree.map(lambda p: jnp.ones_like(p) * 0.2, source_state.params)
    source_state = source_state.apply_gradients(grads=gradients)
    source_state = source_state.apply_gradients(grads=gradients)
    save_checkpoint(source, source_state, source_cfg, {"messages": True})
    original = hashlib.sha256(source.read_bytes()).hexdigest()
    args = SimpleNamespace(
        data=data,
        config=config,
        init=source,
        out=tmp_path / "run",
        groups=2,
        steps=5,
        rehearsal=rehearsal,
        english_train=english,
        english_dev=english_dev,
    )
    if qa_seed is not None:
        args.qa_seed = qa_seed
    if english_seed is not None:
        args.english_seed = english_seed
    import mlm.plasticity_run as driver
    from mlm.model import initial_state
    from mlm.stream import DocumentStream

    real_factory = driver.make_routed_step
    english_calls = []
    english_batches = []

    def observe_factory(config):
        actual = real_factory(config)

        def observe(state, hidden, batch):
            english_batch = "loss_mask" not in batch
            incoming = np.asarray(hidden).copy()
            result = actual(state, hidden, batch)
            if english_batch:
                english_calls.append((incoming, np.asarray(result[1]).copy()))
                english_batches.append(jax.tree.map(np.asarray, batch))
            else:
                np.testing.assert_array_equal(incoming, initial_state(config, 4))
            return result

        return observe

    monkeypatch.setattr(driver, "make_routed_step", observe_factory)
    result = run(args)
    assert result["initial_optimizer_step"] == 0
    assert result["counters"]["qa"]["updates"] == 5
    assert result["counters"]["english"]["updates"] == (5 if rehearsal else 0)
    assert result["optimizer_updates"] == (10 if rehearsal else 5)
    assert result["counters"]["qa"]["supervised_bytes"] == 100
    initial, _, _ = load_checkpoint(args.out / "initial.mlm")
    fresh = create_state(cfg).replace(params=source_state.params)
    for x, y in zip(
        jax.tree.leaves(initial.params), jax.tree.leaves(source_state.params), strict=True
    ):
        np.testing.assert_array_equal(x, y)
    for x, y in zip(
        jax.tree.leaves(initial.opt_state), jax.tree.leaves(fresh.opt_state), strict=True
    ):
        np.testing.assert_array_equal(x, y)
    # Independently derive the sampler contract; English must not consume its RNG.
    questions = json.loads((data / "train-questions.json").read_text())["questions"]
    chosen = set(result["group_ids"])
    batch_ids = list(dict.fromkeys(q["group"] for q in questions if q["group"] in chosen))
    effective_qa = cfg.seed if qa_seed is None else qa_seed
    effective_english = cfg.seed + 1 if english_seed is None else english_seed
    rng = np.random.default_rng(effective_qa)
    expected = [batch_ids[int(i)] for _ in range(3) for i in rng.permutation(2)][:5]
    assert result["update_group_ids"] == expected
    seed_policy = {"subset": cfg.seed, "qa": effective_qa, "english": effective_english}
    assert result["sampling_seeds"] == seed_policy
    assert json.loads((args.out / "protocol.json").read_text())["sampling_seeds"] == seed_policy
    state, _, meta = load_checkpoint(args.out / "final.mlm")
    assert int(state.step) == result["optimizer_updates"]
    assert meta["sampling_seeds"] == seed_policy
    assert (meta["english_stream"] is not None) == rehearsal
    if rehearsal:
        from mlm.data import read_documents

        reference = DocumentStream(read_documents(english), 4, cfg.length, effective_english)
        for actual_batch in english_batches:
            expected_batch = reference.next_batch()
            for key in actual_batch:
                np.testing.assert_array_equal(actual_batch[key], expected_batch[key])
        assert meta["english_stream"] == reference.snapshot()
        assert len(english_calls) == 5
        assert np.linalg.norm(english_calls[0][1]) > 0
        np.testing.assert_array_equal(english_calls[0][0], initial_state(cfg, 4))
        for current, previous in zip(english_calls[1:], english_calls[:-1], strict=True):
            np.testing.assert_array_equal(current[0], previous[1])
        np.testing.assert_array_equal(meta["english_carry"], english_calls[-1][1])
    else:
        assert english_calls == []
    if rehearsal and qa_seed is None and english_seed is None:
        legacy_args = SimpleNamespace(**vars(args))
        legacy_args.out = tmp_path / "explicit-legacy"
        legacy_args.qa_seed = cfg.seed
        legacy_args.english_seed = cfg.seed + 1
        explicit = run(legacy_args)
        explicit_state, _, explicit_meta = load_checkpoint(legacy_args.out / "final.mlm")
        assert explicit["update_group_ids"] == result["update_group_ids"]
        assert explicit["final"] == result["final"]
        assert explicit_meta["english_stream"] == meta["english_stream"]
        for a, b in zip(
            jax.tree.leaves((state.params, state.opt_state)),
            jax.tree.leaves((explicit_state.params, explicit_state.opt_state)),
            strict=True,
        ):
            np.testing.assert_array_equal(a, b)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original
    with pytest.raises(ValueError, match="fresh"):
        run(args)


def test_legacy_checkpoint_with_populated_moments_continues_exactly(tmp_path):
    import zipfile

    cfg = Config(nodes=4, width=8)
    state = create_state(cfg)
    legacy = optax.chain(
        optax.clip_by_global_norm(cfg.clip_norm),
        optax.adamw(
            cfg.learning_rate,
            b1=0.9,
            b2=0.999,
            eps=1e-8,
            weight_decay=cfg.weight_decay,
            mask=decay_mask(state.params),
        ),
    )
    state = state.replace(tx=legacy, opt_state=legacy.init(state.params))
    for value in [0.2, -0.07, 0.03]:
        state = state.apply_gradients(
            grads=jax.tree.map(lambda p: jnp.full_like(p, value), state.params)
        )
    modern = tmp_path / "modern.mlm"
    save_checkpoint(modern, state, cfg, {"mode": "legacy-fixture"})
    with zipfile.ZipFile(modern) as z:
        manifest = json.loads(z.read("manifest.json"))
        payload = z.read("state.msgpack")
    for field in ["shared_lr_scale", "node_lr_scale", "router_lr_scale", "memory_lr_scale"]:
        del manifest["config"][field]
    for field in ["memory", "memory_keys"]:
        del manifest["config"][field]
    old = tmp_path / "legacy.mlm"
    with zipfile.ZipFile(old, "w") as z:
        z.writestr("manifest.json", json.dumps(manifest))
        z.writestr("state.msgpack", payload)
    loaded, restored_cfg, _ = load_checkpoint(old)
    assert (
        restored_cfg.shared_lr_scale
        == restored_cfg.node_lr_scale
        == restored_cfg.router_lr_scale
        == 1
    )
    grads = jax.tree.map(lambda p: jnp.full_like(p, -0.13), state.params)
    expected = state.apply_gradients(grads=grads)
    actual = loaded.apply_gradients(grads=grads)
    for x, y in zip(
        jax.tree.leaves((expected.params, expected.opt_state)),
        jax.tree.leaves((actual.params, actual.opt_state)),
        strict=True,
    ):
        np.testing.assert_array_equal(x, y)


def test_sampling_seed_validation():
    from mlm.plasticity_run import sampling_seeds

    cfg = Config(seed=42)
    assert sampling_seeds(cfg) == {"subset": 42, "qa": 42, "english": 43}
    assert sampling_seeds(cfg, 0, 0) == {"subset": 42, "qa": 0, "english": 0}
    for invalid in [-1, 2**32, True, 1.5]:
        with pytest.raises(ValueError, match="seed"):
            sampling_seeds(cfg, qa_seed=invalid)
        with pytest.raises(ValueError, match="seed"):
            sampling_seeds(cfg, english_seed=invalid)


def test_local_plasticity_freezes_dormant_nodes_against_decay_and_momentum():
    cfg = Config(nodes=4, width=8, local_plasticity=True, weight_decay=0.1)
    state = create_state(cfg)
    params = jax.tree.map(lambda p: p + 0.5, state.params)
    state = state.replace(params=params)
    grads = jax.tree.map(jnp.zeros_like, params)
    # Activate node 1 strongly and node 2 at half strength; keep nodes 0 and 3 dormant (0 grad).
    for name in ["input_kernel", "hidden_kernel", "gru_bias", "read_kernel"]:
        grads[name] = grads[name].at[1].set(0.2).at[2].set(0.1)
    updated = state.apply_gradients(grads=grads)
    for name in ["input_kernel", "hidden_kernel", "gru_bias", "read_kernel"]:
        # Dormant nodes 0 and 3 must remain completely untouched (even with weight_decay=0.1).
        np.testing.assert_array_equal(updated.params[name][0], params[name][0])
        np.testing.assert_array_equal(updated.params[name][3], params[name][3])
        delta_1 = np.linalg.norm(updated.params[name][1] - params[name][1])
        delta_2 = np.linalg.norm(updated.params[name][2] - params[name][2])
        assert delta_1 > delta_2 > 0
    with pytest.raises(ValidationError):
        Config(architecture="gru", local_plasticity=True)


def test_muon_optimizer_orthogonalizes_and_preserves_dormant_slices(tmp_path):
    key = jax.random.PRNGKey(7)
    mat_tall = jax.random.normal(key, (12, 6))
    orth_tall = newton_schulz(mat_tall, steps=5)
    gram_tall = orth_tall.T @ orth_tall
    np.testing.assert_allclose(np.diag(gram_tall), np.ones(6), atol=0.35)

    mat_3d = jax.random.normal(key, (4, 6, 10))
    orth_3d = newton_schulz(mat_3d, steps=5)
    assert orth_3d.shape == (4, 6, 10)

    cfg = Config(nodes=4, width=8, optimizer="muon", weight_decay=0.05)
    state = create_state(cfg)
    # First step: populate momentum on all nodes
    grads_all = jax.tree.map(lambda p: jnp.ones_like(p) * 0.1, state.params)
    state1 = state.apply_gradients(grads=grads_all)

    # Second step: zero gradient on node 0 of input_kernel; active gradient on nodes 1..3
    grads_partial = jax.tree.map(lambda p: jnp.ones_like(p) * 0.1, state1.params)
    grads_partial["input_kernel"] = grads_partial["input_kernel"].at[0].set(0.0)
    state2 = state1.apply_gradients(grads=grads_partial)

    # Node 0 of input_kernel must not change or decay when its slice gradient is zero
    np.testing.assert_array_equal(
        state2.params["input_kernel"][0], state1.params["input_kernel"][0]
    )
    assert np.linalg.norm(state2.params["input_kernel"][1] - state1.params["input_kernel"][1]) > 0

    # Round-trip checkpoint with Muon state
    ckpt = tmp_path / "muon.mlm"
    save_checkpoint(ckpt, state2, cfg, {"messages": True})
    loaded, loaded_cfg, _ = load_checkpoint(ckpt)
    assert loaded_cfg == cfg
    a = state2.apply_gradients(grads=grads_all)
    b = loaded.apply_gradients(grads=grads_all)
    for x, y in zip(jax.tree.leaves(a.params), jax.tree.leaves(b.params), strict=True):
        np.testing.assert_allclose(x, y, rtol=1e-6, atol=1e-7)


def test_adaptive_halt_trains_halt_head_across_rounds():
    cfg = Config(nodes=4, width=8, rounds=3, adaptive_halt=True)
    state = create_state(cfg)
    assert "halt_kernel" in state.params and "halt_bias" in state.params
    batch = {
        "inputs": jnp.array([[256, 65, 66, 67], [256, 68, 69, 70]], dtype=jnp.int32),
        "targets": jnp.array([[65, 66, 67, 257], [68, 69, 70, 257]], dtype=jnp.int32),
        "mask": jnp.ones((2, 4), dtype=jnp.float32),
        "resets": jnp.array([[True, False, False, False], [True, False, False, False]]),
    }
    step = make_train_step(cfg)
    updated, metrics = step(state, batch)
    assert bool(metrics["finite"])
    assert np.linalg.norm(updated.params["halt_kernel"]) > 0
    assert np.linalg.norm(updated.params["halt_bias"] - state.params["halt_bias"]) > 0
    with pytest.raises(ValidationError):
        Config(architecture="gru", adaptive_halt=True)


def test_recombine_grow_nodes_expands_graph_and_trains():
    cfg = Config(
        nodes=4,
        width=8,
        rounds=2,
        node_update="timescale",
        memory="relational",
        memory_keys=4,
        adaptive_halt=True,
    )
    state = create_state(cfg)
    new_cfg, new_params = recombine_grow_nodes(cfg, state.params, add_nodes=2, birth_gate=0.05)
    assert new_cfg.nodes == 6
    assert new_params["input_kernel"].shape == (6, 16, 24)
    assert new_params["hidden_kernel"].shape == (6, 8, 24)
    assert new_params["read_kernel"].shape == (6, 8, 8)
    assert new_params["router_timescale_kernel"].shape == (16, 6)
    assert new_params["router_surprise_kernel"].shape == (6, 8, 1)
    assert new_params["halt_kernel"].shape == (6, 8, 1)
    assert new_params["memory_key_kernel"].shape == (6, 8, 4)

    grown_state = create_state(new_cfg).replace(params=new_params)
    batch = {
        "inputs": jnp.array([[256, 65, 66, 67], [256, 68, 69, 70]], dtype=jnp.int32),
        "targets": jnp.array([[65, 66, 67, 257], [68, 69, 70, 257]], dtype=jnp.int32),
        "mask": jnp.ones((2, 4), dtype=jnp.float32),
        "resets": jnp.array([[True, False, False, False], [True, False, False, False]]),
    }
    step = make_train_step(new_cfg)
    grown_state, metrics = step(grown_state, batch)
    assert bool(metrics["finite"])


def test_generate_spike_frequency_adaptation_suppresses_repetition():
    cfg = Config(nodes=4, width=8)
    state = create_state(cfg)
    # Force vocab_bias to strongly prefer 'a' (97) and slightly less 'b' (98)
    bias = jnp.full((258,), -50.0).at[97].set(10.0).at[98].set(9.2)
    params = {
        **state.params,
        "vocab_kernel": jnp.zeros_like(state.params["vocab_kernel"]),
        "vocab_bias": bias,
    }
    state = state.replace(params=params)
    greedy = generate(state, cfg, "Hi", max_bytes=12, adapt_strength=0.0)
    adapted = generate(state, cfg, "Hi", max_bytes=12, adapt_strength=1.0, adapt_decay=0.9)
    assert greedy == "a" * 12
    assert "b" in adapted and adapted != greedy
    with pytest.raises(ValueError, match="adapt_strength"):
        generate(state, cfg, "Hi", max_bytes=4, adapt_strength=-0.1)
    with pytest.raises(ValueError, match="adapt_decay"):
        generate(state, cfg, "Hi", max_bytes=4, adapt_decay=1.5)
