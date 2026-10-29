import hashlib
import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from pydantic import ValidationError

from mlm.binding_run import warm_start
from mlm.checkpoint import load_checkpoint, save_checkpoint
from mlm.config import Config
from mlm.confirmation_data import prepare
from mlm.generate import LiveLearner, generate
from mlm.graph import sinkhorn_edge_weights, topology
from mlm.hierarchy_run import make_routed_step, routed_objective, run_continual
from mlm.model import initial_state
from mlm.plasticity import (
    LaneSchedule,
    PlasticityController,
    build_grpo_question_group,
    edge_affinity_matrix,
    grpo_objective,
    make_grpo_step,
    maybe_grow_and_prune,
    newton_schulz,
    parameter_roles,
    prune_nodes,
    recombine_grow_nodes,
    rewire_edges,
    update_router_load_bias,
)
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


def test_cross_window_router_balance_detaches_trunk_and_ponder_kl():
    cfg = Config(
        nodes=4,
        width=8,
        rounds=3,
        node_update="timescale",
        router_balance=0.1,
        adaptive_halt=True,
        ponder_beta=0.05,
        halt_prior=0.4,
    )
    state = create_state(cfg)
    batch = {
        "inputs": jnp.array([[256, 65, 66, 67], [256, 68, 69, 70]], dtype=jnp.int32),
        "targets": jnp.array([[65, 66, 67, 257], [68, 69, 70, 257]], dtype=jnp.int32),
        "mask": jnp.ones((2, 4), dtype=jnp.float32),
        "resets": jnp.array([[True, False, False, False], [True, False, False, False]]),
    }
    running_usage = jnp.array([0.7, 0.1, 0.1, 0.1], dtype=jnp.float32)

    def balance_only(params):
        _, (_, metrics) = routed_objective(
            params,
            state.apply_fn,
            batch,
            initial_state(cfg, 2),
            cfg,
            running_usage=running_usage,
        )
        return metrics["router_balance"]

    grads_bal = jax.grad(balance_only)(state.params)
    # Balance penalty must update router weights without distorting trunk weights
    assert np.linalg.norm(grads_bal["router_timescale_kernel"]) > 0
    np.testing.assert_allclose(grads_bal["input_kernel"], 0.0, atol=1e-7)
    np.testing.assert_allclose(grads_bal["hidden_kernel"], 0.0, atol=1e-7)

    total_loss, (_, metrics) = routed_objective(
        state.params,
        state.apply_fn,
        batch,
        initial_state(cfg, 2),
        cfg,
        running_usage=running_usage,
    )
    assert float(metrics["ponder_kl"]) > 0
    assert float(total_loss) > float(metrics["loss"])


def test_warm_start_preserves_moments_and_warmup_ramp(tmp_path):
    base_cfg = Config(nodes=4, width=8, batch_size=2, length=8)
    state = create_state(base_cfg)
    grads = jax.tree.map(lambda p: jnp.ones_like(p) * 0.15, state.params)
    state = state.apply_gradients(grads=grads)
    ckpt = tmp_path / "warm.mlm"
    save_checkpoint(ckpt, state, base_cfg, {"messages": True})

    target_cfg = base_cfg.model_copy(update={"memory": "relational", "memory_keys": 4})
    restored = warm_start(ckpt, target_cfg, preserve_moments=True)
    cold = warm_start(ckpt, target_cfg, preserve_moments=False)
    restored_norm = sum(float(jnp.linalg.norm(x)) for x in jax.tree.leaves(restored.opt_state))
    cold_norm = sum(float(jnp.linalg.norm(x)) for x in jax.tree.leaves(cold.opt_state))
    assert restored_norm > 0 and cold_norm == 0.0

    # Test linear warmup ramp on step 1 vs step 0 without warmup
    warm_cfg = Config(nodes=4, width=8, warmup_steps=10)
    no_warm_cfg = Config(nodes=4, width=8, warmup_steps=0)
    s_warm = create_state(warm_cfg)
    s_nowarm = create_state(no_warm_cfg)
    g = jax.tree.map(lambda p: jnp.ones_like(p) * 0.1, s_warm.params)
    u_warm, _ = s_warm.tx.update(g, s_warm.opt_state, s_warm.params)
    u_nowarm, _ = s_nowarm.tx.update(g, s_nowarm.opt_state, s_nowarm.params)
    np.testing.assert_allclose(
        u_warm["input_kernel"], u_nowarm["input_kernel"] * 0.1, rtol=1e-6, atol=1e-8
    )


def test_plasticity_controller_and_lane_schedule():
    ctrl = PlasticityController(scale=0.4)
    for step_i in range(12):
        ctrl.tick(40)
        ctrl.observe(2.5 - 0.0001 * step_i, se=0.02)
    state_dict = ctrl.state()
    restored = PlasticityController.restore(state_dict)
    assert restored.scale == pytest.approx(ctrl.scale)

    # Trigger a confirmed regime jump (> Floor jump 0.25 across two consecutive readings)
    before = restored.scale
    assert restored.observe(3.1, se=0.02) is None
    note = restored.observe(3.15, se=0.02)
    assert note is not None and "regime change" in note
    assert restored.scale > before

    sched = LaneSchedule(["qa", "english"], sizes={"qa": 500.0, "english": 50000.0})
    for idx in range(1, 8):
        sched.note_read("qa", 200.0)
        sched.note_read("english", 200.0)
        # English improves steeply while QA is flat and repeatedly re-read
        sched.observe({"qa": 1.5 - 0.001 * idx, "english": 3.0 * (idx**-0.35)})
    shares = sched.step()
    assert shares["english"] > shares["qa"]
    picks = [sched.pick() for _ in range(10)]
    assert picks.count("english") >= picks.count("qa")


def test_node_saturation_pruning_and_auto_growth():
    cfg = Config(nodes=6, width=8, degree=2, node_update="timescale")
    state = create_state(cfg)
    # Make nodes 4 and 5 near-zero readout norm so they are pruned
    rk = state.params["read_kernel"].at[4:].set(1e-4)
    params = {**state.params, "read_kernel": rk}
    pruned_cfg, pruned_params, removed = prune_nodes(cfg, params, protect=4, min_norm=0.01)
    assert removed == 2 and pruned_cfg.nodes == 4
    assert pruned_params["input_kernel"].shape[0] == 4

    # Make all 4 remaining nodes active with high routing entropy -> triggers auto-growth
    active_rk = jnp.ones_like(pruned_params["read_kernel"]) * 0.2
    active_params = {**pruned_params, "read_kernel": active_rk}
    usage = np.array([0.25, 0.25, 0.25, 0.25])
    grown_cfg, grown_params, info = maybe_grow_and_prune(
        pruned_cfg,
        active_params,
        usage,
        train_loss=2.0,
        val_loss=2.1,
        add_nodes=2,
        max_nodes=8,
        max_gap=0.5,
    )
    assert info["grew"] == 2 and grown_cfg.nodes == 6
    assert grown_params["input_kernel"].shape[0] == 6

    # Large generalization gap blocks growth (memorization guard)
    held_cfg, _, held_info = maybe_grow_and_prune(
        pruned_cfg,
        active_params,
        usage,
        train_loss=1.0,
        val_loss=2.0,
        add_nodes=2,
        max_nodes=8,
        max_gap=0.5,
    )
    assert held_info["grew"] == 0 and held_cfg.nodes == 4


def test_live_learner_online_chat_stream_updates(tmp_path):
    cfg = Config(nodes=4, width=8, batch_size=1, length=16)
    state = create_state(cfg)
    learner = LiveLearner(state, cfg, chunk=8, context=16)
    reply, steps = learner.chat_turn("Hello world!", max_bytes=8, learn=True)
    assert isinstance(reply, str)
    assert len(steps) >= 1 and int(learner.state.step) >= 1
    out_ckpt = tmp_path / "learned.mlm"
    save_checkpoint(out_ckpt, learner.state, cfg, {"messages": True})
    loaded, _, _ = load_checkpoint(out_ckpt)
    assert int(loaded.step) == int(learner.state.step)


def test_top_k_sparse_node_routing_preserves_dormant_states():
    cfg = Config(
        nodes=4,
        width=8,
        degree=2,
        groups=2,
        node_update="timescale",
        top_k_nodes=2,
        batch_size=2,
        length=4,
    )
    state = create_state(cfg)
    carry = jnp.ones((2, 4, 8), dtype=jnp.float32) * 0.5
    tokens = jnp.asarray([[65], [66]], dtype=jnp.int32)
    resets = jnp.zeros_like(tokens, dtype=bool)
    _, final_state = state.apply_fn({"params": state.params}, tokens, carry, resets)
    # Exactly 2 nodes per batch element should update; 2 nodes must remain identically unchanged
    deltas = np.linalg.norm(np.asarray(final_state - carry), axis=-1)
    assert np.all(np.sum(deltas == 0.0, axis=-1) == 2)
    assert np.all(np.sum(deltas > 0.0, axis=-1) == 2)

    # Verify gradients still flow to the router via straight-through estimator
    batch = {
        "inputs": jnp.asarray([[65, 66, 67, 68], [69, 70, 71, 72]], dtype=jnp.int32),
        "targets": jnp.asarray([[66, 67, 68, 69], [70, 71, 72, 73]], dtype=jnp.int32),
        "mask": jnp.ones((2, 4), dtype=jnp.float32),
        "resets": jnp.zeros((2, 4), dtype=bool).at[:, 0].set(True),
    }
    _, grads = jax.value_and_grad(routed_objective, has_aux=True)(
        state.params, state.apply_fn, batch, carry, cfg
    )
    assert float(jnp.linalg.norm(grads["router_timescale_kernel"])) > 0.0


def test_learned_edge_rewiring_and_custom_topology():
    cfg = Config(
        nodes=4,
        width=8,
        degree=2,
        node_update="timescale",
        edge_aggregation="attention",
        batch_size=2,
        length=8,
    )
    state = create_state(cfg)
    # Boost coupling from node 0 -> node 3 so node 3 wants node 0 in its incoming edges
    params = dict(state.params)
    params["read_kernel"] = params["read_kernel"].at[0].set(5.0)
    params["input_kernel"] = params["input_kernel"].at[3, 8:, :].set(5.0)
    aff = edge_affinity_matrix(cfg, params)
    assert aff.shape == (4, 4)
    assert all(not np.isfinite(aff[i, i]) for i in range(4))

    rewired_cfg, info = rewire_edges(cfg, params, max_rewire_per_node=1, min_gain=1e-6)
    assert info["rewired"] >= 1
    assert rewired_cfg.edges is not None and len(rewired_cfg.edges) == 4
    # Step with rewired topology
    step_fn = make_routed_step(rewired_cfg)
    batch = {
        "inputs": jnp.ones((2, 8), dtype=jnp.int32) * 65,
        "targets": jnp.ones((2, 8), dtype=jnp.int32) * 66,
        "mask": jnp.ones((2, 8), dtype=jnp.float32),
        "resets": jnp.zeros((2, 8), dtype=bool).at[:, 0].set(True),
    }
    new_state, _, metrics = step_fn(state, initial_state(rewired_cfg, 2), batch)
    assert bool(metrics["finite"]) and int(new_state.step) == 1


def test_continual_cli_runner_end_to_end(tmp_path):
    data_dir = tmp_path / "data"
    prepare(data_dir)
    eng_train = tmp_path / "eng_train.txt"
    eng_dev = tmp_path / "eng_dev.txt"
    eng_train.write_text(
        "The quick brown fox jumps over the lazy dog.\n\nPack my box with five dozen liquor jugs.\n"
    )
    eng_dev.write_text(
        "Sphinx of black quartz, judge my vow.\n\nHow vexingly quick daft zebras jump.\n"
    )
    cfg = Config(
        nodes=4,
        width=8,
        degree=2,
        rounds=2,
        groups=2,
        node_update="timescale",
        edge_aggregation="attention",
        memory="relational",
        memory_keys=4,
        local_plasticity=True,
        adaptive_halt=True,
        ponder_beta=1e-3,
        optimizer="muon",
        batch_size=4,
        length=128,
        learning_rate=3e-3,
    )
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(cfg.model_dump_json(indent=2))
    out_dir = tmp_path / "continual_out"
    args = SimpleNamespace(
        data=data_dir,
        config=cfg_path,
        init=None,
        out=out_dir,
        english_train=eng_train,
        english_dev=eng_dev,
        steps=4,
        eval_every=2,
        max_records=4,
        qa_seed=42,
        english_seed=43,
        auto_grow=True,
        rewire=True,
    )
    summary = run_continual(args)
    assert summary["steps"] == 4
    assert (out_dir / "best.mlm").exists()
    assert (out_dir / "latest.mlm").exists()
    assert (out_dir / "summary.json").exists()


def test_deepseek_shared_nodes_and_loss_free_load_balancing():
    cfg = Config(
        nodes=4,
        width=8,
        degree=2,
        groups=2,
        node_update="timescale",
        shared_nodes=1,
        top_k_nodes=1,
        router_bias_gamma=0.05,
        batch_size=2,
        length=4,
    )
    state = create_state(cfg)
    carry = jnp.ones((2, 4, 8), dtype=jnp.float32) * 0.5
    tokens = jnp.asarray([[65], [66]], dtype=jnp.int32)
    resets = jnp.zeros_like(tokens, dtype=bool)
    _, final_state, routes = state.apply_fn(
        {"params": state.params}, tokens, carry, resets, return_routes=True
    )
    deltas = np.linalg.norm(np.asarray(final_state - carry), axis=-1)
    # Shared node 0 must always update; among specialist nodes 1..3, exactly top_k_nodes=1 updates
    assert np.all(deltas[:, 0] > 0.0)
    assert np.all(np.sum(deltas[:, 1:] > 0.0, axis=-1) == 1)
    assert np.all(np.sum(deltas[:, 1:] == 0.0, axis=-1) == 2)
    assert np.allclose(np.asarray(routes[:, :, 0]), 0.25)

    # Test auxiliary-loss-free load bias update
    bias0 = jnp.zeros((4,), dtype=jnp.float32)
    usage = jnp.asarray([0.25, 0.60, 0.10, 0.05], dtype=jnp.float32)
    bias1 = update_router_load_bias(bias0, usage, gamma=0.05, shared_nodes=1)
    assert float(bias1[0]) == 0.0
    assert float(bias1[1]) < 0.0 and float(bias1[2]) > 0.0 and float(bias1[3]) > 0.0


def test_gated_deltanet_memory_decay_and_mhc_sinkhorn_edges():
    cfg = Config(
        nodes=4,
        width=8,
        degree=2,
        rounds=2,
        node_update="timescale",
        edge_aggregation="attention",
        sinkhorn_edges=True,
        memory="relational",
        memory_keys=4,
        memory_decay=True,
        batch_size=2,
        length=6,
    )
    state = create_state(cfg)
    edges = jnp.asarray(topology(4, 2))
    raw_w = jax.nn.softmax(jnp.asarray([[[2.0, 0.1], [2.5, 0.2], [0.1, 1.8], [0.3, 1.5]]]), axis=-1)
    ds_w = sinkhorn_edge_weights(raw_w, edges, 4, iters=5)
    assert np.allclose(np.asarray(ds_w.sum(axis=-1)), 1.0, atol=1e-4)

    batch = {
        "inputs": jnp.asarray(
            [[65, 66, 67, 68, 69, 70], [71, 72, 73, 74, 75, 76]], dtype=jnp.int32
        ),
        "targets": jnp.asarray(
            [[66, 67, 68, 69, 70, 71], [72, 73, 74, 75, 76, 77]], dtype=jnp.int32
        ),
        "mask": jnp.ones((2, 6), dtype=jnp.float32),
        "resets": jnp.zeros((2, 6), dtype=bool).at[:, 0].set(True),
    }
    carry = initial_state(cfg, 2)
    _, grads = jax.value_and_grad(routed_objective, has_aux=True)(
        state.params, state.apply_fn, batch, carry, cfg
    )
    assert float(jnp.linalg.norm(grads["memory_decay_kernel"])) > 0.0
    assert float(jnp.linalg.norm(grads["memory_decay_bias"])) > 0.0


def test_deepseek_v3_causal_mtp_and_r1_grpo_step():
    cfg = Config(
        nodes=4,
        width=8,
        degree=2,
        node_update="timescale",
        mtp_weight=0.2,
        batch_size=3,
        length=32,
        learning_rate=1e-2,
    )
    state = create_state(cfg)
    batch = {
        "inputs": jnp.asarray(
            [[256, 65, 66, 67], [256, 68, 69, 70], [256, 71, 72, 73]], dtype=jnp.int32
        ),
        "targets": jnp.asarray(
            [[65, 66, 67, 68], [68, 69, 70, 71], [71, 72, 73, 74]], dtype=jnp.int32
        ),
        "mask": jnp.ones((3, 4), dtype=jnp.float32),
        "resets": jnp.zeros((3, 4), dtype=bool).at[:, 0].set(True),
    }
    (_, (_, metrics)), grads = jax.value_and_grad(routed_objective, has_aux=True)(
        state.params, state.apply_fn, batch, initial_state(cfg, 3), cfg
    )
    assert float(metrics["mtp_loss"]) > 0.0
    assert float(jnp.linalg.norm(grads["mtp_kernel"])) > 0.0

    # Test DeepSeek-R1 critic-free GRPO step on a candidate question group
    question = {
        "context": "Alice keeps a ruby.",
        "question": "What does Alice keep?",
        "candidates": ["ruby", "coin", "lamp"],
        "answer": "ruby",
    }
    g_batch, g_rewards = build_grpo_question_group(question, length=64)
    ref_params = state.params
    _, init_m = grpo_objective(state.params, ref_params, state.apply_fn, g_batch, g_rewards, cfg)
    init_gap = float(init_m["seq_logp"][0] - jnp.mean(init_m["seq_logp"][1:]))

    grpo_step = make_grpo_step(cfg)
    for _ in range(4):
        state, step_m = grpo_step(state, ref_params, g_batch, g_rewards)
    assert bool(step_m["finite"])
    assert float(step_m["kl_div"]) >= -1e-6
    _, final_m = grpo_objective(state.params, ref_params, state.apply_fn, g_batch, g_rewards, cfg)
    final_gap = float(final_m["seq_logp"][0] - jnp.mean(final_m["seq_logp"][1:]))
    assert final_gap > init_gap


def test_symmetric_span_memory_and_mla_episodic_ring_buffer():
    from mlm.associative import delta_step

    # 1. Verify bidirectional delta_step binds both key->value and rev_key->rev_value
    mem = jnp.zeros((1, 4, 4), dtype=jnp.float32)
    k_fwd = jnp.asarray([[1.0, 0.0, 0.0, 0.0]])
    v_fwd = jnp.asarray([[0.0, 1.0, 0.0, 0.0]])
    k_rev = jnp.asarray([[0.0, 0.0, 1.0, 0.0]])
    v_rev = jnp.asarray([[0.0, 0.0, 0.0, 1.0]])
    beta = jnp.asarray([1.0])
    mem_after, _ = delta_step(mem, k_fwd, v_fwd, k_fwd, beta, rev_key=k_rev, rev_value=v_rev)
    _, recalled_fwd = delta_step(mem_after, k_fwd, v_fwd, k_fwd, jnp.asarray([0.0]))
    _, recalled_rev = delta_step(mem_after, k_fwd, v_fwd, k_rev, jnp.asarray([0.0]))
    assert np.allclose(np.asarray(recalled_fwd), np.asarray(v_fwd), atol=1e-5)
    assert np.allclose(np.asarray(recalled_rev), np.asarray(v_rev), atol=1e-5)

    # 2. Verify GraphLM with symmetric_memory=True and episodic_slots=4
    cfg = Config(
        nodes=4,
        width=8,
        degree=2,
        rounds=2,
        node_update="timescale",
        memory="relational",
        memory_keys=4,
        memory_decay=True,
        symmetric_memory=True,
        episodic_slots=4,
        batch_size=2,
        length=6,
    )
    carry = initial_state(cfg, 2)
    # 4 nodes + 4 memory_keys + 4 trace_horizons + 4 episodic_slots = 16 rows
    assert carry.shape == (2, 16, 8)

    state = create_state(cfg)
    batch = {
        "inputs": jnp.asarray(
            [[65, 66, 67, 68, 69, 70], [71, 72, 73, 74, 75, 76]], dtype=jnp.int32
        ),
        "targets": jnp.asarray(
            [[66, 67, 68, 69, 70, 71], [72, 73, 74, 75, 76, 77]], dtype=jnp.int32
        ),
        "mask": jnp.ones((2, 6), dtype=jnp.float32),
        "resets": jnp.zeros((2, 6), dtype=bool).at[:, 0].set(True),
    }
    (_, (final_carry, _)), grads = jax.value_and_grad(routed_objective, has_aux=True)(
        state.params, state.apply_fn, batch, carry, cfg
    )
    assert final_carry.shape == (2, 16, 8)
    for param_name in (
        "memory_span_kernel",
        "memory_span_bias",
        "memory_rev_key_kernel",
        "memory_rev_value_kernel",
        "episodic_compress_kernel",
        "episodic_write_kernel",
        "episodic_write_bias",
        "episodic_query_kernel",
        "episodic_pos_bias",
        "episodic_output_kernel",
        "episodic_gain",
    ):
        assert float(jnp.linalg.norm(grads[param_name])) > 0.0, f"Zero grad for {param_name}"


def test_surprise_gating_dog_wavelets_and_relational_hops():
    cfg = Config(
        nodes=4,
        width=8,
        degree=2,
        rounds=2,
        node_update="timescale",
        memory="relational",
        memory_keys=4,
        memory_decay=True,
        symmetric_memory=True,
        surprise_gating=True,
        episodic_slots=4,
        relational_hops=2,
        batch_size=2,
        length=6,
    )
    carry = initial_state(cfg, 2)
    state = create_state(cfg)
    batch = {
        "inputs": jnp.asarray(
            [[65, 66, 67, 68, 69, 70], [71, 72, 73, 74, 75, 76]], dtype=jnp.int32
        ),
        "targets": jnp.asarray(
            [[66, 67, 68, 69, 70, 71], [72, 73, 74, 75, 76, 77]], dtype=jnp.int32
        ),
        "mask": jnp.ones((2, 6), dtype=jnp.float32),
        "resets": jnp.zeros((2, 6), dtype=bool).at[:, 0].set(True),
    }
    (_, (final_carry, _)), grads = jax.value_and_grad(routed_objective, has_aux=True)(
        state.params, state.apply_fn, batch, carry, cfg
    )
    assert final_carry.shape == (2, 16, 8)
    for param_name in (
        "surprise_mem_weight",
        "surprise_span_weight",
        "rel_wavelet_kernel",
        "rel_q_kernel",
        "rel_k_kernel",
        "rel_v_kernel",
        "rel_output_kernel",
        "rel_gain",
    ):
        assert float(jnp.linalg.norm(grads[param_name])) > 0.0, f"Zero grad for {param_name}"

    # Verify strict causal masking across document resets when relational_hops > 0
    params_active = dict(state.params, rel_gain=jnp.asarray(1.2, dtype=jnp.float32))
    inputs_a = jnp.asarray([[65, 66, 67, 68, 69, 70]], dtype=jnp.int32)
    inputs_b = jnp.asarray([[65, 66, 67, 90, 91, 92]], dtype=jnp.int32)
    resets_1 = jnp.asarray([[True, False, False, True, False, False]])
    logits_a, _ = state.apply_fn(
        {"params": params_active}, inputs_a, initial_state(cfg, 1), resets_1
    )
    logits_b, _ = state.apply_fn(
        {"params": params_active}, inputs_b, initial_state(cfg, 1), resets_1
    )
    # First 3 tokens (before second reset at index 3) must be strictly identical
    assert np.allclose(np.asarray(logits_a[:, :3]), np.asarray(logits_b[:, :3]), atol=1e-5)
    # And tokens after a reset at index 3 must not depend on tokens before index 3
    inputs_c = jnp.asarray([[80, 81, 82, 68, 69, 70]], dtype=jnp.int32)
    logits_c, _ = state.apply_fn(
        {"params": params_active}, inputs_c, initial_state(cfg, 1), resets_1
    )
    assert np.allclose(np.asarray(logits_a[:, 3:]), np.asarray(logits_c[:, 3:]), atol=1e-5)

    # Verify zero-shot bidirectional (forward + backward) counterfactual relational binding
    from mlm.comprehension import evaluate_questions

    probes = [
        {
            "id": "hat-color-0",
            "pair": "hat-color",
            "category": "attribute",
            "context": "Mia has a red hat.",
            "question": "What color is Mia's hat?",
            "choices": ["red", "blue"],
            "answer_index": 0,
        },
        {
            "id": "hat-color-1",
            "pair": "hat-color",
            "category": "attribute",
            "context": "Mia has a blue hat.",
            "question": "What color is Mia's hat?",
            "choices": ["red", "blue"],
            "answer_index": 1,
        },
        {
            "id": "owner-0",
            "pair": "owner",
            "category": "binding",
            "context": "Nora owns the kite. Lena owns the cup.",
            "question": "Who owns the kite?",
            "choices": ["Nora", "Lena"],
            "answer_index": 0,
        },
        {
            "id": "owner-1",
            "pair": "owner",
            "category": "binding",
            "context": "Lena owns the kite. Nora owns the cup.",
            "question": "Who owns the kite?",
            "choices": ["Nora", "Lena"],
            "answer_index": 1,
        },
    ]
    eval_res = evaluate_questions(state.replace(params=params_active), cfg, probes, format="chat")
    assert eval_res["accuracy"] == 1.0
    assert eval_res["pair_accuracy"] == 1.0
