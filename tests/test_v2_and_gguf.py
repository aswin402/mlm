"""Tests for MLM v2 (Deep Hybrid Liquid-Delta + Rust BPE + SafeTensors + GGUF v3)."""

from pathlib import Path

from mlm.cli import main
from mlm.gguf_export import export_gguf_v2, verify_gguf_with_rust
from mlm.model_v2 import (
    ConfigV2,
    build_layer_execution_schedule,
    compute_ncm_concept_loss_v2,
    compute_ornith_scaffold_mask,
    create_state_v2,
    dsa_lightning_index_mask,
    expand_shortconv_kernel_v2,
    forward_v2,
    forward_with_hidden_v2,
    generate_v2,
    grow_model_v2_depth,
    load_checkpoint_v2,
    make_train_step_v2,
    merge_checkpoints_v2,
    ncm_tied_codebook_logits,
    save_checkpoint_v2,
    sinkhorn_knopp_2x2,
    tempered_topk_kd_loss_v2,
)
from mlm.tokenizer import BPETokenizer
from mlm.train_v2 import (
    build_tokenized_dataset,
    compute_ornith_task_weights_v2,
    format_chat_prompt_v2,
    run_grpo_v2,
    score_grpo_response_v2,
)


def test_bpe_tokenizer_and_byte_span_mask():
    corpus = [
        "<|user|>\nHi\n<|assistant|>\nHello! How can I help you today?",
        "<|user|>\nHow are you?\n<|assistant|>\nI am doing well, thank you for asking!",
        "<|user|>\nWhat is science?\n<|assistant|>\nScience is the study of the natural world.",
    ] * 8
    tok = BPETokenizer.train_from_iterator(corpus, vocab_size=320, min_frequency=1)
    prefix = "<|user|>\nHi\n<|assistant|>\n"
    reply = "Hello! How can I help you today?"
    full = prefix + reply
    span_start = len(prefix.encode("utf-8"))
    span_end = len(full.encode("utf-8"))

    ids, mask = tok.encode_with_mask(full, loss_spans=[[span_start, span_end]], score_eos=True)
    assert len(ids) == len(mask) + 1
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.eos_id
    # First prompt tokens (<|user|>, \n, Hi) must have mask 0.0
    assert mask[0] == 0.0
    # Final reply tokens + EOS must have mask 1.0
    assert mask[-1] == 1.0
    assert mask[-2] == 1.0
    decoded = tok.decode(ids, skip_special_tokens=True)
    assert "Hello!" in decoded


def test_model_v2_hybrid_train_grow_and_gguf(tmp_path: Path):
    records = [
        {
            "text": "<|user|>\nHi\n<|assistant|>\nHello! How can I help you today?",
            "loss_spans": [[20, 52]],
            "score_eos": True,
        },
        {
            "text": "<|user|>\nWho are you?\n<|assistant|>\nI am an AI assistant.",
            "loss_spans": [[29, 50]],
            "score_eos": True,
        },
    ] * 4
    tok = BPETokenizer.train_from_iterator(
        [r["text"] for r in records], vocab_size=300, min_frequency=1
    )
    cfg = ConfigV2(
        arch_version="v2_hybrid",
        vocab_size=tok.vocab_size,
        d_model=64,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        ffn_dim=128,
        max_seq_len=48,
        batch_size=4,
        learning_rate=3e-3,
    )
    ds = build_tokenized_dataset(records, tok, cfg.max_seq_len)
    state = create_state_v2(cfg, total_steps=15)
    step_fn = make_train_step_v2(cfg, total_steps=15)

    initial_loss = None
    last_loss = None
    for s in range(12):
        state.params, state.opt_state, loss, _ = step_fn(
            state.params, state.opt_state, ds["inputs"][:4], ds["targets"][:4], ds["mask"][:4]
        )
        if s == 0:
            initial_loss = float(loss)
        last_loss = float(loss)

    assert last_loss < initial_loss

    # Progressive depth growth (4 -> 6 layers) must preserve forward logits at growth step
    logits_before = forward_v2(state.params, ds["inputs"][:2], cfg)
    grown_params, grown_cfg = grow_model_v2_depth(state.params, cfg, 6)
    logits_after = forward_v2(grown_params, ds["inputs"][:2], grown_cfg)
    assert logits_before.shape == logits_after.shape

    # LFM3 ShortConv receptive-field expansion (K=4 -> K=7) must be 100% lossless at step 0
    k7_params, k7_cfg = expand_shortconv_kernel_v2(grown_params, grown_cfg, new_kernel=7)
    assert k7_cfg.conv_kernel == 7
    assert k7_cfg.mature_conv_kernel == 4
    logits_k7 = forward_v2(k7_params, ds["inputs"][:2], k7_cfg)
    assert float(abs(logits_after - logits_k7).max()) < 1e-6

    # LFM2.5 Tempered Decoupled Top-K KD loss must be 0.0 when student == teacher
    kd_zero = float(
        tempered_topk_kd_loss_v2(logits_k7, logits_after, ds["mask"][:2], top_k=16, temperature=2.0)
    )
    assert abs(kd_zero) < 1e-5
    merged_p = merge_checkpoints_v2(grown_params, k7_params, alpha=0.5)
    assert len(merged_p) == len(k7_params)

    # SenseNova-Loop mHC Sinkhorn-Knopp Birkhoff projection & Contiguous Hybrid Looping
    import jax.numpy as jnp

    ds_mat = sinkhorn_knopp_2x2(jnp.array([[1.2, -0.4], [0.3, 0.9]], dtype=jnp.float32), iters=8)
    assert bool(jnp.all(ds_mat >= 0.0))
    assert float(jnp.max(jnp.abs(jnp.sum(ds_mat, axis=-1) - 1.0))) < 1e-4
    assert float(jnp.max(jnp.abs(jnp.sum(ds_mat, axis=-2) - 1.0))) < 1e-4

    loop_cfg = k7_cfg.model_copy(
        update={
            "loop_start_layer": 1,
            "loop_end_layer": 5,
            "loop_repeats": 2,
            "sliding_window": 8,
            "dsa_top_k": 4,
        }
    )
    sched = build_layer_execution_schedule(loop_cfg)
    assert len(sched) == 10  # 6 physical layers + 4 looped passes = 10 effective layer passes
    logits_loop, hidden_loop = forward_with_hidden_v2(k7_params, ds["inputs"][:2], loop_cfg)
    assert logits_loop.shape == logits_k7.shape

    # GLM-5.3-Flash DSA Lightning Indexer + IndexShare mask test
    dummy_logits = jnp.ones((2, 4, 16, 16), dtype=jnp.float32)
    dsa_m1, dsa_s1 = dsa_lightning_index_mask(dummy_logits, sliding_window=4, dsa_top_k=4)
    dsa_m2, _ = dsa_lightning_index_mask(
        dummy_logits, sliding_window=4, dsa_top_k=4, shared_index_scores=dsa_s1
    )
    assert dsa_m1.shape == (2, 4, 16, 16)
    assert dsa_m2.shape == (2, 4, 16, 16)
    assert "blk.0.deltanet.w_beta.weight" in k7_params
    assert "blk.0.deltanet.channel_gate" in k7_params

    # HazyResearch/ncm Next-Concept Horizon Loss & Tied Codebook Head
    ncm_loss = float(
        compute_ncm_concept_loss_v2(
            k7_params, hidden_loop, ds["targets"][:2], ds["mask"][:2], loop_cfg
        )
    )
    assert 0.0 <= ncm_loss <= 2.0
    tied_id = ncm_tied_codebook_logits(
        hidden_loop,
        k7_params["tok_embeddings.weight"],
        log_scale=jnp.zeros((1,), dtype=jnp.float32),
        norm_alpha=jnp.zeros((1,), dtype=jnp.float32),
    )
    raw_id = jnp.matmul(hidden_loop, jnp.swapaxes(k7_params["tok_embeddings.weight"], 0, 1))
    assert float(jnp.max(jnp.abs(tied_id - raw_id))) < 1e-6

    # Save & load checkpoint with SafeTensors + Rust BPE
    ckpt_path = tmp_path / "model.mlm2"
    save_checkpoint_v2(ckpt_path, grown_params, grown_cfg, tok, step=12)
    loaded_params, loaded_cfg, loaded_tok, manifest = load_checkpoint_v2(ckpt_path)
    assert loaded_cfg.n_layers == 6
    assert manifest["step"] == 12

    # Verify Stateful O(1) Recurrent Decode Cache matches full-sequence forward_with_hidden_v2
    from mlm.model_v2 import _make_cached_recurrent_decode_v2, init_decode_cache_v2

    seq_ids = ds["inputs"][:1, :10]
    full_logits, _ = forward_with_hidden_v2(loaded_params, seq_ids, loaded_cfg)
    _, spec2_fn = _make_cached_recurrent_decode_v2(loaded_cfg)
    carry = init_decode_cache_v2(loaded_cfg)
    for idx in range(0, 8, 2):
        _, (carry, _, _) = spec2_fn(
            loaded_params, carry, seq_ids[0, idx], seq_ids[0, idx + 1]
        )
    (_, logits_a, _), (_, logits_b, _) = spec2_fn(
        loaded_params, carry, seq_ids[0, 8], seq_ids[0, 9]
    )
    assert float(jnp.max(jnp.abs(logits_a - full_logits[0, 8]))) < 1e-4
    assert float(jnp.max(jnp.abs(logits_b - full_logits[0, 9]))) < 1e-4
    assert int(jnp.argmax(logits_a)) == int(jnp.argmax(full_logits[0, 8]))
    assert int(jnp.argmax(logits_b)) == int(jnp.argmax(full_logits[0, 9]))

    reply = generate_v2(
        loaded_params,
        loaded_cfg,
        loaded_tok,
        format_chat_prompt_v2("Hi"),
        max_new_tokens=16,
    )
    assert isinstance(reply, str) and len(reply) > 0

    # Verify 3-Tier Persistent & Episodic Memory Hierarchy (Tier 1 KV, Tier 2 DeltaNet, Tier 3)
    from mlm.model_v2 import MemoryHierarchyV2

    mem_path = tmp_path / "test.mlm2.memory"
    mem1 = MemoryHierarchyV2(loaded_cfg, max_slots=16, episodic_bits=2, storage_path=mem_path)
    store_msg = "Remember that my name is Aswin and the API port is 8080. Let x = 15, y = 7"
    out_store = generate_v2(
        loaded_params,
        loaded_cfg,
        loaded_tok,
        format_chat_prompt_v2(store_msg),
        max_new_tokens=16,
        include_think=True,
        memory_hierarchy=mem1,
    )
    assert "Remembered:" in out_store
    assert len(mem1.slots) == 4
    assert mem_path.exists()
    assert mem1.tier1_active_tokens > 0
    assert mem1.tier2_delta_norm >= 0.0

    # Reload memory in a new session and verify recall, SAO overwrite, and variable substitution
    mem2 = MemoryHierarchyV2(loaded_cfg, max_slots=16, episodic_bits=2, storage_path=mem_path)
    assert len(mem2.slots) == 4
    out_recall = generate_v2(
        loaded_params,
        loaded_cfg,
        loaded_tok,
        format_chat_prompt_v2("What is my name?"),
        max_new_tokens=16,
        include_think=True,
        memory_hierarchy=mem2,
    )
    assert "Aswin" in out_recall
    assert mem2.total_hits == 1

    # State-Action-Overwrite (SAO) on existing slot
    out_sao = generate_v2(
        loaded_params,
        loaded_cfg,
        loaded_tok,
        format_chat_prompt_v2("Remember that the API port is 9090 and x = 20"),
        max_new_tokens=16,
        include_think=True,
        memory_hierarchy=mem2,
    )
    assert "9090" in out_sao
    assert len(mem2.slots) == 4
    assert mem2.total_overwrites == 2
    out_port = generate_v2(
        loaded_params,
        loaded_cfg,
        loaded_tok,
        format_chat_prompt_v2("What is the API port?"),
        max_new_tokens=16,
        memory_hierarchy=mem2,
    )
    assert "9090" in out_port

    # Variable substitution in math query
    out_math = generate_v2(
        loaded_params,
        loaded_cfg,
        loaded_tok,
        format_chat_prompt_v2("What is x + y?"),
        max_new_tokens=16,
        include_think=True,
        memory_hierarchy=mem2,
    )
    assert "[Tier-3 Memory: x=20, y=7]" in out_math
    assert mem2.summary_dict()["tier3_persistent_episodic"]["compression_vs_fp32"] >= 4.0

    # Verify LMCache Hybrid Prefix/Full State Cache & CacheBlend Dynamic SO(2) RoPE Re-Rotation
    from mlm.model_v2 import (
        apply_rope,
        cacheblend_fuse_kv_chunks_v2,
        cacheblend_rope_shift_kv_v2,
        precompute_rope_freqs,
    )

    out_math_repeat = generate_v2(
        loaded_params,
        loaded_cfg,
        loaded_tok,
        format_chat_prompt_v2("What is x + y?"),
        max_new_tokens=16,
        include_think=True,
        memory_hierarchy=mem2,
    )
    assert out_math_repeat == out_math
    assert mem2.lmcache.full_hits + mem2.lmcache.prefix_hits >= 1
    assert mem2.lmcache.saved_prefill_tokens > 0

    cos_tbl, sin_tbl = precompute_rope_freqs(32, loaded_cfg.head_dim, loaded_cfg.rope_theta)
    raw_k = jnp.ones((1, 6, loaded_cfg.n_kv_heads, loaded_cfg.head_dim), dtype=jnp.float32)
    k_at_0 = apply_rope(raw_k, cos_tbl[:6], sin_tbl[:6])
    k_at_10 = apply_rope(raw_k, cos_tbl[10:16], sin_tbl[10:16])
    k_shifted = cacheblend_rope_shift_kv_v2(
        k_at_0,
        jnp.arange(0, 6, dtype=jnp.int32),
        jnp.arange(10, 16, dtype=jnp.int32),
        head_dim=loaded_cfg.head_dim,
        theta=loaded_cfg.rope_theta,
    )
    assert float(jnp.max(jnp.abs(k_shifted - k_at_10))) < 1e-5
    fused = cacheblend_fuse_kv_chunks_v2(
        [(k_at_0[:, :4], raw_k[:, :4], 0), (k_at_10[:, :4], raw_k[:, :4], 10)],
        head_dim=loaded_cfg.head_dim,
        theta=loaded_cfg.rope_theta,
    )
    assert fused["k_fused"].shape == (1, 8, loaded_cfg.n_kv_heads, loaded_cfg.head_dim)
    assert len(fused["hkvd_indices"]) >= 1

    # Verify v0.0.22 True Sparse Index-Gather Execution & Profiler
    from mlm.model_v2 import (
        AutonomousGrowthControllerV2,
        _execute_dense_nodes_step_v2,
        _execute_sparse_nodes_step_v2,
        compute_directed_edge_weights_v2,
        compute_spidernet_graph_regularizers_v2,
        directed_graph_message_passing_v2,
        evaluate_continual_learning_metrics_v2,
        profile_sparse_execution_v2,
        run_spidernet_ablation_suite_v2,
        split_specialist_nodes_orthogonal_v2,
    )

    sp_cfg = loaded_cfg.model_copy(
        update={"n_experts": 6, "top_k_experts": 2, "expert_ffn_dim": 64}
    )
    sp_params = create_state_v2(sp_cfg, total_steps=4).params

    prof = profile_sparse_execution_v2(sp_params, sp_cfg, [1, 2, 3, 4], n_bench_steps=4)
    assert prof["max_logit_diff"] < 1e-4
    assert prof["sparse_specialist_flops_fast_per_tok"] < prof["dense_specialist_flops_per_tok"]
    assert prof["activation_ratio_fast"] <= 0.5

    ffn_x = jnp.ones((1, 1, sp_cfg.d_model), dtype=jnp.float32) * 0.25
    probs_sparse = jnp.zeros((1, 1, sp_cfg.n_experts), dtype=jnp.float32).at[0, 0, :2].set(
        jnp.array([0.65, 0.35], dtype=jnp.float32)
    )
    out_sp = _execute_sparse_nodes_step_v2(
        ffn_x,
        probs_sparse,
        sp_params["blk.0.nodes.gate.weight"],
        sp_params["blk.0.nodes.up.weight"],
        sp_params["blk.0.nodes.down.weight"],
        k_active=2,
    )
    out_dn = _execute_dense_nodes_step_v2(
        ffn_x,
        probs_sparse,
        sp_params["blk.0.nodes.gate.weight"],
        sp_params["blk.0.nodes.up.weight"],
        sp_params["blk.0.nodes.down.weight"],
    )
    assert float(jnp.max(jnp.abs(out_sp - out_dn))) < 1e-5

    # Verify v0.0.23 Directed Graph Message Passing & v0.0.24 Input-Conditioned Edge Routing
    c_active = jnp.full((sp_cfg.n_experts,), 0.85, dtype=jnp.float32)
    out_sp_msg = _execute_sparse_nodes_step_v2(
        ffn_x,
        probs_sparse,
        sp_params["blk.0.nodes.gate.weight"],
        sp_params["blk.0.nodes.up.weight"],
        sp_params["blk.0.nodes.down.weight"],
        k_active=2,
        edge_u=sp_params["blk.0.nodes.edge_u"],
        edge_v=sp_params["blk.0.nodes.edge_v"],
        edge_proj_h=sp_params["blk.0.nodes.edge_proj_h"],
        edge_proj_x=sp_params["blk.0.nodes.edge_proj_x"],
        edge_bias=sp_params["blk.0.nodes.edge_bias"],
        msg_down=sp_params["blk.0.nodes.msg_down"],
        msg_up=sp_params["blk.0.nodes.msg_up"],
        collab_gate=c_active,
        top_k_edges=1,
    )
    out_dn_msg = _execute_dense_nodes_step_v2(
        ffn_x,
        probs_sparse,
        sp_params["blk.0.nodes.gate.weight"],
        sp_params["blk.0.nodes.up.weight"],
        sp_params["blk.0.nodes.down.weight"],
        edge_u=sp_params["blk.0.nodes.edge_u"],
        edge_v=sp_params["blk.0.nodes.edge_v"],
        edge_proj_h=sp_params["blk.0.nodes.edge_proj_h"],
        edge_proj_x=sp_params["blk.0.nodes.edge_proj_x"],
        edge_bias=sp_params["blk.0.nodes.edge_bias"],
        msg_down=sp_params["blk.0.nodes.msg_down"],
        msg_up=sp_params["blk.0.nodes.msg_up"],
        collab_gate=c_active,
        top_k_edges=1,
    )
    assert float(jnp.max(jnp.abs(out_sp_msg - out_dn_msg))) < 1e-5
    assert float(jnp.max(jnp.abs(out_sp_msg - out_sp))) > 1e-5

    dummy_h1 = jnp.ones((1, 2, sp_cfg.n_experts, sp_cfg.d_model), dtype=jnp.float32) * 0.1
    x_ctx = jnp.stack(
        [
            jnp.ones((sp_cfg.d_model,), dtype=jnp.float32),
            -jnp.ones((sp_cfg.d_model,), dtype=jnp.float32),
        ],
        axis=0,
    )[None, :, :]
    a_ij = compute_directed_edge_weights_v2(
        x_ctx,
        dummy_h1,
        sp_params["blk.0.nodes.edge_u"],
        sp_params["blk.0.nodes.edge_v"],
        sp_params["blk.0.nodes.edge_proj_h"],
        sp_params["blk.0.nodes.edge_proj_x"],
        sp_params["blk.0.nodes.edge_bias"],
        top_k_edges=2,
    )
    # Zero self-loop diagonal A_ii == 0 and input-conditioned variation across tokens t=0 vs t=1
    assert float(jnp.max(jnp.abs(jnp.diagonal(a_ij, axis1=-2, axis2=-1)))) == 0.0
    assert float(jnp.max(jnp.abs(a_ij[:, 0] - a_ij[:, 1]))) > 1e-4
    _, a_ij_mp = directed_graph_message_passing_v2(
        x_ctx,
        jnp.full((1, 2, sp_cfg.n_experts), 1.0 / sp_cfg.n_experts, dtype=jnp.float32),
        dummy_h1,
        sp_params["blk.0.nodes.edge_u"],
        sp_params["blk.0.nodes.edge_v"],
        sp_params["blk.0.nodes.edge_proj_h"],
        sp_params["blk.0.nodes.edge_proj_x"],
        sp_params["blk.0.nodes.edge_bias"],
        sp_params["blk.0.nodes.msg_down"],
        sp_params["blk.0.nodes.msg_up"],
        c_active,
        top_k_edges=2,
        return_edge_weights=True,
    )
    assert a_ij_mp.shape == (1, 2, sp_cfg.n_experts, sp_cfg.n_experts)

    # Verify v0.0.23 state-dependent residual gate gamma_j gradient flow through sender & receiver
    import jax
    import numpy as np

    def _mp_loss_fn(h1_in: jnp.ndarray) -> jnp.ndarray:
        mp_h2 = directed_graph_message_passing_v2(
            x_ctx,
            jnp.full((1, 2, sp_cfg.n_experts), 1.0 / sp_cfg.n_experts, dtype=jnp.float32),
            h1_in,
            sp_params["blk.0.nodes.edge_u"],
            sp_params["blk.0.nodes.edge_v"],
            sp_params["blk.0.nodes.edge_proj_h"],
            sp_params["blk.0.nodes.edge_proj_x"],
            sp_params["blk.0.nodes.edge_bias"],
            sp_params["blk.0.nodes.msg_down"],
            sp_params["blk.0.nodes.msg_up"],
            c_active,
            top_k_edges=2,
        )
        return jnp.sum(jnp.square(mp_h2))

    grad_h1 = np.asarray(jax.grad(_mp_loss_fn)(dummy_h1))
    assert float(np.linalg.norm(grad_h1[:, :, 0, :])) > 0.0
    assert float(np.linalg.norm(grad_h1[:, :, 1, :])) > 0.0

    reg_out = compute_spidernet_graph_regularizers_v2(
        sp_params,
        jnp.array([[1, 2, 3, 4]], dtype=jnp.int32),
        jnp.ones((1, 4), dtype=jnp.float32),
        sp_cfg,
    )
    assert float(reg_out["total_reg"]) > 0.0
    assert float(reg_out["load_loss"]) > 0.0
    assert float(reg_out["edge_loss"]) > 0.0
    assert float(reg_out["diversity_loss"]) >= 0.0

    # Verify v0.0.25 Autonomous Growth Controller & v0.0.26 Symmetric Orthogonal Splitting
    split_params, split_cfg, split_diag = split_specialist_nodes_orthogonal_v2(
        sp_params, sp_cfg, add_nodes=1, epsilon=0.02, growth_mode="sibling"
    )
    assert split_cfg.n_experts == sp_cfg.n_experts + 1
    assert split_diag["growth_mode"] == "sibling"
    assert split_diag["max_orthogonality_cosine"] < 1e-4
    assert split_diag["max_mean_reconstruction_err"] < 1e-5
    assert split_params["blk.0.nodes.gate.weight"].shape[0] == sp_cfg.n_experts + 1
    assert split_params["blk.0.nodes.edge_bias"].shape == (
        sp_cfg.n_experts + 1,
        sp_cfg.n_experts + 1,
    )

    child_params, child_cfg, child_diag = split_specialist_nodes_orthogonal_v2(
        sp_params, sp_cfg, parent_indices=[0], add_nodes=1, epsilon=0.02, growth_mode="child"
    )
    assert child_cfg.n_experts == sp_cfg.n_experts + 1
    assert child_diag["growth_mode"] == "child"
    assert float(child_params["blk.0.nodes.edge_bias"][0, sp_cfg.n_experts]) == 1.0
    assert abs(float(child_params["blk.0.nodes.edge_bias"][sp_cfg.n_experts, 0]) - 0.1) < 1e-5

    # Verify v0.0.25 single-step outlier rejection vs sustained domain shift spawning
    ctrl_outlier = AutonomousGrowthControllerV2(
        ema_alpha=0.10, threshold_h=2.50, delta_drift=0.05, cooldown_steps=2
    )
    for s_idx in range(1, 5):
        ctrl_outlier.update_cusum(1.0)
    _, _, outlier_evt = ctrl_outlier.observe_and_maybe_grow(
        step=5,
        loss_val=1.15,  # single noisy outlier spike (+1.7 sigma) < threshold_h=2.50
        params=sp_params,
        grads=None,
        inputs=jnp.array([[1, 2, 3, 4]], dtype=jnp.int32),
        mask=jnp.ones((1, 4), dtype=jnp.float32),
        config=sp_cfg,
        add_nodes=1,
    )
    assert outlier_evt["grew"] is False

    ctrl = AutonomousGrowthControllerV2(threshold_h=0.5, delta_drift=0.05, cooldown_steps=2)
    ctrl.update_cusum(1.0)
    grown_p, grown_c, evt = ctrl.observe_and_maybe_grow(
        step=5,
        loss_val=4.5,
        params=sp_params,
        grads=None,
        inputs=jnp.array([[1, 2, 3, 4]], dtype=jnp.int32),
        mask=jnp.ones((1, 4), dtype=jnp.float32),
        config=sp_cfg,
        add_nodes=1,
    )
    assert evt["grew"] is True
    assert evt["growth_mode"] in ("sibling", "child")
    assert grown_c.n_experts == sp_cfg.n_experts + 1
    assert len(ctrl.growth_events) == 1

    # Verify v0.0.26 Domain Mutual Information I(N; D)
    from mlm.model_v2 import compute_node_domain_mutual_info_v2

    mi_stats = compute_node_domain_mutual_info_v2(
        np.array(
            [
                [0.45, 0.45, 0.05, 0.05],
                [0.05, 0.05, 0.45, 0.45],
            ],
            dtype=np.float64,
        )
    )
    assert mi_stats["I_N_D_bits"] > 0.3
    assert 0.3 < mi_stats["NMI"] <= 1.0

    # Verify v0.0.27 Continual Learning Metrics & v0.0.28 6-Architecture Ablation Suite
    cl_m = evaluate_continual_learning_metrics_v2(
        [
            [0.5, 2.0, 2.0, 2.0],
            [0.6, 0.5, 2.0, 2.0],
            [0.65, 0.55, 0.4, 2.0],
            [0.7, 0.6, 0.45, 0.35],
        ],
        active_k=2,
        total_n=10,
    )
    assert 0.0 < cl_m["AA"] <= 1.0
    assert cl_m["BWT"] < 0.0
    assert cl_m["retention_pct"] > 50.0
    assert cl_m["activation_ratio"] == 0.20

    ablation_res = run_spidernet_ablation_suite_v2(
        vocab_size=loaded_cfg.vocab_size, d_model=64, steps_per_task=3, seeds=(42, 43)
    )
    assert len(ablation_res) == 6
    assert "A_dense_baseline" in ablation_res
    assert "F_spidernet_full_autonomous" in ablation_res
    assert "AA_std" in ablation_res["F_spidernet_full_autonomous"]
    assert "I_N_D_bits" in ablation_res["F_spidernet_full_autonomous"]
    assert "total_params" in ablation_res["A_dense_baseline"]
    assert (
        ablation_res["F_spidernet_full_autonomous"]["final_n_nodes"]
        > ablation_res["B_static_flat_moe"]["final_n_nodes"]
    )

    # Verify v0.0.29 Live Directed Graph Edge Calibration & /graph Telemetry Inspector
    from mlm.model_v2 import (
        inject_and_calibrate_spidernet_graph_edges_v2,
        inspect_directed_graph_edges_v2,
    )
    from mlm.tui import format_graph_inspector_v2

    cal_params = inject_and_calibrate_spidernet_graph_edges_v2(
        sp_params, sp_cfg, collab_strength=0.015
    )
    g_info = inspect_directed_graph_edges_v2(
        cal_params, sp_cfg, loaded_tok, format_chat_prompt_v2("What is (9 + 8) * 2?")
    )
    assert g_info["collab_gate_mean"] > 0.01
    assert g_info["mean_msg_norm"] > 0.0
    assert len(g_info["top_edges"]) >= 1
    assert "──►" in str(g_info["summary_str"])
    g_panel = format_graph_inspector_v2(
        cal_params, sp_cfg, loaded_tok, format_chat_prompt_v2("What is (9 + 8) * 2?"), color=False
    )
    assert "SpiderNet Directed Graph & Edge Inspector (/graph)" in g_panel
    assert "a_ij=" in g_panel

    # Verify v0.0.32 O(1) Ring-Buffer DSA & Recurrent Eviction Bridge
    from mlm.model_v2 import (
        adaptive_multi_hop_message_passing_v2,
        profile_o1_memory_scaling_v2,
        trace_compositional_3hop_chain_v2,
    )

    o1_prof = profile_o1_memory_scaling_v2(
        loaded_params,
        loaded_cfg,
        seq_lengths=(16, 32, 64),
        ring_window=16,
    )
    assert o1_prof["is_strictly_o1_constant_memory"] is True
    assert o1_prof["o1_bytes_per_token_slope"] == 0.0
    assert o1_prof["standard_bytes_per_token_slope"] > 0.0
    assert o1_prof["max_logit_diff_within_window"] < 1e-5
    assert o1_prof["evicted_summary_state_norm"] > 0.0

    # Verify v0.0.33 Adaptive Multi-Hop (1-Hop / 2-Hop / 3-Hop) Message Passing & Halting
    mh_out, mh_diag = adaptive_multi_hop_message_passing_v2(
        x_ctx,
        jnp.full((1, 2, sp_cfg.n_experts), 1.0 / sp_cfg.n_experts, dtype=jnp.float32),
        dummy_h1,
        sp_params["blk.0.nodes.edge_u"],
        sp_params["blk.0.nodes.edge_v"],
        sp_params["blk.0.nodes.edge_proj_h"],
        sp_params["blk.0.nodes.edge_proj_x"],
        sp_params["blk.0.nodes.edge_bias"],
        sp_params["blk.0.nodes.msg_down"],
        sp_params["blk.0.nodes.msg_up"],
        c_active,
        top_k_edges=2,
        max_hops=3,
        return_diagnostics=True,
    )
    assert mh_out.shape == (1, 2, sp_cfg.d_model)
    assert "a_hop1" in mh_diag
    assert "a_hop2" in mh_diag

    hop_suite = trace_compositional_3hop_chain_v2(cal_params, sp_cfg, loaded_tok)
    assert hop_suite["verified_1hop_fast_exit"] is True
    assert hop_suite["verified_2hop_single_domain"] is True
    assert hop_suite["verified_3hop_compositional_chain"] is True
    assert hop_suite["hop_1_greeting"]["active_hops"] == 1
    assert hop_suite["hop_2_science"]["active_hops"] == 2
    assert hop_suite["hop_3_compositional_code_math"]["active_hops"] == 3
    assert hop_suite["hop_3_compositional_code_math"]["chain_nodes"] == [4, 15, 24]

    # Export to GGUF v3 and verify with Rust `mlm-gguf-inspect`
    gguf_path = tmp_path / "model.gguf"
    info = export_gguf_v2(gguf_path, loaded_params, loaded_cfg, loaded_tok)
    assert info["architecture"] == "lfm2"
    rust_check = verify_gguf_with_rust(gguf_path)
    assert rust_check["valid"] is True
    assert rust_check["version"] == 3
    assert rust_check["tensor_count"] == info["tensor_count"]


def test_cli_train_v2_and_chat(tmp_path: Path, capsys):
    import json

    train_p = tmp_path / "train.jsonl"
    val_p = tmp_path / "val.jsonl"
    rec1 = {
        "text": "<|user|>\nHi\n<|assistant|>\nHello there!",
        "loss_spans": [[20, 32]],
        "score_eos": True,
    }
    rec2 = {
        "text": "<|user|>\nHello\n<|assistant|>\nHi! How can I help?",
        "loss_spans": [[23, 42]],
        "score_eos": True,
    }
    train_p.write_text((json.dumps(rec1) + "\n") * 4)
    val_p.write_text((json.dumps(rec2) + "\n") * 2)
    cfg_p = tmp_path / "cfg.json"
    cfg_p.write_text(
        json.dumps(
            {
                "arch_version": "v2_hybrid",
                "vocab_size": 300,
                "d_model": 64,
                "n_layers": 2,
                "n_heads": 4,
                "n_kv_heads": 2,
                "ffn_dim": 128,
                "n_experts": 4,
                "top_k_experts": 2,
                "expert_ffn_dim": 64,
                "max_seq_len": 32,
                "batch_size": 2,
            }
        )
    )
    out_dir = tmp_path / "out_v2"
    gguf_p = tmp_path / "exported.gguf"
    main(
        [
            "train-v2",
            "--train",
            str(train_p),
            "--validation",
            str(val_p),
            "--config",
            str(cfg_p),
            "--out",
            str(out_dir),
            "--steps",
            "8",
            "--eval-every",
            "4",
            "--export-gguf",
            str(gguf_p),
        ]
    )
    assert (out_dir / "best.mlm2").exists()
    assert gguf_p.exists()

    # Grow +2 specialist nodes (4 -> 6 nodes) from the trained checkpoint
    out_grown = tmp_path / "out_v2_grown"
    main(
        [
            "train-v2",
            "--train",
            str(train_p),
            "--validation",
            str(val_p),
            "--init",
            str(out_dir / "latest.mlm2"),
            "--add-nodes",
            "2",
            "--out",
            str(out_grown),
            "--steps",
            "4",
            "--eval-every",
            "4",
        ]
    )
    _, grown_cfg, _, _ = load_checkpoint_v2(out_grown / "latest.mlm2")
    assert grown_cfg.n_experts == 6
    assert grown_cfg.mature_experts == 4
    capsys.readouterr()

    main(["chat", str(out_grown / "latest.mlm2"), "--message", "Hi"])
    out = capsys.readouterr().out
    assert len(out.strip()) > 0

    # Test DeepSeek-R1 + Ornith-1.5 verifiable reward, validity gate, & frontier weights
    import numpy as np

    r_good = score_grpo_response_v2(
        "What is science?",
        "<|think|>Science studies the natural world.<|/think|>\nScience studies nature.",
        "Science studies nature.",
    )
    r_bad = score_grpo_response_v2(
        "What is science?",
        "<|think|>Unclosed tag",
        "Science studies nature.",
    )
    r_parrot = score_grpo_response_v2(
        "What is science?",
        "What is science?",
        "Science studies nature.",
    )
    assert r_good > r_bad
    assert r_bad < 0.0
    assert r_parrot < 0.0

    t_weights = compute_ornith_task_weights_v2(
        np.array([0.35, 1.0, 0.0]),
        np.array([1.0, 1.0, 1.0]),
        p_star=0.35,
    )
    assert t_weights[0] > t_weights[1]

    import jax.numpy as jnp

    s_mask = compute_ornith_scaffold_mask(
        jnp.array([[10, 6, 12, 7, 13]], dtype=jnp.int32), 6, think_close_id=7
    )
    assert float(s_mask[0, 0, 0]) == 0.0
    assert float(s_mask[0, 1, 0]) == 1.0
    assert float(s_mask[0, 2, 0]) == 1.0
    assert float(s_mask[0, 3, 0]) == 0.0
    assert float(s_mask[0, 4, 0]) == 0.0

    out_grpo = tmp_path / "out_v2_grpo"
    grpo_res = run_grpo_v2(
        train_p,
        out_grpo,
        out_grown / "latest.mlm2",
        steps=2,
        group_size=3,
    )
    assert (out_grpo / "latest.mlm2").exists()
    assert grpo_res["grpo_steps"] == 2

    # Test Multi-Node Specialist Cluster & 2-Bank Hierarchical Domain Growth (10 -> 14 Nodes)
    from mlm.model_v2 import compute_fone_abacus_embeddings, grow_model_v2_nodes
    from mlm.node_cluster import MLMNodeCluster

    p_base, c_base, tok_loaded, _ = load_checkpoint_v2(out_grown / "latest.mlm2")
    assert tok_loaded.think_open_id == 6
    assert tok_loaded.think_close_id == 7
    assert len(tok_loaded.get_digit_token_ids()) == 10
    blt_tbl = tok_loaded.compute_blt_ngram_table(64)
    assert blt_tbl.shape == (c_base.vocab_size, 64)
    d_ids = tok_loaded.get_digit_token_ids()
    fone_out = compute_fone_abacus_embeddings(
        jnp.array([[d_ids[9], d_ids[8]]], dtype=jnp.int32),
        p_base,
        c_base,
    )
    assert fone_out.shape == (1, 2, c_base.d_model)

    p_10, c_10 = grow_model_v2_nodes(p_base, c_base, add_nodes=4)
    p_14, c_14 = grow_model_v2_nodes(p_10, c_10, add_nodes=4)
    assert c_14.n_experts == 14
    assert c_14.mature_experts == 10
    logits_14 = forward_v2(p_14, jnp.array([[1, 2, 3, 4]], dtype=jnp.int32), c_14)
    assert logits_14.shape == (1, 4, c_14.vocab_size)

    from mlm.model_v2 import (
        _verify_code_and_logic_v2,
        classify_cognitive_mode_v2,
        get_spidernet_node_role_v2,
        grow_dot2_code_logic_v2,
        inspect_directed_graph_edges_v2,
        trace_reading_node_collaborator_switch_v2,
    )

    p_20, c_20 = grow_model_v2_nodes(p_14, c_14, add_nodes=6)
    assert c_20.n_experts == 20
    p_30, c_30, dot2_m = grow_dot2_code_logic_v2(p_20, c_20)
    assert c_30.n_experts == 30
    assert c_30.mature_experts == 20
    assert dot2_m["flops_reduction_fast_x"] == 15.0
    assert dot2_m["flops_reduction_think_x"] == 7.5
    assert dot2_m["max_orthogonality_cosine"] < 1e-4
    assert dot2_m["ensemble_mean_recon_error"] < 1e-4
    assert get_spidernet_node_role_v2(20) == "Code/Syntax"
    assert get_spidernet_node_role_v2(22) == "Algo/Loops"
    assert get_spidernet_node_role_v2(24) == "Bool/Logic"
    assert get_spidernet_node_role_v2(26) == "Types/Debug"
    assert get_spidernet_node_role_v2(28) == "Code-<think>"
    logits_20 = np.asarray(forward_v2(p_20, jnp.array([[1, 2, 3, 4]], dtype=jnp.int32), c_20))
    logits_30 = np.asarray(forward_v2(p_30, jnp.array([[1, 2, 3, 4]], dtype=jnp.int32), c_30))
    assert float(np.max(np.abs(logits_20 - logits_30))) < 1e-4

    # Verify v0.0.24 input-conditioned collaborator switching from Reading Node 4:
    # Math word problem -> Math Node 15 (Div/GSM8K) vs Analytical/Science -> Node 8 (Eng-<think>)
    sw_math = trace_reading_node_collaborator_switch_v2(
        p_30,
        c_30,
        tok_loaded,
        "A baker makes 4 trays of 6 muffins and sells 9. How many are left?",
    )
    sw_sci = trace_reading_node_collaborator_switch_v2(
        p_30,
        c_30,
        tok_loaded,
        "Why is the sky blue and how do plants use light for photosynthesis?",
    )
    assert sw_math["selected_collaborator"] == 15
    assert sw_math["a_4_to_15"] > sw_math["a_4_to_8"]
    assert sw_sci["selected_collaborator"] == 8
    assert sw_sci["a_4_to_8"] > sw_sci["a_4_to_15"]
    code_q = (
        "<|user|>\nEvaluate the boolean expression: "
        "(True and not False) xor False\n<|assistant|>\n"
    )
    cog_c = classify_cognitive_mode_v2(p_30, c_30, tok_loaded, code_q)
    assert cog_c["domain"] == "code"
    assert cog_c["active_subnodes"] == (24, 25)
    g_c = inspect_directed_graph_edges_v2(p_30, c_30, tok_loaded, code_q)
    assert g_c["n_nodes"] == 30
    assert g_c["domain"] == "code"
    th_bool, ans_bool = _verify_code_and_logic_v2(code_q, "", "")
    assert "True" in th_bool and "True" in ans_bool
    comp_q = (
        "<|user|>\nWhat does `[x * x for x in range(4)]` evaluate to in Python?\n<|assistant|>\n"
    )
    cog_comp = classify_cognitive_mode_v2(p_30, c_30, tok_loaded, comp_q)
    assert cog_comp["domain"] == "code"
    assert cog_comp["active_subnodes"] == (22, 23)
    cog_sci = classify_cognitive_mode_v2(
        p_30,
        c_30,
        tok_loaded,
        (
            "<|user|>\nWhy is the sky blue and how do plants use light "
            "for photosynthesis?\n<|assistant|>\n"
        ),
    )
    assert cog_sci["domain"] in ("english", "math")
    th_comp, ans_comp = _verify_code_and_logic_v2(comp_q, "", "")
    assert "[0, 1, 4, 9]" in ans_comp

    # Verify v0.0.39 Stage-3 Dot-2 Continual Fine-Tuning & 30-Node GGUF v3 Export
    from mlm.gguf_export import export_gguf_v2, verify_gguf_with_rust
    from mlm.train_v2 import run_dot2_continual_stage_v2

    p_30_ft, c_30_ft, stage3_rep = run_dot2_continual_stage_v2(
        p_30, c_30, tok_loaded, steps_per_bucket=2, learning_rate=2.0e-3
    )
    assert stage3_rep["n_experts"] == 30
    assert stage3_rep["loss_reduction"] > 0.0
    assert stage3_rep["max_drift_nodes_0_19"] == 0.0
    assert stage3_rep["max_drift_trunk"] == 0.0
    assert stage3_rep["dot2_update_l2_norm"] > 0.0

    gguf_30_path = tmp_path / "spidernet_30n_q2k.gguf"
    gguf_30_info = export_gguf_v2(gguf_30_path, p_30_ft, c_30_ft, tok_loaded, bits=2)
    assert gguf_30_info["spidernet_experts"] == 30
    assert gguf_30_info["graph_edge_tensors"] > 0
    gguf_30_rust = verify_gguf_with_rust(gguf_30_path)
    assert gguf_30_rust["valid"] is True
    assert gguf_30_rust["tensor_count"] == gguf_30_info["tensor_count"]
    if "expert_count" in gguf_30_rust:
        assert gguf_30_rust["expert_count"] == 30
        assert gguf_30_rust["graph_edge_rank"] == c_30_ft.graph_edge_rank
        assert gguf_30_rust["tensors_aligned"] is True

    cluster = MLMNodeCluster.from_checkpoints(
        [
            ("node_0_english_science", "general", out_dir / "best.mlm2"),
            ("node_1_mathematics", "math", out_grown / "latest.mlm2"),
        ]
    )
    route_res = cluster.route_prompt("Hi")
    assert route_res["selected_node"] in ("node_0_english_science", "node_1_mathematics")
    cluster_reply = cluster.generate("Hi", max_new_tokens=8)
    assert len(cluster_reply["reply"]) > 0

    from mlm.quantize_v2 import (
        dequantize_kv_cache_turboquant,
        quantize_kv_cache_turboquant,
        save_quantized_checkpoint_v2,
    )

    dummy_k = np.random.default_rng(7).normal(0.0, 1.0, size=(1, 16, 2, 32)).astype(np.float32)
    dummy_v = np.random.default_rng(8).normal(0.0, 1.0, size=(1, 16, 2, 32)).astype(np.float32)
    for kv_b in (4, 2, 1):
        packed_kv = quantize_kv_cache_turboquant(dummy_k, dummy_v, bits=kv_b)
        assert packed_kv["bits"] == kv_b
        assert packed_kv["packed_bytes"] < packed_kv["raw_fp32_bytes"]
        k_hat, v_hat = dequantize_kv_cache_turboquant(packed_kv)
        assert k_hat.shape == dummy_k.shape
        assert v_hat.shape == dummy_v.shape

    for q_bits in (16, 8, 4, 2, 1):
        q_path = tmp_path / f"quant_{q_bits}b.mlm2"
        q_info = save_quantized_checkpoint_v2(
            q_path, p_14, c_14, tok_loaded, bits=q_bits, step=10
        )
        assert q_info["bits"] == q_bits
        assert q_info["kv_cache_bits"] in (1, 2, 4)
        assert q_path.exists()
        p_deq, c_deq, _, m_deq = load_checkpoint_v2(q_path)
        assert m_deq["quantization"]["bits"] == q_bits
        assert c_deq.kv_cache_bits in (1, 2, 4)
        logits_deq = forward_v2(p_deq, jnp.array([[1, 2, 3, 4]], dtype=jnp.int32), c_deq)
        assert logits_deq.shape == (1, 4, c_deq.vocab_size)


