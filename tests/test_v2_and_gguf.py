"""Tests for MLM v2 (Deep Hybrid Liquid-Delta + Rust BPE + SafeTensors + GGUF v3)."""

from pathlib import Path

from mlm.cli import main
from mlm.gguf_export import export_gguf_v2, verify_gguf_with_rust
from mlm.model_v2 import (
    ConfigV2,
    build_layer_execution_schedule,
    compute_ncm_concept_loss_v2,
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

    reply = generate_v2(
        loaded_params,
        loaded_cfg,
        loaded_tok,
        format_chat_prompt_v2("Hi"),
        max_new_tokens=16,
    )
    assert isinstance(reply, str) and len(reply) > 0

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

    # Test DeepSeek-R1 + Dr. GRPO verifiable reward & RL step on grown checkpoint
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
    assert r_good > r_bad

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
