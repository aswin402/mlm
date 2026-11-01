"""Google Colab T4 GPU trainer for the 10M Staged-Node Growth MLM (4 -> 6 -> 8 Nodes).

Stages:
  1. Stage 1 (`english_learn`, 4 Specialist Nodes):
     Simple English Wikipedia + SciQ + SQuAD continuous English paragraphs (mask=1.0).
  2. Stage 2 (`english_understand`, Grow 4 -> 6 Specialist Nodes via Crossover):
     Grounded Reading Comprehension & Science QA + 25% Stage 1 English Replay
     (Mature Nodes 0..3 protected at 0.15x LR).
  3. Stage 3 (`english_chat`, Grow 6 -> 8 Specialist Nodes = ~10.8M Params via Crossover):
     SmolTalk Everyday Conversations + Dolly-15k + Alpaca + 25% Stage 1 & 2 Replay
     (Mature Nodes 0..5 protected at 0.15x LR).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax

from mlm.model_v2 import ConfigV2, generate_v2, load_checkpoint_v2
from mlm.parquet_stream import build_staged_english_corpora
from mlm.train_v2 import format_chat_prompt_v2, run_staged_english_growth_v2


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/curated-english-v1"))
    parser.add_argument("--out", type=Path, default=Path("runs/mlm-10m-colab"))
    parser.add_argument("--max-rows-per-source", type=int, default=4000)
    parser.add_argument("--s1-steps", type=int, default=1500)
    parser.add_argument("--s2-steps", type=int, default=1500)
    parser.add_argument("--s3-steps", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    args = parser.parse_args(argv)

    print("=" * 76, flush=True)
    print("⚡ MLM 10M Staged-Node Growth Trainer (4 Learn -> 6 Understand -> 8 Chat)", flush=True)
    print(f"   JAX Devices: {[str(d) for d in jax.devices()]}", flush=True)
    print("=" * 76, flush=True)

    t0 = time.perf_counter()
    print("\n[1/3] Streaming & curating 3-stage English corpora via Rust Polars...", flush=True)
    dataset_counts = build_staged_english_corpora(
        args.data_root, max_rows_per_source=args.max_rows_per_source
    )
    print(json.dumps(dataset_counts, indent=2), flush=True)

    # 10.8M parameter target at 8 specialist nodes (starts at 4 nodes in Stage 1)
    base_cfg = ConfigV2(
        arch_version="v2_hybrid",
        vocab_size=4096,
        d_model=192,
        n_layers=8,
        n_heads=6,
        n_kv_heads=2,
        ffn_dim=384,
        conv_kernel=4,
        attn_interval=4,
        use_gated_deltanet=True,
        n_experts=4,
        top_k_experts=2,
        expert_ffn_dim=192,
        tie_embeddings=True,
        max_seq_len=args.seq_len,
        batch_size=args.batch_size,
        learning_rate=2.0e-3,
        weight_decay=0.03,
        warmup_steps=100,
    )

    gguf_path = args.out / "mlm_10m_v2.gguf"
    print(
        f"\n[2/3] Running 3-stage node growth pipeline "
        f"(steps={args.s1_steps}+{args.s2_steps}+{args.s3_steps})...",
        flush=True,
    )
    summary = run_staged_english_growth_v2(
        args.data_root,
        args.out,
        base_cfg,
        stage_steps=(args.s1_steps, args.s2_steps, args.s3_steps),
        eval_every=args.eval_every,
        gguf_out=gguf_path,
    )

    final_ckpt = args.out / "latest.mlm2"
    named_ckpt = args.out / "mlm_10m_v2.mlm2"
    if final_ckpt.exists():
        named_ckpt.write_bytes(final_ckpt.read_bytes())

    print("\n[3/3] Evaluating 8-Node (10M) Checkpoint on Open-Ended English & Chat...", flush=True)
    params, cfg, tok, _ = load_checkpoint_v2(final_ckpt)
    eval_prompts = [
        "Hi",
        "Hi there",
        "Hello! How are you?",
        "What is science?",
        "What is photosynthesis?",
        "Why is the sky blue?",
        "Can you give me some career advice?",
        "Mia is in the library. Dan is with Mia.\nQuestion: Where is Dan?",
    ]
    chat_results = []
    for p in eval_prompts:
        formatted = format_chat_prompt_v2(p)
        reply = generate_v2(params, cfg, tok, formatted, max_new_tokens=48).strip()
        chat_results.append({"prompt": p, "reply": reply})
        print(f"  User: {p}\n  MLM : {reply}\n", flush=True)

    summary["chat_evaluation"] = chat_results
    summary["total_wall_seconds"] = time.perf_counter() - t0
    (args.out / "colab_10m_report.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("=" * 76, flush=True)
    print(f"✅ Completed in {summary['total_wall_seconds']:.1f}s!", flush=True)
    print(f"   SafeTensors + BPE Checkpoint : {named_ckpt}", flush=True)
    print(f"   llama.cpp GGUF v3 Model      : {gguf_path}", flush=True)
    print("=" * 76, flush=True)


if __name__ == "__main__":
    main()
