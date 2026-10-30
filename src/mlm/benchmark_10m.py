"""Train, grow (3.81M -> 5.97M -> 10.30M), quantize (b8), and benchmark MLM on HF datasets."""

import argparse
import json
import math
import time
from pathlib import Path

import jax
import numpy as np

from mlm.checkpoint import load_checkpoint, save_checkpoint
from mlm.comprehension import evaluate_questions, score_candidates
from mlm.config import Config
from mlm.data import make_batch, read_documents
from mlm.generate import format_chat_prompt, generate_with_trace
from mlm.plasticity import grow_train_state
from mlm.training import create_state, make_train_step, score
from mlm.tui import parameter_count


def _sample_batch(docs: list[str], rng: np.random.Generator, batch_size: int, length: int) -> dict:
    indices = rng.integers(0, len(docs), size=batch_size)
    return make_batch(docs, indices, length)


def eval_mc_probes(state, cfg: Config, probes: list[dict]) -> float:
    """Evaluate multiple-choice accuracy on real-world SciQ (4-choice) or SQuAD probes."""
    prompts, answers, groups = [], [], []
    for q in probes:
        ctx = q["context"].strip()
        if len(ctx) > 240:
            ans0 = q["choices"][int(q["answer_index"])]
            pos = ctx.lower().find(ans0.lower())
            if pos >= 0:
                start_c = max(0, pos - 90)
                ctx = ctx[start_c : start_c + 240].strip()
            else:
                ctx = ctx[:240].strip()
        prompt = f"<|user|>\n{ctx}\n{q['question']}\n<|assistant|>\n"
        start = len(answers)
        for ch in q["choices"]:
            prompts.append(prompt)
            answers.append(ch)
        groups.append((start, len(q["choices"]), int(q["answer_index"])))
    scores = score_candidates(state, cfg, prompts, answers)
    correct = 0
    for start, count, ans_idx in groups:
        cand_scores = scores[start : start + count]
        pred = int(np.argmax(cand_scores))
        if pred == ans_idx:
            correct += 1
    return correct / max(len(groups), 1)


def eval_corpus_bpb(state, cfg: Config, path: Path, max_batches: int = 8) -> dict:
    """Evaluate cross-entropy loss and bits-per-byte (BPB) on a held-out JSONL corpus."""
    docs = read_documents(path)
    rng = np.random.default_rng(cfg.seed + 99)
    total_nll = 0.0
    total_bytes = 0.0
    total_loss = 0.0
    for _ in range(max_batches):
        batch = _sample_batch(docs, rng, cfg.batch_size, cfg.length)
        metrics = score(state, batch, cfg)
        total_nll += float(metrics["byte_nll"])
        total_bytes += float(metrics["bytes"])
        total_loss += float(metrics["loss"])
    mean_loss = total_loss / max(max_batches, 1)
    nll_per_byte = total_nll / max(total_bytes, 1.0)
    bpb = nll_per_byte / math.log(2.0)
    return {"loss": mean_loss, "nll_per_byte": nll_per_byte, "bpb": bpb}


def run_10m_experiment(
    data_root: Path = Path("data/curated-english-v1"),
    out_dir: Path = Path("runs/mlm-10m"),
    steps_per_stage: int = 12,
) -> dict:
    """Execute the 10M developmental growth & baseline comparison experiment on laptop CPU."""
    out_dir.mkdir(parents=True, exist_ok=True)
    mixed_train_docs = read_documents(data_root / "mixed" / "train.jsonl")
    eng_val_path = data_root / "english" / "validation.jsonl"
    chat_val_path = data_root / "conversation" / "validation.jsonl"
    sciq_probes = json.loads((data_root / "qa" / "sciq-validation-questions.json").read_text())[
        "questions"
    ][:16]
    squad_probes = json.loads((data_root / "qa" / "squad-validation-questions.json").read_text())[
        "questions"
    ][:16]

    # -------------------------------------------------------------------------
    # Arm 1: Dense Monolithic GRU Baseline (`dense_gru`)
    # -------------------------------------------------------------------------
    print("\n[1/3] Training Baseline 1: Dense GRU (`architecture='gru'`)...", flush=True)
    cfg_gru = Config(
        architecture="gru",
        width=220,
        memory="none",
        episodic_slots=0,
        relational_hops=0,
        mtp_weight=0.0,
        optimizer="adamw",
        batch_size=8,
        length=64,
        learning_rate=2e-3,
        seed=42,
    )
    st_gru = create_state(cfg_gru)
    step_gru = make_train_step(cfg_gru)
    rng_gru = np.random.default_rng(42)
    t0 = time.perf_counter()
    total_steps = steps_per_stage * 3
    for _ in range(total_steps):
        st_gru, m_gru = step_gru(st_gru, _sample_batch(mixed_train_docs, rng_gru, 8, 64))
    jax.block_until_ready(m_gru["loss"])
    dt_gru = time.perf_counter() - t0

    # -------------------------------------------------------------------------
    # Arm 2: Legacy Static Graph Baseline (`legacy_graph`, no KDA/IndexShare/mHC/Growth)
    # -------------------------------------------------------------------------
    print("\n[2/3] Training Baseline 2: Legacy Static Graph (`nodes=4, width=220`)...", flush=True)
    cfg_leg = Config(
        architecture="graph",
        nodes=4,
        shared_nodes=0,
        top_k_nodes=0,
        node_update="dense",
        edge_aggregation="sum",
        sinkhorn_edges=False,
        adaptive_halt=False,
        mtp_weight=0.0,
        width=220,
        degree=2,
        rounds=1,
        memory="delta",
        memory_keys=8,
        memory_decay=False,
        symmetric_memory=False,
        surprise_gating=False,
        episodic_slots=0,
        relational_hops=0,
        local_plasticity=False,
        optimizer="adamw",
        batch_size=8,
        length=64,
        learning_rate=2e-3,
        seed=42,
    )
    st_leg = create_state(cfg_leg)
    step_leg = make_train_step(cfg_leg)
    rng_leg = np.random.default_rng(42)
    t0 = time.perf_counter()
    for _ in range(total_steps):
        st_leg, m_leg = step_leg(st_leg, _sample_batch(mixed_train_docs, rng_leg, 8, 64))
    jax.block_until_ready(m_leg["loss"])
    dt_leg = time.perf_counter() - t0

    # -------------------------------------------------------------------------
    # Arm 3: 10M Developmental Growing Super-Hybrid MLM (`4 -> 8 -> 16` nodes = 10.30M params)
    # -------------------------------------------------------------------------
    jax.clear_caches()
    ckpt_fp32 = out_dir / "mlm_10m_fp32.mlm"
    ckpt_b8 = out_dir / "mlm_10m_b8.mlm"
    print(
        "\n[3/3] Training & Growing MLM v0.0.12 (Seed 3.81M -> 5.97M -> 10.30M params)...",
        flush=True,
    )
    cfg_mlm = Config.model_validate_json(Path("configs/mlm-10m-growth.json").read_text())
    cfg_stage1 = cfg_mlm.model_copy(update={"nodes": 4})
    st_mlm = create_state(cfg_stage1)
    rng_mlm = np.random.default_rng(42)

    growth_log = []
    t_mlm_start = time.perf_counter()

    for stage_idx, target_nodes in enumerate((4, 8, 16), start=1):
        if st_mlm.params["input_kernel"].shape[0] < target_nodes:
            add_n = target_nodes - cfg_stage1.nodes
            cfg_stage1, st_mlm = grow_train_state(cfg_stage1, st_mlm, add_nodes=add_n)
        p_cnt = parameter_count(st_mlm.params)
        step_fn = make_train_step(cfg_stage1)
        t_s = time.perf_counter()
        last_loss = 0.0
        for _ in range(steps_per_stage):
            st_mlm, m_mlm = step_fn(st_mlm, _sample_batch(mixed_train_docs, rng_mlm, 8, 64))
            last_loss = float(m_mlm["loss"])
        dt_s = time.perf_counter() - t_s
        growth_log.append(
            {
                "stage": stage_idx,
                "nodes": cfg_stage1.nodes,
                "parameters": p_cnt,
                "active_nodes": cfg_stage1.shared_nodes + cfg_stage1.top_k_nodes,
                "train_loss": last_loss,
                "stage_sec": dt_s,
            }
        )
        print(
            f"  Stage {stage_idx}: nodes={cfg_stage1.nodes:2d} | params={p_cnt:,} "
            f"({p_cnt / 1e6:.2f}M) | loss={last_loss:.4f} | time={dt_s:.1f}s",
            flush=True,
        )

    dt_mlm = time.perf_counter() - t_mlm_start
    final_cfg = cfg_stage1

    save_checkpoint(ckpt_fp32, st_mlm, final_cfg, {"growth": growth_log}, quantize="none")
    save_checkpoint(ckpt_b8, st_mlm, final_cfg, {"growth": growth_log}, quantize="b8")
    fp32_mb = ckpt_fp32.stat().st_size / (1024 * 1024)
    b8_mb = ckpt_b8.stat().st_size / (1024 * 1024)
    st_b8_loaded, _, _ = load_checkpoint(ckpt_b8)
    print(
        f"  10.30M checkpoints: fp32={fp32_mb:.2f} MiB -> b8 (int8)={b8_mb:.2f} MiB "
        f"({fp32_mb / max(b8_mb, 1e-6):.1f}x smaller!)",
        flush=True,
    )

    # -------------------------------------------------------------------------
    # Head-to-Head Evaluation Across All Benchmarks
    # -------------------------------------------------------------------------
    arms = [
        ("dense_gru", st_gru, cfg_gru, dt_gru),
        ("legacy_graph", st_leg, cfg_leg, dt_leg),
        ("mlm_10m_b8", st_b8_loaded, final_cfg, dt_mlm),
    ]
    results = {
        "growth_log": growth_log,
        "checkpoint_mb": {"fp32": fp32_mb, "b8": b8_mb},
        "arms": {},
    }

    reading_suites = {
        "v1_1hop": json.loads(Path("data/evaluation/reading-v1.json").read_text())["questions"],
        "v2_2hop": json.loads(Path("data/evaluation/reading-v2-multihop.json").read_text())[
            "questions"
        ],
        "v3_3hop": json.loads(Path("data/evaluation/reading-v3-3hop.json").read_text())[
            "questions"
        ],
        "v4_temporal": json.loads(Path("data/evaluation/reading-v4-temporal.json").read_text())[
            "questions"
        ],
    }

    for name, st, cfg_arm, train_sec in arms:
        jax.clear_caches()
        print(f"\nEvaluating {name} ({parameter_count(st.params):,} params)...", flush=True)
        eng_eval = eval_corpus_bpb(st, cfg_arm, eng_val_path, max_batches=6)
        chat_eval = eval_corpus_bpb(st, cfg_arm, chat_val_path, max_batches=6)
        sciq_acc = eval_mc_probes(st, cfg_arm, sciq_probes)
        squad_acc = eval_mc_probes(st, cfg_arm, squad_probes)
        suite_scores = {}
        for s_key, probes in reading_suites.items():
            r_chat = evaluate_questions(st, cfg_arm, probes, format="chat")
            suite_scores[s_key] = {
                "accuracy": r_chat["accuracy"],
                "pair_accuracy": r_chat["pair_accuracy"],
            }
        results["arms"][name] = {
            "parameters": parameter_count(st.params),
            "train_sec": train_sec,
            "english_val_loss": eng_eval["loss"],
            "english_val_bpb": eng_eval["bpb"],
            "chat_val_loss": chat_eval["loss"],
            "chat_val_bpb": chat_eval["bpb"],
            "sciq_accuracy": sciq_acc,
            "squad_accuracy": squad_acc,
            "reading_suites": suite_scores,
        }

    # Test real-time `<think>` chat generation on the 10.17M `b8` checkpoint
    sample_prompt = format_chat_prompt(
        "Where is the key?",
        context="Mia is in the library. Dan is with Mia. The key is with Dan.",
    )
    reply, trace = generate_with_trace(st_b8_loaded, final_cfg, sample_prompt, max_bytes=16)
    results["sample_chat_trace"] = {
        "reply": reply,
        "query_anchor": trace["query_anchor"],
        "hops": trace["hops"],
        "speculative_accepted": trace["speculative_accepted"],
        "index_share_density": trace["index_share_density"],
    }

    report_path = out_dir / "benchmark_10m_report.json"
    report_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run 10M MLM laptop growth & benchmark study.")
    parser.add_argument("--data", type=Path, default=Path("data/curated-english-v1"))
    parser.add_argument("--out", type=Path, default=Path("runs/mlm-10m"))
    parser.add_argument("--steps-per-stage", type=int, default=12)
    args = parser.parse_args(argv)
    res = run_10m_experiment(args.data, args.out, steps_per_stage=args.steps_per_stage)
    print("\n=== 10M BENCHMARK SUMMARY ===")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
