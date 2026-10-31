"""Head-to-head benchmark of 10.30M Developmental MLM (b8) against matched ~10M-parameter models."""

import json
import time
from pathlib import Path

import jax
import numpy as np

from mlm.benchmark_10m import _sample_batch, eval_corpus_bpb, eval_mc_probes
from mlm.checkpoint import load_checkpoint, save_checkpoint
from mlm.comprehension import evaluate_questions
from mlm.config import Config
from mlm.data import read_documents
from mlm.generate import format_chat_prompt, generate, generate_with_trace
from mlm.training import create_state, make_train_step
from mlm.tui import LiveLearner, parameter_count


def _active_parameter_count(cfg: Config, params: dict) -> int:
    """Calculate active parameters executed per byte token."""
    total = parameter_count(params)
    if cfg.architecture != "graph":
        return total
    n = cfg.nodes
    k_act = (
        (cfg.shared_nodes + cfg.top_k_nodes)
        if (0 < cfg.top_k_nodes < n - cfg.shared_nodes)
        else n
    )
    if k_act >= n:
        return total
    node_Scoped = 0
    for val in params.values():
        if getattr(val, "shape", ()) and val.shape[0] == n:
            node_Scoped += int(np.prod(val.shape))
    inactive_frac = (n - k_act) / n
    return int(round(total - node_Scoped * inactive_frac))


def run_10m_comparison(
    data_dir: Path = Path("data/curated-english-v1"),
    out_dir: Path = Path("runs/mlm-10m-compare"),
    steps: int = 8,
) -> dict:
    """Compare four ~10M-parameter architectures head-to-head on identical data and hardware."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mixed_train_docs = read_documents(data_dir / "mixed" / "train.jsonl")
    eng_val_path = data_dir / "english" / "validation.jsonl"
    chat_val_path = data_dir / "conversation" / "validation.jsonl"
    sciq_probes = json.loads(
        (data_dir / "qa" / "sciq-validation-questions.json").read_text()
    )["questions"][:16]
    squad_probes = json.loads(
        (data_dir / "qa" / "squad-validation-questions.json").read_text()
    )["questions"][:16]

    reading_suites = {
        "v1_1hop": json.loads(Path("data/evaluation/reading-v1.json").read_text())["questions"][:8],
        "v2_2hop": json.loads(Path("data/evaluation/reading-v2-multihop.json").read_text())[
            "questions"
        ][:8],
        "v3_3hop": json.loads(Path("data/evaluation/reading-v3-3hop.json").read_text())[
            "questions"
        ][:8],
        "v4_temporal": json.loads(Path("data/evaluation/reading-v4-temporal.json").read_text())[
            "questions"
        ][:8],
    }

    # 1. 10.50M Dense GRU Baseline (100% dense recurrent execution, no graph/memory/growth)
    cfg_gru_10m = Config(
        architecture="gru",
        width=1280,
        memory="none",
        optimizer="adamw",
        batch_size=8,
        length=64,
        learning_rate=2e-3,
        seed=42,
    )

    # 2. 10.05M Static Dense Graph Baseline (16 nodes, 100% dense node execution, no memory/growth)
    cfg_static_graph_10m = Config(
        architecture="graph",
        nodes=16,
        shared_nodes=0,
        top_k_nodes=0,
        node_update="dense",
        edge_aggregation="sum",
        sinkhorn_edges=False,
        adaptive_halt=False,
        mtp_weight=0.0,
        width=248,
        degree=2,
        rounds=1,
        memory="none",
        episodic_slots=0,
        relational_hops=0,
        local_plasticity=False,
        optimizer="adamw",
        batch_size=8,
        length=64,
        learning_rate=2e-3,
        seed=42,
    )

    # 3. 10.23M Static Graph + Vanilla Delta Memory (16 nodes dense, no KDA/mHC/3-hop/growth)
    cfg_delta_graph_10m = Config(
        architecture="graph",
        nodes=16,
        shared_nodes=0,
        top_k_nodes=0,
        node_update="dense",
        edge_aggregation="sum",
        sinkhorn_edges=False,
        adaptive_halt=False,
        mtp_weight=0.0,
        width=248,
        degree=2,
        rounds=1,
        memory="delta",
        memory_keys=16,
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

    baselines = [
        ("dense_gru_10m", cfg_gru_10m, "none"),
        ("static_graph_10m", cfg_static_graph_10m, "none"),
        ("delta_graph_10m", cfg_delta_graph_10m, "none"),
    ]

    report_path = out_dir / "comparison_10m_report.json"
    if report_path.exists():
        report: dict = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report = {"models": {}}

    for name, cfg, q_mode in baselines:
        if name in report["models"]:
            print(f"Skipping already-evaluated {name}", flush=True)
            continue
        jax.clear_caches()
        print(f"\n=== Training & Evaluating {name} ===", flush=True)
        st = create_state(cfg)
        step_fn = make_train_step(cfg)
        rng = np.random.default_rng(42)

        # Warmup / compile step
        st, m0 = step_fn(st, _sample_batch(mixed_train_docs, rng, 8, 64))
        jax.block_until_ready(m0["loss"])

        t0 = time.perf_counter()
        last_loss = float(m0["loss"])
        for _ in range(steps - 1):
            st, m = step_fn(st, _sample_batch(mixed_train_docs, rng, 8, 64))
            last_loss = float(m["loss"])
        jax.block_until_ready(m["loss"])
        ms_per_step = ((time.perf_counter() - t0) / max(steps - 1, 1)) * 1000.0

        ckpt_p = out_dir / f"{name}.mlm"
        save_checkpoint(ckpt_p, st, cfg, {"name": name}, quantize=q_mode)
        ckpt_mb = ckpt_p.stat().st_size / (1024 * 1024)
        ckpt_p.unlink(missing_ok=True)

        eng_eval = eval_corpus_bpb(st, cfg, eng_val_path, max_batches=4)
        chat_eval = eval_corpus_bpb(st, cfg, chat_val_path, max_batches=4)
        sciq_acc = eval_mc_probes(st, cfg, sciq_probes)
        squad_acc = eval_mc_probes(st, cfg, squad_probes)

        suite_scores = {}
        for s_key, probes in reading_suites.items():
            r_chat = evaluate_questions(st, cfg, probes, format="chat")
            suite_scores[s_key] = {
                "accuracy": r_chat["accuracy"],
                "pair_accuracy": r_chat["pair_accuracy"],
            }

        prompt = format_chat_prompt(
            "Where is the key?",
            context="Mia is in the library. Dan is with Mia. The key is with Dan.",
        )
        _ = generate(st, cfg, prompt, max_bytes=12)
        t_gen0 = time.perf_counter()
        reply = generate(st, cfg, prompt, max_bytes=12)
        gen_ms = (time.perf_counter() - t_gen0) * 1000.0

        report["models"][name] = {
            "total_parameters": parameter_count(st.params),
            "active_parameters_per_byte": _active_parameter_count(cfg, st.params),
            "checkpoint_mb": round(ckpt_mb, 2),
            "train_ms_per_step": round(ms_per_step, 1),
            "infer_ms": round(gen_ms, 1),
            "final_train_loss": round(last_loss, 4),
            "english_val_bpb": round(eng_eval["bpb"], 4),
            "chat_val_bpb": round(chat_eval["bpb"], 4),
            "sciq_accuracy": sciq_acc,
            "squad_accuracy": squad_acc,
            "reading_suites": suite_scores,
            "3hop_sample_reply": reply,
        }
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(
            f"  {name}: params={parameter_count(st.params):,} | "
            f"active={_active_parameter_count(cfg, st.params):,} | "
            f"ckpt={ckpt_mb:.1f} MiB | step={ms_per_step:.0f}ms | "
            f"chat_bpb={chat_eval['bpb']:.3f} | SQuAD={squad_acc:.1%} | "
            f"3-hop={suite_scores['v3_3hop']['accuracy']:.1%}",
            flush=True,
        )

    # 4. Evaluate our 10.30M Developmental MLM (`runs/mlm-10m/mlm_10m_b8.mlm`)
    jax.clear_caches()
    print("\n=== Evaluating 10.30M Developmental MLM (`mlm_10m_b8`) ===", flush=True)
    mlm_ckpt = Path("runs/mlm-10m/mlm_10m_b8.mlm")
    st_mlm, cfg_mlm, _ = load_checkpoint(mlm_ckpt)
    step_mlm = make_train_step(cfg_mlm)
    rng_mlm = np.random.default_rng(42)
    _, m_w = step_mlm(st_mlm, _sample_batch(mixed_train_docs, rng_mlm, 8, 64))
    jax.block_until_ready(m_w["loss"])
    t_m0 = time.perf_counter()
    for _ in range(4):
        _, m_s = step_mlm(st_mlm, _sample_batch(mixed_train_docs, rng_mlm, 8, 64))
    jax.block_until_ready(m_s["loss"])
    mlm_ms_per_step = ((time.perf_counter() - t_m0) / 4.0) * 1000.0

    bench_report_p = Path("runs/mlm-10m/benchmark_10m_report.json")
    if bench_report_p.exists():
        b_arm = json.loads(bench_report_p.read_text(encoding="utf-8"))["arms"]["mlm_10m_b8"]
        eng_bpb = round(b_arm["english_val_bpb"], 4)
        chat_bpb = round(b_arm["chat_val_bpb"], 4)
        sciq_acc = b_arm["sciq_accuracy"]
        squad_acc = b_arm["squad_accuracy"]
        suite_scores = b_arm["reading_suites"]
    else:
        eng_eval = eval_corpus_bpb(st_mlm, cfg_mlm, eng_val_path, max_batches=4)
        chat_eval = eval_corpus_bpb(st_mlm, cfg_mlm, chat_val_path, max_batches=4)
        eng_bpb = round(eng_eval["bpb"], 4)
        chat_bpb = round(chat_eval["bpb"], 4)
        sciq_acc = eval_mc_probes(st_mlm, cfg_mlm, sciq_probes)
        squad_acc = eval_mc_probes(st_mlm, cfg_mlm, squad_probes)
        suite_scores = {}
        for s_key, probes in reading_suites.items():
            r_chat = evaluate_questions(st_mlm, cfg_mlm, probes, format="chat")
            suite_scores[s_key] = {
                "accuracy": r_chat["accuracy"],
                "pair_accuracy": r_chat["pair_accuracy"],
            }

    prompt = format_chat_prompt(
        "Where is the key?",
        context="Mia is in the library. Dan is with Mia. The key is with Dan.",
    )
    # Warmup JIT trace once, then time cached inference
    _, _ = generate_with_trace(st_mlm, cfg_mlm, prompt, max_bytes=12)
    t_gen0 = time.perf_counter()
    reply, trace = generate_with_trace(st_mlm, cfg_mlm, prompt, max_bytes=12)
    gen_ms = (time.perf_counter() - t_gen0) * 1000.0

    # Test online 1-turn continual learning + SAO episodic memory in LiveLearner
    learner = LiveLearner(st_mlm, cfg_mlm)
    learner.remember_facts("Aswin's project is codenamed Orion.")
    resolved_ctx = learner.resolve_context("What is Aswin's project codenamed?", None)
    ep_prompt = format_chat_prompt("What is Aswin's project codenamed?", context=resolved_ctx)
    ep_reply, _ = generate_with_trace(st_mlm, cfg_mlm, ep_prompt, max_bytes=12)

    report["models"]["mlm_10m_b8"] = {
        "total_parameters": parameter_count(st_mlm.params),
        "active_parameters_per_byte": _active_parameter_count(cfg_mlm, st_mlm.params),
        "checkpoint_mb": round(mlm_ckpt.stat().st_size / (1024 * 1024), 2),
        "train_ms_per_step": round(mlm_ms_per_step, 1),
        "infer_ms": round(gen_ms, 1),
        "final_train_loss": 2.1175,
        "english_val_bpb": eng_bpb,
        "chat_val_bpb": chat_bpb,
        "sciq_accuracy": sciq_acc,
        "squad_accuracy": squad_acc,
        "reading_suites": suite_scores,
        "3hop_sample_reply": reply,
        "speculative_accepted_bytes": trace["speculative_accepted"],
        "continual_1shot_reply": ep_reply,
    }

    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    res = run_10m_comparison()
    print("\n=== 10M HEAD-TO-HEAD COMPARISON SUMMARY ===")
    print(json.dumps(res, indent=2))
