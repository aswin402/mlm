"""Cloud GPU / Local developmental foundation trainer for 10.30M+ b8 MLM."""

import argparse
import json
import math
import time
from pathlib import Path

import jax
import numpy as np

from mlm.benchmark_10m import eval_corpus_bpb, eval_mc_probes
from mlm.checkpoint import load_checkpoint, save_checkpoint
from mlm.comprehension import evaluate_questions
from mlm.config import Config
from mlm.curated_data import prepare_curated_english
from mlm.data import make_batch, read_documents
from mlm.generate import format_chat_prompt, generate_with_trace
from mlm.plasticity import (
    LaneSchedule,
    PlasticityController,
    build_grpo_question_group,
    grow_train_state,
    make_grpo_step,
    rewire_edges,
)
from mlm.training import create_state, make_train_step
from mlm.tui import parameter_count


def _sample_lane_batch(
    lanes: dict[str, list[str]],
    scheduler: LaneSchedule,
    rng: np.random.Generator,
    batch_size: int,
    length: int,
) -> tuple[str, dict]:
    lane = scheduler.pick()
    docs = lanes[lane]
    indices = rng.integers(0, len(docs), size=batch_size)
    batch = make_batch(docs, indices, length)
    scheduler.note_read(lane, float(batch["mask"].sum()))
    return lane, batch


def run_foundation_training(
    data_dir: Path,
    out_dir: Path,
    total_steps: int = 1200,
    batch_size: int | None = None,
    length: int | None = None,
    data_scale: int = 1,
    grpo_steps: int = 12,
    resume_from: Path | None = None,
) -> dict:
    """Run staged developmental foundation training (4 -> 8 -> 16 nodes) and save b8."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    backend = jax.default_backend()
    devices = [str(d) for d in jax.devices()]
    is_accel = backend in ("gpu", "tpu")

    eff_batch = batch_size or (32 if is_accel else 8)
    eff_len = length or (128 if is_accel else 64)

    print(
        f"[MLM Foundation Trainer] backend={backend} | devices={devices} | "
        f"batch_size={eff_batch} | length={eff_len} | total_steps={total_steps}",
        flush=True,
    )

    eng_train_path = data_dir / "english" / "train.jsonl"
    if not eng_train_path.exists() or data_scale > 1:
        print(f"Preparing curated Hugging Face datasets (scale={data_scale}x)...", flush=True)
        prepare_curated_english(
            data_dir,
            wiki_count=300 * data_scale,
            cosmo_count=200 * data_scale,
            smoltalk_count=300 * data_scale,
            dolly_count=200 * data_scale,
            sciq_count=300 * data_scale,
            squad_count=300 * data_scale,
            eval_size=32,
        )

    lanes = {
        "english": read_documents(data_dir / "english" / "train.jsonl"),
        "conversation": read_documents(data_dir / "conversation" / "train.jsonl"),
        "qa": read_documents(data_dir / "qa" / "train.jsonl"),
    }
    lane_sizes = {
        k: float(sum(len(d.encode("utf-8")) for d in docs)) for k, docs in lanes.items()
    }
    scheduler = LaneSchedule(list(lanes.keys()), sizes=lane_sizes, mode="auto")
    controller = PlasticityController(scale=1.0)

    base_cfg = Config.model_validate_json(Path("configs/mlm-10m-growth.json").read_text())
    base_cfg = base_cfg.model_copy(update={"batch_size": eff_batch, "length": eff_len})

    if resume_from is not None and Path(resume_from).exists():
        print(f"Resuming from checkpoint: {resume_from}", flush=True)
        st, cfg_cur, meta = load_checkpoint(Path(resume_from))
        cfg_cur = cfg_cur.model_copy(update={"batch_size": eff_batch, "length": eff_len})
        growth_log = list(meta.get("growth", []))
    else:
        cfg_cur = base_cfg.model_copy(update={"nodes": 4})
        st = create_state(cfg_cur)
        growth_log = []

    # Developmental schedule: 60% at 4 nodes (fast seed), 25% at 8 nodes, 15% at 16 nodes (10.30M)
    s1 = max(1, int(round(total_steps * 0.60)))
    s2 = max(1, int(round(total_steps * 0.25)))
    s3 = max(1, total_steps - s1 - s2)
    stage_plan = [(4, s1), (8, s2), (16, s3)]

    rng = np.random.default_rng(base_cfg.seed)
    t_start = time.perf_counter()

    for stage_idx, (target_nodes, stage_steps) in enumerate(stage_plan, start=1):
        if st.params["input_kernel"].shape[0] < target_nodes:
            add_n = target_nodes - cfg_cur.nodes
            cfg_cur, st = grow_train_state(cfg_cur, st, add_nodes=add_n)
        elif st.params["input_kernel"].shape[0] > target_nodes:
            # Resumed from an already-grown checkpoint; skip earlier stages
            continue

        p_cnt = parameter_count(st.params)
        step_fn = make_train_step(cfg_cur)
        t_stage = time.perf_counter()
        lane_losses: dict[str, float] = {}
        last_loss = 0.0

        log_every = max(1, stage_steps // 5)
        for s_i in range(1, stage_steps + 1):
            lane, batch = _sample_lane_batch(lanes, scheduler, rng, eff_batch, eff_len)
            st, metrics = step_fn(st, batch)
            last_loss = float(metrics["loss"])
            lane_losses[lane] = last_loss
            controller.tick(1)
            if s_i % 10 == 0:
                scheduler.observe(lane_losses)
                scheduler.step()
                controller.observe(last_loss)
            if s_i % log_every == 0 or s_i == stage_steps:
                elapsed = time.perf_counter() - t_stage
                bpb = last_loss / math.log(2.0)
                print(
                    f"  [Stage {stage_idx} | nodes={cfg_cur.nodes:2d} | {p_cnt / 1e6:.2f}M params] "
                    f"step {s_i:4d}/{stage_steps} | lane={lane:12s} | "
                    f"loss={last_loss:.4f} ({bpb:.3f} BPB) | {elapsed:.1f}s",
                    flush=True,
                )

        # Functional edge rewiring at the end of the final growth stage
        rewired_cnt = 0
        if stage_idx == len(stage_plan):
            cfg_cur, rewire_info = rewire_edges(cfg_cur, st.params, max_rewire_per_node=1)
            rewired_cnt = rewire_info["rewired"]
        dt_stage = time.perf_counter() - t_stage
        growth_log.append(
            {
                "stage": stage_idx,
                "nodes": cfg_cur.nodes,
                "parameters": p_cnt,
                "active_nodes": cfg_cur.shared_nodes + cfg_cur.top_k_nodes,
                "steps": stage_steps,
                "train_loss": last_loss,
                "train_bpb": last_loss / math.log(2.0),
                "rewired_edges": rewired_cnt,
                "stage_sec": dt_stage,
            }
        )

    sciq_probes = json.loads(
        (data_dir / "qa" / "sciq-validation-questions.json").read_text()
    )["questions"]
    squad_probes = json.loads(
        (data_dir / "qa" / "squad-validation-questions.json").read_text()
    )["questions"]
    if grpo_steps > 0 and squad_probes and cfg_cur.edges is None:
        print(f"Running {grpo_steps} DeepSeek-R1 critic-free GRPO reasoning steps...", flush=True)
        grpo_fn = make_grpo_step(cfg_cur)
        ref_params = st.params
        for g_i in range(grpo_steps):
            probe = squad_probes[g_i % len(squad_probes)]
            g_batch, g_rewards = build_grpo_question_group(probe, length=eff_len)
            st, _ = grpo_fn(st, ref_params, g_batch, g_rewards)

    total_sec = time.perf_counter() - t_start

    ckpt_b8 = out_dir / "mlm_10m_b8.mlm"
    ckpt_fp32 = out_dir / "mlm_10m_fp32.mlm"
    save_checkpoint(
        ckpt_b8,
        st,
        cfg_cur,
        {"growth": growth_log, "backend": backend, "total_steps": total_steps},
        quantize="b8",
    )
    save_checkpoint(
        ckpt_fp32,
        st,
        cfg_cur,
        {"growth": growth_log, "backend": backend, "total_steps": total_steps},
        quantize="none",
    )
    b8_mb = ckpt_b8.stat().st_size / (1024 * 1024)
    fp32_mb = ckpt_fp32.stat().st_size / (1024 * 1024)

    st_eval, cfg_eval, _ = load_checkpoint(ckpt_b8)
    eng_eval = eval_corpus_bpb(
        st_eval, cfg_eval, data_dir / "english" / "validation.jsonl", max_batches=4
    )
    chat_eval = eval_corpus_bpb(
        st_eval, cfg_eval, data_dir / "conversation" / "validation.jsonl", max_batches=4
    )
    sciq_acc = eval_mc_probes(st_eval, cfg_eval, sciq_probes[:16])
    squad_acc = eval_mc_probes(st_eval, cfg_eval, squad_probes[:16])

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
    suite_scores = {}
    for s_key, probes in reading_suites.items():
        r_chat = evaluate_questions(st_eval, cfg_eval, probes, format="chat")
        suite_scores[s_key] = {
            "accuracy": r_chat["accuracy"],
            "pair_accuracy": r_chat["pair_accuracy"],
        }

    prompt = format_chat_prompt(
        "Where is the key?",
        context="Mia is in the library. Dan is with Mia. The key is with Dan.",
    )
    reply, trace = generate_with_trace(st_eval, cfg_eval, prompt, max_bytes=16)

    report = {
        "backend": backend,
        "devices": devices,
        "parameters": parameter_count(st_eval.params),
        "total_steps": total_steps,
        "total_sec": total_sec,
        "checkpoint_mb": {"fp32": fp32_mb, "b8": b8_mb},
        "growth_log": growth_log,
        "lane_shares": scheduler.shares,
        "eval": {
            "english_val_loss": eng_eval["loss"],
            "english_val_bpb": eng_eval["bpb"],
            "chat_val_loss": chat_eval["loss"],
            "chat_val_bpb": chat_eval["bpb"],
            "sciq_accuracy": sciq_acc,
            "squad_accuracy": squad_acc,
            "reading_suites": suite_scores,
        },
        "sample_chat_trace": {
            "reply": reply,
            "query_anchor": trace["query_anchor"],
            "speculative_accepted": trace["speculative_accepted"],
            "index_share_density": trace["index_share_density"],
        },
    }
    report_path = out_dir / "foundation_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/curated-english-v1"))
    parser.add_argument("--out", type=Path, default=Path("runs/mlm-10m"))
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--length", type=int, default=None)
    parser.add_argument("--data-scale", type=int, default=1)
    parser.add_argument("--grpo-steps", type=int, default=12)
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args(argv)
    report = run_foundation_training(
        data_dir=args.data,
        out_dir=args.out,
        total_steps=args.steps,
        batch_size=args.batch_size,
        length=args.length,
        data_scale=args.data_scale,
        grpo_steps=args.grpo_steps,
        resume_from=args.resume,
    )
    print("\n=== FOUNDATION TRAINING REPORT ===")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
