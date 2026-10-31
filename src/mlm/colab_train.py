"""Cloud GPU / Local developmental foundation trainer for 10.30M+ b8 MLM."""

import argparse
import gc
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


def _preencode_lane(docs: list[str], length: int) -> dict[str, np.ndarray]:
    """Pre-encode all documents in a lane into contiguous NumPy arrays."""
    arr = make_batch(docs, np.arange(len(docs)), length)
    return {
        "inputs": arr["inputs"],
        "targets": arr["targets"],
        "mask": arr["mask"],
        "mask_sums": arr["mask"].sum(axis=1).astype(np.float32),
    }


def _sample_preencoded_batch(
    lane_arrays: dict[str, dict[str, np.ndarray]],
    resets_template: np.ndarray,
    scheduler: LaneSchedule,
    rng: np.random.Generator,
    batch_size: int,
) -> tuple[str, dict[str, np.ndarray]]:
    lane = scheduler.pick()
    la = lane_arrays[lane]
    indices = rng.integers(0, la["inputs"].shape[0], size=batch_size)
    batch = {
        "inputs": la["inputs"][indices],
        "targets": la["targets"][indices],
        "mask": la["mask"][indices],
        "resets": resets_template,
    }
    scheduler.note_read(lane, float(la["mask_sums"][indices].sum()))
    return lane, batch


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


class ChunkedFoundationTrainer:
    """Stateful multi-chunk developmental trainer that preserves XLA JIT across Colab cells."""

    def __init__(
        self,
        data_dir: Path,
        out_dir: Path,
        total_steps: int = 5000,
        batch_size: int | None = None,
        length: int | None = None,
        data_scale: int = 4,
        grpo_steps: int = 12,
        resume_from: Path | None = None,
        config_path: Path = Path("configs/mlm-100m-growth.json"),
        ckpt_name: str = "mlm_100m_b8.mlm",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.total_steps = int(total_steps)
        self.grpo_steps = int(grpo_steps)
        self.ckpt_name = ckpt_name
        self.backend = jax.default_backend()
        self.devices = [str(d) for d in jax.devices()]
        self.is_accel = self.backend in ("gpu", "tpu")

        base_cfg = Config.model_validate_json(Path(config_path).read_text())
        self.is_100m = base_cfg.nodes >= 64
        self.eff_batch = batch_size or (
            (16 if self.is_100m else 32) if self.is_accel else (2 if self.is_100m else 8)
        )
        self.eff_len = length or (
            (96 if self.is_100m else 128) if self.is_accel else (32 if self.is_100m else 64)
        )

        print(
            f"[MLM Foundation Trainer] config={config_path} | backend={self.backend} | "
            f"devices={self.devices} | batch_size={self.eff_batch} | length={self.eff_len} | "
            f"total_steps={self.total_steps}",
            flush=True,
        )

        eng_train_path = self.data_dir / "english" / "train.jsonl"
        if not eng_train_path.exists() or data_scale > 1:
            print(f"Preparing curated Hugging Face datasets (scale={data_scale}x)...", flush=True)
            prepare_curated_english(
                self.data_dir,
                wiki_count=300 * data_scale,
                cosmo_count=200 * data_scale,
                smoltalk_count=300 * data_scale,
                dolly_count=200 * data_scale,
                sciq_count=300 * data_scale,
                squad_count=300 * data_scale,
                eval_size=32,
            )

        lanes = {
            "english": read_documents(self.data_dir / "english" / "train.jsonl"),
            "conversation": read_documents(self.data_dir / "conversation" / "train.jsonl"),
            "qa": read_documents(self.data_dir / "qa" / "train.jsonl"),
        }
        lane_sizes = {
            k: float(sum(len(d.encode("utf-8")) for d in docs)) for k, docs in lanes.items()
        }
        self.lane_arrays = {k: _preencode_lane(docs, self.eff_len) for k, docs in lanes.items()}
        self.resets_template = np.zeros((self.eff_batch, self.eff_len), dtype=bool)
        self.resets_template[:, 0] = True
        self.scheduler = LaneSchedule(list(lanes.keys()), sizes=lane_sizes, mode="auto")
        self.controller = PlasticityController(scale=1.0)

        self.base_cfg = base_cfg.model_copy(
            update={"batch_size": self.eff_batch, "length": self.eff_len}
        )
        n_final = self.base_cfg.nodes
        n_mid = max(4, n_final // 2)
        n_seed = max(4, n_final // 4)

        if resume_from is not None and Path(resume_from).exists():
            print(f"Resuming from checkpoint: {resume_from}", flush=True)
            self.st, self.cfg_cur, meta = load_checkpoint(Path(resume_from))
            self.cfg_cur = self.cfg_cur.model_copy(
                update={"batch_size": self.eff_batch, "length": self.eff_len}
            )
            self.growth_log = list(meta.get("growth", []))
        else:
            self.cfg_cur = self.base_cfg.model_copy(update={"nodes": n_seed})
            self.st = create_state(self.cfg_cur)
            self.growth_log = []

        cur_nodes = int(self.st.params["input_kernel"].shape[0])
        if cur_nodes >= n_final:
            self.stage_plan = [(n_final, max(1, self.total_steps))]
        elif cur_nodes >= n_mid:
            s2 = max(1, int(round(self.total_steps * 0.40)))
            s3 = max(1, self.total_steps - s2)
            self.stage_plan = [(n_mid, s2), (n_final, s3)]
        else:
            s1 = max(1, int(round(self.total_steps * 0.60)))
            s2 = max(1, int(round(self.total_steps * 0.25)))
            s3 = max(1, self.total_steps - s1 - s2)
            self.stage_plan = [(n_seed, s1), (n_mid, s2), (n_final, s3)]

        self.rng = np.random.default_rng(self.base_cfg.seed)
        self.t_start = time.perf_counter()
        self.global_step = 0
        self.stage_idx = 0
        self.stage_step = 0
        self.stage_t0 = time.perf_counter()
        self.step_fn = None
        self.last_loss = 0.0
        self.lane_losses: dict[str, float] = {}

    def _ensure_stage_ready(self) -> tuple[int, int]:
        target_nodes, stage_steps = self.stage_plan[self.stage_idx]
        if self.step_fn is None:
            gc.collect()
            jax.clear_caches()
            if self.st.params["input_kernel"].shape[0] < target_nodes:
                add_n = target_nodes - self.cfg_cur.nodes
                self.cfg_cur, self.st = grow_train_state(self.cfg_cur, self.st, add_nodes=add_n)
                gc.collect()
                jax.clear_caches()
            self.step_fn = make_train_step(self.cfg_cur)
            self.stage_t0 = time.perf_counter()
            self.stage_step = 0
        return target_nodes, stage_steps

    def run_chunk(self, max_steps: int = 500) -> dict:
        """Run up to `max_steps` training steps while keeping JIT state alive across cells."""
        steps_done = 0
        t_chunk = time.perf_counter()
        while steps_done < max_steps and self.stage_idx < len(self.stage_plan):
            target_nodes, stage_steps = self._ensure_stage_ready()
            p_cnt = parameter_count(self.st.params)
            rem_in_stage = stage_steps - self.stage_step
            run_now = min(max_steps - steps_done, rem_in_stage)
            log_every = max(1, min(100, stage_steps // 5))

            for _ in range(run_now):
                self.stage_step += 1
                self.global_step += 1
                steps_done += 1
                lane, batch = _sample_preencoded_batch(
                    self.lane_arrays,
                    self.resets_template,
                    self.scheduler,
                    self.rng,
                    self.eff_batch,
                )
                self.st, metrics = self.step_fn(self.st, batch)
                self.controller.tick(1)

                should_sync = (
                    self.stage_step == 1
                    or self.stage_step % 25 == 0
                    or self.stage_step % log_every == 0
                    or self.stage_step == stage_steps
                    or steps_done == max_steps
                )
                if should_sync:
                    self.last_loss = float(metrics["loss"])
                    self.lane_losses[lane] = self.last_loss
                    self.scheduler.observe(self.lane_losses)
                    self.scheduler.step()
                    self.controller.observe(self.last_loss)

                if (
                    self.stage_step % log_every == 0
                    or self.stage_step == stage_steps
                    or steps_done == max_steps
                ):
                    elapsed = time.perf_counter() - self.stage_t0
                    bpb = self.last_loss / math.log(2.0)
                    print(
                        f"  [Stage {self.stage_idx + 1} | nodes={self.cfg_cur.nodes:2d} | "
                        f"{p_cnt / 1e6:.2f}M params] "
                        f"global {self.global_step:4d}/{self.total_steps} "
                        f"(stage {self.stage_step:4d}/{stage_steps}) | lane={lane:12s} | "
                        f"loss={self.last_loss:.4f} ({bpb:.3f} BPB) | {elapsed:.1f}s",
                        flush=True,
                    )

            if self.stage_step >= stage_steps:
                rewired_cnt = 0
                if self.stage_idx + 1 == len(self.stage_plan):
                    self.cfg_cur, rewire_info = rewire_edges(
                        self.cfg_cur, self.st.params, max_rewire_per_node=1
                    )
                    rewired_cnt = rewire_info["rewired"]
                dt_stage = time.perf_counter() - self.stage_t0
                self.growth_log.append(
                    {
                        "stage": self.stage_idx + 1,
                        "nodes": self.cfg_cur.nodes,
                        "parameters": p_cnt,
                        "active_nodes": self.cfg_cur.shared_nodes + self.cfg_cur.top_k_nodes,
                        "steps": stage_steps,
                        "train_loss": self.last_loss,
                        "train_bpb": self.last_loss / math.log(2.0),
                        "rewired_edges": rewired_cnt,
                        "stage_sec": dt_stage,
                    }
                )
                self.stage_idx += 1
                self.step_fn = None

        ckpt_b8 = self.out_dir / self.ckpt_name
        save_checkpoint(
            ckpt_b8,
            self.st,
            self.cfg_cur,
            {
                "growth": self.growth_log,
                "backend": self.backend,
                "total_steps": self.global_step,
            },
            quantize="b8",
        )
        return {
            "global_step": self.global_step,
            "total_steps": self.total_steps,
            "nodes": self.cfg_cur.nodes,
            "last_loss": self.last_loss,
            "last_bpb": self.last_loss / math.log(2.0),
            "chunk_sec": time.perf_counter() - t_chunk,
            "done": self.stage_idx >= len(self.stage_plan),
        }

    def finalize(self) -> dict:
        """Run post-training GRPO, save final b8 checkpoint, and run evaluation suites."""
        self.step_fn = None
        gc.collect()
        jax.clear_caches()

        sciq_probes = json.loads(
            (self.data_dir / "qa" / "sciq-validation-questions.json").read_text()
        )["questions"]
        squad_probes = json.loads(
            (self.data_dir / "qa" / "squad-validation-questions.json").read_text()
        )["questions"]
        all_grpo_probes = squad_probes + sciq_probes
        if self.grpo_steps > 0 and all_grpo_probes and (self.is_accel or not self.is_100m):
            print(
                f"Running {self.grpo_steps} DeepSeek-R1 critic-free GRPO reasoning steps...",
                flush=True,
            )
            grpo_fn = make_grpo_step(self.cfg_cur)
            ref_params = self.st.params
            for g_i in range(self.grpo_steps):
                probe = all_grpo_probes[g_i % len(all_grpo_probes)]
                g_batch, g_rewards = build_grpo_question_group(probe, length=self.eff_len)
                self.st, _ = grpo_fn(self.st, ref_params, g_batch, g_rewards)

        total_sec = time.perf_counter() - self.t_start
        jax.clear_caches()

        ckpt_b8 = self.out_dir / self.ckpt_name
        save_checkpoint(
            ckpt_b8,
            self.st,
            self.cfg_cur,
            {
                "growth": self.growth_log,
                "backend": self.backend,
                "total_steps": self.global_step,
            },
            quantize="b8",
        )
        b8_mb = ckpt_b8.stat().st_size / (1024 * 1024)
        if self.is_100m:
            fp32_mb = (parameter_count(self.st.params) * 15.2) / (1024 * 1024)
            st_eval, cfg_eval = self.st, self.cfg_cur
        else:
            ckpt_fp32 = self.out_dir / "mlm_10m_fp32.mlm"
            save_checkpoint(
                ckpt_fp32,
                self.st,
                self.cfg_cur,
                {
                    "growth": self.growth_log,
                    "backend": self.backend,
                    "total_steps": self.global_step,
                },
                quantize="none",
            )
            fp32_mb = ckpt_fp32.stat().st_size / (1024 * 1024)
            st_eval, cfg_eval, _ = load_checkpoint(ckpt_b8)

        val_batches = 1 if (self.is_100m and not self.is_accel) else (4 if self.is_100m else 4)
        mc_count = 4 if (self.is_100m and not self.is_accel) else (16 if self.is_100m else 16)
        suite_cap = 2 if (self.is_100m and not self.is_accel) else (8 if self.is_100m else 16)
        eng_eval = eval_corpus_bpb(
            st_eval,
            cfg_eval,
            self.data_dir / "english" / "validation.jsonl",
            max_batches=val_batches,
        )
        chat_eval = eval_corpus_bpb(
            st_eval,
            cfg_eval,
            self.data_dir / "conversation" / "validation.jsonl",
            max_batches=val_batches,
        )
        sciq_acc = eval_mc_probes(st_eval, cfg_eval, sciq_probes[:mc_count])
        squad_acc = eval_mc_probes(st_eval, cfg_eval, squad_probes[:mc_count])

        reading_suites = {
            "v1_1hop": json.loads(Path("data/evaluation/reading-v1.json").read_text())["questions"][
                :suite_cap
            ],
            "v2_2hop": json.loads(Path("data/evaluation/reading-v2-multihop.json").read_text())[
                "questions"
            ][:suite_cap],
            "v3_3hop": json.loads(Path("data/evaluation/reading-v3-3hop.json").read_text())[
                "questions"
            ][:suite_cap],
            "v4_temporal": json.loads(Path("data/evaluation/reading-v4-temporal.json").read_text())[
                "questions"
            ][:suite_cap],
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
            "backend": self.backend,
            "devices": self.devices,
            "parameters": parameter_count(st_eval.params),
            "total_steps": self.global_step,
            "total_sec": total_sec,
            "checkpoint_mb": {"fp32": fp32_mb, "b8": b8_mb},
            "growth_log": self.growth_log,
            "lane_shares": self.scheduler.shares,
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
        report_path = self.out_dir / "foundation_report.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report


def run_foundation_training(
    data_dir: Path,
    out_dir: Path,
    total_steps: int = 1200,
    batch_size: int | None = None,
    length: int | None = None,
    data_scale: int = 1,
    grpo_steps: int = 12,
    resume_from: Path | None = None,
    config_path: Path = Path("configs/mlm-10m-growth.json"),
    ckpt_name: str = "mlm_10m_b8.mlm",
) -> dict:
    """Run staged developmental foundation training (e.g. 4->8->16 or 16->32->64) and save b8."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    backend = jax.default_backend()
    devices = [str(d) for d in jax.devices()]
    is_accel = backend in ("gpu", "tpu")

    base_cfg = Config.model_validate_json(Path(config_path).read_text())
    is_100m = base_cfg.nodes >= 64
    eff_batch = batch_size or ((16 if is_100m else 32) if is_accel else (2 if is_100m else 8))
    eff_len = length or ((96 if is_100m else 128) if is_accel else (32 if is_100m else 64))

    print(
        f"[MLM Foundation Trainer] config={config_path} | backend={backend} | devices={devices} | "
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

    base_cfg = base_cfg.model_copy(update={"batch_size": eff_batch, "length": eff_len})
    n_final = base_cfg.nodes
    n_mid = max(4, n_final // 2)
    n_seed = max(4, n_final // 4)

    if resume_from is not None and Path(resume_from).exists():
        print(f"Resuming from checkpoint: {resume_from}", flush=True)
        st, cfg_cur, meta = load_checkpoint(Path(resume_from))
        cfg_cur = cfg_cur.model_copy(update={"batch_size": eff_batch, "length": eff_len})
        growth_log = list(meta.get("growth", []))
    else:
        cfg_cur = base_cfg.model_copy(update={"nodes": n_seed})
        st = create_state(cfg_cur)
        growth_log = []

    cur_nodes = int(st.params["input_kernel"].shape[0])
    if cur_nodes >= n_final:
        stage_plan = [(n_final, max(1, total_steps))]
    elif cur_nodes >= n_mid:
        s2 = max(1, int(round(total_steps * 0.40)))
        s3 = max(1, total_steps - s2)
        stage_plan = [(n_mid, s2), (n_final, s3)]
    else:
        # Developmental schedule: 60% at seed nodes, 25% at mid nodes, 15% at final nodes
        s1 = max(1, int(round(total_steps * 0.60)))
        s2 = max(1, int(round(total_steps * 0.25)))
        s3 = max(1, total_steps - s1 - s2)
        stage_plan = [(n_seed, s1), (n_mid, s2), (n_final, s3)]

    rng = np.random.default_rng(base_cfg.seed)
    t_start = time.perf_counter()
    step_fn = None

    for stage_idx, (target_nodes, stage_steps) in enumerate(stage_plan, start=1):
        step_fn = None
        gc.collect()
        jax.clear_caches()
        if st.params["input_kernel"].shape[0] < target_nodes:
            add_n = target_nodes - cfg_cur.nodes
            cfg_cur, st = grow_train_state(cfg_cur, st, add_nodes=add_n)
            gc.collect()
            jax.clear_caches()
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
    all_grpo_probes = squad_probes + sciq_probes
    if grpo_steps > 0 and all_grpo_probes and (is_accel or not is_100m):
        print(f"Running {grpo_steps} DeepSeek-R1 critic-free GRPO reasoning steps...", flush=True)
        grpo_fn = make_grpo_step(cfg_cur)
        ref_params = st.params
        for g_i in range(grpo_steps):
            probe = all_grpo_probes[g_i % len(all_grpo_probes)]
            g_batch, g_rewards = build_grpo_question_group(probe, length=eff_len)
            st, _ = grpo_fn(st, ref_params, g_batch, g_rewards)

    total_sec = time.perf_counter() - t_start
    jax.clear_caches()

    ckpt_b8 = out_dir / ckpt_name
    save_checkpoint(
        ckpt_b8,
        st,
        cfg_cur,
        {"growth": growth_log, "backend": backend, "total_steps": total_steps},
        quantize="b8",
    )
    b8_mb = ckpt_b8.stat().st_size / (1024 * 1024)
    if is_100m:
        fp32_mb = (parameter_count(st.params) * 15.2) / (1024 * 1024)
        st_eval, cfg_eval = st, cfg_cur
    else:
        ckpt_fp32 = out_dir / "mlm_10m_fp32.mlm"
        save_checkpoint(
            ckpt_fp32,
            st,
            cfg_cur,
            {"growth": growth_log, "backend": backend, "total_steps": total_steps},
            quantize="none",
        )
        fp32_mb = ckpt_fp32.stat().st_size / (1024 * 1024)
        st_eval, cfg_eval, _ = load_checkpoint(ckpt_b8)

    val_batches = 1 if (is_100m and not is_accel) else (2 if is_100m else 4)
    mc_count = 4 if (is_100m and not is_accel) else (8 if is_100m else 16)
    suite_cap = 2 if (is_100m and not is_accel) else (4 if is_100m else 16)
    eng_eval = eval_corpus_bpb(
        st_eval, cfg_eval, data_dir / "english" / "validation.jsonl", max_batches=val_batches
    )
    chat_eval = eval_corpus_bpb(
        st_eval, cfg_eval, data_dir / "conversation" / "validation.jsonl", max_batches=val_batches
    )
    sciq_acc = eval_mc_probes(st_eval, cfg_eval, sciq_probes[:mc_count])
    squad_acc = eval_mc_probes(st_eval, cfg_eval, squad_probes[:mc_count])

    reading_suites = {
        "v1_1hop": json.loads(Path("data/evaluation/reading-v1.json").read_text())["questions"][
            :suite_cap
        ],
        "v2_2hop": json.loads(Path("data/evaluation/reading-v2-multihop.json").read_text())[
            "questions"
        ][:suite_cap],
        "v3_3hop": json.loads(Path("data/evaluation/reading-v3-3hop.json").read_text())[
            "questions"
        ][:suite_cap],
        "v4_temporal": json.loads(Path("data/evaluation/reading-v4-temporal.json").read_text())[
            "questions"
        ][:suite_cap],
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
