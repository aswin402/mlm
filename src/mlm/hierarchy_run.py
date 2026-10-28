"""Soft hierarchical recurrent updates; dense execution, no sparse-speed claim."""

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from mlm.binding_run import _copy_matching_opt_state, better, build_batches, warm_start
from mlm.checkpoint import save_checkpoint
from mlm.cli import environment, write_json
from mlm.comprehension import evaluate_questions
from mlm.config import Config
from mlm.data import read_documents, read_training_data
from mlm.gradual_eval import order_robust
from mlm.model import initial_state
from mlm.plasticity import (
    LaneSchedule,
    PlasticityController,
    build_grpo_question_group,
    make_grpo_step,
    maybe_grow_and_prune,
    rewire_edges,
    update_router_load_bias,
)
from mlm.stream import DocumentStream, evaluate_stream
from mlm.training import create_state


def routed_objective(
    params,
    apply_fn,
    batch,
    carry,
    config,
    return_logits=False,
    running_usage=None,
    router_bias=None,
):
    use_ponder = config.adaptive_halt and config.ponder_beta > 0
    use_mtp = config.architecture == "graph" and config.mtp_weight > 0
    kwargs = {"valid": batch["mask"], "return_routes": True}
    if use_ponder:
        kwargs["return_ponder"] = True
    if use_mtp:
        kwargs["return_mtp"] = True
        kwargs["next_tokens"] = jnp.minimum(batch["targets"], 258)
    if router_bias is not None:
        kwargs["router_bias"] = router_bias
    out = apply_fn(
        {"params": params},
        batch["inputs"],
        carry,
        batch["resets"],
        **kwargs,
    )
    logits, final, probabilities = out[0], out[1], out[2]
    idx = 3
    ponder_kl = out[idx] if use_ponder else None
    if use_ponder:
        idx += 1
    mtp_logits = out[idx] if use_mtp else None

    losses = optax.softmax_cross_entropy_with_integer_labels(logits, batch["targets"])
    valid = batch["mask"]
    targets = valid * batch.get("loss_mask", valid)
    count = jnp.maximum(targets.sum(), 1)
    language = (losses * targets).sum() / count
    positions = jnp.maximum(valid.sum(), 1)
    usage = (probabilities * valid[..., None]).sum(axis=(0, 1)) / positions
    if running_usage is not None:
        ref_usage = jax.lax.stop_gradient(running_usage)
        balance_raw = config.nodes * (ref_usage * usage).sum() - 1.0
    else:
        balance_raw = config.nodes * jnp.square(usage).sum() - 1.0
    balance = jnp.where(valid.sum() > 0, jnp.maximum(0.0, balance_raw), 0.0)
    entropy = -(probabilities * jnp.log(jnp.maximum(probabilities, 1e-9))).sum(-1)
    weight = config.router_balance if config.node_update in {"hierarchical", "timescale"} else 0.0
    ponder_loss = (
        (ponder_kl * valid).sum() / positions
        if (use_ponder and ponder_kl is not None)
        else jnp.asarray(0.0, language.dtype)
    )
    if use_mtp and mtp_logits is not None and batch["targets"].shape[1] > 1:
        mtp_targets = batch["targets"][:, 1:]
        mtp_valid = (
            valid[:, :-1]
            * targets[:, 1:]
            * (~batch["resets"][:, 1:])
            * (batch["targets"][:, :-1] == batch["inputs"][:, 1:])
        )
        mtp_ce = optax.softmax_cross_entropy_with_integer_labels(mtp_logits[:, :-1], mtp_targets)
        mtp_loss = (mtp_ce * mtp_valid).sum() / jnp.maximum(mtp_valid.sum(), 1)
    else:
        mtp_loss = jnp.asarray(0.0, language.dtype)
    payload = targets * (batch["targets"] < 256)
    metrics = {
        "loss": language,
        "mtp_loss": mtp_loss,
        "router_balance": balance,
        "ponder_kl": ponder_loss,
        "route_usage": usage,
        "route_entropy": (entropy * valid).sum() / positions,
        "input_positions": valid.sum(),
        "positions": targets.sum(),
        "bytes": payload.sum(),
        "consumed_bytes": (valid * (batch["targets"] < 256)).sum(),
    }
    aux = (final, metrics, logits) if return_logits else (final, metrics)
    total_loss = (
        language
        + weight * balance
        + config.ponder_beta * ponder_loss
        + config.mtp_weight * mtp_loss
    )
    return total_loss, aux


def make_routed_step(config):
    @jax.jit
    def step(state, carry, batch, running_usage=None, router_bias=None):
        (_, (final, metrics)), gradients = jax.value_and_grad(routed_objective, has_aux=True)(
            state.params,
            state.apply_fn,
            batch,
            jax.lax.stop_gradient(carry),
            config,
            False,
            running_usage,
            router_bias,
        )
        candidate = state.apply_gradients(grads=gradients)
        leaves = jax.tree.leaves((candidate.params, candidate.opt_state, gradients, final, metrics))
        finite = jnp.all(jnp.stack([jnp.isfinite(x).all() for x in leaves]))
        updated = finite & (metrics["positions"] > 0)
        new_state = jax.lax.cond(updated, lambda: candidate, lambda: state)
        hidden = jax.lax.cond(finite, lambda: final, lambda: carry)
        return new_state, hidden, {**metrics, "finite": finite, "updated": updated}

    return step


def run(args):
    if args.steps < 1 or args.eval_every < 1:
        raise ValueError("Positive budget/interval required")
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Use an empty output directory")
    cfg = Config.model_validate_json(args.config.read_text())
    if cfg.architecture != "graph" or cfg.node_readout != "sum":
        raise ValueError("Keep sum graph unchanged")
    train = args.data / "train.jsonl"
    records = [json.loads(line) for line in train.read_text().splitlines()]
    questions = json.loads((args.data / "train-questions.json").read_text())["questions"]
    batches, ids = build_batches(records, questions, cfg)
    dev, dev_masks, _ = read_training_data(args.data / "validation.jsonl")
    dev_questions = json.loads((args.data / "validation-questions.json").read_text())["questions"]
    if {r["text"] for r in records} & set(dev):
        raise ValueError("Task split overlap")
    english_train = read_documents(args.english_train)
    english_dev = read_documents(args.english_dev)
    if set(english_train) & set(english_dev):
        raise ValueError("English split overlap")
    state = warm_start(args.init, cfg)
    args.out.mkdir(parents=True, exist_ok=True)
    paths = [
        train,
        args.data / "train-questions.json",
        args.data / "validation.jsonl",
        args.data / "validation-questions.json",
        args.english_train,
        args.english_dev,
        args.config,
    ]
    provenance = {
        "mode": "hierarchical-routing",
        "node_update": cfg.node_update,
        "router_balance": cfg.router_balance,
        "messages": True,
        "rehearsal": args.rehearsal,
        "source": str(args.init),
        "source_sha256": hashlib.sha256(args.init.read_bytes()).hexdigest(),
        "files": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "qa_budget": args.steps,
        "eval_every_qa": args.eval_every,
        "group_ids": ids,
        "resume": "not supported",
        "selection": "macro order-robust development accuracy, then QA loss; initial eligible",
    }
    write_json(args.out / "protocol.json", provenance)
    write_json(args.out / "config.json", cfg.model_dump())
    write_json(args.out / "environment.json", environment())
    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(len(batches))
    position = 0
    stream = DocumentStream(english_train, cfg.batch_size, cfg.length, cfg.seed + 1)
    carry = initial_state(cfg, cfg.batch_size)
    counters = {
        name: {"updates": 0, "input_bytes": 0, "supervised_bytes": 0} for name in ["qa", "english"]
    }
    qa_step = 0
    routing = {
        name: {
            "positions": 0.0,
            "usage_sum": np.zeros(cfg.nodes),
            "entropy_sum": 0.0,
            "balance_sum": 0.0,
        }
        for name in counters
    }

    def metadata():
        return {
            **provenance,
            "qa_step": qa_step,
            "counters": counters,
            "qa_order": order.tolist(),
            "qa_position": position,
            "qa_rng": rng.bit_generator.state,
            "english_stream": stream.snapshot(),
            "english_carry": np.array(carry).tolist(),
        }

    def validate():
        ranked = evaluate_questions(state, cfg, dev_questions, format="chat")
        pools = {
            name: order_robust([q for q in ranked["items"] if q["category"] == name])
            for name in ["familiar-names-new-pairs", "new-names"]
        }
        strata = {}
        for name in pools:
            for kind in ["within", "cross"]:
                subset = [
                    q for q in ranked["items"] if q["category"] == name and q["pair_type"] == kind
                ]
                strata[name + "/" + kind] = order_robust(subset)
        loss = evaluate_stream(state, cfg, dev, loss_masks=dev_masks)["loss"]
        english = evaluate_stream(state, cfg, english_dev)["loss"]
        score = sum(p["accuracy"] for p in pools.values()) / 2
        return {
            "qa_step": qa_step,
            "optimizer_step": int(state.step),
            "pools": pools,
            "strata": strata,
            "score": score,
            "qa_loss": loss,
            "english_development_loss": english,
            "counters": json.loads(json.dumps(counters)),
            "routing": {
                key: {
                    "positions": value["positions"],
                    "mean_usage": np.asarray(
                        value["usage_sum"] / max(value["positions"], 1)
                    ).tolist(),
                    "mean_entropy": value["entropy_sum"] / max(value["positions"], 1),
                    "mean_batch_balance": value["balance_sum"] / max(value["positions"], 1),
                }
                for key, value in routing.items()
            },
        }

    initial = validate()
    best = {name: (initial["score"], initial["qa_loss"]) for name in ["best"]}
    selected = {name: 0 for name in best}
    for name in best:
        save_checkpoint(args.out / f"{name}.mlm", state, cfg, metadata())
    logs = [initial]
    (args.out / "metrics.jsonl").write_text(json.dumps(initial) + "\n")
    print(json.dumps(initial), flush=True)
    step = make_routed_step(cfg)

    def update(batch, hidden, objective):
        nonlocal state
        state, hidden, raw = step(state, hidden, jax.tree.map(jnp.asarray, batch))
        raw = jax.device_get(raw)
        if not bool(raw["finite"]) or not bool(raw["updated"]):
            raise FloatingPointError("Rejected update")
        counters[objective]["updates"] += 1
        counters[objective]["input_bytes"] += int(raw["consumed_bytes"])
        counters[objective]["supervised_bytes"] += int(raw["bytes"])
        positions = float(raw["input_positions"])
        routing[objective]["positions"] += positions
        routing[objective]["usage_sum"] += np.asarray(raw["route_usage"]) * positions
        routing[objective]["entropy_sum"] += float(raw["route_entropy"]) * positions
        routing[objective]["balance_sum"] += float(raw["router_balance"]) * positions
        return hidden

    for qa_step in range(1, args.steps + 1):
        if position == len(order):
            order = rng.permutation(len(batches))
            position = 0
        batch = batches[int(order[position])]
        position += 1
        update(batch, initial_state(cfg, 4), "qa")
        if args.rehearsal:
            carry = update(stream.next_batch(), carry, "english")
        eligible = qa_step % args.eval_every == 0
        if eligible or qa_step == args.steps:
            row = validate()
            row["selection_eligible"] = eligible
            for name in best:
                if eligible and better(row["score"], row["qa_loss"], best[name]):
                    best[name] = (row["score"], row["qa_loss"])
                    selected[name] = qa_step
                    save_checkpoint(args.out / f"{name}.mlm", state, cfg, metadata())
            save_checkpoint(args.out / "latest.mlm", state, cfg, metadata())
            logs.append(row)
            (args.out / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in logs))
            print(json.dumps(row), flush=True)
    summary = {
        "qa_updates": qa_step,
        "optimizer_updates": int(state.step),
        "rehearsal": args.rehearsal,
        "selected_qa_steps": selected,
        "selected_scores": {k: v[0] for k, v in best.items()},
        "counters": counters,
        "parameters": sum(p.size for p in jax.tree.leaves(state.params)),
        "nominal_write_mass": cfg.nodes if cfg.node_update == "dense" else 1,
        "execution": "dense; all candidate nodes computed",
    }
    write_json(args.out / "summary.json", summary)
    print(json.dumps(summary), flush=True)


def run_continual(args) -> dict:
    """Autonomous continual loop with LaneSchedule, PlasticityController, growth, and rewiring."""
    if args.steps < 1 or args.eval_every < 1:
        raise ValueError("Positive budget and eval-every required")
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Use an empty output directory")
    cfg = Config.model_validate_json(args.config.read_text())
    if cfg.architecture != "graph":
        raise ValueError("Continual graph runner requires graph architecture")
    max_rec = getattr(args, "max_records", None)
    train_lines = (args.data / "train.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in (train_lines[:max_rec] if max_rec else train_lines)]
    all_questions = json.loads((args.data / "train-questions.json").read_text())["questions"]
    questions = all_questions[: len(records)]
    batches, ids = build_batches(records, questions, cfg)
    dev_docs, dev_masks, _ = read_training_data(args.data / "validation.jsonl")
    dev_questions = json.loads((args.data / "validation-questions.json").read_text())["questions"]
    if max_rec:
        dev_docs = dev_docs[:max_rec]
        if dev_masks is not None:
            dev_masks = dev_masks[:max_rec]
        dev_questions = dev_questions[:max_rec]
    english_train, _, _ = read_training_data(args.english_train)
    english_dev, _, _ = read_training_data(args.english_dev)

    init_path = getattr(args, "init", None)
    state = (
        warm_start(init_path, cfg, preserve_moments=True)
        if init_path is not None
        else create_state(cfg)
    )
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "config.json", cfg.model_dump())
    write_json(args.out / "environment.json", environment())

    qa_seed = getattr(args, "qa_seed", None)
    eng_seed = getattr(args, "english_seed", None)
    rng = np.random.default_rng(cfg.seed if qa_seed is None else int(qa_seed))
    order = rng.permutation(len(batches))
    position = 0
    stream = DocumentStream(
        english_train,
        cfg.batch_size,
        cfg.length,
        (cfg.seed + 1) if eng_seed is None else int(eng_seed),
    )
    carry = initial_state(cfg, cfg.batch_size)
    running_usage = jnp.full((cfg.nodes,), 1.0 / cfg.nodes, dtype=jnp.float32)
    router_bias = jnp.zeros((cfg.nodes,), dtype=jnp.float32)
    step_fn = make_routed_step(cfg)
    use_grpo = bool(getattr(args, "grpo", False))
    grpo_step_fn = make_grpo_step(cfg) if use_grpo else None
    grpo_groups = (
        [
            build_grpo_question_group(questions[i : i + 4], cfg.length)
            for i in range(0, len(questions), 4)
        ]
        if use_grpo
        else []
    )
    ref_params = state.params if use_grpo else None

    qa_corpus_bytes = float(sum(int(np.sum(b["mask"])) for b in batches))
    eng_corpus_bytes = float(sum(len(d.encode("utf-8")) for d in english_train))
    lanes = LaneSchedule(
        ["qa", "english"],
        sizes={"qa": qa_corpus_bytes, "english": eng_corpus_bytes},
    )
    controller = PlasticityController(scale=1.0)
    counters = {
        name: {"updates": 0, "input_bytes": 0, "supervised_bytes": 0} for name in ["qa", "english"]
    }
    latest_losses: dict[str, float] = {}
    logs = []
    growth_events = []
    rewire_events = []

    def rebuild_state_for_config(new_cfg: Config, new_params: dict):
        fresh = create_state(new_cfg)
        preserved_opt = _copy_matching_opt_state(
            fresh.opt_state, state.opt_state, set(new_params.keys())
        )
        return fresh.replace(step=state.step, params=new_params, opt_state=preserved_opt)

    def evaluate_step(step_num: int) -> dict:
        train_eval = evaluate_questions(state, cfg, questions, format="chat")
        dev_eval = evaluate_questions(state, cfg, dev_questions, format="chat")
        qa_val = float(evaluate_stream(state, cfg, dev_docs, loss_masks=dev_masks)["loss"])
        eng_val = float(evaluate_stream(state, cfg, english_dev)["loss"])
        controller.observe(qa_val)
        return {
            "step": step_num,
            "optimizer_step": int(state.step),
            "nodes": cfg.nodes,
            "train_qa_accuracy": float(train_eval["accuracy"]),
            "dev_qa_accuracy": float(dev_eval["accuracy"]),
            "qa_validation_loss": qa_val,
            "english_validation_loss": eng_val,
            "lane_shares": dict(lanes.shares),
            "controller_scale": float(controller.scale),
            "counters": json.loads(json.dumps(counters)),
        }

    initial_row = evaluate_step(0)
    best_key = (initial_row["dev_qa_accuracy"], initial_row["qa_validation_loss"])
    save_checkpoint(args.out / "best.mlm", state, cfg, {"step": 0, "lanes": lanes.shares})
    save_checkpoint(args.out / "latest.mlm", state, cfg, {"step": 0, "lanes": lanes.shares})
    logs.append(initial_row)

    for step_idx in range(1, args.steps + 1):
        lane = lanes.pick()
        if lane == "qa":
            if position >= len(order):
                order = rng.permutation(len(batches))
                position = 0
            batch_idx = int(order[position])
            batch = batches[batch_idx]
            position += 1
            state, _, raw = step_fn(
                state,
                initial_state(cfg, cfg.batch_size),
                jax.tree.map(jnp.asarray, batch),
                running_usage,
                router_bias if cfg.router_bias_gamma > 0 else None,
            )
            if use_grpo and grpo_step_fn is not None and grpo_groups:
                ref_params = state.params
                g_batch, g_rewards = grpo_groups[batch_idx % len(grpo_groups)]
                state, _ = grpo_step_fn(state, ref_params, g_batch, g_rewards)
        else:
            batch = stream.next_batch()
            state, carry, raw = step_fn(
                state,
                carry,
                jax.tree.map(jnp.asarray, batch),
                running_usage,
                router_bias if cfg.router_bias_gamma > 0 else None,
            )
        raw = jax.device_get(raw)
        if not bool(raw["finite"]) or not bool(raw["updated"]):
            raise FloatingPointError("Rejected continual update")
        running_usage = 0.95 * running_usage + 0.05 * jnp.asarray(raw["route_usage"])
        if cfg.router_bias_gamma > 0:
            router_bias = update_router_load_bias(
                router_bias, running_usage, cfg.router_bias_gamma, cfg.shared_nodes
            )
        counters[lane]["updates"] += 1
        counters[lane]["input_bytes"] += int(raw["consumed_bytes"])
        counters[lane]["supervised_bytes"] += int(raw["bytes"])
        lanes.note_read(lane, float(raw["consumed_bytes"]))
        latest_losses[lane] = float(raw["loss"])

        if step_idx % 10 == 0:
            lanes.observe(latest_losses)
            lanes.step()

        if step_idx % args.eval_every == 0 or step_idx == args.steps:
            lanes.observe(latest_losses)
            lanes.step()
            prev_qa_val = logs[-1]["qa_validation_loss"] if logs else None
            row = evaluate_step(step_idx)
            plateaued = prev_qa_val is not None and row["qa_validation_loss"] >= prev_qa_val - 0.02

            if step_idx < args.steps and plateaued and getattr(args, "auto_grow", False):
                new_cfg, new_params, g_info = maybe_grow_and_prune(
                    cfg,
                    state.params,
                    running_usage,
                    train_loss=latest_losses.get("qa"),
                    val_loss=row["qa_validation_loss"],
                )
                if new_cfg.nodes != cfg.nodes:
                    cfg = new_cfg
                    state = rebuild_state_for_config(cfg, new_params)
                    batches, ids = build_batches(records, questions, cfg)
                    carry = initial_state(cfg, cfg.batch_size)
                    running_usage = jnp.full((cfg.nodes,), 1.0 / cfg.nodes, dtype=jnp.float32)
                    router_bias = jnp.zeros((cfg.nodes,), dtype=jnp.float32)
                    step_fn = make_routed_step(cfg)
                    if use_grpo:
                        grpo_step_fn = make_grpo_step(cfg)
                        ref_params = state.params
                    growth_events.append({"step": step_idx, **g_info, "nodes": cfg.nodes})
                    row["nodes"] = cfg.nodes

            if step_idx < args.steps and plateaued and getattr(args, "rewire", False):
                new_cfg, r_info = rewire_edges(cfg, state.params, max_total_rewires=1)
                if r_info["rewired"] > 0:
                    cfg = new_cfg
                    step_fn = make_routed_step(cfg)
                    if use_grpo:
                        grpo_step_fn = make_grpo_step(cfg)
                    rewire_events.append(
                        {"step": step_idx, "rewired": r_info["rewired"], "edges": r_info["edges"]}
                    )

            meta = {
                "step": step_idx,
                "lanes": dict(lanes.shares),
                "controller": controller.state(),
            }
            write_json(args.out / "config.json", cfg.model_dump())
            save_checkpoint(args.out / "latest.mlm", state, cfg, meta)
            if better(row["dev_qa_accuracy"], row["qa_validation_loss"], best_key):
                best_key = (row["dev_qa_accuracy"], row["qa_validation_loss"])
                save_checkpoint(args.out / "best.mlm", state, cfg, meta)
            logs.append(row)
            (args.out / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in logs))

    summary = {
        "steps": args.steps,
        "optimizer_updates": int(state.step),
        "nodes": cfg.nodes,
        "edges": cfg.edges,
        "parameters": sum(int(p.size) for p in jax.tree.leaves(state.params)),
        "counters": counters,
        "final_lane_shares": dict(lanes.shares),
        "controller_scale": float(controller.scale),
        "best_dev_qa_accuracy": best_key[0],
        "best_qa_validation_loss": best_key[1],
        "final_train_qa_accuracy": logs[-1]["train_qa_accuracy"],
        "final_dev_qa_accuracy": logs[-1]["dev_qa_accuracy"],
        "final_english_validation_loss": logs[-1]["english_validation_loss"],
        "growth_events": growth_events,
        "rewire_events": rewire_events,
    }
    write_json(args.out / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/hierarchy-v1"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--english-train", type=Path, default=Path("data/hf/stories/train.jsonl"))
    parser.add_argument(
        "--english-dev", type=Path, default=Path("data/hf/stories/validation.jsonl")
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.set_defaults(rehearsal=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
