"""Interactive llama.cpp-style TUI for real-time MLM chat, thinking trace, and online learning."""

import argparse
import sys
from pathlib import Path

import jax
import numpy as np

from mlm.checkpoint import load_checkpoint, save_checkpoint
from mlm.config import Config
from mlm.generate import LiveLearner
from mlm.graph import topology
from mlm.plasticity import node_saturation
from mlm.training import create_state

# ANSI styling helpers
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
BRIGHT_CYAN = "\033[96m"
GREEN = "\033[32m"
BRIGHT_GREEN = "\033[92m"
YELLOW = "\033[33m"
BRIGHT_YELLOW = "\033[93m"
MAGENTA = "\033[35m"
BRIGHT_MAGENTA = "\033[95m"
BLUE = "\033[34m"
RED = "\033[31m"
GRAY = "\033[90m"


def parameter_count(params: dict) -> int:
    return int(sum(int(np.prod(v.shape)) for v in jax.tree.leaves(params)))


def _c(text: str, code: str, color: bool = True) -> str:
    return f"{code}{text}{RESET}" if color else text


def _bar(val: float, width: int = 8) -> str:
    v = float(np.clip(val, 0.0, 1.0))
    filled = int(round(v * width))
    return "█" * filled + "░" * (width - filled)


def format_model_banner(
    config: Config,
    params: dict,
    ckpt_label: str,
    step: int,
    learn_enabled: bool,
    show_thinking: bool,
    context_str: str = "",
    color: bool = True,
) -> str:
    """Format a llama.cpp-inspired model metadata and system banner."""
    n_params = parameter_count(params)
    devs = ", ".join(str(d) for d in jax.devices())
    n_nodes = 1 if config.architecture == "gru" else config.nodes
    active_k = config.top_k_nodes if config.top_k_nodes > 0 else n_nodes
    mem_desc = (
        f"KDA-{config.memory} ({config.memory_keys} keys"
        + (", channel-decay + ShortConv" if config.memory_decay else "")
        + (", 4-scale DoG wavelets" if config.symmetric_memory else "")
        + ")"
        if config.memory != "none"
        else "none"
    )
    rel_desc = (
        f"{config.relational_hops}-hop Disjoint-Clause Bridge + IndexShare + 4-Stream mHC"
        if config.relational_hops > 0
        else "disabled"
    )
    arch_str = _c(config.architecture.upper(), BRIGHT_GREEN, color)
    param_str = _c(f"{n_params:,}", BRIGHT_YELLOW, color)
    think_c = BRIGHT_GREEN if show_thinking else YELLOW
    learn_c = BRIGHT_GREEN if learn_enabled else YELLOW
    think_str = _c("ON" if show_thinking else "OFF", think_c, color)
    learn_str = _c("ON" if learn_enabled else "OFF", learn_c, color)
    ctx_suffix = f" | context={_c(repr(context_str), BRIGHT_MAGENTA, color)}" if context_str else ""
    top_border = "╭" + "─" * 78 + "╮"
    mid_border = "├" + "─" * 78 + "┤"
    bot_border = "╰" + "─" * 78 + "╯"
    lines = [
        _c(top_border, CYAN, color),
        _c("│ ", CYAN, color)
        + _c("⚡ MLM Interactive Neural Console", BOLD + BRIGHT_CYAN, color)
        + _c("  (llama.cpp-style TUI + Live <think>)", DIM, color),
        _c(mid_border, CYAN, color),
        (
            f"  {_c('model_load_meta:', BOLD, color)}  arch={arch_str} | "
            f"params={param_str} ({n_params * 4 / 1024:.1f} KiB fp32) | "
            f"step={step} | src={ckpt_label}"
        ),
        (
            f"  {_c('system_backend: ', BOLD, color)}  jax_devices=[{devs}] | "
            f"vocab=258 UTF-8 bytes | optimizer={config.optimizer.upper()}"
        ),
        (
            f"  {_c('graph_topology: ', BOLD, color)}  nodes={n_nodes} "
            f"(shared={config.shared_nodes}, active_top_k={active_k}, "
            f"degree={config.degree}, rounds={config.rounds}, "
            f"sinkhorn={config.sinkhorn_edges})"
        ),
        (
            f"  {_c('memory_system:  ', BOLD, color)}  fast_weights={mem_desc} | "
            f"mla_episodic_slots={config.episodic_slots}"
        ),
        f"  {_c('relational_ckt: ', BOLD, color)}  {rel_desc}",
        (
            f"  {_c('runtime_modes:  ', BOLD, color)}  thinking={think_str} | "
            f"online_learn={learn_str} | speculative_span_decode=ON{ctx_suffix}"
        ),
        _c(mid_border, CYAN, color),
        (
            f"  {_c('Autonomous Chat:', BOLD, color)} "
            f"Type facts or questions naturally (type 'exit' or 'quit' to leave)"
        ),
        _c(bot_border, CYAN, color),
    ]
    return "\n".join(lines)


def format_thinking_panel(
    trace: dict,
    config: Config,
    color: bool = True,
) -> str:
    """Render the `<think>` panel showing multi-hop reasoning, node routing, and state norms."""
    header = "╭─ 🧠 Model Thinking & Telemetry (<think>) " + "─" * 27 + "╮"
    footer = "╰" + "─" * 69 + "╯"
    lines = [_c(header, MAGENTA, color)]

    if trace.get("relational_active"):
        g_q = trace.get("g_question", 0.0)
        g_bwd = trace.get("g_bwd", 0.0)
        direction = trace.get("direction", "forward")
        clauses = trace.get("clauses", {})
        lines.append(
            _c("│ ", MAGENTA, color)
            + _c("Mode: ", BOLD, color)
            + _c("Relational Multi-Hop QA", BRIGHT_GREEN, color)
            + f"  (q_gate={g_q:.3f}, dir={direction}, bwd_prob={g_bwd:.2f})"
        )
        if clauses:
            clause_parts = [f"[C{cid}] {txt}" for cid, txt in sorted(clauses.items())]
            lines.append(
                _c("│ ", MAGENTA, color)
                + _c("Context Clauses: ", BOLD, color)
                + _c(" | ".join(clause_parts), DIM, color)
            )
        anchor = trace.get("query_anchor") or "?"
        lines.append(
            _c("│ ", MAGENTA, color)
            + _c("Query Salience Anchor: ", BOLD, color)
            + _c(f"'{anchor}'", BRIGHT_YELLOW, color)
        )
        hops = trace.get("hops", [])
        prev_target = anchor
        for h in hops:
            hop_num = h["hop"]
            cid = h["clause_id"]
            tgt = h["target_word"] or "?"
            mass = h["target_mass"]
            if hop_num == 1:
                src_s = _c(repr(prev_target), BRIGHT_YELLOW, color)
                tgt_s = _c(repr(tgt), BRIGHT_GREEN, color)
                lines.append(
                    _c("│ ", MAGENTA, color)
                    + _c(f"  Hop 1 [Clause C{cid}]: ", BRIGHT_CYAN, color)
                    + f"anchor {src_s} ──► target {tgt_s} (attn_mass={mass:.3f})"
                )
                prev_target = tgt
            elif h["active"]:
                br = h.get("bridge_word") or prev_target
                gate = h.get("gate", 0.0)
                br_s = _c(repr(br), BRIGHT_YELLOW, color)
                tgt_s = _c(repr(tgt), BRIGHT_GREEN, color)
                lines.append(
                    _c("│ ", MAGENTA, color)
                    + _c(f"  Hop {hop_num} [Clause C{cid}]: ", BRIGHT_CYAN, color)
                    + f"bridge {br_s} (gate={gate:.3f}) ──► target {tgt_s} (attn_mass={mass:.3f})"
                )
                prev_target = tgt
            else:
                gate = h.get("gate", 0.0)
                halt_msg = f"gate={gate:.3f} (halt — hold target {prev_target!r})"
                lines.append(
                    _c("│ ", MAGENTA, color)
                    + _c(f"  Hop {hop_num} [passthrough]: ", GRAY, color)
                    + _c(halt_msg, DIM, color)
                )
        onset_w = trace.get("onset_word") or "?"
        onset_m = trace.get("onset_mass", 0.0)
        eos_m = trace.get("eos_mass", 0.0)
        idx_s = trace.get("index_share_density", 0.0) * 100.0
        spec_acc = trace.get("speculative_accepted", 0)
        onset_s = _c(repr(onset_w), BOLD + BRIGHT_GREEN, color)
        lines.append(
            _c("│ ", MAGENTA, color)
            + _c("Word-Onset Readout: ", BOLD, color)
            + f"selected={onset_s} (onset_conf={onset_m:.3f}, eos_gate={eos_m:.3f}, "
            + f"idx_share={idx_s:.1f}%, spec_accepted={spec_acc}B)"
        )
    else:
        g_q = trace.get("g_question", 0.0)
        lines.append(
            _c("│ ", MAGENTA, color)
            + _c("Mode: ", BOLD, color)
            + _c("Recurrent Byte-Stream Continuation", BRIGHT_CYAN, color)
            + f"  (q_gate={g_q:.3f})"
        )

    routes = trace.get("routes_mean", [])
    if routes:
        node_chips = []
        for idx, r_val in enumerate(routes):
            star = "*" if idx < config.shared_nodes else ""
            chip = f"N{idx}{star}[{_bar(r_val, 5)}] {r_val * 100:4.1f}%"
            node_chips.append(chip)
        lines.append(
            _c("│ ", MAGENTA, color)
            + _c("Graph Routing: ", BOLD, color)
            + "  ".join(node_chips[:4])
        )
        for row_idx in range(4, len(node_chips), 4):
            lines.append(
                _c("│ ", MAGENTA, color)
                + "               "
                + "  ".join(node_chips[row_idx : row_idx + 4])
            )

    ponder_m = trace.get("ponder_mean", 0.0)
    mem_n = trace.get("memory_norm", 0.0)
    wav_n = trace.get("wavelet_norm", 0.0)
    ep_n = trace.get("episodic_norm", 0.0)
    lines.append(
        _c("│ ", MAGENTA, color)
        + _c("Neural State:  ", BOLD, color)
        + f"ponder_KL={ponder_m:.4f} | KDA ||M||_F={mem_n:.3f} | "
        + f"DoG ||W||={wav_n:.3f} | MLA ||E||_F={ep_n:.3f}"
    )

    ep_facts = trace.get("episodic_facts", [])
    if ep_facts:
        comp_n = trace.get("compacted_facts", 0)
        comp_str = f", compacted={comp_n}" if comp_n > 0 else ""
        lines.append(
            _c("│ ", MAGENTA, color)
            + _c("SAO Episodic Bank: ", BOLD, color)
            + f"{len(ep_facts)} fact(s){comp_str}: "
            + " ".join(ep_facts[-2:])
        )

    tok_steps = trace.get("token_steps", [])
    if tok_steps:
        emitted = [
            f"{s['char']!r}({s['prob'] * 100:.0f}%)"
            if s["char"] != "<EOS>"
            else f"<EOS>({s['prob'] * 100:.0f}%)"
            for s in tok_steps[:8]
        ]
        lines.append(
            _c("│ ", MAGENTA, color) + _c("Byte Stream:   ", BOLD, color) + " → ".join(emitted)
        )

    lines.append(_c(footer, MAGENTA, color))
    return "\n".join(lines)


def format_perf_footer(
    trace: dict,
    learner: LiveLearner,
    learn_steps: list[dict],
    color: bool = True,
) -> str:
    """Format a llama.cpp-style timing and telemetry footer."""
    p_bytes = trace.get("prompt_bytes", 0)
    p_ms = trace.get("prompt_ms", 0.0)
    p_bps = trace.get("prompt_bps", 0.0)
    e_bytes = trace.get("eval_bytes", 0)
    e_ms = trace.get("eval_ms", 0.0)
    e_bps = trace.get("eval_bps", 0.0)
    spec_acc = trace.get("speculative_accepted", 0)
    mem_n = trace.get("memory_norm", 0.0)
    ep_n = trace.get("episodic_norm", 0.0)
    if learn_steps:
        last_loss = learn_steps[-1]["loss"]
        learn_str = (
            f"online: +{len(learn_steps)} steps (total={learner.steps}, loss={last_loss:.3f})"
        )
    else:
        learn_str = f"online: {learner.steps} steps (buf={len(learner.buf)}B)"
    footer = (
        f"[ prompt eval: {p_bytes} B in {p_ms:.1f} ms ({p_bps:.1f} B/s) | "
        f"eval: {e_bytes} B (spec={spec_acc}B) in {e_ms:.1f} ms ({e_bps:.1f} B/s) | "
        f"||M||_F: {mem_n:.2f} | ||E||_F: {ep_n:.2f} | {learn_str} ]"
    )
    return _c(footer, GRAY, color)


def format_nodes_inspector(
    config: Config,
    params: dict,
    last_trace: dict | None,
    color: bool = True,
) -> str:
    """Detailed Graph-GRU node routing, readout norm, and topology inspector (`/nodes`)."""
    n_nodes = 1 if config.architecture == "gru" else config.nodes
    routes = (
        last_trace.get("routes_mean", [1.0 / n_nodes] * n_nodes)
        if last_trace
        else [1.0 / n_nodes] * n_nodes
    )
    node_norms = last_trace.get("node_norms", [0.0] * n_nodes) if last_trace else [0.0] * n_nodes
    read_k = np.asarray(params.get("read_kernel", np.zeros((n_nodes, config.width, config.width))))
    rk_norms = np.linalg.norm(read_k, axis=(1, 2))
    sat = (
        node_saturation(routes, read_k)
        if config.architecture == "graph"
        else {"entropy_frac": 1.0, "idle": 0, "idle_by_routing": 0, "peak_over_ideal": 1.0}
    )
    edges = (
        config.edges if config.edges is not None else topology(n_nodes, min(config.degree, n_nodes))
    )
    hdr = "╭─ 🕸️  Graph-GRU Node & Topology Inspector (/nodes) " + "─" * 17 + "╮"
    ftr = "╰" + "─" * 69 + "╯"
    lines = [
        _c(hdr, CYAN, color),
        (
            f"  Nodes: {n_nodes} (shared={config.shared_nodes}, "
            f"active_top_k={config.top_k_nodes}) | "
            f"Routing Balance: {sat['entropy_frac'] * 100:.1f}% (idle={sat['idle']})"
        ),
    ]
    for i in range(n_nodes):
        role = "SHARED    " if i < config.shared_nodes else "SPECIALIST"
        r_p = routes[i] if i < len(routes) else 0.0
        h_n = node_norms[i] if i < len(node_norms) else 0.0
        rk_n = float(rk_norms[i]) if i < len(rk_norms) else 0.0
        in_edges = [int(x) for x in edges[i]] if i < len(edges) else []
        lines.append(
            f"  • Node {i:2d} [{role}]  route=[{_bar(r_p, 8)}] {r_p * 100:5.1f}%  "
            f"|h|={h_n:5.2f}  |W_read|={rk_n:5.2f}  in_edges={in_edges}"
        )
    lines.append(_c(ftr, CYAN, color))
    return "\n".join(lines)


def format_memory_inspector(
    config: Config,
    last_trace: dict | None,
    color: bool = True,
) -> str:
    """Detailed fast-weight Gated DeltaNet, DoG wavelet, and MLA episodic inspector (`/memory`)."""
    hdr = "╭─ 💾 Multi-Tier Continual Memory Inspector (/memory) " + "─" * 16 + "╮"
    ftr = "╰" + "─" * 69 + "╯"
    lines = [_c(hdr, CYAN, color)]
    if not last_trace:
        lines.append("  No forward pass recorded yet. Send a chat message first.")
        lines.append(_c(ftr, CYAN, color))
        return "\n".join(lines)

    mem_n = last_trace.get("memory_norm", 0.0)
    key_norms = last_trace.get("memory_key_norms", [])
    span_norms = last_trace.get("span_norms", [])
    ep_n = last_trace.get("episodic_norm", 0.0)
    ep_slots = last_trace.get("episodic_slot_norms", [])

    lines.append(
        f"  1. Fast-Weight Gated DeltaNet ({config.memory}, {config.memory_keys} keys): "
        f"||M||_F = {mem_n:.4f}"
    )
    if key_norms:
        max_k = max(max(key_norms), 1e-6)
        chips = [f"K{i}[{_bar(v / max_k, 4)}]{v:.2f}" for i, v in enumerate(key_norms)]
        for row_start in range(0, len(chips), 4):
            lines.append("     " + "  ".join(chips[row_start : row_start + 4]))
    if span_norms:
        lines.append(
            "  2. 4-Scale Laplace / DoG Wavelet Traces: "
            + "  ".join(f"H{i}={v:.3f}" for i, v in enumerate(span_norms))
        )
    if ep_slots:
        max_e = max(max(ep_slots), 1e-6)
        ep_chips = [f"S{i}[{_bar(v / max_e, 4)}]{v:.2f}" for i, v in enumerate(ep_slots)]
        lines.append(
            f"  3. MLA Compressed Episodic Cache ({config.episodic_slots} slots): "
            f"||E||_F = {ep_n:.4f}"
        )
        for row_start in range(0, len(ep_chips), 4):
            lines.append("     " + "  ".join(ep_chips[row_start : row_start + 4]))
    lines.append(_c(ftr, CYAN, color))
    return "\n".join(lines)


def run_chat_tui(
    state,
    config: Config,
    ckpt_label: str = "live-init",
    max_bytes: int = 160,
    adapt_strength: float = 0.0,
    adapt_decay: float = 0.88,
    learn: bool = False,
    chunk: int = 16,
    show_thinking: bool = True,
    initial_context: str = "",
    save_path: Path | None = None,
    input_stream=None,
    output_stream=None,
    color: bool | None = None,
) -> tuple[object, Config]:
    """Run the interactive real-time TUI chat loop."""
    in_s = input_stream if input_stream is not None else sys.stdin
    out_s = output_stream if output_stream is not None else sys.stdout
    use_color = color if color is not None else (hasattr(out_s, "isatty") and bool(out_s.isatty()))

    learner = LiveLearner(state, config, chunk=chunk)
    learn_enabled = bool(learn)
    thinking_enabled = bool(show_thinking)
    persistent_context = initial_context.strip()

    def writeln(text: str = ""):
        out_s.write(text + "\n")
        out_s.flush()

    writeln(
        format_model_banner(
            learner.config,
            learner.state.params,
            ckpt_label=ckpt_label,
            step=int(learner.state.step),
            learn_enabled=learn_enabled,
            show_thinking=thinking_enabled,
            context_str=persistent_context,
            color=use_color,
        )
    )
    tip_msg = (
        'Tip: Type facts or questions naturally (e.g. "Mia is in the attic. Dan is with Mia. '
        'Where is Dan?"). Episodic memory & SAO compaction run automatically.'
    )
    writeln(_c(tip_msg, DIM, use_color))

    while True:
        prompt_prefix = _c("\nYou ❯ ", BOLD + BRIGHT_GREEN, use_color)
        out_s.write(prompt_prefix)
        out_s.flush()
        try:
            line = in_s.readline()
        except (KeyboardInterrupt, EOFError):
            writeln()
            break
        if not line:
            writeln()
            break
        user_text = line.strip()
        if not user_text:
            continue

        if user_text.lower() in ("quit", "exit", "q"):
            writeln(_c("Exiting MLM interactive session.", DIM, use_color))
            break

        # Standard chat turn: support inline `\n` escapes if typed literally
        clean_user = user_text.replace("\\n", "\n")

        def _handle_trace(tr: dict):
            if not thinking_enabled:
                out_s.write(_c("MLM ❯ ", BOLD + BRIGHT_CYAN, use_color))
                out_s.flush()

        def _handle_token(ch: str):
            if not thinking_enabled:
                out_s.write(ch)
                out_s.flush()

        reply, trace, learn_steps = learner.chat_turn_with_trace(
            clean_user,
            max_bytes=max_bytes,
            adapt_strength=adapt_strength,
            adapt_decay=adapt_decay,
            learn=learn_enabled,
            context=persistent_context,
            on_trace=_handle_trace,
            on_token=_handle_token,
        )

        if thinking_enabled:
            writeln(format_thinking_panel(trace, learner.config, color=use_color))
            writeln(_c("MLM ❯ ", BOLD + BRIGHT_CYAN, use_color) + reply)
        else:
            writeln()

        writeln(format_perf_footer(trace, learner, learn_steps, color=use_color))

    if save_path is not None:
        save_checkpoint(
            save_path,
            learner.state,
            learner.config,
            {"chat_steps": learner.steps},
        )
    return learner.state, learner.config


def load_or_init_chat_state(
    checkpoint: Path | None,
    config_path: Path | None = None,
    seed: int | None = None,
) -> tuple[object, Config, str]:
    """Load an `.mlm` checkpoint or initialize a fresh model from `--config` or default."""
    if checkpoint is not None and checkpoint.exists():
        state, config, _ = load_checkpoint(checkpoint)
        return state, config, str(checkpoint)
    default_cfg = Path("configs/deepseek-v1.json")
    cfg_file = (
        config_path if config_path is not None else (default_cfg if default_cfg.exists() else None)
    )
    if cfg_file is not None and cfg_file.exists():
        config = Config.model_validate_json(cfg_file.read_text())
        label = f"init({cfg_file})"
    else:
        config = Config(relational_hops=3)
        label = "init(default-3hop)"
    if seed is not None:
        config = config.model_copy(update={"seed": int(seed)})
    state = create_state(config)
    return state, config, label


def chat_main(argv: list[str] | None = None) -> None:
    """Standalone `mlm-chat` CLI entrypoint."""
    parser = argparse.ArgumentParser(
        prog="mlm-chat",
        description="Interactive real-time chat TUI for MLM with <think> telemetry.",
    )
    parser.add_argument(
        "checkpoint",
        nargs="?",
        type=Path,
        default=None,
        help="Optional path to .mlm checkpoint (initializes from --config if omitted)",
    )
    parser.add_argument(
        "--config", type=Path, default=None, help="Config JSON if no checkpoint is given"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed when initializing without a checkpoint"
    )
    parser.add_argument(
        "--message",
        type=str,
        default=None,
        help="Run a single non-interactive chat turn and exit",
    )
    parser.add_argument("--context", type=str, default="", help="Initial persistent context facts")
    parser.add_argument("--max-bytes", type=int, default=160)
    parser.add_argument("--adapt-strength", type=float, default=0.0)
    parser.add_argument("--adapt-decay", type=float, default=0.88)
    parser.add_argument(
        "--learn", action="store_true", help="Enable online continual learning during chat"
    )
    parser.add_argument("--no-learn", action="store_true", help="Disable online continual learning")
    parser.add_argument("--chunk", type=int, default=16)
    parser.add_argument(
        "--show-thinking",
        action="store_true",
        help="Show <think> telemetry in single-turn mode",
    )
    parser.add_argument(
        "--no-think", action="store_true", help="Start TUI with <think> panel hidden"
    )
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors in TUI")
    parser.add_argument("--save", type=Path, default=None, help="Save updated checkpoint on exit")
    parser.add_argument("--stats", type=Path, default=None, help="Write JSON stats on exit")
    args = parser.parse_args(argv)

    state, config, label = load_or_init_chat_state(args.checkpoint, args.config, seed=args.seed)
    learn_flag = bool(args.learn) and not bool(args.no_learn)

    if args.message is not None:
        learner = LiveLearner(state, config, chunk=args.chunk)
        clean_msg = args.message.replace("\\n", "\n")
        if args.show_thinking:
            reply, trace, steps = learner.chat_turn_with_trace(
                clean_msg,
                max_bytes=args.max_bytes,
                adapt_strength=args.adapt_strength,
                adapt_decay=args.adapt_decay,
                learn=learn_flag,
                context=args.context,
            )
            sys.stdout.write(format_thinking_panel(trace, config, color=False) + "\n")
            sys.stdout.write(reply + "\n")
            sys.stdout.write(format_perf_footer(trace, learner, steps, color=False) + "\n")
        else:
            reply, steps = learner.chat_turn(
                clean_msg,
                max_bytes=args.max_bytes,
                adapt_strength=args.adapt_strength,
                adapt_decay=args.adapt_decay,
                learn=learn_flag,
                context=args.context,
            )
            sys.stdout.write(reply)
        if args.save is not None:
            save_checkpoint(args.save, learner.state, learner.config, {"chat_steps": learner.steps})
        return

    run_chat_tui(
        state,
        config,
        ckpt_label=label,
        max_bytes=args.max_bytes,
        adapt_strength=args.adapt_strength,
        adapt_decay=args.adapt_decay,
        learn=learn_flag,
        chunk=args.chunk,
        show_thinking=not args.no_think,
        initial_context=args.context,
        save_path=args.save,
        color=False if args.no_color else None,
    )
