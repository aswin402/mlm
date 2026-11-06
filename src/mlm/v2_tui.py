"""Interactive llama.cpp-style Neural Console & Telemetry Inspectors for MLM v2 SpiderNet."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import jax
import numpy as np

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


def _c(text: str, code: str, color: bool = True) -> str:
    return f"{code}{text}{RESET}" if color else text


def _bar(val: float, width: int = 8) -> str:
    v = float(np.clip(val, 0.0, 1.0))
    filled = int(round(v * width))
    return "█" * filled + "░" * (width - filled)


def _parameter_count_v2(params: dict) -> int:
    return int(sum(int(np.prod(v.shape)) for v in jax.tree.leaves(params)))


def format_memory_inspector_v2(
    mem_h,
    color: bool = True,
) -> str:
    """Render the 3-Tier Persistent External & Episodic Memory Hierarchy Inspector (`/memory`)."""
    info = mem_h.summary_dict()
    t1 = info["tier1_immediate_kv"]
    t2 = info["tier2_working_deltanet"]
    t3 = info["tier3_persistent_episodic"]
    lmc = info["lmcache_hybrid"]
    hdr = "╭─ 💾 MLM v2 3-Tier Persistent Memory Hierarchy Inspector (/memory) " + "─" * 4 + "╮"
    ftr = "╰" + "─" * 71 + "╯"
    lines = [
        _c(hdr, CYAN, color),
        (
            f"  {_c('Tier 1 (Immediate Context):', BOLD + BRIGHT_GREEN, color)} "
            f"{t1['mechanism']} (TurboQuant {t1['kv_bits']}-bit + CacheGen + QJL)"
        ),
        (
            f"     active_tokens={t1['active_tokens']} | "
            f"||KV||_F={t1['kv_norm']:.3f} | ||K_idx||_F={t1['dsa_idx_norm']:.3f} | "
            f"LMCache(chunks={lmc['cached_chunks']}, hits={lmc['prefix_hits'] + lmc['full_hits']}, "
            f"saved={lmc['saved_prefill_tokens']} tok)"
        ),
        (
            f"  {_c('Tier 2 (Working State):    ', BOLD + BRIGHT_YELLOW, color)} "
            f"{t2['mechanism']} (decay={t2['working_decay']:.2f})"
        ),
        f"     ||S_work||_F={t2['delta_norm']:.3f}",
    ]
    lnorms = t2.get("layer_norms", [])
    if lnorms:
        max_ln = max(max(lnorms), 1e-6)
        chips = [f"L{i}[{_bar(v / max_ln, 4)}]{v:.2f}" for i, v in enumerate(lnorms)]
        for r_idx in range(0, len(chips), 4):
            lines.append("     " + "  ".join(chips[r_idx : r_idx + 4]))
    lines.append(
        f"  {_c('Tier 3 (Persistent Bank):  ', BOLD + BRIGHT_CYAN, color)} "
        f"{t3['mechanism']}"
    )
    lines.append(
        f"     slots={t3['occupied_slots']}/{t3['max_slots']} | "
        f"packed={t3['packed_bytes'] / 1024.0:.1f} KiB "
        f"({t3['compression_vs_fp32']:.2f}x vs fp32) | "
        f"hits={t3['total_hits']} | SAO_overwrites={t3['total_overwrites']}"
    )
    if t3.get("storage_path"):
        lines.append(f"     disk_store={t3['storage_path']}")
    if mem_h.slots:
        for sl in mem_h.slots:
            lines.append(
                f"     • [slot #{sl['slot_id']:02d} | {sl['kind']}] "
                f"{sl['subject']} {sl['relation']} {sl['value']} "
                f"(hits={sl.get('hits', 0)}, sao={sl.get('overwrites', 0)})"
            )
    else:
        lines.append("     (No persistent episodic slots stored yet)")
    lines.append(_c(ftr, CYAN, color))
    return "\n".join(lines)


def format_graph_inspector_v2(
    params_v2: dict,
    cfg_v2=None,
    tok_v2=None,
    prompt_text: str = "<|user|>\nHi\n<|assistant|>\n",
    color: bool = True,
) -> str:
    """Render the SpiderNet v0.0.29 Directed Graph & Edge Inspector (`/graph`)."""
    from mlm.model_v2 import get_spidernet_node_role_v2, inspect_directed_graph_edges_v2

    if cfg_v2 is None or "n_nodes" in params_v2:
        g_info = params_v2
        edge_rank = g_info.get("edge_rank", 16)
        top_k_edges = g_info.get("top_k_edges", 2)
    else:
        g_info = inspect_directed_graph_edges_v2(params_v2, cfg_v2, tok_v2, prompt_text)
        edge_rank = cfg_v2.graph_edge_rank
        top_k_edges = cfg_v2.graph_edge_top_k
    hdr = "╭─ 🕸️  MLM v2 SpiderNet Directed Graph & Edge Inspector (/graph) " + "─" * 6 + "╮"
    ftr = "╰" + "─" * 71 + "╯"
    n_nodes = int(g_info["n_nodes"])
    k_act = int(g_info["k_active"])
    flops_red = float(g_info["flops_reduction_x"])
    act_sub = tuple(g_info["active_subnodes"])  # type: ignore[arg-type]
    sub_str = ", ".join(f"N{i}({get_spidernet_node_role_v2(i)})" for i in act_sub)
    lines = [
        _c(hdr, CYAN, color),
        (
            f"  {_c('Graph Topology:', BOLD + BRIGHT_GREEN, color)} "
            f"nodes={n_nodes} | active_k={k_act} ({flops_red:.1f}x FLOPs reduction) | "
            f"edge_rank={edge_rank} | top_k_edges={top_k_edges}"
        ),
        (
            f"  {_c('Cognitive Route:', BOLD + BRIGHT_YELLOW, color)} "
            f"mode={str(g_info['mode']).upper()} | domain={g_info['domain']} "
            f"(seq_dl={g_info['seq_dl']}) | primary=[{sub_str}]"
        ),
        (
            f"  {_c('2-Hop Message Passing (N_i -> N_j, A_ii=0):', BOLD + BRIGHT_CYAN, color)} "
            f"collab_gate={g_info['collab_gate_mean']:.4f} | "
            f"mean_||M||_2={g_info['mean_msg_norm']:.4f}"
        ),
    ]
    top_edges = g_info.get("top_edges", [])
    if top_edges:
        for e in top_edges:
            a_val = float(e["a_ij"])
            lines.append(
                f"     • N{e['src']:02d} [{e['src_role']:<15s}] "
                f"──[{_bar(a_val, 6)}] a_ij={a_val:.3f} (flux={e['flux']:.3f})──► "
                f"N{e['dst']:02d} [{e['dst_role']}]"
            )
    else:
        lines.append("     (No directed inter-node edges active)")
    lnorms = g_info.get("layer_msg_norms", [])
    if lnorms:
        max_m = max(max(lnorms), 1e-6)
        chips = [f"L{i}[{_bar(v / max_m, 4)}]{v:.3f}" for i, v in enumerate(lnorms)]
        lines.append(f"  {_c('Per-Layer Message Norms ||M^(l)||_2:', BOLD, color)}")
        for r_idx in range(0, len(chips), 4):
            lines.append("     " + "  ".join(chips[r_idx : r_idx + 4]))
    lines.append(_c(ftr, CYAN, color))
    return "\n".join(lines)


def format_multihop_inspector_v2(
    params_v2: dict,
    cfg_v2,
    tok_v2,
    prompt_text: str | None = None,
    color: bool = True,
) -> str:
    """Render the SpiderNet v0.0.42 Spectral Multi-Hop Chain Inspector (`/hops`)."""
    from mlm.model_v2 import trace_compositional_3hop_chain_v2

    raw_info = trace_compositional_3hop_chain_v2(params_v2, cfg_v2, tok_v2, prompt_text)
    hop_info = (
        raw_info["hop_3_compositional_code_math"]
        if "hop_3_compositional_code_math" in raw_info
        else raw_info
    )
    hdr = "╭─ 🔗 MLM v2 Spectral Multi-Hop Chain Inspector (/hops) " + "─" * 16 + "╮"
    ftr = "╰" + "─" * 71 + "╯"
    act_hops = int(hop_info["active_hops"])  # type: ignore[index]
    edge_stages = int(hop_info["edge_stages"])  # type: ignore[index]
    spec_rank = float(hop_info["spectral_effective_rank"])  # type: ignore[index]
    a_h1 = float(hop_info["a_hop1"])  # type: ignore[index]
    a_h2 = float(hop_info["a_hop2"])  # type: ignore[index]
    flux = float(hop_info["path_flux"])  # type: ignore[index]
    pathway = str(hop_info["pathway"])  # type: ignore[index]
    lines = [
        _c(hdr, CYAN, color),
        (
            f"  {_c('Spectral Halting:', BOLD + BRIGHT_GREEN, color)} "
            f"R_eff(X)={spec_rank:.2f} | active_hops={act_hops} (3-Hop Chain) | "
            f"edge_stages={edge_stages}"
        ),
        (
            f"  {_c('Chain Pathway:   ', BOLD + BRIGHT_YELLOW, color)} "
            f"{pathway}"
        ),
        (
            f"  {_c('Edge Flux:       ', BOLD + BRIGHT_CYAN, color)} "
            f"a_hop1=[{_bar(a_h1, 6)}] {a_h1:.4f} | "
            f"a_hop2=[{_bar(a_h2, 6)}] {a_h2:.4f} | "
            f"path_flux={flux:.4f}"
        ),
        _c(ftr, CYAN, color),
    ]
    return "\n".join(lines)


def run_chat_tui_v2(
    checkpoint_path: Path,
    max_new_tokens: int = 64,
    show_thinking: bool = True,
    initial_context: str = "",
    input_stream=None,
    output_stream=None,
    color: bool | None = None,
    kv_cache_bits: int | None = None,
    memory_path: Path | None = None,
    reset_memory: bool = False,
) -> None:
    """Interactive llama.cpp-style TUI for MLM v2 (.mlm2) checkpoints."""
    from mlm.model_v2 import (
        MemoryHierarchyV2,
        generate_v2,
        inspect_directed_graph_edges_v2,
        load_checkpoint_v2,
    )
    from mlm.train_v2 import format_chat_prompt_v2

    in_s = input_stream if input_stream is not None else sys.stdin
    out_s = output_stream if output_stream is not None else sys.stdout
    use_color = color if color is not None else (hasattr(out_s, "isatty") and bool(out_s.isatty()))

    params_v2, cfg_v2, tok_v2, manifest = load_checkpoint_v2(checkpoint_path)
    if kv_cache_bits is not None:
        cfg_v2 = cfg_v2.model_copy(update={"kv_cache_bits": int(kv_cache_bits)})
    eff_mem_path = (
        Path(memory_path)
        if memory_path is not None
        else checkpoint_path.with_suffix(".mlm2.memory")
    )
    if reset_memory and eff_mem_path.exists():
        eff_mem_path.unlink(missing_ok=True)
    mem_h = MemoryHierarchyV2(
        cfg_v2,
        max_slots=64,
        episodic_bits=2,
        storage_path=eff_mem_path,
    )
    q_info = manifest.get("quantization", {})
    w_bits = int(
        q_info.get("bits", manifest.get("metadata", {}).get("quantization_bits", 32))
    )
    q_method = q_info.get("method", manifest.get("metadata", {}).get("quantization_method", ""))
    st_bytes = int(
        manifest.get(
            "safetensors_bytes",
            sum(int(np.asarray(p).nbytes) for p in jax.tree.leaves(params_v2)),
        )
    )
    st_mib = st_bytes / (1024.0 * 1024.0)
    if q_method:
        size_desc = f"{st_mib:.2f} MiB {w_bits}-bit {q_method}"
    else:
        first_leaf = next(iter(jax.tree.leaves(params_v2)), None)
        dt_str = str(first_leaf.dtype) if first_leaf is not None else "float32"
        size_desc = f"{st_mib:.2f} MiB {dt_str}"
    kv_str = (
        f"TurboQuant {cfg_v2.kv_cache_bits}-bit + QJL"
        if cfg_v2.kv_cache_bits in (1, 2, 4, 8)
        else "unquantized"
    )
    n_params = _parameter_count_v2(params_v2)
    step = int(manifest.get("step", 0))
    devs = ", ".join(str(d) for d in jax.devices())

    def writeln(text: str = ""):
        out_s.write(text + "\n")
        out_s.flush()

    top_border = "╭" + "─" * 78 + "╮"
    mid_border = "├" + "─" * 78 + "┤"
    bot_border = "╰" + "─" * 78 + "╯"
    arch_str = _c(cfg_v2.arch_version.upper(), BRIGHT_GREEN, use_color)
    param_str = _c(f"{n_params:,}", BRIGHT_YELLOW, use_color)
    banner = [
        _c(top_border, CYAN, use_color),
        _c("│ ", CYAN, use_color)
        + _c("⚡ MLM v2 Interactive Neural Console", BOLD + BRIGHT_CYAN, use_color)
        + _c("  (Hybrid Liquid-Delta + Rust BPE)", DIM, use_color),
        _c(mid_border, CYAN, use_color),
        (
            f"  {_c('model_load_meta:', BOLD, use_color)}  arch={arch_str} | "
            f"params={param_str} ({size_desc}) | kv={kv_str} | "
            f"step={step} | src={checkpoint_path}"
        ),
        (
            f"  {_c('system_backend: ', BOLD, use_color)}  jax_devices=[{devs}] | "
            f"vocab={cfg_v2.vocab_size} Rust BPE subwords | format=SafeTensors (.mlm2)"
        ),
        (
            f"  {_c('hybrid_stack:   ', BOLD, use_color)}  layers={cfg_v2.n_layers} | "
            f"d_model={cfg_v2.d_model} | heads={cfg_v2.n_heads} (kv={cfg_v2.n_kv_heads}) | "
            f"core_ffn={cfg_v2.ffn_dim} | shortconv_k={cfg_v2.conv_kernel}"
        ),
        (
            f"  {_c('graph_nodes:    ', BOLD, use_color)}  specialist_nodes={cfg_v2.n_experts} "
            f"(active_top_k={cfg_v2.top_k_experts}, mature_protected={cfg_v2.mature_experts}, "
            f"node_dim={cfg_v2.expert_ffn_dim})"
        ),
        (
            f"  {_c('memory_tiers:   ', BOLD, use_color)}  "
            f"T1=DSA+{kv_str} | T2=Gated DeltaNet Fast-Weights | "
            f"T3=Episodic ({len(mem_h.slots)}/{mem_h.max_slots} slots, 2b TurboQuant)"
        ),
        _c(mid_border, CYAN, use_color),
        (
            f"  {_c('Autonomous Chat:', BOLD, use_color)} "
            f"Type messages naturally ('/graph', '/hops', '/memory', '/forget', 'exit')"
        ),
        _c(bot_border, CYAN, use_color),
    ]
    writeln("\n".join(banner))
    last_prompt_for_graph = format_chat_prompt_v2("What is (9 + 8) * 2?", initial_context)

    while True:
        out_s.write(_c("\nYou ❯ ", BOLD + BRIGHT_GREEN, use_color))
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
        if user_text.lower() in ("quit", "exit", "q", "/exit", "/quit"):
            writeln(_c("Exiting MLM v2 interactive session.", DIM, use_color))
            break
        if user_text.lower() in ("/graph", "/nodes"):
            writeln(
                format_graph_inspector_v2(
                    params_v2, cfg_v2, tok_v2, last_prompt_for_graph, color=use_color
                )
            )
            continue
        if user_text.lower() == "/hops":
            writeln(
                format_multihop_inspector_v2(
                    params_v2, cfg_v2, tok_v2, last_prompt_for_graph, color=use_color
                )
            )
            continue
        if user_text.lower() == "/memory":
            writeln(format_memory_inspector_v2(mem_h, color=use_color))
            continue
        if user_text.lower() == "/forget":
            mem_h.clear()
            writeln(
                _c(
                    "Cleared Tier-2 working fast weights and Tier-3 persistent episodic bank.",
                    DIM,
                    use_color,
                )
            )
            continue

        prompt = format_chat_prompt_v2(user_text.replace("\\n", "\n"), initial_context)
        last_prompt_for_graph = prompt
        p_toks = len(tok_v2.encode(prompt, add_bos=True, add_eos=False))
        t0 = time.perf_counter()
        raw_reply = generate_v2(
            params_v2,
            cfg_v2,
            tok_v2,
            prompt,
            max_new_tokens=max_new_tokens,
            include_think=True,
            memory_hierarchy=mem_h,
        ).strip()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        r_toks = max(1, len(tok_v2.encode(raw_reply, add_bos=False, add_eos=False)))
        tps = r_toks / max(elapsed_ms / 1000.0, 1e-6)

        think_trace = ""
        reply = raw_reply
        if "<|/think|>" in raw_reply:
            t_part, a_part = raw_reply.split("<|/think|>", 1)
            think_trace = t_part.replace("<|think|>", "").strip()
            reply = a_part.replace("<|think|>", "").strip()
        else:
            reply = raw_reply.replace("<|think|>", "").replace("<|/think|>", "").strip()

        if show_thinking:
            g_meta = inspect_directed_graph_edges_v2(
                params_v2, cfg_v2, tok_v2, mem_h.last_resolved_prompt or prompt
            )
            hdr = "╭─ 🧠 MLM v2 Hybrid Telemetry (<think>) " + "─" * 30 + "╮"
            ftr = "╰" + "─" * 69 + "╯"
            writeln(_c(hdr, MAGENTA, use_color))
            writeln(
                _c("│ ", MAGENTA, use_color)
                + _c("Stack: ", BOLD, use_color)
                + f"{cfg_v2.n_layers} layers (LFM2 ShortConv + Parallel Gated DeltaNet + GQA) | "
                + f"weights={w_bits}b | kv={kv_str}"
            )
            writeln(
                _c("│ ", MAGENTA, use_color)
                + _c("Routing: ", BOLD, use_color)
                + f"mode={str(g_meta['mode']).upper()} | domain={g_meta['domain']} | "
                + f"subnodes={g_meta['active_subnodes']}"
            )
            writeln(
                _c("│ ", MAGENTA, use_color)
                + _c("Graph: ", BOLD, use_color)
                + f"{g_meta['summary_str']}"
            )
            writeln(
                _c("│ ", MAGENTA, use_color)
                + _c("Memory: ", BOLD, use_color)
                + f"T1(KV)={mem_h.tier1_kv_bits}b[{mem_h.tier1_active_tokens} tok, "
                + f"||KV||={mem_h.tier1_kv_norm:.1f}] | "
                + f"T2(DeltaNet ||S||_F)={mem_h.tier2_delta_norm:.2f} | "
                + f"T3(Episodic)={len(mem_h.slots)}/{mem_h.max_slots} slots "
                + f"(hits={mem_h.total_hits}, sao={mem_h.total_overwrites}) | "
                + f"LMCache(saved={mem_h.lmcache.saved_prefill_tokens} tok)"
            )
            if think_trace:
                writeln(
                    _c("│ ", MAGENTA, use_color)
                    + _c("Thought: ", BOLD + BRIGHT_CYAN, use_color)
                    + think_trace
                )
            writeln(
                _c("│ ", MAGENTA, use_color)
                + _c("Subwords: ", BOLD, use_color)
                + f"prompt={p_toks} BPE tokens → generated={r_toks} BPE tokens"
            )
            writeln(_c(ftr, MAGENTA, use_color))

        writeln(_c("MLM ❯ ", BOLD + BRIGHT_CYAN, use_color) + reply)
        footer = (
            f"[ prompt: {p_toks} tok | eval: {r_toks} tok in {elapsed_ms:.1f} ms "
            f"({tps:.1f} tok/s) | vocab={cfg_v2.vocab_size} BPE ]"
        )
        writeln(_c(footer, GRAY, use_color))
