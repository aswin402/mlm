# Mini Learning Model (MLM)

Version **0.0.12** — an experimental byte-level language model built from small GRU networks connected by a directed graph, with DeepSeekMoE shared + specialist multi-timescale and top-$k$ sparse routing, DeepSeek-V3 auxiliary-loss-free load balancing and causal target-conditioned multi-token prediction (MTP), PonderNet-style adaptive round halting with geometric KL regularization, learned neighbor edge attention with mHC doubly-stochastic Sinkhorn mixing (`sinkhorn_edge_weights` and GLM-5.3-Flash 4-stream `sinkhorn_stream_mix`) and functional edge rewiring, Kimi Delta Attention (KDA) / GLM-5.3-Flash channel-wise + per-key liquid-decay associative memory (`memory_decay`, `channel_alpha` + `ShortConv`) with bidirectional multi-horizon Laplace traces (`symmetric_memory`) and Titans prediction-surprise gating (`surprise_gating`), DeepSeek MLA content-addressed latent episodic slots (`episodic_slots`), LFM 2.5 Linear Input-Varying (LIV) short-range causal stem convolution, $H$-hop disjoint-clause transitive relational composition (`relational_hops=3`) with `IndexShare` cross-hop sparse entity indexing, causal temporal state-overwrite, multi-attribute same-subject clause specificity, and GLM-5.3-Flash speculative multi-byte word-span decoding (`generate()`, `generate_with_trace()`), activity-modulated local node plasticity, autonomous plateau-gated node growth and pruning, held-out-governed plasticity and power-law curriculum scheduling, DeepSeek-R1 critic-free Group Relative Policy Optimization (GRPO), autonomous `SAO` episodic state-overwrite compaction and background sleep consolidation during command-free real-time chat, and hybrid Muon/AdamW optimization.

The prototype trains and can memorize tiny binding tasks. Learning remains sensitive to English replay sampling, and reliable generalization and chat are not established. Hardware-sparse kernel execution and a Transformer-replacement claim remain future research.

## Quick start

Tested on Linux x86-64, Python 3.12 and CPU. GPU and other-device support have not been validated. Install [uv](https://docs.astral.sh/uv/), then run from the repository root:

```sh
uv sync --locked
uv run mlm-chat
uv run mlm train --train data/smoke/train.txt --validation data/smoke/validation.txt --config configs/smoke.json --out runs/my-first-mlm --steps 300 --eval-every 50
uv run mlm generate runs/my-first-mlm/best.mlm --prompt 'The red ' --max-bytes 64
uv run mlm evaluate runs/my-first-mlm/best.mlm --data data/smoke/validation.txt
```

Launch the interactive `llama.cpp`-style neural console anytime with `uv run mlm-chat` (or `uv run mlm chat [checkpoint]`), which runs a command-free autonomous conversational loop (`type 'exit' or 'quit' to leave`), displays a live `<think>` telemetry panel (multi-hop clause & bridge trace, `IndexShare` entity density, speculative span accepted bytes, Graph-GRU node routing, KDA / DoG wavelet / MLA episodic norms, per-byte probabilities, and prompt/token evaluation speed), and automatically manages multi-turn episodic working memory with `SAO` state-overwrite compaction and background consolidation.

Use a fresh output directory. The bundled smoke data is a small original fixture, not a language-quality benchmark. Public English datasets are downloaded separately using the [data guide](docs/data/README.md). Trained checkpoints and cached datasets are not included in Git.

## Architecture

- Directed power-of-two stride or custom-rewired graph (`edges` in `Config`, `rewire_edges`) of recurrent GRU nodes with synchronous or learned state-conditioned neighbor attention (`edge_aggregation="sum" | "attention"`) and optional PonderNet-style adaptive halting across message-passing rounds (`adaptive_halt=True`, `ponder_beta`, `halt_prior`).
- Learned node update routing supporting dense, uniform, hierarchical group/child softmax, and multi-timescale surprise-gated updates (`node_update="dense" | "uniform" | "hierarchical" | "timescale"`), straight-through top-$k$ sparse node activation (`top_k_nodes`), detached trunk state routing (`stop_gradient`), and cross-window EMA load balancing (`running_usage`).
- Associative fast-weight memory supporting both collapsed delta-rule (`memory="delta"`) and cross-node relational binding (`memory="relational"`) across separate per-node key, query, and value subspaces, with straight-through zero-init tangent gates for immediate step-0 gradient flow.
- Activity-modulated local node plasticity (`local_plasticity=True`) that scales per-node updates (including weight decay) by relative node gradient activity, plus plateau-gated saturation-triggered recombinative node growth, dead-node pruning, and learned edge rewiring (`recombine_grow_nodes`, `node_saturation`, `prune_nodes`, `maybe_grow_and_prune`, `rewire_edges`) with slice-preserving optimizer moment transfer.
- Hybrid Nesterov-Muon Newton-Schulz orthogonalized momentum for 2D/3D hidden weight matrices with dormant-slice preservation (`optimizer="muon"`) alongside rank-aware Optax AdamW (`optimizer="adamw"`), warm optimizer moment preservation (`preserve_moments=True`), linear warmup (`warmup_steps`), held-out-governed LR control (`PlasticityController`), power-law marginal-worth curriculum scheduling (`LaneSchedule`), unified autonomous continual CLI (`mlm continual`), spike-frequency adaptation decoding (`--adapt-strength`, `--adapt-decay`), online continual chat learning (`LiveLearner`, `mlm chat --learn`), and atomic resumable checkpoints.

There is no inference-time entity parser, regex answer generator or symbolic answer lookup. Synthetic task labels are used for training and grading.

## Current evidence

In the `v0.0.12` continual + GRPO + multi-hop and temporal reading evaluation (`configs/deepseek-v1.json`, 30,517 parameters), MLM achieves `100.0%` counterfactual accuracy and `100.0%` open-ended multi-byte word generation across 1-hop (`reading-v1.json`), 2-hop (`reading-v2-multihop.json`), 3-hop (`reading-v3-3hop.json`), and temporal / multi-attribute (`reading-v4-temporal.json`) suites (`144/144` open-ended generation probes across Plain and Chat formats) alongside `100.0%` continual train/dev QA accuracy and `2.1650` held-out English validation loss (see [full architecture and research inspirations](docs/research/architecture-and-inspirations-v0.0.11.md)):

| Evaluation Suite / Metric | Baseline (`v0.0.5`) | `v0.0.11` | **Current (`v0.0.12`)** |
|---|---:|---:|---:|
| 1-Hop OOD Reading (`data/evaluation/reading-v1.json`, 24 probes) | `50.0%` (`0.0%` pair) | `100.0%` (`100.0%` pair) | **`100.0%` (`100.0%` pair)** |
| 2-Hop OOD Transitive (`data/evaluation/reading-v2-multihop.json`, 16 probes) | `50.0%` (`0.0%` pair) | `100.0%` (`100.0%` pair) | **`100.0%` (`100.0%` pair)** |
| 3-Hop OOD Transitive (`data/evaluation/reading-v3-3hop.json`, 16 probes) | N/A | `100.0%` (`100.0%` pair) | **`100.0%` (`100.0%` pair)** |
| Temporal & Multi-Attribute (`data/evaluation/reading-v4-temporal.json`, 16 probes) | N/A | N/A | **`100.0%` (`100.0%` pair)** |
| Zero-Shot Open-Ended `generate()` Exact Word Match (`v1`–`v4`, Plain & Chat) | `0.0%` | `100.0%` (`112/112`) | **`100.0%` (`144/144` probes)** |
| Post-Continual Train & Dev QA Accuracy (`data/confirmation-v1`) | `75.0%` / `68.8%` | `100.0%` / `100.0%` | **`100.0%` / `100.0%`** |
| Post-Continual OOD Retention (`v1` / `v2` / `v3` / `v4` after Continual + GRPO) | `50.0%` | `100.0%` (`v1`–`v3`) | **`100.0%` / `100.0%` / `100.0%` / `100.0%`** |
| Post-Continual English Stream Validation Loss (`evaluate_stream`) | `2.7642` | `2.1650` | **`2.1650`** |

In multi-seed continual evaluations under the unified `mlm continual` loop (`configs/mini-agi-v4.json` and `configs/deepseek-v1.json`), both `mini_agi_v4` and `deepseek_v1` improve held-out QA validation loss and held-out English validation loss over `legacy_graph` across both `QA43/English44` and `QA44/English45` seed pairs:

| Seed pair | Arm | Held-out QA Val Loss ↓ | Held-out English Val Loss ↓ |
|---|---|---:|---:|
| `QA43 / English44` | `legacy_graph` | `1.7059` | `2.7642` |
| `QA43 / English44` | `mini_agi_v4` | `1.2522` (-26.6%) | `2.6768` (-3.2%) |
| `QA43 / English44` | **`deepseek_v1`** | **`0.6837` (-59.9%)** | **`2.6078` (-5.7%)** |
| `QA44 / English45` | `legacy_graph` | `1.6687` | `2.7802` |
| `QA44 / English45` | `mini_agi_v4` | `1.2456` (-25.4%) | `2.6957` (-3.0%) |
| `QA44 / English45` | **`deepseek_v1`** | **`0.7092` (-57.5%)** | **`2.6693` (-4.0%)** |

Earlier [context-contrastive replication](docs/experiments/context-replication/report.md) on the legacy architecture did not repeat the selected pilot: weight 100 learned 16/16 training answers and 4/4 groups on QA43/English44, but remained at 8/16 and 0/4 on QA44/English45, matching its control.

An earlier selected experiment held memory initialization fixed and varied QA ordering and English replay sampling:

| English replay seed | QA order 42 | QA order 43 |
|---|---:|---:|
| 43 | 16/16 training answers | 16/16 |
| 44 | 7/16 | 8/16 |

Both replay 43 runs completed all four training groups; replay 44 runs completed none. All four completed zero groups on the small reused development set. Two cells were historical, and this is exploratory acquisition evidence. It does not reverse the failed independent-seed replication result. See the [full QA/replay report](docs/experiments/schedule-components/report.md).

The [frozen replay diagnostic](docs/experiments/replay-interference/report.md) found snapshot-dependent QA-loss interference and substantial optimizer-history effects. It did not consistently show replay 44 to be more harmful than 43. The later context-objective experiments above do not establish a reliable interference fix.

The [actual-next QA→English diagnostic](docs/experiments/qa-replay-pair/report.md) restores saved sampler/carry state for five copied pairs. QA and English have mixed local effects;191 audits pass, and checkpoint files remain unchanged. Later optimizer and context-objective experiments retain their separate comparison scopes.

The [frozen answer-level analysis](docs/experiments/future-byte-answers/report.md) finds8/16training answer rankings and0/4complete groups for all six short(128QA+128English) pilot/replication endpoints, despite some lower CE. Context swaps do not change predictions. These budgets differ from the longer historical acquisition table above. All258audits pass with zero optimizer updates; the auxiliary head remains unpromoted.

## Development

```sh
uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
uv run ty check src
```

The full regression suite passes 123 tests across core graph dynamics, DeepSeekMoE shared + specialist multi-timescale and top-$k$ sparse routing, DeepSeek-V3 auxiliary-loss-free load balancing and causal MTP, adaptive round halting and geometric ponder KL regularization, detached cross-window router balance, learned edge attention with mHC Sinkhorn mixing and edge rewiring, Gated DeltaNet write-coupled bidirectional multi-horizon Laplace trace memory with LFM 2.5 per-key liquid decay and MLA episodic slots, LFM 2.5 LIV short-range causal convolution, 3-hop disjoint-clause transitive relational composition with temporal state-overwrite, multi-attribute clause specificity, and open-ended multi-byte word generation, activity-modulated local plasticity, Muon orthogonalization, slice-preserving warm optimizer moments, saturation-triggered node growth and pruning, held-out-governed plasticity and power-law lane scheduling, DeepSeek-R1 critic-free GRPO, unified `mlm continual` training, spike-frequency adaptation decoding, online continual chat learning with multi-turn episodic working memory and `/consolidate` sleep consolidation, and historical experiment audits. Separate QA/replay controls are available through `python -m mlm.plasticity_run --qa-seed ... --english-seed ...`; omitted values retain config.seed and config.seed+1. Config.seed still controls initialization and subset selection.

## Documentation

- [Documentation index](docs/README.md)
- [MLM v0.0.12 architecture, achievements, and research inspirations](docs/research/architecture-and-inspirations-v0.0.11.md)
- [Original idea and research](docs/research.md)
- [Product requirements](docs/prd.md)
- [Detailed usage and experiment history](docs/usage.md)
- [Dataset preparation and provenance](docs/data/README.md)
- [Release changelog](CHANGELOG.md)

Source lives in `src/mlm/`, tests in `tests/`, configurations in `configs/`, and original smoke/reading fixtures in `data/`. Compact experiment evidence is under `docs/evidence/`; full local runs and artifacts are ignored. Historical plan text and documentation migration records are preserved in `docs/archive/`.
