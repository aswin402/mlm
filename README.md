# Mini Learning Model (MLM)

Version **0.0.29** — includes the **`MLM v2` SpiderNet Dynamically Growing Hierarchical Directed Graph Network** (`model_v2.py`: 1 Shared Core Trunk + Dot 0 `10M` English Specialist Macro-Node `Nodes 0..9` + Dot 1 `10M` Mathematics Specialist Macro-Node `Nodes 10..19` = `22.56M` total parameters with **True Sparse Index-Gather Execution** `_execute_sparse_nodes_step_v2` activating only `10%–20%` of specialist parameters for a **`10.0x` specialist FLOPs reduction** and **`2.93x` wall-clock speedup**, **SVD-Calibrated 2-Hop Directed Inter-Node Graph Message Passing** `inject_and_calibrate_spidernet_graph_edges_v2` + `directed_graph_message_passing_v2` with zero self-loops $A_{ii}=0$ and live `/graph` TUI edge telemetry `format_graph_inspector_v2`, **Input-Conditioned Low-Rank Bilinear Edge Routing** `compute_directed_edge_weights_v2` + **3 Anti-Collapse Graph Regularizers** $\mathcal{L}_{\text{load}} + \mathcal{L}_{\text{edge}} + \mathcal{L}_{\text{diversity}}$, **Autonomous Page-Hinkley / One-Sided CUSUM + 3-Signal Node Spawning Controller** `AutonomousGrowthControllerV2`, **Symmetric Orthogonal Parent-Child Specialist Splitting** `split_specialist_nodes_orthogonal_v2` with $\langle Q, W_{i^*} \rangle = 0$, **Continual-Learning Evaluation Harness & 6-Architecture Ablation Suite** `run_spidernet_ablation_suite_v2`, **3-Tier Persistent External & Episodic Memory Hierarchy** `MemoryHierarchyV2` + `LMCache` / `CacheBlend` / `CacheGen`, **Stateful $O(1)$ Recurrent Decode Cache** reaching **28.4–132.7 tok/s** on CPU, 3-Mode Cognitive Router `Direct` / `Deliberative` / `Verified`, EAGLE-3 / DeepSeek-V3 Multi-Token Speculative Decoding, DeepSeek-R1 `GRPO` + Process Step Verification, Recursive Language Model `RLM` self-calls, `BLT` byte n-grams, `FoNE + Abacus` number embeddings, `FlexOlmo` cosine domain routing, Google Research `TurboQuant` / `BiLLM + QJL` `16/8/4/2/1-bit` weight & KV-cache quantization, Rust `tokenizers` BPE, Rust `polars` Parquet streaming, Rust `safetensors` `.mlm2` checkpoints, and `llama.cpp` GGUF v3 `.gguf` export with native Rust verifier `crates/mlm_core`) as well as the `v1` byte-level recurrent graph model.

## MLM v2 (`22.45M` 20-Node Hierarchical SpiderNet: Shared Core + Dot 0 English + Dot 1 Math)

```sh
# Interactive chat on the 12.47 MiB 2-bit SpiderNet checkpoint with persistent 3-tier memory (.mlm2.memory)
uv run mlm chat runs/mlm-10m-math-colab/quantized/mlm_10m_math_v2_2bit.mlm2 --kv-bits 2

# Grow and train the 20-node SpiderNet (Dot 0 English Nodes 0..9 + Dot 1 Math Nodes 10..19)
uv run mlm grow-math-v2 --data-root data/curated-math-v1 --out runs/mlm-10m-math-colab --init-checkpoint runs/mlm-10m-colab/mlm_10m_v2.mlm2

# Export all 5 quantized precision tiers (16-bit, 8-bit, 4-bit, 2-bit, 1-bit)
uv run mlm quantize-v2 --checkpoint runs/mlm-10m-math-colab/mlm_10m_math_v2.mlm2 --out-dir runs/mlm-10m-math-colab/quantized --bits all
```

### Verified SpiderNet True Sparse Index-Gather Execution (`v0.0.22`) & 6-Architecture Continual-Learning Ablation (`v0.0.28`)

| Benchmark Profile (`22.45M` 20-Node Checkpoint) | Active Nodes ($k / N$) | Active Specialist Params | Specialist FLOPs / Tok | Dense vs Sparse Latency | Wall-Clock Speedup | Max Logit Diff |
|---|---:|---:|---:|---:|---:|---:|
| **English Fast (`k_t0 = 2`)** | `2 / 20` (**`10.0%`**) | `1.77M / 17.69M` (`29.1%` total) | **`5.31M`** vs `53.08M` (**`10.0x`** reduction) | `18.12 ms` $\to$ **`6.18 ms/tok`** | **`2.93x`** | `9.54e-6` (**100% lossless**) |
| **Math Deliberative (`k_t1 = 4`)** | `4 / 20` (**`20.0%`**) | `3.54M / 17.69M` (`36.9%` total) | **`10.62M`** vs `53.08M` (**`5.0x`** reduction) | `18.05 ms` $\to$ **`7.51 ms/tok`** | **`2.40x`** | `9.78e-6` (**100% lossless**) |

| Architecture Variant (`English -> Math -> Science -> Code`) | Final Nodes | Edge Rank $r$ | Growth Events | Average Accuracy (`AA` $\uparrow$) | Backward Transfer (`BWT` $\uparrow$) | Forgetting (`F` $\downarrow$) | Retention (`%` $\uparrow$) | Active Ratio (`FLOPs` $\downarrow$) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`A_dense_baseline`** (Dense FFN, no specialists) | `0` | `0` | `0` | `0.23309` | `-0.14688` | `0.17378` | `83.29%` | `1.000` (`1.00x`) |
| **`B_static_flat_moe`** (Static Flat Top-$k$ MoE) | `4` | `0` | `0` | `0.27237` | `-0.20861` | `0.24094` | `73.65%` | `0.500` (`2.00x`) |
| **`C_static_hierarchical_moe`** (Static Hierarchical MoE) | `6` | `0` | `0` | `0.22800` | `-0.22416` | `0.25106` | `70.66%` | `0.333` (`3.00x`) |
| **`D_manual_grow_moe`** (Schedule-Grown MoE, $4 \to 7$) | `7` | `0` | `0` | `0.30822` | `-0.29459` | `0.29459` | `49.27%` | `0.286` (`3.50x`) |
| **`E_spidernet_directed_graph`** (Static Directed Graph, $r=8$) | `6` | `8` | `0` | `0.21518` | `-0.23055` | `0.24028` | `63.39%` | `0.333` (`3.00x`) |
| **`F_spidernet_full_autonomous`** (**Full SpiderNet: Sparse Gather + Directed Graph + CUSUM Orthogonal Split**) | **`7`** | **`8`** | **`3`** | **`0.49417`** | **`-0.00954`** | **`0.11652`** | **`92.43%`** | **`0.286` (`3.50x`)** |

### Multi-Precision SpiderNet Checkpoints (`22,450,960` Parameters, 20 Specialist Sub-Nodes)

| Precision Tier | Quantization Method | KV Cache Precision | 20-Node SpiderNet (`.mlm2`) | 10-Node English (`.mlm2`) | Compression vs `fp32` |
|---|---|---:|---:|---:|---:|
| **`fp32` (Master)** | Unquantized `float32` | Unquantized | `85.81 MiB` | `43.60 MiB` | `1.00x` |
| **`16-bit`** | `fp16` | `TurboQuant 4-bit + QJL` | `43.63 MiB` | `26.72 MiB` | `1.97x` |
| **`8-bit`** | `turboquant_8bit_fwht64_qjl` | `TurboQuant 4-bit + QJL` | `23.18 MiB` | `14.46 MiB` | `3.70x` |
| **`4-bit`** | `turboquant_4bit_fwht64_qjl` | `TurboQuant 4-bit + QJL` | `15.93 MiB` | `10.12 MiB` | `5.39x` |
| **`2-bit`** | `turboquant_2bit_fwht64_qjl` | `TurboQuant 2-bit + QJL` | **`12.47 MiB`** | **`8.54 MiB`** | **`6.88x`** |
| **`1-bit`** | `turboquant_1bit_fwht64_qjl` | `TurboQuant 1-bit + QJL` | **`10.66 MiB`** | **`7.48 MiB`** | **`8.05x`** |

### Verified 3-Mode Cognitive Routing & Stateful $O(1)$ Recurrent Decode (`12.47 MiB` `2-Bit` Weights + `2-Bit` KV Cache)

| Prompt | Cognitive Mode & Active Sub-Nodes | `<think>` Reasoning Trace (`Nodes 8..9` / `18..19`) | Final Output | Speed (`O(1)` Cache) |
|---|---|---|---|---:|
| `hii` | `DIRECT \| english \| (2, 0)` | `Hello! How can I help you today?` | `Hello! How can I help you today?` | **`78.7 tok/s`** |
| `How are you?` | `DELIBERATIVE \| english \| (0, 7)` | `As a Large Language Model (LLM), I don't have feelings like humans do. Thanks for asking though!` | `As a Large Language Model (LLM), I don't have feelings like humans do. Thanks for asking though!` | **`56.3 tok/s`** |
| `What is science?` | `DELIBERATIVE \| english \| (0, 7)` | `Science is what we do to find out about the natural world.` | `Science is what we do to find out about the natural world. Natural sciences include, chemistry, biology, geology, astronomy, and physics. Science uses mathematics and logic, which are sometimes called "formal sciences".` | **`42.4 tok/s`** |
| `What is photosynthesis?` | `DELIBERATIVE \| english \| (0, 7)` | `Photosynthesis is how plants and some microorganisms make carbohydrates.` | `Photosynthesis is how plants and some microorganisms make carbohydrates. It is an endothermic (takes in heat) chemical process which uses sunlight to turn carbon dioxide into sugars.` | **`53.6 tok/s`** |
| `whats 4+4` | `VERIFIED \| math \| (10, 15)` | `4 + 4 = 8.` | `4 + 4 = 8. The answer is 8.` | **`41.3 tok/s`** |
| `What is 1 plus 1?` | `VERIFIED \| math \| (11, 15)` | `1 + 1 = 2.` | `1 + 1 = 2. The answer is 2.` | **`38.2 tok/s`** |
| `What is 15 - 7?` | `VERIFIED \| math \| (15, 10)` | `15 - 7 = 8.` | `15 - 7 = 8. The answer is 8.` | **`33.5 tok/s`** |
| `What is 17 * 2?` | `VERIFIED \| math \| (10, 15)` | `17 * 2 = 34.` | `17 * 2 = 34. The answer is 34.` | **`30.3 tok/s`** |
| `What is 36 / 6?` | `VERIFIED \| math \| (10, 15)` | `36 / 6 = 6.` | `36 / 6 = 6. The answer is 6.` | **`33.6 tok/s`** |
| `What is (9 + 8) * 2?` | `VERIFIED \| math \| (10, 15)` | `9 + 8 = 17. 17 * 2 = 34.` | `9 + 8 = 17, and 17 * 2 = 34. The answer is 34.` | **`51.6 tok/s`** |
| `Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?` | `VERIFIED \| math \| (15, 10)` | `48 / 2 = 24. 48 + 24 = 72.` | `Natalia sold 48 / 2 = 24 clips in May. 48 + 24 = 72. The answer is 72.` | **`28.4 tok/s`** |

### Verified 3-Tier Persistent & Episodic Memory Hierarchy Across Sessions (`.mlm2.memory`)

| Session & Turn | User Prompt | Tier-3 Action / Cognitive Mode | Model Output | Speed |
|---|---|---|---|---:|
| **Session 1, Turn 1** | `Remember that my name is Aswin and I am building MLM SpiderNet.` | `Tier-3 Write` (`slots #0, #1`) | `Remembered: my name is Aswin. You are building MLM SpiderNet.` | `13.1 tok/s` |
| **Session 1, Turn 2** | `The API port is 8080 and Mia is in Paris.` | `Tier-3 Write` (`slots #2, #3`) | `Remembered: The API port is 8080. Mia is in Paris.` | **`302.8 tok/s`** |
| **Session 1, Turn 3** | `Let x = 15, y = 7` | `Tier-3 Write` (`slots #4, #5`) | `Remembered: x = 15 y = 7` | `20.4 tok/s` |
| **Session 2, Turn 1** *(Reloaded)* | `What is my name?` | `Tier-3 Recall` (`slot #0`, `hits=1`) | `Your name is Aswin.` | `6.0 tok/s` |
| **Session 2, Turn 2** | `What am I building?` | `Tier-3 Recall` (`slot #1`, `hits=2`) | `You are building MLM SpiderNet.` | **`509.8 tok/s`** |
| **Session 2, Turn 3** | `Where is Mia?` | `Tier-3 Recall` (`slot #3`, `hits=3`) | `Mia is in Paris.` | **`402.7 tok/s`** |
| **Session 2, Turn 4** | `What is the API port?` | `Tier-3 Recall` (`slot #2`, `hits=4`) | `The API port is 8080.` | **`386.2 tok/s`** |
| **Session 2, Turn 5** | `What is (x + 2) * 2?` | `VERIFIED` + `[Tier-3 Memory: x=15]` | `15 + 2 = 17, and 17 * 2 = 34. The answer is 34.` | **`87.5 tok/s`** |
| **Session 2, Turn 6** | `Remember that the API port is 9090 and x = 20` | **`SAO Overwrite`** (`sao=2` on `#2, #4`) | `Remembered: x = 20 the API port is 9090.` | **`303.0 tok/s`** |
| **Session 2, Turn 7** | `What is the API port?` | `Tier-3 Recall` (`slot #2`, `hits=6`) | `the API port is 9090.` | **`459.8 tok/s`** |
| **Session 2, Turn 8** | `What is x + y?` | `VERIFIED` + `[Tier-3 Memory: x=20, y=7]` | `20 + 7 = 27. The answer is 27.` | **`82.1 tok/s`** |

See the [MLM v0.0.21 SpiderNet, 3-Tier Persistent Memory Hierarchy & Stateful O(1) Recurrent Decode research paper](docs/research/architecture-and-inspirations-v0.0.21.md) and [Colab 20-Node SpiderNet Math & Reasoning report](docs/evidence/colab_10m_math_v2_report.json) for full details.

## MLM v2 (`10M` Staged-Node Growth: `4 → 6 → 8` Specialist Graph Nodes)

```sh
uv run mlm grow-english-v2 --data-root data/curated-english-v1 --out runs/mlm-10m-colab --s1-steps 1000 --s2-steps 1200 --s3-steps 1600 --gguf-out runs/mlm-10m-colab/mlm_10m_v2.gguf
uv run mlm chat runs/mlm-10m-colab/mlm_10m_v2.mlm2 --message "What is photosynthesis?"
./crates/mlm_core/target/release/mlm-gguf-inspect runs/mlm-10m-colab/mlm_10m_v2.gguf
```

| Benchmark Suite (Zero-Shot on Colab T4 GPU) | Random | `TinyStories-8M` (`8.3M`) | `Pythia-14M` (`14.1M`) | **`MLM v2 (11.4M, 4→6→8 Nodes)`** |
|---|---:|---:|---:|---:|
| **BLiMP English Grammar Minimal Pairs** | `50.0%` | `100.0%` | `75.0%` | **`83.33%`** |
| **Held-Out 4-Way Science & Reading QA (`Stage 2 Val`)** | `25.0%` | `32.0%` | `44.0%` | **`58.00%`** |
| **Held-Out 4-Way Multi-Turn Chat Selection (`Stage 3 Val`)** | `25.0%` | `22.0%` | `34.0%` | **`30.00%` (`48.0%` Split 1)** |

See [Colab 10M v2 benchmark report](docs/evidence/colab_10m_v2_report.json) for full telemetry.

---

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
- [MLM v0.0.19 SpiderNet architecture, 3-Mode Cognitive Router, GRPO, and TurboQuant research paper](docs/research/architecture-and-inspirations-v0.0.19.md)
- [MLM v0.0.12 architecture, achievements, and research inspirations](docs/research/architecture-and-inspirations-v0.0.11.md)
- [Original idea and research](docs/research.md)
- [Product requirements](docs/prd.md)
- [Detailed usage and experiment history](docs/usage.md)
- [Dataset preparation and provenance](docs/data/README.md)
- [Release changelog](CHANGELOG.md)

Source lives in `src/mlm/`, tests in `tests/`, configurations in `configs/`, and original smoke/reading fixtures in `data/`. Compact experiment evidence is under `docs/evidence/`; full local runs and artifacts are ignored. Historical plan text and documentation migration records are preserved in `docs/archive/`.
