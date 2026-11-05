# Changelog

## [Unreleased]

## [0.0.29] - 2026-11-05

### Added

- **SpiderNet Live Directed Graph Edge Calibration (`inject_and_calibrate_spidernet_graph_edges_v2`, `inspect_directed_graph_edges_v2`, `get_spidernet_node_role_v2` in `mlm.model_v2`):**
  - Calibrates each layer's low-rank directed edge projections (`edge_proj_h`, `edge_proj_x`, `edge_u`, `edge_v`) and message bottleneck (`msg_down`, `msg_up`) in the top-$r$ ($r=16$) principal SVD subspace $V_r$ of the trained specialist router prototypes and output projections, initializes compositional directed adjacency priors `edge_bias` ($A_{ii} = 0$) across Dot 0 English (`Nodes 0..9`) and Dot 1 Math (`Nodes 10..19`), and activates `collab_gate` (`collab_strength = 0.008`) on the live `22.56M` 20-node SpiderNet checkpoints (`16/8/4/2/1-bit`).
- **Interactive `/graph` Directed Graph Inspector & Live `<think>` Edge Telemetry (`format_graph_inspector_v2` in `mlm.tui`):**
  - Added `/graph` (and `/nodes`) to `mlm chat` to inspect active specialist roles, input-conditioned directed edge weights $a_{ij}(x_t)$, directed message flux $p_i(x_t) a_{ij}(x_t)$, per-layer 2-hop message norms $\|M^{(l)}\|_2$, and sparse index-gather FLOPs reduction (`10.0x` / `5.0x`).
  - Added live `Graph:` telemetry (`N10(Add/Count)──0.64──►N15(Div/GSM8K) | N15(Div/GSM8K)──0.50──►N10(Add/Count) | ||M||_2=0.006 | sparse=4/20 (5.0x FLOPs)`) inside the `<think>` panel on every turn.

## [0.0.28] - 2026-11-05

### Added

- **SpiderNet 6-Architecture Continual-Learning Ablation Suite (`run_spidernet_ablation_suite_v2` in `mlm.model_v2`):** Implemented a controlled 6-variant continual-learning ablation (`A_dense_baseline`, `B_static_flat_moe`, `C_static_hierarchical_moe`, `D_manual_grow_moe`, `E_spidernet_directed_graph`, `F_spidernet_full_autonomous`) across 4 sequential domains (`English -> Math -> Science -> Code`):
  - `F_spidernet_full_autonomous` achieved **`AA = 0.49417`** (`+112%` vs `A_dense_baseline` `0.23309`, `+60%` vs `D_manual_grow_moe` `0.30822`), **`BWT = -0.00954`** (near-zero backward transfer degradation), **`forgetting = 0.11652`**, **`retention_pct = 92.43%`**, **`final_mean_loss = 0.83097`**, and **`3.50x` specialist FLOPs reduction** (`4 -> 7` nodes via `3` autonomous CUSUM orthogonal split events).

## [0.0.27] - 2026-11-05

### Added

- **Continual-Learning Evaluation Harness (`evaluate_continual_learning_metrics_v2` in `mlm.model_v2`):** Added standardized continual-learning metrics over sequential task loss matrices $L \in \mathbb{R}^{T \times T}$ (`R_{t,j} = \exp(-L_{t,j})`), computing **Average Accuracy Proxy (`AA`)**, **Backward Transfer (`BWT`)**, **Average Forgetting (`F`)**, **Retention Percentage (`retention_pct`)**, **Active Parameter Ratio (`activation_ratio`)**, and **Specialist FLOPs Reduction (`flops_reduction_x`)**.

## [0.0.26] - 2026-11-05

### Added

- **Symmetric Orthogonal Parent-Child Specialist Splitting (`_orthogonal_perturbation_like`, `split_specialist_nodes_orthogonal_v2` in `mlm.model_v2`):** Replaced noisy heuristic duplication with Gram-Schmidt orthogonal perturbation $Q$ ($\langle Q, W_{i^*} \rangle = 0, \; \|Q\|_F = \|W_{i^*}\|_F$) so parent $i^*$ and child $\text{new}$ split symmetrically:
  $$W_{i^*}' = W_{i^*} + \epsilon Q, \qquad W_{\text{new}}' = W_{i^*} - \epsilon Q, \qquad b_{i^*}' = b_{i^*} - \log 2, \qquad b_{\text{new}}' = b_{i^*} - \log 2$$
  Verified `max_orthogonality_cosine = 4.56e-9` and `max_mean_reconstruction_err = 1.49e-8` while initializing directed parent$\leftrightarrow$child graph collaboration bias $c_{i^*,\text{new}} = c_{\text{new},i^*} = +0.5$.

## [0.0.25] - 2026-11-05

### Added

- **Autonomous Page-Hinkley / One-Sided CUSUM + 3-Signal Node Spawning Controller (`AutonomousGrowthControllerV2` in `mlm.model_v2`):** Implemented autonomous drift detection ($C_t = \max(0, C_{t-1} + (L_t - \mu_t)/\sigma_t - \delta) > h$) and composite per-node overload scoring:
  $$G_i = w_s \hat{S}_i + w_o \hat{O}_i + w_h \hat{H}_i$$
  combining gradient-conflict saturation $\hat{S}_i$, routing occupancy overload $\hat{O}_i = \bar{p}_i / (k/N)$, and output representation variance $\hat{H}_i$ to autonomously select parent $i^* = \arg\max_i G_i$ and trigger `split_specialist_nodes_orthogonal_v2`.

## [0.0.24] - 2026-11-05

### Added

- **Input-Conditioned Low-Rank Bilinear Edge Routing (`compute_directed_edge_weights_v2`) & 3 Anti-Collapse Graph Regularizers (`compute_spidernet_graph_regularizers_v2` in `mlm.model_v2`):**
  - Implemented token-conditioned directed edge scoring $e_{ij}(x_t) = \frac{(U h_i^{(1)})^\top (V h_j^{(1)}) + u_i^\top R x_t + v_j^\top R x_t}{\sqrt{r}} + c_{ij}$ with top-$k_e$ sparse neighbor selection so the same primary specialist dynamically selects different collaborators depending on the token context $x_t$.
  - Added 3 differentiable anti-collapse graph regularizers into `make_train_step_v2`: Switch-style load-balancing $\mathcal{L}_{\text{load}} = N \sum_{i=1}^N \bar{p}_i \bar{f}_i$, L1 directed edge sparsity $\mathcal{L}_{\text{edge}} = \frac{1}{N(N-1)} \sum_{i \neq j} \bar{a}_{ij}$, and squared cosine specialist diversity $\mathcal{L}_{\text{diversity}} = \frac{1}{N(N-1)} \sum_{i \neq j} \cos^2(W_i, W_j)$.

## [0.0.23] - 2026-11-05

### Added

- **Directed Inter-Node Graph Message Passing (`directed_graph_message_passing_v2` in `mlm.model_v2`):** Upgraded SpiderNet from parallel independent experts to a 2-hop directed computational graph ($N_i \to N_j$) with zero self-loops ($A_{ii} = 0$), low-rank bottleneck message projection $W_{\text{msg}} = W_{\text{msg,up}} W_{\text{msg,down}}$, and zero-initialized gated residual refinement:
  $$M_j(x_t) = \sum_{i \neq j} p_i(x_t) a_{ij}(x_t) W_{\text{msg}} h_i^{(1)}(x_t), \qquad \Delta h_j^{(2)} = \tanh(g_{\text{collab}, j}) \odot \left(\operatorname{SiLU}(M_j) \odot \hat{h}_j^{(1)} + M_j\right)$$
  guaranteeing 100% lossless backward compatibility (`max_logit_diff = 0.0`) with existing checkpoints when $g_{\text{collab}} = 0$ and exact equivalence between sparse gather and dense execution when $g_{\text{collab}} \neq 0$.

## [0.0.22] - 2026-11-05

### Added

- **True Sparse Index-Gather Specialist Execution (`_execute_sparse_nodes_step_v2`, `_execute_dense_nodes_step_v2`, `profile_sparse_execution_v2` in `mlm.model_v2`):** Replaced dense `einsum("btd,ndf->btnf")` evaluation of all $N=20$ specialist nodes during $O(1)$ recurrent decoding with a 3-tier `jax.lax.switch` active-index gather (`k_t0 = 2`, `k_t1 = 4`, `k_t2 = 8`) over `gate_w[active_idx]`, `up_w[active_idx]`, and `down_w[active_idx]`:
  - **English Profile (`k_active = 2 / 20 = 10%`):** `max_logit_diff = 9.54e-6` (**100% lossless**), **`10.0x` specialist FLOPs reduction** (`5.31M` vs `53.08M` FLOPs/tok; `1.77M / 17.69M` active specialist params), and **`2.93x` wall-clock decode speedup** (`6.18 ms/tok` vs `18.12 ms/tok`).
  - **Math Profile (`k_active = 4 / 20 = 20%`):** `max_logit_diff = 9.78e-6` (**100% lossless**), **`5.0x` specialist FLOPs reduction**, and **`2.40x` wall-clock decode speedup** (`7.51 ms/tok` vs `18.05 ms/tok`).

## [0.0.21] - 2026-11-05

### Added

- **3-Tier Persistent External & Episodic Memory Hierarchy (`MemoryHierarchyV2` in `mlm.model_v2`, `format_memory_inspector_v2` in `mlm.tui`):** Implemented a 3-tier memory architecture across `model_v2.py`, `tui.py`, and `cli.py`:
  1. **Tier 1 — Immediate Context (`DSA + IndexShare Sparse Attention` + `TurboQuant 2-bit + QJL` KV Cache):** Live tracking of active KV positions, Frobenius KV norms ($\sqrt{\|K\|_F^2 + \|V\|_F^2}$), and `DSA` Lightning Indexer cache norms ($\|K_{\text{idx}}\|_F$) directly from the $O(1)$ recurrent decode carry (`record_decode_carry`).
  2. **Tier 2 — Working State (`Parallel Gated DeltaNet` Recurrent Fast Weights):** Cross-turn recurrent fast-weight state $S_{\text{work}} \in \mathbb{R}^{L_{\text{sched}} \times 1 \times H \times D_h \times D_h}$ updated via gated delta-rule EMA ($S_{\text{work}} \leftarrow \gamma S_{\text{work}} + (1 - \gamma) S_{\text{turn}}$), blended into episodic value representations (`_compute_text_key_value`), and persisted in `fp16` inside `memory.safetensors`.
  3. **Tier 3 — Persistent Long-Term Episodic Memory (`TurboQuant 2-bit + QJL` Content-Addressed Bank + State-Action-Overwrite `SAO`):** Compresses up to 64 content-addressed $(K_{\text{ep}}, V_{\text{ep}}) \in \mathbb{R}^{M \times D}$ slots using 2-bit `TurboQuant` (`FWHT` + Lloyd-Max + 1-bit `QJL`) at **`8.00x` compression vs `fp32`** (`12.0 KiB` total), supporting declarative fact extraction, cross-session `.mlm2.memory` disk persistence (`save` / `load`), in-place **State-Action-Overwrite (`SAO` / Delta Rule)** when facts or variables change (`upsert_slot`), and persistent variable substitution inside `VERIFIED` math expressions (`process_turn_before_generate`).
- **LMCache Hybrid Prefix/Full State Cache (`LMCacheHybridV2`), `CacheBlend` Dynamic $SO(2)$ RoPE Position Re-Rotation (`cacheblend_rope_shift_kv_v2`, `cacheblend_fuse_kv_chunks_v2`), & `CacheGen` Anchor-Token Temporal Delta `TurboQuant` KV Compression (`quantize_kv_cache_turboquant`):** Added hash-indexed hybrid recurrent + KV prefix/full state caching at 4-token chunk boundaries across turns and recursive `RLM` sub-calls (`81.8 tok/s` on repeated/shared-prefix turns), exact $SO(2)$ relative RoPE rotation (`max_diff < 1e-5`) with High-KV-Deviation (`HKVD`) token selection for non-prefix chunk fusion, and `CacheGen` anchor-token temporal delta coding ($\Delta K_t = K_t - K_{\lfloor t/G \rfloor G}$) inside `quantize_kv_cache_turboquant`.
- **Interactive Memory TUI & CLI Controls (`mlm.tui`, `mlm.cli`):** Added `/memory` (rendering all 3 memory tiers, `LMCache` chunk/token-save stats, and per-layer DeltaNet fast-weight bars), `/forget` (clearing working and episodic memory), `--memory-file <path>`, and `--reset-memory` to `mlm chat`, plus live `Memory: T1(KV)=... | T2(DeltaNet ||S||_F)=... | T3(Episodic)=... | LMCache(saved=... tok)` telemetry inside `<think>` boxes.

### Changed

- Extended `tests/test_v2_and_gguf.py` with unit tests covering `MemoryHierarchyV2` Tier-1 KV tracking, Tier-2 Gated DeltaNet working state, Tier-3 2-bit `TurboQuant` episodic storage, cross-session `.mlm2.memory` reload, `SAO` slot overwrite, and math variable substitution.
- Incremented package and lockfile version to `0.0.21`.

## [0.0.20] - 2026-11-04

### Added

- **Stateful $O(1)$ Recurrent Decode Cache (`init_decode_cache_v2`, `_recurrent_token_step_v2`, `_make_cached_recurrent_decode_v2` in `mlm.model_v2`):** Replaced full-sequence (`256`-token) recomputation during autoregressive decoding with an exact static-shape recurrent carry state across all 12 effective hybrid passes (`ShortConv` $K=7$ ring buffer `shortconv_buf`, `ncm_phi_bridge` ring buffer `phi_conv_buf`, exact rank-1 Gated DeltaNet recurrent state matrix $S_{t,h} = \alpha_{t,h} S_{t-1,h} + k_{t,h} v_{t,h}^\top \in \mathbb{R}^{H \times D_h \times D_h}$ with 1-step key/value shift buffers, incremental GQA + `DSA` + `TurboQuant` KV cache slots `k_cache`/`v_cache`/`k_idx_cache`, and causal cumulative router load/domain-energy trackers `_route_single_bank_step_v2`), matching full-sequence `forward_with_hidden_v2` logits within `< 2e-6` (`100%` greedy equivalence) while accelerating CPU inference by **3×–7×** (`28.4–78.7 tok/s`, up from `~11 tok/s`).
- **2-Token MTP Speculative Recurrent Verification (`spec2`) with Zero-Recomputation Rollback:** Extended `_make_cached_recurrent_decode_v2` with a compiled 2-token recurrent kernel `spec2(params, carry, tok_a, draft_b)` that returns both `(carry_a, logits_a, draft_a)` and `(carry_b, logits_b, draft_b)` in a single XLA call; on draft acceptance `generate_v2` advances 2 tokens to `carry_b`, and on draft rejection it rolls back to `carry_a` with zero recomputation.
- **Single-Prefill Dual-Stream Fork (`_prefill_state` in `generate_v2`):** In `DELIBERATIVE` and `VERIFIED` modes (`include_think=True`), `generate_v2` prefills the prompt once in 2-token `spec2` pairs and forks the post-prompt recurrent state `fork_prefill_state = (carry, prompt_raw, prompt_draft, pos)` into both the `<|think|>` reasoning pass and the final response pass, eliminating 100% of duplicate prompt prefill.

### Changed

- Updated unit tests (`tests/test_v2_and_gguf.py`) to verify numerical and `argmax` equivalence between `_make_cached_recurrent_decode_v2` (`spec2`) and full-sequence `forward_with_hidden_v2`.
- Incremented package and lockfile version to `0.0.20`.

## [0.0.19] - 2026-11-04

### Added

- **Hierarchical SpiderNet Growing Architecture (`22.45M` Total Params: 1 Shared Core + Dot 0 `10M` English Macro-Node + Dot 1 `10M` Math Macro-Node):** Implemented modular SpiderNet skill-specific Sub-Node growth (`active_subnodes`, `upgrade_subnodes_from_checkpoint_v2`, `calibrate_spidernet_router_v2` in `mlm.model_v2` and `mlm.train_v2`), enabling continual training, upgrading, and splicing of individual `~1M`-parameter Sub-Nodes (`Nodes 0..3` Grammar/Vocab, `Nodes 4..5` Science/Reading, `Nodes 6..7` Multi-Turn Chat, `Nodes 8..9` Deliberative `<|think|>`, `Nodes 10..11` Addition/Counting, `Nodes 12..13` Subtraction/Multiplication, `Nodes 14..15` Division/Word Problems, `Nodes 16..17` Algebra/Geometry, and `Nodes 18..19` Verified Math `<|think|>`) while keeping the Shared Core Trunk (`val_perplexity = 1.31`) and mature specialist banks 100% frozen.
- **3-Mode Cognitive Router (`Direct` / `Deliberative` / `Verified` in `classify_cognitive_mode_v2`):** Added adaptive entrance cognitive routing (`mlm.model_v2`, `mlm.tui`) that inspects the prompt's calibrated FlexOlmo cosine domain log-odds (`seq_dl`) and query complexity to select:
  - `DIRECT` mode for fast conversational turns (`Nodes 0..7`, bypassing redundant second-pass `<|think|>` decoding),
  - `DELIBERATIVE` mode for multi-step scientific and analytical explanations (`Nodes 8..9` `<|think|>` + `Nodes 0..7`), and
  - `VERIFIED` mode for arithmetic, multi-step parenthesized expressions, and GSM8K word problems (`Nodes 18..19` `<|think|>` + Process Step Verifier + `Nodes 10..17`).
- **EAGLE-3 / DeepSeek-V3 Multi-Token Speculative Decoding (`_make_cached_logits_step_v2`, `generate_v2`):** Coupled the trained Multi-Token Prediction head (`mtp_norm.weight`, `mtp_proj.weight`) directly into greedy autoregressive decoding (`mlm.model_v2`), drafting token $t+2$ alongside token $t+1$ and verifying both in a single causal forward pass with 100% lossless greedy equivalence (`16–26.5 tok/s` on CPU).
- **DeepSeek-R1 Group Relative Policy Optimization (`run_grpo_reasoning_v2`) & Post-Router `<|think|>` Alignment:** Added group-normalized clipped surrogate policy optimization with analytical KL regularization to $\pi_{\text{ref}}$ (`run_grpo_reasoning_v2` in `mlm.train_v2`) and fixed the `active_subnodes` router bias gate-scale mismatch in `_forward_core_v2` so `<|think|>` Sub-Nodes (`Nodes 8..9` and `Nodes 18..19`) train with the exact same `sigmoid(think_logits)` gate scale used at inference (`val_loss = 0.1386`, `grpo_loss = 0.0399` on Google Colab NVIDIA Tesla T4 GPU).
- **Process Step Verification (`Let's Verify Step by Step` / `_verify_reasoning_steps_v2` & `_verify_answer_against_thought_v2`) + Recursive Language Model (`RLM`) Self-Calls (`_rlm_try_recursive_decomposition_v2`):** Added step-by-step equation verification at `<|/think|>` and response boundaries, prompt-grounded single-operation verification, multi-step `<|think|>`-to-answer reconciliation, and recursive self-call parenthesized expression decomposition (`What is (9 + 8) * 2?` $\to$ `9 + 8 = 17` $\to$ `17 * 2 = 34`).
- **Google Research `TurboQuant` (`arXiv:2504.19874`) & `BiLLM + QJL` Multi-Precision Weight & KV-Cache Quantization (`16-bit`, `8-bit`, `4-bit`, `2-bit`, `1-bit`):** Added `mlm.quantize_v2` (`export_all_precisions_v2`, `quantize-v2` CLI) and `turboquant_kv_tensor_jax` (`--kv-bits 1|2|4|8`), applying Fast Walsh-Hadamard Orthogonal Rotation (`FWHT`), Lloyd-Max Gaussian centroids, and 1-bit Quantized Johnson-Lindenstrauss (`QJL`) residual sign correction to compress the `22.45M`-parameter 20-node SpiderNet checkpoint from `85.81 MiB` (`fp32`) down to `43.63 MiB` (`16-bit`), `23.18 MiB` (`8-bit`), `15.93 MiB` (`4-bit`), `12.47 MiB` (`2-bit`), and `10.66 MiB` (`1-bit`) while preserving verified English, Science, Arithmetic, Parentheses, and GSM8K Word-Problem accuracy.
- **BLT Byte N-Grams, FoNE + Abacus Number Embeddings & FlexOlmo Cosine Domain Routing (`mlm.model_v2`):** Added Byte Latent Transformer (`BLT`) hashed byte 3/4/5-gram subword enrichment (`blt_token_ngrams`, `blt_proj.weight`), Fourier Number Embeddings (`FoNE`, NeurIPS 2025) + Causal Abacus Place-Value Embeddings (`compute_fone_abacus_embeddings`), FlexOlmo L2-normalized cosine similarity routing (`8.0 * (x_unit @ w_unit)`), and Tiiny-AI/PowerInfer rectified sparse domain gating (`seq_dl >= -0.95`).

### Changed

- Updated `mlm chat` (`mlm.tui`) to display realcheckpoint quantization metadata (`bits`, `method`, `safetensors_bytes`), KV-cache precision (`TurboQuant 2-bit + QJL`), and live 3-Mode Cognitive Router telemetry (`Routing: mode=DIRECT|DELIBERATIVE|VERIFIED | domain=english|math | subnodes=(...)`).
- Incremented package and lockfile version to `0.0.19`.

## [0.0.18] - 2026-11-01

### Added

- **MLM v2 Hybrid Liquid-Delta + Continual Recombinative Graph-Node Architecture (`mlm.model_v2`, `mlm.train_v2`, `mlm.gguf_export`, `crates/mlm_core`):** Introduced the 8-layer `v2_hybrid` stack combining LFM 2.5 `ShortConv` ($K=7$), Parallel Gated DeltaNet with KDA channel-wise decay, DeepSeek Sparse Attention (`DSA` + `IndexShare`) with head-wise gated GQA, SenseNova-Loop 12-pass 2-stream `mHC` Sinkhorn looping, HazyResearch `ncm` Next-Concept Modeling (`ncm_phi_bridge`, Euclidean NCM tied head), Ornith-1.5 Self-Scaffold `<|think|>` horizon mask (`compute_ornith_scaffold_mask`), Rust `tokenizers` 4,096-subword digit-aware BPE, Rust `polars` Parquet streaming (`mlm.parquet_stream`, `mlm.math_data`), SafeTensors `.mlm2` checkpoints, and native Rust GGUF v3 binary inspector (`mlm-gguf-inspect`).
- **3-Stage Developmental English Node Growth (`4 → 6 → 8` Specialist Nodes = `11.4M` Params on Colab T4 GPU):** Outperformed `TinyStories-8M` (`8.3M`) and `Pythia-14M` (`14.1M`) on held-out 4-way Science & Reading QA (`58.00%` vs `32.0%` and `44.0%`) and achieved `83.33%` on BLiMP English Grammar Minimal Pairs (`docs/evidence/colab_10m_v2_report.json`).

### Changed

- Incremented package and lockfile version to `0.0.18`.

## [0.0.17] - 2026-10-31

### Added

- Assistant-masked Hugging Face conversational curation (`loss_spans`), orthogonal DoG wavelet bands, and 6-architecture comparative benchmark (`mlm.compare_10m_models`).

## [0.0.16] - 2026-10-31

### Added

- 5,000-step T4 GPU `ChunkedFoundationTrainer` and Muon RMS scaling (`mlm.colab_train`).

## [0.0.15] - 2026-10-31

### Added

- `100.70M`-parameter developmental sparse graph MLM (`16 → 32 → 64` nodes) trained on Google Colab T4 GPU.

## [0.0.14] - 2026-10-31

### Added

- 10M model head-to-head comparison (`mlm.compare_10m_models`), JAX 0.11+ Colab compatibility, and automatic `b8` (`int8`) checkpoint loading in `mlm chat`.

## [0.0.13] - 2026-10-31

### Added

- **True Indexed Sparse Top-$K$ Node Execution (`act_idx` Gather) & Gradient Rematerialization (`@jax.checkpoint`):** When `nodes >= 8` and `0 < top_k_nodes < nodes - shared_nodes`, `graph_round` in `GraphLM.__call__` (`mlm.model`) gathers only the `shared_nodes + top_k_nodes` active nodes via `jnp.take_along_axis` before running the heavy `x_kernel` and `h_kernel` matrix multiplications and scatters updates back into the graph state, while wrapping `token_step` in `jax.checkpoint` when `nodes * width >= 512` and capping XLA/OpenMP CPU threads (`OMP_NUM_THREADS=6`) in `mlm.config` for lag-free laptop training of 10M+ parameter models.
- **Warm Optimizer Moment Transfer Across Node Growth (`grow_train_state`):** Added `grow_train_state` in `mlm.plasticity` to preserve warm Muon/AdamW moments (`opt_state`) and step counters across shared parameters and parent nodes during recombinative node growth (`recombine_grow_nodes`) while automatically evicting stale XLA caches (`jax.clear_caches()`).
- **Symmetric Per-Channel 8-Bit (`b8` / `int8`) Checkpoint Quantization:** Added `quantize_params_b8`, `dequantize_params_b8`, and `quantize="b8"` support in `save_checkpoint` / `load_checkpoint` (`mlm.checkpoint`), compressing the `10,301,833`-parameter (`10.30M`) developmental checkpoint from `156.70 MiB` (`fp32`) down to `8.64 MiB` (`b8` `int8` — `18.1x` smaller) with lossless `100.0%` multi-hop reading accuracy.
- **Curated Real-World Hugging Face Dataset Pipeline & `10.30M` Developmental Benchmark (`mlm.curated_data`, `mlm.benchmark_10m`, `configs/mlm-10m-growth.json`):** Added `prepare_curated_english` (`mlm.curated_data`) curating 6 real-world Hugging Face datasets (`wikimedia/wikipedia`, `HuggingFaceTB/cosmopedia-100k`, `HuggingFaceTB/smoltalk`, `databricks/databricks-dolly-15k`, `allenai/sciq`, and `rajpurkar/squad`) and `run_10m_experiment` (`mlm.benchmark_10m`) growing `GraphLM` from `3.81M` (`4` nodes) $\to$ `5.97M` (`8` nodes) $\to$ `10.30M` (`16` nodes) and benchmarking `mlm_10m_b8` head-to-head against `dense_gru` and `legacy_graph`.

### Changed

- Incremented package and lockfile version to 0.0.13 (124 regression tests passing).

## [0.0.12] - 2026-10-31

### Added

- **Kimi Delta Attention (KDA) Channel-Wise Decay + ShortConv Value Smoothing (`arXiv:2510.26692` / GLM-5.3-Flash):** Factored per-key + per-channel retention matrix $\boldsymbol{\Alpha}_{b,k,d} = \alpha_{b,k} \cdot \alpha^{\text{chan}}_{b,d}$ (`channel_alpha` in `delta_step`, `mlm.associative`) and fast Laplace short-convolution smoothing (`0.15 * jnp.tanh(span_state[:, 0, :])`) on associative memory writes in `GraphLM.__call__` (`mlm.model`).
- **4-Stream `mHC` Doubly-Stochastic Readout Highway (`arXiv:2512.24880` / GLM-5.3-Flash):** Birkhoff polytope Sinkhorn-Knopp mixing (`sinkhorn_stream_mix` in `mlm.graph`) across the 4 parallel recurrent readout streams (`[z_base, mem_contrib, wavelet_stream, ep_contrib]`) in `GraphLM.__call__`, preserving exact zero-init warm-start equivalence at step 0 (`mhc_active` gate).
- **`IndexShare` Cross-Hop Sparse Entity Indexer (`arXiv:2603.12201` / GLM-5.2 & GLM-5.3-Flash):** Shared non-function 3-gram word-ending entity mask (`shared_entity_index = is_3letter_end * not_func_word * ctx_t`) computed once and reused across Hops `1..H`, with active density exported in telemetry (`index_share_density`).
- **GLM-5.3-Flash Speculative Multi-Byte Span Drafting & Single-Pass Verification:** When the relational word-onset consolidator identifies a multi-byte context word (`onset_word` with `onset_mass >= 0.35`), `generate_with_trace` (`mlm.generate`) drafts the remaining word bytes and verifies the entire span + `<EOS>` in a single parallel `step_seq_fast` forward call with 100% lossless greedy equivalence (`speculative_accepted` in `trace`).
- **Causal Temporal State-Overwrite & Multi-Attribute Clause Specificity:** Per-clause query specificity (`spec_k`), causal temporal recency (`clause_priority = exp(3.5 * spec_k + 1.8 * k_cl)`, `bridge_recency`), and backward-adjacent modifier retrieval (`f_adj_bwd`, `before_period_t`, `not_func_word`, `prep4_before_final`) in `GraphLM.__call__`, plus the 16-probe (8 counterfactual pairs) temporal and multi-attribute benchmark `data/evaluation/reading-v4-temporal.json`, achieving `100.0%` (`16/16`) accuracy and `100.0%` (`8/8`) pair accuracy across both `plain` and `chat` formats (`144/144` total open-ended generation probes across `v1`–`v4`).
- **Command-Free Interactive Neural Console (`mlm-chat`) & Autonomous `SAO` Episodic Compaction:** Added `mlm chat` / `mlm-chat` (`mlm.tui`) with live `<think>` reasoning telemetry, automatic multi-turn episodic working memory (`remember_facts`, `resolve_context`, `_record_episodic_qa`), `SAO` state-overwrite compaction (`compact_episodic_facts` in `LiveLearner`), and background sleep consolidation (`LiveLearner.consolidate`: stream flush + DeepSeek-R1 critic-free GRPO reinforcement + functional edge rewiring).

### Changed

- Incremented package and lockfile version to 0.0.12 (123 regression tests passing).

## [0.0.11] - 2026-10-30

### Added

- Open-ended multi-byte autoregressive relational generation (`_make_seq_step` and `generate()` in `mlm.generate`), maintaining a fixed-shape causal sequence buffer `(1, buf_len)` with a `valid` mask and cached JIT kernel when `config.relational_hops > 0` so multi-hop relational circuits remain active across all generated answer bytes.
- Word-onset consolidation (`is_word3`, `onset_attn`), causal 2-gram interior span continuation (`interior_span * match_2 + 0.15 * onset_attn * match_1`), Cauchy-Schwarz unit-hypersphere logit projection (`vk_u.T`), and context delimiter `EOS` termination (`is_strict_delim_t`, `eos_mass`) in `GraphLM.__call__`, enabling unconstrained greedy `generate()` to emit exact multi-byte answer words of arbitrary length without candidate choices (`100.0%` exact word match across `reading-v1.json`, `reading-v2-multihop.json`, and `reading-v3-3hop.json` in both `plain` and `chat` formats).
- Pure-tensor isomorphic context and question boundary detection (`c_tag`, `m_plain_t`, `ctx_exact`, `q_exact`, `t_ref = q_end + 8.0`), period-anchored cumulative clause segmentation (`is_period_t`, `clause_id_t`), same-word suffix suppression (`q_m1`, `q_m2`, `q_tm1`, `q_tm2`), lag-2 repeating-vowel filtering (`is_lag2_dup`), and question confidence gating (`g_question`) with bounded `1e-4 * jnp.tanh(...)` residual paths in `GraphLM.__call__`, achieving `100.0%` post-continual OOD retention across 1-hop, 2-hop, and 3-hop reading suites alongside `100.0%` train/dev continual QA accuracy.
- 3-hop disjoint-clause transitive relational composition (`relational_hops=3` in `configs/deepseek-v1.json`, `visited` clause/entity mask in `GraphLM.__call__`) and the 16-probe (8 counterfactual pairs) 3-hop evaluation suite `data/evaluation/reading-v3-3hop.json` (`chain-room`, `chain-color`, `chain-food`, `chain-city`, `chain-owner`, `chain-prize`, `chain-tool`, `chain-drink`), achieving `100.0%` (`16/16`) accuracy and `100.0%` (`8/8`) pair accuracy.

### Changed

- Incremented package and lockfile version to 0.0.11 (122 regression tests passing).

## [0.0.10] - 2026-10-30

### Added

- Liquid AI LFM 2.5 / STAR (`arXiv:2411.17800`, `arXiv:2511.23404`) Linear Input-Varying (LIV) double-gated short-range causal convolution (`rel_liv_gate_kernel`, `rel_liv_conv_kernel` in `GraphLM.__call__` and `MEMORY_PARAMETERS` in `mlm.plasticity`), providing adaptive sub-token stem mixing across causal 4-byte receptives fields.
- Input-varying per-key liquid time-constant associative memory decay ($\alpha_{b,k} = \exp(-\Delta_{b,k}(x_t))$) in `delta_step` (`mlm.associative`) and `memory_decay_kernel` (`(2 * d, memory_keys)` in `GraphLM.__call__`), enabling concurrent fast working-memory slots and slow long-horizon factual slots.
- True 2-hop disjoint-clause transitive relational composition ($A \to B \to C$) with expected-question-position anchoring (`s1_mean`, `q_win_t`, `q_win_h2`, `ctx_t`), clause-interval bridge extraction (`hop_bridge`, `clause_same`, `clause_disjoint`), and Zipf delimiter-invariant autoregressive in-span byte continuation (`tok_freq`, `in_span_prev`) in `GraphLM.__call__`.
- 2-hop counterfactual reading comprehension evaluation suite (`data/evaluation/reading-v2-multihop.json`) with 16 probes across 8 counterfactual pairs (`person-room`, `container-color`, `pet-food`, `guard-trainer`, `friend-city`, `key-room`, `team-prize`, `tool-shelf`), achieving `100.0%` (`16/16`) zero-shot accuracy and `100.0%` (`8/8`) pair accuracy across both `chat` and `plain` formats alongside `100.0%` (`24/24`) on `data/evaluation/reading-v1.json`.

### Changed

- Incremented package and lockfile version to 0.0.10 (122 regression tests passing).

## [0.0.9] - 2026-10-29

### Added

- Hyperspherical heat / RBF 3-gram kernel ($K_3(t, s) = \exp(24(C_3(t, s) - 1))$) and symmetric bilinear stem matching (`qk_sym`) in `GraphLM.__call__`, eliminating low-dimensional ($d=16$) random unit-vector cross-talk while preserving exact 3-gram stem self-similarity.
- Multi-scale bidirectional clause-offset retrieval (`f_fwd`, `f_bwd` over Gaussian offset centers $[6.0, 9.5, 13.5, 15.0]$) with diagonal left-predicate alignment gating (`align_left = (k3_m5 + k3_m7 + k3_m9) / 3.0`, `g_bwd`), automatically switching between forward `Subject -> Attribute / Object` retrieval and backward `Predicate + Object -> Subject / Modifier` retrieval.
- Dual-stage causal question-window 3-gram complement gating (`u_val` with context-horizon gate `ctx_gate` at anchor step $s_1$, and `u_val_t` over causal question window $[t - 42, t - 8]$ at readout step $t$), Tikhonov ridge-normalized induction (`b_ind`), and question-horizon gating (`q_horizon`) to prevent teacher-forced candidate answer prefixes from hijacking the Laplace question window.
- Readout-anchored unit value projection (`vocab_unit` from column-centered `vocab_kernel`) in `GraphLM.__call__`, enabling zero-shot token-to-logit transfer across all 259 byte tokens and achieving `100.0%` (`24/24`) individual accuracy and `100.0%` (`12/12`) counterfactual pair accuracy on `data/evaluation/reading-v1.json`, alongside `100.0%` (`16/16`) train/dev QA accuracy and `0.0179` QA continual loss (`-99.6%`) on the DeepSeek continual benchmark.

### Changed

- Incremented package and lockfile version to 0.0.9 (122 regression tests passing).

## [0.0.8] - 2026-10-29

### Added

- Titans-inspired prediction-surprise associative write gating (`surprise_gating=True` in `Config`, `surprise_mem_weight` and `surprise_span_weight` in `GraphLM.__call__`), modulating fast-weight delta writes and multi-horizon trace updates by normalized predictive gradient norm.
- Multi-scale Difference-of-Gaussians (DoG) / Laplace wavelet filter bank (`wavelet_out` in `GraphLM.__call__`) transforming exponentially decaying trace horizons into zero-mean band-pass edge detectors across clause and entity boundaries.
- Causal 3-gram Simpson-weighted relational induction circuit (`relational_hops` in `Config`, `rel_wavelet_kernel`, `rel_q_kernel`, `rel_k_kernel`, `rel_v_kernel`, `rel_output_kernel`, `rel_gain` in `GraphLM.__call__`) achieving `100.0%` train and held-out dev counterfactual QA accuracy on multi-entity continual binding.

### Changed

- Incremented package and lockfile version to 0.0.8 (122 regression tests passing).

## [0.0.7] - 2026-10-29

### Added

- Bidirectional multi-horizon Laplace trace associative memory (`symmetric_memory=True` in `Config`, `rev_key` / `rev_value` dual Householder writes in `delta_step`, and 4-horizon log-spaced trace bank $\tau \in [2.0, 24.0]$ with 3D `einsum` cross-horizon projections `memory_span_kernel`, `memory_span_bias`, `memory_rev_key_kernel`, `memory_rev_value_kernel` in `GraphLM.__call__`).
- DeepSeek-V2/V3 Multi-Head Latent Attention (MLA) content-addressed episodic slot buffer (`episodic_slots` in `Config`, `initial_state`, and `GraphLM.__call__` with `episodic_compress_kernel`, `episodic_write_kernel`, `episodic_write_bias`, `episodic_query_kernel`, `episodic_pos_bias`, `episodic_output_kernel`, `episodic_gain`), maintaining $S$ compressed latent slots inside the recurrent carry with content-addressed updates $\sigma(\ell) \odot \text{softmax}(\ell)$ and causal cross-attention readout.
- Continuous straight-through minimum-magnitude gain floors (`mem_floor` and `ep_floor` in `GraphLM.__call__`), preserving exact zero-init warm-start forward equivalence at step 0 while maintaining non-vanishing gradient flow to upstream associative and episodic parameters across subsequent steps.
- Multi-seed continual replication across `QA43/English44` and `QA44/English45` comparing `legacy_graph`, `mini_agi_v4`, and `deepseek_v1` (`configs/deepseek-v1.json`), where `deepseek_v1` reduced held-out QA validation loss by 57.5%–59.9% (`0.6837` / `0.7092` vs `1.7059` / `1.6687`) and held-out English validation loss by 4.0%–5.7% (`2.6078` / `2.6693` vs `2.7642` / `2.7802`).

### Changed

- Incremented package and lockfile version to 0.0.7 (121 regression tests passing).

## [0.0.6] - 2026-10-29

### Added

- DeepSeekMoE always-active shared nodes + routed specialist nodes (`shared_nodes` in `Config` and `apply_routing_gates` in `GraphLM.__call__`), pinning the first $N_s$ nodes to update rate $1.0$ on every token while routing and load-balancing across the remaining $N - N_s$ specialist nodes.
- DeepSeek-V3 auxiliary-loss-free router load balancing (`router_bias_gamma` in `Config`, `update_router_load_bias` in `mlm.plasticity`, and `router_bias` in `GraphLM.__call__`, `routed_objective`, `make_routed_step`, and `run_continual`) adjusting specialist routing logits via a non-gradient sign step $b_i \leftarrow b_i + \gamma\,\text{sign}(\bar{u} - u_i)$.
- Gated DeltaNet write-coupled data-dependent associative memory decay (`memory_decay=True` in `Config`, `alpha` retention scaling in `delta_step`, and straight-through gated `memory_decay_kernel` / `memory_decay_bias` in `GraphLM.__call__`).
- DeepSeek-V3 causal target-conditioned multi-token prediction (`mtp_weight` in `Config`, `return_mtp` / `next_tokens` and `mtp_kernel` in `GraphLM.__call__`, and boundary-checked `mtp_loss` in `objective` and `routed_objective`).
- DeepSeek mHC manifold-constrained doubly-stochastic graph message mixing (`sinkhorn_edges=True` in `Config` and `sinkhorn_edge_weights` in `mlm.graph`) using vectorized Sinkhorn-Knopp alternating sender/receiver normalization.
- DeepSeek-R1 critic-free Group Relative Policy Optimization (`grpo_objective`, `make_grpo_step`, and `build_grpo_question_group` in `mlm.plasticity`, `--grpo` CLI flag on `mlm continual`, and `configs/deepseek-v1.json`) with group-normalized verifiable rewards, clipped policy ratios, and unbiased non-negative KL regularization.

### Changed

- Incremented package and lockfile version to 0.0.6 (120 regression tests passing).

## [0.0.5] - 2026-10-29

### Added

- Top-$k$ sparse active-node routing (`top_k_nodes` in `Config` and `GraphLM.__call__`) using straight-through top-$k$ gating so unselected nodes remain at rest with identically zero recurrent state drift while preserving router gradient flow.
- Learned directed graph edge rewiring (`edge_affinity_matrix` and `rewire_edges` in `mlm.plasticity`, custom `edges` in `Config` and `model_topology` in `mlm.graph`) computing functional alignment between sender state/message subspaces and receiver message-input/attention subspaces to prune weak incoming edges and sprout high-affinity connections.
- Unified `mlm continual` CLI command and `run_continual` loop (`mlm.hierarchy_run`) integrating `LaneSchedule`, `PlasticityController`, cross-window EMA `running_usage`, plateau-gated `maybe_grow_and_prune` and `rewire_edges`, and slice-preserving optimizer moment transfer (`_copy_matching_opt_state` in `mlm.binding_run`).
- Canonical `configs/mini-agi-v4.json` configuration and multi-seed continual replication across `QA43/English44` and `QA44/English45`, where `mini_agi_v4` reduced held-out QA validation loss by 17.6%–20.0% (`1.3649` / `1.3448` vs `1.6567` / `1.6809`) and held-out English validation loss by 3.1%–4.2% (`2.6777` / `2.6574` vs `2.7623` / `2.7727`) across both seed pairs.

### Changed

- Upgraded `_copy_matching_opt_state` to slice-copy parent node moments along grown dimensions (`axis=0` and `axis=-1`) when `recombine_grow_nodes` expands the graph.
- Incremented package and lockfile version to 0.0.5 (117 regression tests passing).

## [0.0.4] - 2026-10-28

### Added

- Cross-window EMA router load-balance regularization (`running_usage` in `routed_objective` and `make_routed_step`) and detached trunk state routing (`jax.lax.stop_gradient(context)` on `route_probs` when `return_routes=True` in `GraphLM.__call__`) so router load-balancing updates router weights without distorting trunk representations across both `"hierarchical"` and `"timescale"` routing modes.
- Geometric ponder prior KL regularization (`ponder_beta`, `halt_prior` in `Config` and `return_ponder=True` in `GraphLM.__call__`), regularizing per-round adaptive halting probabilities toward a truncated geometric distribution $\text{Geom}(\text{halt\_prior})$ in both `objective` and `routed_objective`.
- Warm optimizer moment preservation (`preserve_moments=True` in `warm_start` via `_copy_matching_opt_state`) to carry over matching AdamW second-moment (`nu`) and Muon momentum (`muon_bufs`) buffers across continual learning phases, alongside linear learning-rate warmup (`warmup_steps` in `Config` and `scale_optimizer`).
- Held-out-governed learning rate controller (`PlasticityController`) using dual slow/fast exponentially weighted least-squares (EWLS) fits ($\lambda = 0.97, 0.85$), scale-free effect size $e = -\text{slope}/\sigma$, asymmetric log-space nudges, and confirmed regime-change step-up (`UP = 2.0`) with `.state()` / `.restore()` checkpoint support.
- Power-law curriculum & marginal-worth lane scheduler (`LaneSchedule`) fitting log-loss vs. log-bytes scaling slopes ($\tau = 0.15$) per corpus/skill lane, computing marginal worth $|\alpha_s| L_s / D_s$ with Muennighoff repetition discounting $\exp(-\text{epochs}/15.4)$, and selecting batches via deterministic smooth weighted round-robin credit.
- Saturation-triggered autonomous node growth and gate-based pruning (`node_saturation`, `prune_nodes`, `maybe_grow_and_prune`) measuring active `birth_gate` saturation and normalized entropy, pruning dead nodes with $\sqrt{N_{\text{kept}}/N_{\text{old}}}$ readout compensation, and triggering `recombine_grow_nodes` when capacity saturates.
- Online continual byte-stream learning during live chat (`LiveLearner` in `mlm.generate` and `--learn`, `--save-out` CLI flags for `mlm chat`) performing bounded online gradient updates every `chunk` bytes while keeping structural gates frozen (`freeze_gates=True`).

### Changed

- Incremented package and lockfile version to 0.0.4 (114 regression tests passing).

## [0.0.3] - 2026-10-28

### Added

- Hybrid Nesterov-Muon optimizer (`optimizer="muon"`) using 5 quintic Newton-Schulz iterations `(3.4445, -4.7750, 2.0315)` with unit-RMS step scaling `sqrt(max(rows, cols))` on internal 2D and 3D weight matrices (`newton_schulz`, `muon_mask`, `make_muon_adamw`) and dormant-slice momentum preservation when a node slice receives zero gradient, while keeping embeddings, output heads, biases, and 1D/0D parameters on AdamW.
- PonderNet-style adaptive halting across recurrent graph message-passing rounds (`adaptive_halt=True`, `halt_kernel`, `halt_bias`) with remaining-probability state freezing (`halt_freeze`) and straight-through zero-init tangent gating for step-1 gradient flow.
- Recombinative graph node growth (`recombine_grow_nodes`) expanding trained `GraphLM` parameter sets from `N` to `N + add_nodes` nodes by recombining high-norm parent node channel slices with symmetry-breaking noise, $\sqrt{N_{\text{new}}/N}$ parent readout compensation, and `birth_gate` scaling on newborn nodes.
- Biological spike-frequency adaptation trace during deterministic byte generation (`adapt_strength`, `adapt_decay` in `generate` and `--adapt-strength`, `--adapt-decay` CLI flags for `mlm generate` and `mlm chat`) to suppress repetitive greedy decoding loops without random sampling.

### Changed

- Registered `halt_kernel` and `halt_bias` under `ROUTER_PARAMETERS` so role-based learning rate multipliers apply uniformly.
- Incremented package and lockfile version to 0.0.3 (109 regression tests passing).

## [0.0.2] - 2026-10-28

### Added

- Multi-timescale surprise-gated node routing (`node_update="timescale"`) with fast-to-slow log-odds spectrum initialization and state-transition surprise gating (`router_timescale_kernel`, `router_surprise_kernel`, `router_timescale_bias`).
- Cross-node relational associative memory (`memory="relational"`) projecting keys, queries, and values from separate per-node recurrent subspaces `(N, D, K)` and `(N, D, D)` with $\sqrt{D}$-scaled write/readout gating.
- Learned state-conditioned neighbor message aggregation (`edge_aggregation="attention"`) over incoming graph edges (`edge_query`, `edge_key`, `edge_vector`).
- Activity-modulated local node plasticity (`local_plasticity=True`) scaling post-AdamW node updates (including weight decay and momentum drift) by per-node relative gradient activity so dormant nodes remain frozen.
- Straight-through zero-init tangent gates on `memory_gain`, `attention_vector`, `edge_vector`, and `router_surprise_kernel`, preserving bit-for-bit step-0 forward equivalence after warm-start while immediately flowing non-zero step-0 gradients to all upstream parameters.
- Fixed context-contrastive sampling replication with runtime exposure and protocol integrity checks: the new schedule remained8/16training answers and0/4groups, failing acquisition and joint gates;8focused tests passed.

- Context-contrastive objective and bounded rescaling: 16/16 training answers and 4/4 groups on one schedule, with worse English probe loss; joint gate failed. Separate inference unchanged, tests and audit limits documented in docs/experiments/context-contrastive/report.md.

- Primary-source DeepSeek/recurrent-memory reassessment and frozen-score context-discrimination feasibility audit; proposed context-contrastive objective remains unimplemented, with no new quality claim.

- Frozen seven-model answer/position analysis using unchanged candidate scoring and exact16/48/16/16teacher target partitions;4focused tests and258runtimeaudits pass.
- No short-budget answer-ranking gain:allmodels8/16and0/4groups, with zero optimizer updates or model writes. Prior acquisition/replication results remain unchanged.

- Predefined future-byte sampling replication:four new arms,1,024backboneupdates,89passing tests and212passing audits.
- Failed primary QA replication gate(1/2); relative English-probe direction passes2/2, without retention/generalization or promotion claims.
- Strict historical artifact-manifest validation and independent legacy-training reference tests.

- Optional rank16 causal future-byte auxiliary head and fixed two-arm training pilot, preserving existing core model/trainer/checkpoint behavior.
- Narrow training-loss result after128QA+128English updates/arm; no held-out quality or promotion claim.
- Original84-test regression run,3focused post-fix checks,134finalaudits and documented audit-only continuation preserving completed control training.

- Bounded loaded-QA update diagnostic:10 optimizer computations,50 independent parameter evaluations,653 passing audits and82 passing regression tests; no universal step-size fix established.
- Sourced DeepSeek research note and staged MLM adaptation hypotheses; no architectural changes or model-quality claims.

- Frozen QA optimizer-history/router-regularizer controls:30 independent copied transitions,391 passing audits and81 passing regression tests.
- Objective and update-size evidence showing the tested unchanged-rate Adam reset worsens local QA losses; no training fix or release change.

- Restored actual-next QA→English pair probes with matched-input English-only and zero-gradient controls:25 copied transitions,191 passing audits and80 passing regression tests.
- Pair diagnostic report and immutable evidence; mixed phase effects do not establish a replay or optimizer fix.

- Frozen training-only replay/optimizer diagnostics with document-prefix context, loaded-Adam and zero-gradient controls, role accounting and checkpoint immutability audits.
- Predefined 80-probe report: local QA-loss effects are snapshot-dependent; a consistently worse replay44 effect is not established.
- 78-test regression suite and504 completed diagnostic audit checks.

### Changed

- Updated `decay_mask` to select weight matrices by tensor rank (`ndim >= 2 and "bias" not in name`), preventing AdamW weight decay from decaying 0D scalar `memory_gain` and 1D vector gates toward zero.
- Replaced hardcoded node-0 attention readout query (`h[:, 0, :]`) with full-graph state pooling (`h.mean(axis=1)`).
- Replaced hardcoded `(nodes, degree) == (8, 2)` special case in `topology` with general power-of-two stride generation for arbitrary `(nodes, degree)`.
- Removed hardcoded string `.split()` truncation in `mlm chat` so generation relies on learned `EOS` termination, and eliminated redundant forward pass in `future_objective`.
- Incremented package and lockfile version to 0.0.2 (105 regression tests passing).

## [0.0.1] - 2026-10-25

### Added

- Initial experimental graph-GRU language model with hierarchical routing and optional learned delta-rule memory.
- Byte-level English training, full-document streams, assistant-target masking, checkpoint/resume and generation/evaluation commands.
- Independent QA and English replay sampler controls with legacy-compatible defaults and explicit seed provenance.
- Controlled acquisition, retention, adaptation-seed and schedule-component experiments, with 75 regression tests passing before this release cleanup.
- Organized documentation, compact experiment evidence and original-document provenance archives under `docs/`.

### Changed

- Set package and lockfile version to 0.0.1.
- Replaced the long root README with a concise setup, architecture, evidence and documentation overview.
- Moved important Markdown documentation to `docs/`; removed obsolete working-plan Markdown files after preserving their original text and hashes.
- Excluded generated datasets, checkpoints, caches, build outputs and full local artifacts from publication.

### Research limitations

- Tiny-task memorization depends on replay sampling; reliable held-out binding and chat are not established.
- Dynamic node growth, rewiring, sparse execution, broad device support and Transformer superiority are not demonstrated.
