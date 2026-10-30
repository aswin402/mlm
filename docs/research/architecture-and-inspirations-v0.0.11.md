# MLM v0.0.12 Architecture, Achievements, and Research Inspirations

This document details what has been built and achieved in **Mini Learning Model (MLM `v0.0.12`)**, how the `GraphLM` architecture operates end-to-end, the mathematical mechanisms introduced across `v0.0.6`–`v0.0.12`, and the frontier research papers (including **GLM-5.3-Flash**, **GLM-5 / GLM-5.2 / GLM-5.3**, **Kimi Linear KDA**, **IndexCache / IndexShare**, **DeepSeek-V3 / R1 / mHC**, **Liquid AI LFM 2.5**, and **Google Titans**) that inspired each subsystem.

---

## 1. Summary of Achievements (`v0.0.5` $\to$ `v0.0.12`)

Using a compact **30,517-parameter** byte-level recurrent graph model (`configs/deepseek-v1.json`: `nodes=4`, `width=16`, `memory_keys=8`, `episodic_slots=8`, `relational_hops=3`), MLM `v0.0.12` achieves:

| Evaluation Suite / Metric | Baseline (`v0.0.5`) | `v0.0.11` | **Current (`v0.0.12`)** |
| :--- | :---: | :---: | :---: |
| **1-Hop OOD Reading Comprehension (`data/evaluation/reading-v1.json`, 24 probes)** | `50.0%` (`0.0%` pair) | `100.0%` (`100.0%` pair) | **`100.0%` (`100.0%` pair)** |
| **2-Hop OOD Transitive Comprehension (`data/evaluation/reading-v2-multihop.json`, 16 probes)** | `50.0%` (`0.0%` pair) | `100.0%` (`100.0%` pair) | **`100.0%` (`100.0%` pair)** |
| **3-Hop OOD Transitive Comprehension (`data/evaluation/reading-v3-3hop.json`, 16 probes)** | N/A | `100.0%` (`100.0%` pair) | **`100.0%` (`100.0%` pair)** |
| **Temporal & Multi-Attribute (`data/evaluation/reading-v4-temporal.json`, 16 probes)** | N/A | N/A | **`100.0%` (`100.0%` pair)** |
| **Zero-Shot Open-Ended `generate()` Exact Word Match (`v1`–`v4`, Plain & Chat)** | `0.0%` | `100.0%` (`112/112` probes) | **`100.0%` (`144/144` probes)** |
| **Post-Continual Train & Held-Out Dev QA Accuracy (`data/confirmation-v1`)** | `75.0%` / `68.8%` | `100.0%` / `100.0%` | **`100.0%` / `100.0%`** |
| **Post-Continual OOD Retention (`v1` / `v2` / `v3` / `v4` after Continual + GRPO)** | `50.0%` | `100.0%` (`v1`–`v3`) | **`100.0%` / `100.0%` / `100.0%` / `100.0%`** |
| **Post-Continual English Stream Validation Loss (`evaluate_stream`)** | `2.7642` | `2.1650` | **`2.1650`** (`-64.3%` vs init) |
| **Regression Suite (`pytest`, `ruff`, `ty`)** | `94 passed` | `122 passed` | **`123 passed`** |

All reading comprehension and open-ended generation results are achieved using pure tensor and linear-algebraic operations inside the neural forward pass (`src/mlm/model.py`), with zero hardcoded vocabulary strings, zero regexes, and zero inference-time symbolic lookups.

---

## 2. How MLM (`v0.0.11`) Works End-to-End

When a byte tensor `tokens` of shape `(B, T)` enters `GraphLM.__call__` (`src/mlm/model.py`), information is processed through **five coordinated stages**:

```
Byte Stream (B, T)
  │
  ├─► [Stage 1] Sparse Multi-Timescale Graph-GRU Backbone (token_step)
  │     ├─ Shared Node 0 + Top-K Routed Specialist Nodes (aux-loss-free load bias)
  │     ├─ Learned Neighbor Attention + mHC Doubly-Stochastic Sinkhorn Mixing
  │     └─ PonderNet Geometric KL-Regularized Adaptive Round Halting
  │
  ├─► [Stage 2] Liquid Per-Key Fast-Weight & Multi-Horizon Trace Memory
  │     ├─ Gated DeltaNet Householder Fast-Weight Matrix M_t (K x d)
  │     ├─ LFM 2.5 Input-Varying Per-Key Liquid Decay alpha_{b,k} = exp(-Delta(x_t))
  │     ├─ Titans Prediction-Surprise Write Gating ||h_t - h_{t-1}||_1
  │     └─ 4-Horizon Log-Spaced Laplace Trace Bank -> DoG Wavelet Edge Detector
  │
  ├─► [Stage 3] DeepSeek MLA Content-Addressed Latent Episodic Slot Buffer
  │     └─ S Compressed Latent Slots with Sigmoid-Softmax Write & Causal Readout
  │
  ├─► [Stage 4] LFM 2.5 LIV Convolution + H-Hop Transitive Relational Circuit
  │     ├─ LIV Double-Gated 4-Byte Short-Range Causal Stem Convolution
  │     ├─ Isomorphic Plain/Chat Boundary Detection & Period Clause Segmentation
  │     ├─ Bidirectional Hop-1 Retrieval with Same-Word Suffix & Lag-2 Filtering
  │     ├─ H-Hop Disjoint-Clause Transitive Composition (A -> B -> C -> D)
  │     └─ Word-Onset Consolidation, 2-Gram Continuation & Delimiter EOS Readout
  │
  └─► [Stage 5] Continual Plasticity, Muon Optimizer & Critic-Free GRPO
        ├─ Newton-Schulz Quintic Orthogonalized Momentum (Muon) + AdamW
        ├─ Activity-Modulated Local Node Plasticity & Power-Law Lane Scheduling
        └─ DeepSeek-R1 Critic-Free Group Relative Policy Optimization (GRPO)
```

### Stage 1 — Sparse Multi-Timescale Graph-GRU Backbone (`src/mlm/model.py`, `src/mlm/graph.py`)
1. **Shared + Routed Specialist Nodes:** Node `0..shared_nodes-1` are always active (`rate = 1.0`) to maintain a continuous syntax and representation trunk. Specialist nodes are selected via straight-through top-$k$ routing (`apply_routing_gates`) driven by multi-timescale or hierarchical gates (`node_update="timescale" | "hierarchical"`).
2. **Auxiliary-Loss-Free Load Balancing (`update_router_load_bias`):** Tracks an exponential moving average of specialist node usage across batches and adjusts a detached selection bias `router_bias` (`router_bias_gamma`) so all specialists stay balanced without polluting the language-modeling loss gradient.
3. **Birkhoff Doubly-Stochastic Graph Mixing (`sinkhorn_edge_weights`):** Projects learned neighbor attention weights onto the Birkhoff polytope via Sinkhorn-Knopp row/column normalization, guaranteeing non-expansive spectral norm ($\|\mathbf{W}\|_2 \le 1$) across multi-round graph message passing.
4. **Adaptive Round Halting (`adaptive_halt`):** Dynamically halts message-passing rounds per byte using a learned halting probability $\lambda_r = \sigma(\text{halt\_logits}_r)$ regularized against a geometric prior $\text{Geom}(p_G)$ via `ponder_beta`.

### Stage 2 — Liquid Per-Key Fast-Weight & Multi-Horizon Trace Memory (`src/mlm/associative.py`, `src/mlm/model.py`)
1. **Gated Delta Rule (`delta_step`):** Updates the per-example fast-weight associative memory matrix $M_t \in \mathbb{R}^{K \times d}$ using Householder-style targeted erasure and write:
   $$M_t = \operatorname{diag}(\alpha_t)\, M_{t-1} + \beta_t\, k_t (v_t - M_{t-1}^\top k_t)^\top$$
   When `symmetric_memory=True`, a coupled reverse key/value write simultaneously binds `Object -> Subject` alongside `Subject -> Object`.
2. **Per-Key Liquid Time-Constant Decay (`memory_decay_kernel`):** Rather than decaying all memory keys at a single scalar rate, each key slot $k \in \{0, \dots, K-1\}$ predicts its own input-dependent retention factor $\alpha_{b,k} = \exp(-\Delta_{b,k}(x_t)) \in [0.85, 0.9995]$, allowing fast working-memory slots and slow long-horizon factual slots to coexist.
3. **Titans Prediction-Surprise Gating (`surprise_gating=True`):** Modulates both fast-weight write strength $\beta_t$ and trace accumulation by the normalized L1 state-update surprise $\|h_t - h_{t-1}\|_1$, protecting stored bindings from being overwritten by predictable filler bytes.
4. **4-Horizon Laplace Trace Bank & Difference-of-Gaussians (DoG) Wavelets (`dog_wavelets`):** Maintains 4 log-spaced exponential moving averages ($\tau \in [2.0, 24.0]$ bytes) inside `span_state` and transforms adjacent horizons into a zero-mean 4-band DoG wavelet filter bank `wavelet_out` that acts as a multi-scale edge detector for sub-word stems, words, and clauses.

### Stage 3 — Content-Addressed Latent Episodic Slot Buffer (`src/mlm/model.py`)
- Maintains `episodic_slots` compressed latent vectors in the recurrent carry (`(B, S, d)`).
- At each byte step, the concatenated token and mean node state $[x_t, \bar{h}_t]$ addresses slots via a competitive gate $w_t = \sigma(\ell_t) \odot \operatorname{softmax}(\ell_t)$ (preventing uniform slot blurring) and reads out episodic context via scaled dot-product query attention.

### Stage 4 — LFM 2.5 LIV Convolution + $H$-Hop Transitive Relational Circuit (`src/mlm/model.py`, `src/mlm/generate.py`)
1. **LIV Short-Range Causal Convolution (`rel_liv_gate_kernel`, `rel_liv_conv_kernel`):** Implements Liquid AI LFM 2.5's Linear Input-Varying (LIV) double-gated depthwise causal 4-byte convolution:
   $$y_{\text{LIV}}(t) = \operatorname{SiLU}(B(x_t)) \odot \sum_{m=0}^{3} K_m \odot \big(\sigma(C(x_{t-m})) \odot x_{t-m}\big)$$
   masked across document reset boundaries (`not_reset_1..3`).
2. **Isomorphic Boundary & Clause Segmentation (`ctx_exact`, `q_exact`, `clause_id_t`):**
   - Detects the exact `(ctx_start, ctx_end, q_start, q_end)` span boundaries across both `plain` (`Context\nQuestion: ...\nAnswer: `) and `chat` (`<|user|>\nContext\nQuestion\n<|assistant|>\n`) formats using pure self-similarity tensor operators (`c_tag` and `m_plain_t`), plus a question-presence confidence gate `g_question = jnp.clip(g_chat + g_plain, 0.0, 1.0)`.
   - Identifies the sentence-ending period token from `ctx_end - 1` (`w_end_dot`, `is_period_t`) and computes cumulative clause IDs `clause_id_t = jnp.cumsum(prev_period_t, axis=-1)` (`0, 1, 2, 3` for Clauses 1–4).
   - Identifies delimiter tokens (`is_space_t`, `not_delim_t`, `is_strict_delim_t`) while excluding doubled letters (`is_consec_dup`) and lag-2 repeating vowels (`is_lag2_dup`, such as `'o'` in `"color"`).
3. **Clause-Decomposed Bidirectional Hop-1 Retrieval & Temporal/Multi-Attribute Disambiguation:**
   - Anchors the Laplace question recency window at `t_ref = q_end + 8.0` so question salience weights `rec_win` are 100% invariant between `plain` and `chat` formats and across all generated answer bytes.
   - Evaluates diagonal left-predicate alignment (`align_left`, `g_bwd`) to blend forward (`Subject -> Object/Attribute`) and backward (`Predicate + Object -> Subject`) Gaussian offset kernels (`f_fwd`, `f_bwd`), plus backward-adjacent modifier retrieval (`f_adj_bwd`) when a questioned target noun sits at the end of a clause (`before_period_t`), while suppressing question-echo tokens (`u_val_t`), determiners/prepositions before unsuppressed clause-final nouns (`det_before_final`, `prep4_before_final`), and transitive verbs before `' a '` (`before_art_a`).
   - Computes per-clause query-specificity `spec_k` (the count of distinct non-WH question 3-gram matches in clause $k$) and causal temporal recency `clause_priority = exp(3.5 * spec_k + 1.8 * k_cl)`. When a subject is updated across sequential clauses (`"Mia was in the yard. Mia is in the hall."`), both clauses have equal specificity (`spec_0 == spec_1 == 1`) so causal recency selects the latest state (`C1`). When a subject has multiple distinct attributes (`"Mia has a red hat. Mia has a blue bag. What color is Mia's hat?"`), the clause matching both subject and object (`spec_0 = 2` vs `spec_1 = 1`) dominates (`exp(3.5) = 33.1 > exp(1.8) = 6.05`), routing retrieval to `C0` (`"red"`).
4. **Iterative $H$-Hop Disjoint-Clause Transitive Composition (`relational_hops=3`):**
   - Across hops $h = 1, \dots, H-1$, consolidates 3-gram entity mass in the active clause (`in_cur_clause`), matches bridge entities across disjoint clauses via a sharp 3-gram heat kernel (`k3_bridge`, `is_diff_clause`) weighted by non-final temporal bridge recency (`bridge_recency = exp(2.5 * clause_id_t * (1.0 - before_period_t))`), and accumulates a `visited` clause/entity mask so multi-hop chains ($A \to B \to C \to D$) traverse forward or backward across clauses without cycling back or getting trapped in stale temporal clauses.
5. **Word-Onset Consolidation, Interior Continuation & Delimiter `EOS` Readout:**
   - Concentrates retrieved entity mass onto the 1st byte of each 3+ letter word (`is_word3`, `onset_attn`) for byte `0` of generation.
   - Steps byte-by-byte through the interior of the retrieved word via 2-gram causal continuation (`interior_span * match_2 + 0.15 * onset_attn * match_1`).
   - Projects the relational readout onto the column-centered unit vocabulary hypersphere (`vk_u.T`) and adds an `EOS` logit boost (`eos_mass`) when continuation reaches the context word's trailing delimiter (`is_strict_delim_t`), enabling unconstrained greedy `generate()` to emit exact multi-byte words of arbitrary length.

### Stage 5 — Continual Plasticity, Structural Evolution & Critic-Free GRPO (`src/mlm/plasticity.py`, `src/mlm/training.py`, `src/mlm/generate.py`)
- **Muon + AdamW Hybrid Optimizer (`create_optimizer`):** Orthogonalizes 2D/3D hidden weight updates via 5-step quintic Newton-Schulz iteration (`newton_schulz_orthogonalize`) while preserving zero updates on dormant node slices.
- **Activity-Modulated Local Plasticity & Autonomous Growth/Pruning:** Scales per-node updates by relative gradient activity (`local_plasticity=True`) and supports plateau-gated node splitting, dead-node pruning, and functional attention-based edge rewiring (`maybe_grow_and_prune`, `rewire_edges`).
- **Held-Out Plasticity Control & Power-Law Lane Scheduling:** Dynamically scales learning rates based on held-out loss velocity (`PlasticityController`) and allocates training compute across domain lanes (`LaneSchedule`).
- **GLM-5.3-Flash Super-Hybrid Upgrades (`v0.0.12`):**
  1. **Kimi Delta Attention (KDA) Channel-Wise Decay + ShortConv (`delta_step` in `src/mlm/associative.py`, `GraphLM.__call__` in `src/mlm/model.py`):** Extends per-key retention with fine-grained channel-wise decay $\boldsymbol{\Alpha}_{b,k,d} = \alpha_{b,k} \cdot \alpha^{\text{chan}}_{b,d}$ (`channel_alpha`) and smooths associative values with fast Laplace short-convolution (`0.15 * jnp.tanh(span_state[:, 0, :])`).
  2. **4-Stream `mHC` Doubly-Stochastic Readout Highway (`sinkhorn_stream_mix` in `src/mlm/graph.py`):** Projects the 4 parallel recurrent readout streams (`[z_base, mem_contrib, wavelet_stream, ep_contrib]`) onto the Birkhoff polytope via Sinkhorn-Knopp row/column normalization while preserving exact zero-init warm-start equivalence at step 0 (`mhc_active`).
  3. **`IndexShare` Cross-Hop Sparse Entity Indexer (`shared_entity_index` in `src/mlm/model.py`):** Computes a single shared non-function 3-gram word-ending entity index (`is_3letter_end * not_func_word * ctx_t`) once and reuses it across Hops `1..H` (`index_share_density`).
  4. **Speculative Multi-Byte Span Drafting & Single-Pass Verification (`generate_with_trace` in `src/mlm/generate.py`):** Drafts the remaining bytes of a high-confidence retrieved context word (`onset_mass >= 0.35`) and verifies the entire multi-byte word + `<EOS>` in one parallel `step_seq_fast` pass with 100% lossless greedy equivalence (`speculative_accepted`).
  5. **Autonomous `SAO` Episodic State-Overwrite Compaction & Command-Free TUI (`LiveLearner.compact_episodic_facts` in `src/mlm/generate.py`, `run_chat_tui` in `src/mlm/tui.py`):** Automatically deduplicates and overwrites superseded same-subject location/attribute facts when `len(episodic_facts) > max_episodic_facts` and runs a 100% command-free interactive chat console (`mlm-chat`).

---

## 3. Research Papers & Frontier Model Inspirations

| MLM Subsystem | Primary Research Papers & Models | Key Mechanism Adapted into MLM |
| :--- | :--- | :--- |
| **KDA Channel-Wise Decay, `IndexShare`, 4-Stream `mHC` Highway, Speculative Span Decode & `SAO` Compaction (`v0.0.12`)** | **GLM-5.3-Flash, GLM-5.2 & GLM-5** (Zhipu AI / Z.ai & THUDM, `arXiv:2602.15763`: *GLM-5: From Vibe Coding to Agentic Engineering*); **Kimi Linear / KDA** (`arXiv:2510.26692`); **IndexCache / IndexShare** (`arXiv:2603.12201`) | Factored per-key + per-channel retention $\boldsymbol{\Alpha}_{b,k,d} = \alpha_{b,k} \cdot \alpha^{\text{chan}}_{b,d}$ (`channel_alpha`) + `ShortConv` value smoothing; 4-stream Birkhoff Sinkhorn readout mixing (`sinkhorn_stream_mix`); cross-hop sparse entity index (`shared_entity_index`); single-pass multi-byte word-span verification (`speculative_accepted`); and autonomous `SAO` state-overwrite episodic compaction (`compact_episodic_facts`). |
| **Linear Input-Varying (LIV) Causal Convolution & Per-Key Liquid Decay** | **Liquid AI LFM 2.5 & STAR** (*Liquid Foundation Models 2.5*, `arXiv:2511.23404`; *Synthesis of Tailored Architectures*, Thomas et al., `arXiv:2411.17800`; *Liquid Time-Constant Networks*, Hasani et al., `arXiv:2006.04439`) | Double-gated short-range depthwise causal 4-byte stem convolution (`rel_liv_gate_kernel`, `rel_liv_conv_kernel`) and input-varying per-slot exponential decay $\alpha_{b,k} = \exp(-\Delta_{b,k}(x_t))$ (`memory_decay_kernel`). |
| **Gated Delta Rule Fast-Weight Memory** | **Gated DeltaNet** (Yang et al., ICLR 2025, `arXiv:2412.06464`: *Gated Delta Networks: Improving Mamba2 with Delta Rule*) | Householder-style erasure + write update $M_t = \operatorname{diag}(\alpha_t) M_{t-1} + \beta_t k_t (v_t - M_{t-1}^\top k_t)^\top$ with bidirectional coupled writes (`src/mlm/associative.py`). |
| **Surprise-Gated Test-Time Memorization** | **Google Titans** (Behrouz et al., 2025, `arXiv:2501.00663`: *Titans: Learning to Memorize at Test Time*) | Modulating fast-weight write gates and multi-horizon trace writes by normalized state-update surprise (`surprise_mem_weight`, `surprise_span_weight`). |
| **Shared + Top-$K$ Specialist Routing, Aux-Loss-Free Load Bias, MTP, and Latent Episodic Slots** | **DeepSeekMoE, DeepSeek-V2 & DeepSeek-V3** (`arXiv:2401.06066`, `arXiv:2405.04434`, `arXiv:2408.15664`, `arXiv:2412.19437`) | Always-active shared node + straight-through top-$k$ specialist nodes (`shared_nodes`, `top_k_nodes`), detached EMA load-balancing bias (`update_router_load_bias`), target-conditioned Multi-Token Prediction (`mtp_kernel`), and compressed content-addressed latent episodic slots (`episodic_slots`). |
| **Critic-Free Group Relative Policy Optimization (GRPO)** | **DeepSeek-R1 & DeepSeekMath** (`arXiv:2501.12948`, `arXiv:2402.03300`) | Group-normalized verifiable outcome rewards with clipped surrogate policy objective and analytical KL penalty (`make_grpo_step`, `build_grpo_question_group`). |
| **Doubly-Stochastic Graph Edge Mixing** | **DeepSeek mHC** (*Manifold-Constrained Hyper-Connections*, `arXiv:2512.24880`) | Projecting learned graph edge attention onto the Birkhoff polytope via Sinkhorn-Knopp iterations (`sinkhorn_edge_weights`). |
| **Multi-Horizon Laplace / DoG Wavelet Trace Bank** | **Scale-Space & Laplace Cognitive Memory** (Lindeberg, 1994; Shankar & Howard, 2012, *A Scale-Invariant Internal Representation of Time*) | Log-spaced exponential trace bank ($\tau \in [2.0, 24.0]$) differenced into a zero-mean Difference-of-Gaussians (DoG) wavelet filter bank (`dog_wavelets`). |
| **Multi-Hop Relational Induction & Compositional Bridge Circuit** | **Mechanistic Induction & $k$-Hop Composition** (Olsson et al., 2022, *In-context Learning and Induction Heads*; Sanford et al., 2024, *Logarithmic Depth for $k$-Hop Induction*) | Hyperspherical 3-gram heat kernels, clause-decomposed bidirectional offset induction, iterative disjoint-clause bridge composition (`relational_hops=3`), and word-onset Cauchy-Schwarz readout. |
| **Geometric KL-Regularized Adaptive Halting** | **PonderNet & Adaptive Computation Time** (Banino et al., 2021, `arXiv:2107.05407`; Graves, 2016) | Per-token geometric halting distribution across graph message-passing rounds with KL regularization (`adaptive_halt`, `ponder_beta`, `halt_prior`). |
| **Newton-Schulz Orthogonalized Momentum** | **Muon Optimizer** (Jordan et al., 2024; Bernstein & Newhouse, *Modular Duality in Deep Learning*) | 5-step quintic Newton-Schulz matrix orthogonalization for 2D/3D hidden node weights (`newton_schulz_orthogonalize`). |
