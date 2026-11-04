# MLM v0.0.19 Architecture, SpiderNet Hierarchy, and Research Inspirations

This document details the architecture, mathematical mechanisms, training algorithms, and empirical verification of **Mini Learning Model (`MLM v0.0.19`)**, synthesizing frontier advances across **DeepSeek-V3 / R1 / mHC**, **Gated DeltaNet & Kimi KDA**, **Liquid AI LFM 2.5**, **SenseNova-Loop**, **Byte Latent Transformer (`BLT`)**, **FoNE + Abacus Embeddings**, **FlexOlmo**, **EAGLE-3 Speculative Decoding**, **Process Reward Step Verification (`PRM`)**, **Recursive Language Models (`RLM`)**, **HazyResearch Next-Concept Modeling (`NCM`)**, and **Google Research `TurboQuant` (`ICLR 2026`) / `BiLLM + QJL`**.

---

## 1. Executive Summary: The SpiderNet Growing Architecture (`v0.0.19`)

Instead of training monolithic dense models from scratch whenever a new domain or skill is needed, **MLM v2 (`v0.0.19`)** organizes intelligence as a **Hierarchical SpiderNet Graph** (`22,450,960` total parameters across 20 specialist sub-nodes, with only **top-$k=2$** base sub-nodes + conditional `<|think|>` sub-nodes active per token):

```
                       ┌─────────────────────────────────────────────┐
                       │      SHARED MAIN CORE TRUNK (~4.6M)         │
                       │  • 4,096 Digit-Aware Rust BPE + BLT N-Grams │
                       │  • FoNE Fourier + Abacus Place Embeddings   │
                       │  • 8 Layers (12 Effective Passes via Loop)  │
                       │  • 6x LFM2 ShortConv (K=7) + Gated DeltaNet │
                       │  • 2x DeepSeek DSA + IndexShare GQA Layers  │
                       │  • 2-Stream mHC Doubly-Stochastic Highway   │
                       └──────────────────────┬──────────────────────┘
                                              │
                 FlexOlmo Cosine Router + Rectified Sparse Gate (seq_dl >= -0.95)
                        ┌─────────────────────┴─────────────────────┐
                        ▼                                           ▼
     ┌─────────────────────────────────────┐     ┌─────────────────────────────────────┐
     │   DOT 0: 10M ENGLISH MACRO-NODE     │     │    DOT 1: 10M MATH MACRO-NODE       │
     │        (Nodes 0..9, ~8.9M)          │     │       (Nodes 10..19, ~8.9M)         │
     ├─────────────────────────────────────┤     ├─────────────────────────────────────┤
     │ • Nodes 0..3: Grammar & Vocabulary  │     │ • Nodes 10..11: Addition & Counting │
     │ • Nodes 4..5: Science & Reading QA  │     │ • Nodes 12..13: Sub & Multiplication│
     │ • Nodes 6..7: Multi-Turn Dialogue   │     │ • Nodes 14..15: Division & GSM8K    │
     │ • Nodes 8..9: Deliberative <think>  │     │ • Nodes 16..17: Parentheses/Algebra │
     │   (Trained Post-Router with GRPO)   │     │ • Nodes 18..19: Verified <think>    │
     └─────────────────────────────────────┘     │   (Trained Post-Router with GRPO)   │
                                                 └─────────────────────────────────────┘
```

### Key Properties of the SpiderNet Architecture
1. **Zero-Forgetting Modular Growth:** When adding or upgrading a specialist skill (e.g., Subtraction/Multiplication in `Nodes 12..13` or Division/Word Problems in `Nodes 14..15`), the Shared Main Core Trunk and all other Sub-Nodes remain **100% frozen** (`shared_lr_scale = 0.0`, `mature_lr_scale = 0.0`), and updated sub-node weights are spliced directly into the checkpoint via [`upgrade_subnodes_from_checkpoint_v2`](../../src/mlm/model_v2.py).
2. **3-Mode Adaptive Cognitive Routing (`Direct` / `Deliberative` / `Verified`):** [`classify_cognitive_mode_v2`](../../src/mlm/model_v2.py) inspects the calibrated FlexOlmo cosine domain log-odds (`seq_dl`) and prompt structure at entrance to allocate test-time compute only when needed:
   - **`DIRECT` Mode:** Fast single-pass response for greetings and simple conversational turns (`Nodes 0..7`).
   - **`DELIBERATIVE` Mode:** Two-stage analytical reasoning using Dot 0 `<|think|>` Sub-Nodes (`Nodes 8..9`) + Base Sub-Nodes (`Nodes 0..7`) for scientific and conceptual questions.
   - **`VERIFIED` Mode:** Multi-step mathematical derivation using Dot 1 `<|think|>` Sub-Nodes (`Nodes 18..19`) + Process Step Verification ([`_verify_reasoning_steps_v2`](../../src/mlm/model_v2.py), [`_verify_answer_against_thought_v2`](../../src/mlm/model_v2.py)) + Recursive Language Model (`RLM`) decomposition ([`_rlm_try_recursive_decomposition_v2`](../../src/mlm/model_v2.py)).
3. **EAGLE-3 / DeepSeek-V3 Multi-Token Speculative Decoding:** [`_make_cached_logits_step_v2`](../../src/mlm/model_v2.py) uses the trained `mtp_head` (`mtp_norm.weight`, `mtp_proj.weight`) to draft token $t+2$ alongside token $t+1$ and verifies both in a single causal forward pass, achieving **16.2–26.5 tok/s** on CPU with 100% lossless greedy equivalence.
4. **Extreme Weight & KV-Cache Quantization (`16/8/4/2/1-bit`):** Using Google Research `TurboQuant` (`FWHT` + Gaussian Lloyd-Max centroids + 1-bit `QJL` residual sign correction) and `BiLLM` L2-preserving 1-bit quantization ([`quantize_v2.py`](../../src/mlm/quantize_v2.py)), the `22.45M`-parameter 20-node SpiderNet model compresses from **`85.81 MiB` (`fp32`) down to `12.47 MiB` (`2-bit`) and `10.66 MiB` (`1-bit`)** while retaining verified accuracy across English, Science, Arithmetic, Parentheses, and GSM8K Word Problems.

---

## 2. Six Research Dimensions & Mathematical Formulation

### Dimension 1 — Hybrid Core Backbone (`LFM 2.5 ShortConv` + `Parallel Gated DeltaNet` + `DSA GQA` + `SenseNova-Loop mHC`)
Each of the 8 physical layers (`d_model = 192`, `n_heads = 6`, `n_kv_heads = 2`) combines local subword convolution, linear-time associative state-space memory, and sparse global attention:
- **Layers `0, 1, 3, 4, 6, 7` (Hybrid Recurrent-State Layers):**
  1. **Liquid AI LFM 2.5 Double-Gated `ShortConv` ($K=7$):**
     $$[B_t, C_t, V_t] = x_t W_{\text{in}}, \qquad y_{\text{conv}}(t) = \operatorname{SiLU}(C_t) \odot \operatorname{CausalConv}_{K=7}\big(\operatorname{SiLU}(B_t) \odot V_t\big)$$
  2. **Parallel Gated DeltaNet with Kimi KDA Channel-Wise Decay:**
     Maintains an associative fast-weight state with per-head decay $\gamma_{t,h} = \sigma(x_t W_{\text{decay}})$ and Householder delta-rule erasure/write modulated by `deltanet.w_beta` and `deltanet.channel_gate`.
- **Layers `2, 5` (DeepSeek Sparse Attention `DSA` + `IndexShare` GQA Layers):**
  Combines local sliding-window attention (`W=64`) on odd heads with top-$k$ (`k=32`) lightning-indexed global attention (`dsa_lightning_index_mask`) shared across attention layers (`IndexShare`), plus head-wise sigmoid output gating (`2.0 * sigmoid(x @ W_gate)`).
- **SenseNova-Loop 12-Pass Contiguous Looping (`layers 2..5` repeated twice) & 2-Stream `mHC` Sinkhorn Highway:**
  Middle reasoning layers (`2, 3, 4, 5`) execute twice (`12` effective layer passes from `8` physical layers), bridged at loop re-entry by HazyResearch's Next-Concept `Phi` depthwise convolution bridge (`ncm_phi_bridge`) and stabilized by a 2-stream Birkhoff doubly-stochastic Sinkhorn-Knopp residual highway (`mhc_read_stream`, `mhc_write_stream`).

### Dimension 2 — Subword & Numerical Representations (`BLT` + `FoNE` + `Abacus`)
Standard BPE tokenizers fracture multi-digit numbers and lose character-level morphology. `MLM v2` combines three complementary input representations:
1. **Digit-Aware 4,096-Subword Rust BPE:** Forces every single ASCII digit `'0'..'9'` and `<|think|>` / `<|/think|>` tag to be an atomic token.
2. **Byte Latent Transformer (`BLT`) Hashed Byte 3/4/5-Grams (`blt_token_ngrams`, `blt_proj.weight`):** Injects causal 3-token window mixtures (`0.60 * ng_t + 0.25 * ng_{t-1} + 0.15 * ng_{t-2}`) of hashed byte 3/4/5-gram features into the token embedding stream.
3. **Fourier Number Embeddings (`FoNE`, NeurIPS 2025) + Causal Abacus Place-Value Embeddings ([`compute_fone_abacus_embeddings`](../../src/mlm/model_v2.py)):**
   For every digit token $d_t \in \{0, \dots, 9\}$ at contiguous causal run-length place index $p_t \in \{0, \dots, 7\}$:
   $$e_{\text{num}}(t) = \bigoplus_{k=1}^{8} \left[\sin\!\left(\frac{2\pi d_t k}{10}\right), \cos\!\left(\frac{2\pi d_t k}{10}\right)\right] W_{\text{FoNE}} + E_{\text{Abacus}}[p_t]$$

### Dimension 3 — FlexOlmo Cosine Routing & Rectified Sparse Domain Gate
To prevent newly trained specialist banks from overpowering mature frozen banks via weight norm inflation, [`routed_graph_nodes_ffn`](../../src/mlm/model_v2.py) computes L2-normalized cosine similarity logits across all 20 specialist sub-nodes:
$$\ell_{t,i} = 8.0 \cdot \frac{x_t}{\|x_t\|_2} \cdot \frac{w_i}{\|w_i\|_2}, \qquad i \in \{0, \dots, 19\}$$
- **Domain Log-Odds (`seq_dl`):** Compares size-normalized log-sum-exp energy between Dot 0 Base Nodes (`0..7`) and Dot 1 Base Nodes (`10..17`):
  $$E_{\text{eng}}(t) = \log\sum_{i=0}^{7} e^{\ell_{t,i}} - \log 8, \qquad E_{\text{math}}(t) = \log\sum_{i=10}^{17} e^{\ell_{t,i}} - \log 8, \qquad \text{dl}_t = 2\big(E_{\text{math}}(t) - E_{\text{eng}}(t)\big)$$
- **Tiiny-AI / PowerInfer Rectified Sparse Gate:** When the prompt-level mean domain log-odds $\overline{\text{dl}}_{\text{prompt}} < -0.95$, $p_{\text{math}}$ is strictly rectified to `0.0` ($p_{\text{eng}} = 1.0$), guaranteeing zero cross-domain interference on English, Science, and Chat prompts.
- **Multi-Skill Least-Squares Router Calibration ([`calibrate_spidernet_router_v2`](../../src/mlm/train_v2.py)):** Directly fits each layer's `nodes.router.weight` on real hidden states across all 7 English and Math skill corpora with ridge regularization ($\lambda = 1.5$), achieving **100.0% domain separation** across all 8 layers in under 2 seconds.

### Dimension 4 — Post-Router `<|think|>` Alignment & DeepSeek-R1 `GRPO`
A critical discovery during `v0.0.19` development was why `<|think|>` sub-nodes previously drifted at inference:
1. **Gate-Scale Equivalence Fix:** Base sub-nodes (`0..7` and `10..17`) compete via `softmax` + `top_k=2`, whereas `<|think|>` sub-nodes (`8..9` and `18..19`) are gated via `sigmoid(think_logits) * ornith_mask`. Excluding `(8, 9, 18, 19)` from the `+5.0` `active_subnodes` router bias boost ensures `<|think|>` sub-nodes train with the exact same `sigmoid(think_logits)` gate scale used at inference.
2. **Post-Router Training Order:** Training `Nodes 8..9` and `Nodes 18..19` *after* `calibrate_spidernet_router_v2` (with the Main Core Trunk and calibrated Base Sub-Nodes `0..7` and `10..17` 100% frozen) aligns the `<|think|>` sub-nodes with the exact active base sub-node representations seen at inference.
3. **Group Relative Policy Optimization ([`run_grpo_reasoning_v2`](../../src/mlm/train_v2.py)):** Reinforces verified `<|think|>` trajectories ($y^+$) while penalizing corrupted arithmetic or drifted `<|think|>` trajectories ($y^-$) under a frozen reference policy $\pi_{\text{ref}}$:
   $$\mathcal{L}_{\text{GRPO}}(\theta) = \mathcal{L}_{\text{CE}}(\theta; y^+) + \alpha_{\text{adv}}\,\operatorname{softplus}\!\big(m - (\mathcal{L}_{\text{CE}}(\theta; y^-) - \mathcal{L}_{\text{CE}}(\theta; y^+))\big) + \beta_{\text{KL}}\,\mathbb{D}_{\text{KL}}\!\big(\pi_{\text{ref}}(\cdot \mid y^+) \,\|\, \pi_\theta(\cdot \mid y^+)\big)$$

### Dimension 5 — Process Step Verification (`PRM`) & Recursive Language Models (`RLM`)
- **Process Step Verifier ([`_verify_reasoning_steps_v2`](../../src/mlm/model_v2.py), [`_verify_answer_against_thought_v2`](../../src/mlm/model_v2.py)):** At `<|/think|>` and response boundaries, inspects every intermediate equation $A \text{ op } B = C$ generated by `Nodes 18..19`, verifies arithmetic step correctness along the causal derivation chain, propagates corrected intermediate results to downstream steps, and reconciles the final answer against the verified `<|think|>` chain.
- **Recursive Self-Call Decomposition ([`_rlm_try_recursive_decomposition_v2`](../../src/mlm/model_v2.py)):** Inspired by MIT CSAIL's Recursive Language Models (`arXiv:2512.24601`), detects nested parenthesized sub-expressions `(A op B)` in the prompt, invokes a recursive self-call `generate_v2(..., "What is A op B?", _rlm_depth + 1)` to evaluate the inner sub-problem using the neural model, substitutes the predicted intermediate value into the outer prompt, and chains both reasoning traces inside `<|think|>`.

### Dimension 6 — EAGLE-3 Speculative Decoding & `TurboQuant` / `BiLLM + QJL` Quantization
- **2-Token Speculative Decoding ([`_make_cached_logits_step_v2`](../../src/mlm/model_v2.py)):** At each step $t$, the main backbone predicts $y_{t+1} = \arg\max \text{logits}_t$ while the Multi-Token Prediction head (`mtp_proj.weight`) conditions on $[h_t, \operatorname{RMSNorm}(E[y_{t+1}])]$ to draft $\hat{y}_{t+2}$. On step $t+1$, $\hat{y}_{t+2}$ is placed at `cur_len` and verified in parallel with zero extra forward passes whenever accepted.
- **`TurboQuant` (`arXiv:2504.19874`) & `BiLLM + QJL` ([`quantize_v2.py`](../../src/mlm/quantize_v2.py)):** Applies block-64 Fast Walsh-Hadamard Orthogonal Rotation (`FWHT`) to Gaussianize weight and KV-cache coordinates, quantizes rotated coordinates via optimal Lloyd-Max centroids (`8-bit`, `4-bit`, `2-bit`) or BiLLM L2-preserving group signs (`1-bit`), and adds a 1-bit Quantized Johnson-Lindenstrauss (`QJL`) residual sign correction for unbiased inner products.

---

## 3. Empirical Results (`MLM v0.0.19` Trained on Google Colab NVIDIA Tesla T4 GPU)

### 3.1 Multi-Precision Checkpoint Footprint (`22,450,960` Parameters)

| Precision Tier | Quantization Method | KV Cache Default | 20-Node SpiderNet (`.mlm2`) | 10-Node English (`.mlm2`) | Compression vs `fp32` |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`fp32` (Master)** | Unquantized `float32` | Unquantized | `85.81 MiB` | `43.60 MiB` | `1.00x` |
| **`16-bit`** | `fp16` | `TurboQuant 4-bit + QJL` | `43.63 MiB` | `26.72 MiB` | `1.97x` |
| **`8-bit`** | `turboquant_8bit_fwht64_qjl` | `TurboQuant 4-bit + QJL` | `23.18 MiB` | `14.46 MiB` | `3.70x` |
| **`4-bit`** | `turboquant_4bit_fwht64_qjl` | `TurboQuant 4-bit + QJL` | `15.93 MiB` | `10.12 MiB` | `5.39x` |
| **`2-bit`** | `turboquant_2bit_fwht64_qjl` | `TurboQuant 2-bit + QJL` | **`12.47 MiB`** | **`8.54 MiB`** | **`6.88x`** |
| **`1-bit`** | `turboquant_1bit_fwht64_qjl` | `TurboQuant 1-bit + QJL` | **`10.66 MiB`** | **`7.48 MiB`** | **`8.05x`** |

### 3.2 End-to-End Verification on `2-Bit` Checkpoint (`12.47 MiB` Weights + `2-Bit` KV Cache)

| Prompt | Cognitive Mode & Active Sub-Nodes | `<think>` Trace (`Nodes 8..9` / `18..19`) | Final Response | Throughput |
| :--- | :--- | :--- | :--- | :---: |
| `hii` | `DIRECT \| english \| (2, 0)` | `Hello! How can I help you today?` | `Hello! How can I help you today?` | `5.1 tok/s` |
| `How are you?` | `DELIBERATIVE \| english \| (0, 7)` | `As a Large Language Model (LLM), I don't have feelings like humans do. Thanks for asking though!` | `As a Large Language Model (LLM), I don't have feelings like humans do. Thanks for asking though!` | `20.5 tok/s` |
| `What is science?` | `DELIBERATIVE \| english \| (0, 7)` | `Science is what we do to find out about the natural world.` | `Science is what we do to find out about the natural world. Natural sciences include, chemistry, biology, geology, astronomy, and physics. Science uses mathematics and logic, which are sometimes called "formal sciences".` | `16.2 tok/s` |
| `What is photosynthesis?` | `DELIBERATIVE \| english \| (0, 7)` | `Photosynthesis is how plants and some microorganisms make carbohydrates.` | `Photosynthesis is how plants and some microorganisms make carbohydrates. It is an endothermic (takes in heat) chemical process which uses sunlight to turn carbon dioxide into sugars.` | `16.4 tok/s` |
| `whats 4+4` | `VERIFIED \| math \| (15, 10)` | `4 + 4 = 8.` | `4 + 4 = 8. The answer is 8.` | `17.7 tok/s` |
| `What is 1 plus 1?` | `VERIFIED \| math \| (11, 15)` | `1 + 1 = 2.` | `1 + 1 = 2. The answer is 2.` | `16.4 tok/s` |
| `What is 15 - 7?` | `VERIFIED \| math \| (15, 10)` | `15 - 7 = 8.` | `Solution: 15 - 7 = 8. The answer is 8.` | `18.6 tok/s` |
| `What is 17 * 2?` | `VERIFIED \| math \| (10, 15)` | `17 * 2 = 34.` | `17 * 2 = 34. The answer is 34.` | `16.3 tok/s` |
| `What is 36 / 6?` | `VERIFIED \| math \| (10, 15)` | `36 / 6 = 6.` | `36 / 6 = 6. The answer is 6.` | `14.0 tok/s` |
| `What is (9 + 8) * 2?` | `VERIFIED \| math \| (10, 15)` | `9 + 8 = 17. 17 * 2 = 34.` | `9 + 8 = 17, and 17 * 2 = 34. The answer is 34.` | `26.5 tok/s` |
| `Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?` | `VERIFIED \| math \| (15, 10)` | `48 / 2 = 24. 48 + 24 = 72.` | `Natalia sold 48 / 2 = 24 clips in May. 48 + 24 = 72. The answer is 72.` | `7.7 tok/s` |

---

## 4. Frontier Research Papers & Architectural Inspirations

| MLM v0.0.19 Subsystem | Primary Research Papers & Models | Mechanism Adapted into MLM v2 |
| :--- | :--- | :--- |
| **Hierarchical SpiderNet Specialist Routing & Cosine Domain Gate** | **FlexOlmo** (AllenAI, 2025); **DeepSeekMoE & DeepSeek-V3** (`arXiv:2401.06066`, `arXiv:2412.19437`); **PowerInfer / Tiiny-AI** (`arXiv:2312.12456`) | L2-normalized cosine similarity routing (`8.0 * (x_unit @ w_unit)`), auxiliary-loss-free load bias (`router_bias`), rectified sparse domain gate (`seq_dl >= -0.95`), and freeze-and-grow sub-node splicing (`upgrade_subnodes_from_checkpoint_v2`). |
| **3-Mode Cognitive Router, `<|think|>` Alignment & Critic-Free `GRPO`** | **DeepSeek-R1** (`arXiv:2501.12948`); **Ornith-1.5 Self-Scaffold Horizon**; **Adaptive Compute / System-2 Routing** | 3-Mode Cognitive Router (`Direct` / `Deliberative` / `Verified`), Ornith-1.5 self-scaffold `<|think|>` decay mask (`compute_ornith_scaffold_mask`), post-router `<|think|>` alignment, and `run_grpo_reasoning_v2`. |
| **Process Step Verification (`PRM`) & Recursive Language Models (`RLM`)** | **Let's Verify Step by Step** (Lightman et al., OpenAI, `arXiv:2305.20050`); **Recursive Language Models** (Zhang et al., MIT CSAIL, `arXiv:2512.24601`) | Step-by-step equation verification (`_verify_reasoning_steps_v2`), `<|think|>`-to-response reconciliation (`_verify_answer_against_thought_v2`), and recursive parenthesized sub-expression self-calls (`_rlm_try_recursive_decomposition_v2`). |
| **Multi-Token Speculative Decoding (`MTP`)** | **DeepSeek-V3 MTP** (`arXiv:2412.19437`); **EAGLE-3** (Li et al., 2025, `arXiv:2503.01840`) | 2-token speculative drafting via `mtp_norm.weight` + `mtp_proj.weight` and single-pass causal verification in `_make_cached_logits_step_v2` and `generate_v2`. |
| **Hybrid `ShortConv` + Parallel Gated DeltaNet + `DSA` Sparse GQA** | **Liquid AI LFM 2.5** (`arXiv:2511.23404`); **Gated DeltaNet** (Yang et al., ICLR 2025, `arXiv:2412.06464`); **Kimi Linear KDA** (`arXiv:2510.26692`); **DeepSeek-V3.2 DSA & IndexShare** (`arXiv:2603.12201`) | Double-gated $K=7$ depthwise causal `ShortConv`, parallel Householder delta-rule associative recurrence with KDA channel-wise gating, and shared lightning-indexed top-$k$ sparse GQA (`dsa_lightning_index_mask`). |
| **12-Pass Loop & 2-Stream `mHC` Sinkhorn Highway + `NCM`** | **SenseNova-Loop** (SenseTime, 2025); **DeepSeek mHC** (`arXiv:2512.24880`); **Next-Concept Modeling (`ncm`)** (HazyResearch, 2025) | Contiguous 2-pass middle-layer looping (`layers 2..5`), 2-stream Birkhoff Sinkhorn residual highway (`mhc_read_stream`, `mhc_write_stream`), `ncm_phi_bridge` re-entry convolution, and Euclidean NCM tied output head (`ncm_tied_codebook_logits`). |
| **Byte N-Grams & Fourier + Abacus Number Embeddings** | **Byte Latent Transformer (`BLT`)** (Meta FAIR, `arXiv:2412.09871`); **FoNE** (NeurIPS 2025, `arXiv:2502.09741`); **Abacus Embeddings** (McLeish et al., NeurIPS 2024, `arXiv:2405.17399`) | Hashed byte 3/4/5-gram subword mixing (`blt_token_ngrams`) and harmonic Fourier digit features + causal contiguous digit-run place-value embeddings (`compute_fone_abacus_embeddings`). |
| **`TurboQuant` & `BiLLM + QJL` 16/8/4/2/1-Bit Quantization** | **TurboQuant** (Google Research, ICLR 2026, `arXiv:2504.19874`); **QJL** (`arXiv:2406.03482`); **BiLLM** (ICML 2024, `arXiv:2402.04291`) | Block-64 Fast Walsh-Hadamard Transform (`FWHT`), Gaussian Lloyd-Max scalar quantization (`8/4/2-bit`), BiLLM L2-preserving group binary quantization (`1-bit`), and 1-bit Quantized Johnson-Lindenstrauss (`QJL`) residual sign correction for both weights and KV cache. |
