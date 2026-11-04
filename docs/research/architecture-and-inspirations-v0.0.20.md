# MLM v0.0.20 Architecture, SpiderNet Hierarchy, Stateful O(1) Recurrent Decode Cache, and Research Inspirations

This document details the architecture, mathematical mechanisms, training algorithms, and empirical verification of **Mini Learning Model (`MLM v0.0.20`)**, synthesizing frontier advances across **DeepSeek-V3 / R1 / mHC**, **Gated DeltaNet & Kimi KDA**, **Liquid AI LFM 2.5**, **SenseNova-Loop**, **Byte Latent Transformer (`BLT`)**, **FoNE + Abacus Embeddings**, **FlexOlmo**, **EAGLE-3 Speculative Decoding**, **Process Reward Step Verification (`PRM`)**, **Recursive Language Models (`RLM`)**, **HazyResearch Next-Concept Modeling (`NCM`)**, and **Google Research `TurboQuant` (`ICLR 2026`) / `BiLLM + QJL`**.

---

## 1. Executive Summary: The SpiderNet Growing Architecture & O(1) Recurrent Inference (`v0.0.20`)

Instead of training monolithic dense models from scratch whenever a new domain or skill is needed, **MLM v2 (`v0.0.20`)** organizes intelligence as a **Hierarchical SpiderNet Graph** (`22,450,960` total parameters across 20 specialist sub-nodes, with only **top-$k=2$** base sub-nodes + conditional `<|think|>` sub-nodes active per token) paired with an exact **Stateful $O(1)$ Recurrent Decode Cache**:

```
                       ┌─────────────────────────────────────────────┐
                       │      SHARED MAIN CORE TRUNK (~4.6M)         │
                       │  • 4,096 Digit-Aware Rust BPE + BLT N-Grams │
                       │  • FoNE Fourier + Abacus Place Embeddings   │
                       │  • 8 Layers (12 Effective Passes via Loop)  │
                       │  • 6x LFM2 ShortConv (K=7) + Gated DeltaNet │
                       │  • 2x DeepSeek DSA + IndexShare GQA Layers  │
                       │  • 2-Stream mHC Doubly-Stochastic Highway   │
                       │  • Stateful O(1) Recurrent Decode Cache     │
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

### Key Properties of `MLM v0.0.20`
1. **Zero-Forgetting Modular Growth:** When adding or upgrading a specialist skill (e.g., Subtraction/Multiplication in `Nodes 12..13` or Division/Word Problems in `Nodes 14..15`), the Shared Main Core Trunk and all other Sub-Nodes remain **100% frozen** (`shared_lr_scale = 0.0`, `mature_lr_scale = 0.0`), and updated sub-node weights are spliced directly into the checkpoint via [`upgrade_subnodes_from_checkpoint_v2`](../../src/mlm/model_v2.py).
2. **Stateful $O(1)$ Recurrent Decode Cache ([`init_decode_cache_v2`](../../src/mlm/model_v2.py), [`_recurrent_token_step_v2`](../../src/mlm/model_v2.py), [`_make_cached_recurrent_decode_v2`](../../src/mlm/model_v2.py)):**
   Maintains a live static-shape JAX PyTree carry across all 12 effective hybrid passes (`ShortConv` $K=7$ ring buffer, `ncm_phi_bridge` ring buffer, exact rank-1 Gated DeltaNet recurrent state matrix $S_{t,h} \in \mathbb{R}^{D_h \times D_h}$, incremental `TurboQuant` GQA KV cache slots, and causal cumulative router load/domain-energy trackers), reducing per-token complexity from $O(T^2 H D_h)$ to $O(H D_h^2)$ with `< 2e-6` numerical deviation (`100%` greedy equivalence) and accelerating CPU inference to **28.4–78.7 tok/s** (**3×–7× faster** than full-sequence decoding).
3. **2-Token MTP Speculative Verification (`spec2`) + Single-Prefill Dual-Stream Fork (`_prefill_state`):**
   - `spec2(params, carry, tok_a, draft_b)` executes two recurrent steps in one XLA call and returns both `(carry_a, logits_a, draft_a)` and `(carry_b, logits_b, draft_b)`. On draft rejection, `generate_v2` rolls back to `carry_a` with **zero recomputation**.
   - In `DELIBERATIVE` and `VERIFIED` modes, `generate_v2` prefills the prompt **once** in 2-token `spec2` pairs and forks `fork_prefill_state = (carry, prompt_raw, prompt_draft, pos)` into both the `<|think|>` pass and the final response pass, eliminating 100% of duplicate prompt prefill.
4. **3-Mode Adaptive Cognitive Routing (`Direct` / `Deliberative` / `Verified`):** [`classify_cognitive_mode_v2`](../../src/mlm/model_v2.py) inspects the calibrated FlexOlmo cosine domain log-odds (`seq_dl`) and prompt structure at entrance to allocate test-time compute only when needed.
5. **Extreme Weight & KV-Cache Quantization (`16/8/4/2/1-bit`):** Using Google Research `TurboQuant` (`FWHT` + Gaussian Lloyd-Max centroids + 1-bit `QJL` residual sign correction) and `BiLLM` L2-preserving 1-bit quantization ([`quantize_v2.py`](../../src/mlm/quantize_v2.py)), the `22.45M`-parameter 20-node SpiderNet model compresses from **`85.81 MiB` (`fp32`) down to `12.47 MiB` (`2-bit`) and `10.66 MiB` (`1-bit`)** while retaining verified accuracy across English, Science, Arithmetic, Parentheses, and GSM8K Word Problems.

---

## 2. Seven Research Dimensions & Mathematical Formulation

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

### Dimension 2 — Stateful $O(1)$ Recurrent Formulation (`_recurrent_token_step_v2`)
During autoregressive generation (`generate_v2`), the full-sequence parallel operators factor into exact step-by-step recurrences across each of the $s \in \{0, \dots, 11\}$ scheduled layer passes:
1. **Exact Rank-1 Gated DeltaNet Matrix Recurrence:**
   Given the 1-step key/value Householder delta update $\tilde{k}_{t,h}, \tilde{v}_{t,h} \in \mathbb{R}^{D_h}$ and composite decay factor $\alpha_{t,h} = \gamma_{t,h}(1 - 0.15\beta_{t,h}) \in (0, 1)$, the parallel prefix operator
   $$\text{out}_{t,h} = \sum_{j=0}^{t} \left(\prod_{m=j+1}^{t} \alpha_{m,h}\right) \big(q_{t,h}^\top \tilde{k}_{j,h}\big)\tilde{v}_{j,h}$$
   is computed in $O(H D_h^2)$ time via the recurrent state matrix $S_{t,h} \in \mathbb{R}^{D_h \times D_h}$:
   $$S_{t,h} = \alpha_{t,h} S_{t-1,h} + \tilde{k}_{t,h}\tilde{v}_{t,h}^\top, \qquad \text{out}_{t,h} = q_{t,h}^\top S_{t,h}$$
2. **Causal `ShortConv` & `NCM` $\phi$-Bridge Shift Buffers:**
   Maintains fixed-size FIFO shift buffers `shortconv_buf` $\in \mathbb{R}^{(K-1) \times D}$ and `phi_conv_buf` $\in \mathbb{R}^{(W-1) \times D}$, updating via a single $K$-tap depthwise dot product per step.
3. **Incremental `DSA` + `TurboQuant` GQA KV Cache:**
   Writes the RoPE-rotated key/value vectors `(k_rot, v_cur)` (after optional `turboquant_kv_tensor_jax` compression) and lightning indexer key `k_idx` into slot `pos` of `k_cache`, `v_cache`, and `k_idx_cache`, masking positions $j > \text{pos}$.

### Dimension 3 — Subword & Numerical Representations (`BLT` + `FoNE` + `Abacus`)
1. **Digit-Aware 4,096-Subword Rust BPE:** Forces every single ASCII digit `'0'..'9'` and `<|think|>` / `<|/think|>` tag to be an atomic token.
2. **Byte Latent Transformer (`BLT`) Hashed Byte 3/4/5-Grams (`blt_token_ngrams`, `blt_proj.weight`):** Injects causal 3-token window mixtures (`0.60 * ng_t + 0.25 * ng_{t-1} + 0.15 * ng_{t-2}`) of hashed byte 3/4/5-gram features into the token embedding stream.
3. **Fourier Number Embeddings (`FoNE`, NeurIPS 2025) + Causal Abacus Place-Value Embeddings ([`compute_fone_abacus_embeddings`](../../src/mlm/model_v2.py)):**
   $$e_{\text{num}}(t) = \bigoplus_{k=1}^{8} \left[\sin\!\left(\frac{2\pi d_t k}{10}\right), \cos\!\left(\frac{2\pi d_t k}{10}\right)\right] W_{\text{FoNE}} + E_{\text{Abacus}}[p_t]$$

### Dimension 4 — FlexOlmo Cosine Routing & Rectified Sparse Domain Gate
[`routed_graph_nodes_ffn`](../../src/mlm/model_v2.py) and [`_route_single_bank_step_v2`](../../src/mlm/model_v2.py) compute L2-normalized cosine similarity logits across all 20 specialist sub-nodes:
$$\ell_{t,i} = 8.0 \cdot \frac{x_t}{\|x_t\|_2} \cdot \frac{w_i}{\|w_i\|_2}, \qquad i \in \{0, \dots, 19\}$$
- **Domain Log-Odds (`seq_dl`):**
  $$E_{\text{eng}}(t) = \log\sum_{i=0}^{7} e^{\ell_{t,i}} - \log 8, \qquad E_{\text{math}}(t) = \log\sum_{i=10}^{17} e^{\ell_{t,i}} - \log 8, \qquad \text{dl}_t = 2\big(E_{\text{math}}(t) - E_{\text{eng}}(t)\big)$$
- **Tiiny-AI / PowerInfer Rectified Sparse Gate:** When $\overline{\text{dl}}_{\text{prompt}} < -0.95$, $p_{\text{math}}$ is strictly rectified to `0.0` ($p_{\text{eng}} = 1.0$).

### Dimension 5 — Post-Router `<|think|>` Alignment & DeepSeek-R1 `GRPO`
- **Gate-Scale Equivalence & Post-Router Training Order:** `<|think|>` Sub-Nodes (`Nodes 8..9` and `Nodes 18..19`) are trained *after* `calibrate_spidernet_router_v2` using the exact `sigmoid(think_logits) * ornith_mask` gate scale used at inference, followed by [`run_grpo_reasoning_v2`](../../src/mlm/train_v2.py):
  $$\mathcal{L}_{\text{GRPO}}(\theta) = \mathcal{L}_{\text{CE}}(\theta; y^+) + \alpha_{\text{adv}}\,\operatorname{softplus}\!\big(m - (\mathcal{L}_{\text{CE}}(\theta; y^-) - \mathcal{L}_{\text{CE}}(\theta; y^+))\big) + \beta_{\text{KL}}\,\mathbb{D}_{\text{KL}}\!\big(\pi_{\text{ref}}(\cdot \mid y^+) \,\|\, \pi_\theta(\cdot \mid y^+)\big)$$

### Dimension 6 — Process Step Verification (`PRM`) & Recursive Language Models (`RLM`)
- **Process Step Verifier ([`_verify_reasoning_steps_v2`](../../src/mlm/model_v2.py), [`_verify_answer_against_thought_v2`](../../src/mlm/model_v2.py)):** Verifies intermediate equations $A \text{ op } B = C$ at `<|/think|>` and response boundaries and reconciles the final answer against the verified `<|think|>` derivation chain.
- **Recursive Self-Call Decomposition ([`_rlm_try_recursive_decomposition_v2`](../../src/mlm/model_v2.py)):** Evaluates nested parenthesized sub-expressions `(A op B)` via recursive neural self-calls `generate_v2(..., "What is A op B?", _rlm_depth + 1)`.

### Dimension 7 — `TurboQuant` & `BiLLM + QJL` Multi-Precision Quantization
- **`TurboQuant` (`arXiv:2504.19874`) & `BiLLM + QJL` ([`quantize_v2.py`](../../src/mlm/quantize_v2.py)):** Block-64 Fast Walsh-Hadamard Orthogonal Rotation (`FWHT`) + optimal Gaussian Lloyd-Max centroids (`8/4/2-bit`) or BiLLM L2-preserving group signs (`1-bit`) + 1-bit Quantized Johnson-Lindenstrauss (`QJL`) residual sign correction.

---

## 3. Empirical Results (`MLM v0.0.20`)

### 3.1 Multi-Precision Checkpoint Footprint (`22,450,960` Parameters)

| Precision Tier | Quantization Method | KV Cache Default | 20-Node SpiderNet (`.mlm2`) | 10-Node English (`.mlm2`) | Compression vs `fp32` |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`fp32` (Master)** | Unquantized `float32` | Unquantized | `85.81 MiB` | `43.60 MiB` | `1.00x` |
| **`16-bit`** | `fp16` | `TurboQuant 4-bit + QJL` | `43.63 MiB` | `26.72 MiB` | `1.97x` |
| **`8-bit`** | `turboquant_8bit_fwht64_qjl` | `TurboQuant 4-bit + QJL` | `23.18 MiB` | `14.46 MiB` | `3.70x` |
| **`4-bit`** | `turboquant_4bit_fwht64_qjl` | `TurboQuant 4-bit + QJL` | `15.93 MiB` | `10.12 MiB` | `5.39x` |
| **`2-bit`** | `turboquant_2bit_fwht64_qjl` | `TurboQuant 2-bit + QJL` | **`12.47 MiB`** | **`8.54 MiB`** | **`6.88x`** |
| **`1-bit`** | `turboquant_1bit_fwht64_qjl` | `TurboQuant 1-bit + QJL` | **`10.66 MiB`** | **`7.48 MiB`** | **`8.05x`** |

### 3.2 End-to-End Verification & Stateful $O(1)$ Recurrent Decode Speed (`2-Bit` Checkpoint on CPU)

| Prompt | Cognitive Mode & Active Sub-Nodes | `<think>` Trace (`Nodes 8..9` / `18..19`) | Final Response | `v0.0.19` Speed | **`v0.0.20` `O(1)` Cache Speed** |
| :--- | :--- | :--- | :--- | :---: | :---: |
| `hii` | `DIRECT \| english \| (2, 0)` | `Hello! How can I help you today?` | `Hello! How can I help you today?` | `5.1 tok/s` | **`78.7 tok/s`** |
| `How are you?` | `DELIBERATIVE \| english \| (0, 7)` | `As a Large Language Model (LLM), I don't have feelings like humans do. Thanks for asking though!` | `As a Large Language Model (LLM), I don't have feelings like humans do. Thanks for asking though!` | `20.5 tok/s` | **`56.3 tok/s`** |
| `What is science?` | `DELIBERATIVE \| english \| (0, 7)` | `Science is what we do to find out about the natural world.` | `Science is what we do to find out about the natural world. Natural sciences include, chemistry, biology, geology, astronomy, and physics. Science uses mathematics and logic, which are sometimes called "formal sciences".` | `16.2 tok/s` | **`42.4 tok/s`** |
| `What is photosynthesis?` | `DELIBERATIVE \| english \| (0, 7)` | `Photosynthesis is how plants and some microorganisms make carbohydrates.` | `Photosynthesis is how plants and some microorganisms make carbohydrates. It is an endothermic (takes in heat) chemical process which uses sunlight to turn carbon dioxide into sugars.` | `16.4 tok/s` | **`53.6 tok/s`** |
| `whats 4+4` | `VERIFIED \| math \| (10, 15)` | `4 + 4 = 8.` | `4 + 4 = 8. The answer is 8.` | `17.7 tok/s` | **`41.3 tok/s`** |
| `What is 1 plus 1?` | `VERIFIED \| math \| (11, 15)` | `1 + 1 = 2.` | `1 + 1 = 2. The answer is 2.` | `16.4 tok/s` | **`38.2 tok/s`** |
| `What is 15 - 7?` | `VERIFIED \| math \| (15, 10)` | `15 - 7 = 8.` | `15 - 7 = 8. The answer is 8.` | `18.6 tok/s` | **`33.5 tok/s`** |
| `What is 17 * 2?` | `VERIFIED \| math \| (10, 15)` | `17 * 2 = 34.` | `17 * 2 = 34. The answer is 34.` | `16.3 tok/s` | **`30.3 tok/s`** |
| `What is 36 / 6?` | `VERIFIED \| math \| (10, 15)` | `36 / 6 = 6.` | `36 / 6 = 6. The answer is 6.` | `14.0 tok/s` | **`33.6 tok/s`** |
| `What is (9 + 8) * 2?` | `VERIFIED \| math \| (10, 15)` | `9 + 8 = 17. 17 * 2 = 34.` | `9 + 8 = 17, and 17 * 2 = 34. The answer is 34.` | `26.5 tok/s` | **`51.6 tok/s`** |
| `Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?` | `VERIFIED \| math \| (15, 10)` | `48 / 2 = 24. 48 + 24 = 72.` | `Natalia sold 48 / 2 = 24 clips in May. 48 + 24 = 72. The answer is 72.` | `7.7 tok/s` | **`28.4 tok/s`** |
