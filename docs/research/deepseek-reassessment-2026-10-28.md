# DeepSeek and recurrent-memory research reassessment

Reviewed2026-10-28 after the completed MLM replication and answer analysis. **There are useful research directions, but no new improvement is demonstrated by this review.** The numerical check below only reuses frozen neural scores; no new optimizer, architecture, training or model inference was run.

## What changed since the first review

We already tested a disposable future-byte adaptation inspired by MTP. Its pilot lowered main QA CE, but benefit repeated in only1/2new comparisons, failing the predefined primary/joint gates. All six128-pair endpoints rank8/16training answers and0/4groups, with no context-swap prediction flips. Existing longer-budget selected acquisition remains real and is not refuted by these short runs. The immediate bottleneck is answer/context discrimination under the present budget, not demonstrated KV-cache cost. See [measured answer analysis](../experiments/future-byte-answers/report.md).

## R1: outcome training requires a capable base

[R1](https://arxiv.org/html/2501.12948v1) starts from pretrained DeepSeek-V3-Base. R1-Zero removes preliminary SFT, not language pretraining; R1 combines cold-start examples, RL and additional supervision. GRPO uses grouped rewards without a separately trained critic. Distillation results use pretrained Qwen/Llama students much larger than MLM.

Project inference: use verifiable task outcomes and short supervised examples before considering reasoning RL. A teacher can provide training data, but is not an inference dependency. Full GRPO and long reasoning traces are poor next bets for a384,784parameter backbone without stable elementary binding. Neither R1 nor its distillation demonstrates spontaneous network growth.

## V4 Flash: distinguish model efficiency from small-device feasibility

The [V4 report](https://arxiv.org/html/2606.19348v1) describes a Transformer/MoE Flash model with284Btotal/13Bactive parameters, hybrid compressed attention, mHC and Muon-based optimization. Its expert layers combine shared and selected routed experts. Those active counts still require large model storage. KV-cache compression does not directly target our fixed-size recurrent state. The [V4.1 report](https://arxiv.org/html/2609.19969v1) is a distinct successor with causal encoder-decoder/cache innovations; do not attribute its architecture to original V4.

Our application assessment: preserve the graph-GRU baseline and borrow one mechanism at a time. The parameter counts do not imply a Flash-like network fits every device, and DeepSeek's gains do not transfer automatically to tiny recurrent graphs. Muon needs a separate matched-initializer comparison with each logically independent GRU matrix handled correctly; the failed Adam reset does not favor Muon.

## DeepSeek architecture ideas ranked for this project

| Mechanism | Paper observation | MLM application and limitation |
|---|---|---|
| Shared/routed experts | [DeepSeekMoE](https://arxiv.org/html/2401.06066v1) separates common computation from specialist capacity | An always-updated shared recurrent path plus selected specialists could fit the web idea. Specialization is learned, not manually named grammar/brain categories. Define stale-state/message behavior; masking computed nodes is not sparse execution. |
| Loss-free balance | [Wang et al.](https://arxiv.org/html/2408.15664v1) adjust selection biases using historical load | Useful within a genuine top-k routing design. Bias influences selection, not output weighting. Prior-batch statistics preserve causality; current/future-sequence aggregate bias risks leakage. Our all-node execution has no sparse hardware-load problem yet. |
| MTP | [V3](https://arxiv.org/html/2412.19437v2) uses causal sequential extra-token modules | Our simpler head is not that implementation. It has already failed the short-budget answer-benefit checks; do not propose the same trial as new evidence. |
| mHC | [mHC](https://arxiv.org/html/2512.24880v1) constrains residual mixing to doubly stochastic maps | Candidate for controlled message/stream mixing. The residual-operator property does not prove whole-GRU stability or binding. Graph topology and projection cost need an explicit design. |
| Engram | [Engram](https://arxiv.org/html/2601.07372v1) combines static learned N-gram tables and contextual gating | May help repeated language patterns. It is not document-specific person–object memory. Hashing is an algorithm over learned embeddings, not hand-entered answers; storage/collisions/overfitting still matter. |
| Sparse/compressed attention | [NSA](https://arxiv.org/abs/2502.11089) and V4 target token-attention costs | Lower priority for our short fixed-state experiments. A small local-attention/recurrent hybrid is a separate architectural hypothesis, not a drop-in cache optimization or proven speedup. |

V3's main balancing mechanism is called auxiliary-loss-free but retains a small sequence-level balance loss. Simply deleting MLM's router penalty is not equivalent and already had mixed local results. All experts/nodes being computed under soft gates also differs from discrete MoE selection.

## Other papers: more direct matches to our failure

### Preference learning and conditional dependence

[DPO](https://arxiv.org/html/2305.18290v2) turns preference alignment into classification relative to a reference policy. [SimPO](https://arxiv.org/html/2405.14734v2) instead uses length-normalized sequence log probabilities without that reference and adds a target margin. Our current answer scorer already uses mean answer-byte log probability, making an adaptation technically natural. This is not evidence that their large-model chat gains transfer to MLM.

[mDPO](https://arxiv.org/html/2406.11839v1) documents a different setting—multimodal models neglecting images—and uses conditional preference plus anchoring. It motivates checking whether a model uses its conditioning input. A text-context contrast below is our proposed analogy, not a reproduction, a DeepSeek algorithm, or an established remedy for this graph model.

### Gated DeltaNet and Titans

[Gated DeltaNet](https://arxiv.org/html/2412.06464v1) combines data-dependent state decay with targeted delta-rule updates and studies recurrent/attention hybrids. Our memory already has error-correcting writes and a beta write gate; an independent alpha decay gate would be an additional mechanism. Specify read-before-write timing and preserve useful facts rather than assuming more forgetting helps.

[Titans](https://arxiv.org/html/2501.00663v1) learns an online neural memory with momentum/forgetting and retrieval. This is closer to dynamic memory weights than growing graph topology. Both papers justify investigating memory control, not automatically adding nodes. Prior MLM [chrono-style GRU retention initialization](../experiments/memory-retention/report.md) is a different intervention; it did not establish reliable understanding. The [addressing traces](../experiments/addressing/report.md) suggest weak query discrimination in that historical pilot, without proving a unique cause for current endpoints.

## Proposed first experiment: require the answer to depend on its facts

Let s(x,a) be the unchanged neural scorer's mean answer-byte log probability. For a known context-swap pair, compare the SAME candidate answer under the context where it is correct and the paired context where it is wrong:

    gap_context = s(context_correct, answer) - s(context_swapped, answer)
    L_context = mean softplus(-beta * gap_context)
    L_training = existing routed/SFT objective + lambda * L_context

This is a context-contrastive supervised objective proposal. Evaluate both assignment directions/answers; retain ordinary SFT to anchor absolute likelihood. Labels and pair metadata supply training targets only, never inference answers. The predictor still consumes bytes and uses learned parameters. No entity parser, keyword rule, answer lookup, teacher API or new inference head is needed.

An additive answer-only prior cancels in this same-answer context gap. With counterbalanced directions, a context-only additive offset also cannot satisfy both answers. However, the network can still memorize a tiny fixture, or optimize a surrogate without crossing decision boundaries. This objective cannot itself guarantee binding, free generation or generalization. Finite nonzero gradients and their relative scale need a preflight before training.

Compare unchanged SFT against SFT+one context loss with identical source, sampling schedules,128QA+128English exposure, optimizer and parameter count; keep MTP disabled and add no routing/memory/optimizer changes simultaneously. Freeze beta/lambda, masks, budget, seeds and gates BEFORE running—none is selected by this score audit. Judge primary success by more correct answer rankings AND at least one complete context/query group, then require independent-seed repetition and separately defined English retention/held-out checks. CE or preference-loss reduction alone cannot pass. This is a suggested design, not an implemented or executed training protocol.

## What the frozen-score check actually shows

We revalidated all188prior input hashes and the published/raw answer-analysis manifest. Only existing scores were read. For comparison we calculated ordinary answer-preference softplus and the proposed same-answer context softplus with beta1 and no margin. These are descriptive diagnostics, not fitted hyperparameters or results from training a new objective.

| Frozen model | Answer-preference loss | Same-answer context loss | Correct rankings /16 | Complete groups /4 |
|---|---:|---:|---:|---:|
| initial | 1.020962596 | 0.693198096 | 8 | 0 |
| pilot42_control | 0.709637153 | 0.693153518 | 8 | 0 |
| pilot42_future_byte | 0.705593790 | 0.693150407 | 8 | 0 |
| rep43_control | 0.693789828 | 0.693146199 | 8 | 0 |
| rep43_future_byte | 0.694237955 | 0.693146268 | 8 | 0 |
| rep44_control | 0.694290741 | 0.693145787 | 8 | 0 |
| rep44_future_byte | 0.693403485 | 0.693142006 | 8 | 0 |

A fully context-independent scorer gives a context loss of log(2)=0.693147181 on these balanced comparisons. The observed values are near that reference. Some score-level context response exists, but no context-swap predictions flip; do not call the probabilities exactly context-independent. Adding a constant per-answer score bias leaves all context gaps unchanged, checked arithmetically. The ordinary answer-preference surrogate can fall while rankings stay8/16, reinforcing the need for outcome gates even in preference training.

[Score audit](../evidence/deepseek-reassessment-2026-10-28/score-audit.json) records full precision, original-source hash, all seven models and the analytic sanity check. It contains no new model predictions, gradients or optimizer updates. No new quality improvement is demonstrated. The old failed primary/joint gates and longer-budget selected acquisition observations remain unchanged.

## Recommended order and decision

1. Test a carefully isolated context-contrastive objective with answer/group gates, if its gradient preflight is viable.
2. If architectural work is needed, prioritize learned query preservation and selective memory reads/writes, with a Gated-DeltaNet-inspired control or small local-attention hybrid as separate alternatives. Existing traces motivate investigation, not a diagnosis of the new models.
3. Explore shared/routed recurrent specialists after establishing baseline acquisition and measuring real dispatch cost. Keep balancing and mHC as separate ablations.
4. Defer Engram for language-pattern capacity, large-scale GRPO, Muon, distributed kernels and graph growth until their prerequisites are measured. No technology-stack change is required merely to test these ideas.

Worth trying: yes. Already proven better in MLM: no new architecture or objective from this reassessment. Do not claim that copying Flash, MoE, online memory or a preference loss makes a capable LLM. The next evidence must include answer correctness, complete groups, independent schedules and eventually held-out understanding, alongside runtime and memory measurements.

## Source scope

[Primary source registry](../evidence/deepseek-reassessment-2026-10-28/sources.json) records the14papers and inspected topics. Architecture facts use the reports, not speculative pull requests, third-party claims or benchmark marketing. No new dependencies, source code, model weights, training data or checkpoint selection were introduced by this research update.
