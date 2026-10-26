# DeepSeek research: what is transferable to MLM?

Official sources reviewed2026-10-26; project assessment2026-10-27. These are research hypotheses, not implemented improvements or validated transfer claims. No teacher API calls, model downloads, new datasets or training changes were made for this review.

## Project starting point

MLM is a byte-level fixed graph of8 GRU nodes, hierarchical soft update gates and optional learned document-local delta memory. All nodes still execute. Narrow training binding has been acquired under selected schedules, but independent-seed replication and held-out binding gates remain failed. Existing [QA controls](../experiments/qa-step-controls/report.md) reject the tested unchanged-rate Adam reset; simply deleting router regularization gives mixed local effects. Training stability and generalization remain open.

## R1 and DeepSeekMath: training sequence before reasoning RL

[R1](https://arxiv.org/html/2501.12948v1) applies reasoning post-training to an already pretrained DeepSeek-V3 base. R1-Zero uses reinforcement learning without supervised cold-start fine-tuning, not without language pretraining. R1 adds cold-start supervision, reasoning RL, rejection-sampled supervision and further RL. Its distillation experiments use already pretrained Qwen/Llama students much larger than MLM.

Transfer hypothesis: verified short teacher-generated examples may be a more practical early supervised experiment than direct reasoning RL. This does not supply the missing base language ability automatically. Keep a genuine-data control, record teacher/data provenance and evaluate disjoint entities, facts and wording. Teacher generation would be offline training-data production, not an inference-time answer service. Short answers and concise explanations are more proportionate than importing long reasoning traces now.

[DeepSeekMath](https://arxiv.org/abs/2402.03300) introduces GRPO; R1 uses group-relative rewards instead of a separately trained critic. Later, verifiable synthetic tasks could supply outcome rewards. RL is deferred until MLM can produce meaningful candidate answers. A correctness verifier used for training/evaluation is distinct from a handwritten inference answer generator; rewards alone do not create general language competence.

## DeepSeekMoE: shared processing and learned specialization

[DeepSeekMoE](https://arxiv.org/html/2401.06066v1) separates shared experts from fine-grained routed experts to reduce redundant learning and improve specialization. This is the closest connection to the original connected-mini-network idea, but its experts are feed-forward modules, not stateful GRU webs or growing nodes.

MLM hypothesis: maintain an always-updated shared recurrent path and route selected work to a few specialists. Learn specialization rather than manually declaring verb/science/brain categories. Define inactive-state persistence and stale-message handling explicitly. First compare soft gating against discrete routing while retaining dense execution to isolate representation effects; then separately test actual selective execution and latency. Masking already-computed outputs is not compute sparsity. Dispatch overhead can outweigh arithmetic savings at our tiny scale. Match parameter budgets and separately report active compute, total storage, runtime and quality.

## Load balancing: change its mechanism, do not just remove it

[Loss-free balancing](https://arxiv.org/html/2408.15664v1) updates per-expert selection biases from recent load: overloaded experts become harder to select, underused ones easier. This balances dispatch without that mechanism adding an auxiliary-loss gradient.

[V3](https://arxiv.org/html/2412.19437v2) uses the correction for top-k selection but original affinity scores for expert-output weighting; it retains a small complementary sequence-level auxiliary loss. Thus auxiliary-loss-free does not mean that every balancing loss disappears. Our soft recurrent gates are not the same routing system. A bias-based extension to soft gates would alter amplitudes and needs its own specification; it is not a faithful transplant.

Transfer hypothesis: compare balancing mechanisms only within an explicit routing design, holding other changes fixed. Measure utilization and collapse as well as QA/English quality. Our mixed coefficient-removal findings motivate this question but do not establish balance-gradient interference as the unique cause of failure.

## Multi-token prediction: bounded extra supervision

[V3 MTP](https://arxiv.org/html/2412.19437v2#S2.SS2) sequentially predicts additional future tokens with causal prediction modules and shared embedding/output components. The auxiliary modules can be removed for ordinary inference; speculative decoding is a separate possible use.

MLM hypothesis: first test one additional future-byte objective after the update-size baseline. A simple extra head is an adaptation, not an exact V3 reproduction. Prevent leakage into the main next-byte predictor, shift masks correctly, and exclude document boundaries and invalid assistant targets. Compare against added-parameter and training-compute controls. Byte-level extra supervision might primarily improve local spelling; it does not establish reasoning or promise faster decoding. Predefine seeds, budget and weighting before measuring; no coefficient search on the final test set.

## V4 Flash: efficiency mechanisms have specific targets

[Official release](https://deepseek.com/en/news/v4-preview/) and [V4 report](https://arxiv.org/html/2606.19348v1) describe a Transformer/MoE model with284B total and13B active parameters for Flash. CSA/HCA compressed attention addresses long-context attention and cache costs. That is not MLM's current recurrent-state structure or a drop-in speedup. Compression-inspired recurrent summaries would be a new design with a different accuracy/memory tradeoff.

V4 uses Muon for many matrices and AdamW for selected other parameter classes. Future MLM optimizer comparisons must handle each logically independent GRU matrix correctly, especially node-batched tensors and concatenated gates. Do not switch optimizers on the existing checkpoint as if histories were interchangeable. Use matched initialization and predeclared budgets/rates, and account for optimizer-state memory and wall time. The rejected Adam reset is not evidence for Muon.

The newer [V4.1 Flash report](https://arxiv.org/html/2609.19969v1) adds causal encoder-decoder and further KV-cache efficiency mechanisms. These still target a very different attention-based system. None demonstrates automatic node growth or broad-device feasibility for MLM. Model aliases/deployment availability can change; the linked papers are the technical basis here, not a deployment recommendation.

## mHC: constrained interaction, not a general recurrent stability proof

[mHC](https://arxiv.org/html/2512.24880v1) constrains residual-stream mixing using nonnegative doubly stochastic matrices. This controls the residual mixing operator and composition, not every nonlinear component of the model.

MLM hypothesis: test constrained mixing of node messages against the existing graph aggregation. Respect graph topology and any residual identity path; arbitrary sparsity masks can make the desired doubly stochastic constraints infeasible or alter connectivity. Sinkhorn-style projection adds cost. Stability of that mixing matrix does not prove stability of a GRU over long sequences, prevent oversmoothing, or guarantee correct factual binding. Defer until a specific communication bottleneck is demonstrated.

## Engram: static learned patterns versus changing document facts

[Engram](https://arxiv.org/html/2601.07372v1) retrieves learned embeddings through hashed local token patterns, with contextual gating and lightweight causal refinement. The [official repository](https://github.com/deepseek-ai/Engram) supplies a demonstration with mocked backbone components, not a complete small-model training recipe.

MLM hypothesis: a small byte-pattern embedding bank could help local English patterns. Learned table values would be trained parameters; hash addressing would be a deterministic algorithm, not manually entered answers. It adds storage and collision/overfitting risks. Compare under matched memory/parameter budgets and disjoint strings. Keep original byte identity rather than blindly copying normalization intended for another tokenizer. Static lookup does not replace document-local delta memory or solve changing person–object assignments; it must not become a shortcut for memorizing the tiny binding fixture.

## Ordered research gates

1. Complete the [bounded loaded-update control](../experiments/qa-update-scale/protocol.md). Distinguish reduced local harm from genuine improvement and from a trajectory-level fix.
2. If justified, freeze a single extra-future-byte objective with appropriate causal/mask tests and parameter/compute controls. If baseline instability remains unresolved, report that rather than bundling a new architecture into a supposed fix.
3. Separately specify shared recurrent processing and learned specialist routing; then compare balancing mechanisms and actual selective execution.
4. Consider constrained messages, Engram and teacher-data training as separate studies. Defer reasoning RL and a Muon comparison until prerequisites and budgets are explicit.

Every intervention must report all predefined seeds, held-out binding, English retention, actual runtime and memory. Do not pool unrelated historical runs as an independent replication, choose favorable replay seeds, or call training memorization generalization. Transformer replacement, category webs, graph growth and any-device execution remain unproven.


## Implementation status:2026-10-27

The [future-byte pilot](../experiments/future-byte/report.md) now implements the optional rank16 causal-output adaptation. It observed lower main training loss at one fixed schedule and short budget, without held-out/ranking evaluation or promotion. It is not a faithful V3 sequential-MTP implementation. Other architectural proposals above remain unimplemented; the next gate is predefined replication rather than declaring general capability.
