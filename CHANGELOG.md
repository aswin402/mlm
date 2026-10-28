# Changelog

## [Unreleased]

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
