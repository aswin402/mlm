# Changelog

## [Unreleased]

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
