# Mini Learning Model (MLM): Product Requirements

Date: 2026-10-23
Version: 0.1
Status: Initial draft for research prototype; no implementation or performance claim.
Research basis: [research.md](research.md).

## 1. Vision and original intent

Build an experimental language-model architecture composed of small neural networks connected like a spiderweb. Nodes can contain subnetworks; connections can change; capacity can grow within resource limits. Investigate whether selective computation and adaptation can improve language modeling or continual learning efficiency.

The long-term ambition is a small, fast, efficient alternative to standard Transformer models, accessible across multiple device classes. This is an aspiration to validate, not a promised outcome. The project name is Mini Learning Model; MLM must be defined explicitly to avoid confusion with masked language modeling.

The first product is a reproducible research system and local language-model prototype. A hosted assistant, consumer app, and universal deployment runtime are outside the initial release.

## 2. Problem

We need a way to test whether a network of small recurrent modules can allocate computation and capacity more effectively than fixed reference models. Without controlled baselines, explicit state semantics, and full resource accounting, apparent growth or sparsity could conceal worse latency, higher memory, or lower language quality.

The system must make both positive and negative results easy to reproduce.

## 3. Users and jobs

| User | Need |
|---|---|
| Founder / architecture researcher | Preserve the idea, implement variants, and assess evidence |
| Experiment contributor | Run a documented experiment with validated configuration |
| Reviewer | Compare models fairly and inspect topology, metrics, and structural events |
| Future local user | Run an explicitly supported model within a measured device budget |

Representative workflows:

1. Validate an experiment configuration and estimate whether it fits a declared capacity limit.
2. Train a baseline or graph model using the same data pipeline.
3. Resume a checkpoint with the same graph, optimizer, and recurrent-state policy.
4. Compare quality, runtime, memory, and adaptation metrics.
5. Generate text locally from a saved model.
6. Later, enable rewiring or bounded growth and inspect every structural event.

## 4. Decisions and assumptions

### Confirmed from the discussion

- Preserve the spiderweb concept and small neural modules.
- Investigate new architecture rather than assume a Transformer replacement already exists.
- Treat growth, dynamic connectivity, and eventual adaptation as core research directions.
- Prioritize measured efficiency and small-device accessibility.
- Produce research and requirements documents before implementation.

### Working proposals, requiring technical validation

- Primary research stack: Python, JAX, Flax, and Optax.
- Development tools: uv, Ruff, ty, and Pydantic.
- Polars when useful for dataset metadata and result analysis.
- A flat fixed graph before hierarchical or growing variants.
- TinyStories as the first restricted language experiment.
- Approximately 1–10 million total parameters as an exploratory starting scale.

Burn remains the Rust-first alternative. Candle, tinygrad, PyTorch, Mojo, and Rust extensions remain options with the roles described in research.md. Versions and hardware have not been selected. No dependency installation, kernel implementation, or framework port is authorized by this document alone.

## 5. Goals and non-goals

### Initial goals

- Train and evaluate a causal graph language model.
- Compare against small Transformer and recurrent baselines.
- Reproduce results from configuration, source revision, and saved artifacts.
- Measure actual executed work and resource use.
- Establish safe structural boundaries for later rewiring and growth.
- Keep the original vision visible as staged research requirements.

### Later research goals

- Input-dependent module routing.
- Learned rewiring under a bounded edge budget.
- Bounded capacity growth with optimizer-state migration.
- Continual learning and neural-memory updates during use.
- Hierarchical modules and cross-device inference.

### Initial non-goals

- Frontier assistant quality or broad factual competence.
- Guaranteed superiority to Transformers.
- Unlimited growth or unrestricted persistent learning from every input.
- Useful execution on every device.
- Simultaneous implementations in multiple ML frameworks.
- Web UI, hosted API, distributed training, or custom kernels before a justified need.

## 6. System outline

Data pipeline → shared tokenizer → selected model → causal training loop → evaluation → checkpoint and experiment report.

Graph model: token embedding → input injection → bounded recurrent message passing → readout → next-token logits.

The structural controller is separate from the token update path and initially disabled. It later runs between training intervals. Graph topology is represented explicitly as data, with invariants and resource limits.

## 7. Functional requirements

| ID | Priority / stage | Requirement and acceptance evidence |
|---|---|---|
| FR-01 | Must / foundation | Validate configuration before training; reject invalid dimensions, degree, capacity, and unknown settings with actionable errors |
| FR-02 | Must / foundation | Save resolved configuration, dataset identity, tokenizer identity, seed, code revision when available, versions, and device information for each run |
| FR-03 | Must / foundation | Use reproducible train/validation/test boundaries; log transformations and exclude known split overlap |
| FR-04 | Must / foundation | Use the same tokenizer and evaluation protocol for models in a direct perplexity comparison |
| FR-05 | Must / foundation | Implement a small causal Transformer and gated recurrent baseline |
| FR-06 | Must / fixed graph | Implement configurable graph modules, bounded degree, state width, update rounds, input injection, and readout |
| FR-07 | Must / fixed graph | Predict future tokens without observing them; state reset and update order must be explicit and tested |
| FR-08 | Must / fixed graph | Train with logged loss, gradient health, optimizer configuration, token count, elapsed time, and checkpoint intervals |
| FR-09 | Must / fixed graph | Save and restore parameters, topology, optimizer, schedule, RNG, data progress, and applicable controller state |
| FR-10 | Must / fixed graph | Provide local generation with documented decoding controls and explicit reset/continuation behavior |
| FR-11 | Must / evaluation | Produce a machine-readable metrics record and readable comparison report |
| FR-12 | Must / evaluation | Report allocated, trainable, active, and serialized parameter/resource counts separately |
| FR-13 | Should / routing | Enable routing independently; report module usage, selected work, and routing overhead |
| FR-14 | Should / rewiring | Enable rewiring independently; enforce edge budgets and log removed/added edges, scoring method, timing, and cost |
| FR-15 | Should / growth | Enforce maximum nodes, edges, memory policy, and growth frequency; log every growth decision |
| FR-16 | Should / growth | Initialize new modules and optimizer state explicitly; preserve existing state and verify checkpoint round trips after structural changes |
| FR-17 | Should / adaptation | Keep recurrent state, fast memory, and persistent parameter learning separately configurable |
| FR-18 | Should / adaptation | Evaluate old-domain retention and new-domain learning with declared reset and adaptation boundaries |
| FR-19 | Later / hierarchy | Allow nested modules only after a flat-graph comparison establishes a useful reference |
| FR-20 | Later / deployment | Add a second runtime or API only after local correctness and a measured deployment need |

## 8. Nonfunctional requirements

### Correctness

- No causal leakage through token inputs, graph states, routing features, controller decisions, or adaptive memory.
- Finite loss and gradients on the smoke-training fixture.
- Graph indices valid; active edges cannot reference invalid nodes.
- Synchronous graph updates read prior-round state.
- Independent sequences reset state according to the declared policy.
- Structural checkpoint resume preserves topology and optimizer semantics.

### Reproducibility

Use a uv lockfile once implementation begins. Pin the main framework stack and record backend versions. Distinguish numerical reproducibility within a documented tolerance from bitwise identity, which may depend on device kernels. Use at least three seeds for headline research comparisons.

### Performance and resource accounting

No absolute throughput or RAM promise is set until target hardware is known. Before main evaluation, record a target-device profile containing RAM ceiling, model scale, batch size, context length, dtype, and latency goals.

Separate compilation, warm-up, steady execution, data loading, controller time, and kernel-call overhead. Synchronize accelerator timing. Report both host and accelerator memory where applicable. A masked dense implementation must not be labeled sparse execution.

### Maintainability

Separate data, model updates, training, evaluation, structural control, and artifact handling. Keep configuration and serialization outside numerical hot paths. Maintain one primary training implementation until a second has a concrete purpose.

## 9. Experiment and artifact contract

Each run must produce enough information to answer: what ran, on what data and device, under which budget, and with what result?

Required artifacts:

- Resolved experiment configuration and run metadata.
- Tokenizer identity and data/split manifest.
- Training and evaluation metrics.
- Checkpoint and checkpoint-format version.
- Graph summary and structural-event log when applicable.
- Timing and memory measurement methodology.
- Comparison report including failed runs and material deviations.

Use appropriate binary tensor storage for weights; JSON serializers are not the model-weight format. Checkpoints must distinguish resumable training state from inference-only exports.

## 10. Milestones and exit gates

| Milestone | Deliverable | Exit gate |
|---|---|---|
| M0: Research record | These documents | Original idea, evidence, proposals, and unknowns are recorded |
| M1: Experiment foundation | Configuration, data, baselines, metrics | Baselines train and evaluate using a shared protocol |
| M2: Fixed graph | Working causal graph model | Causality/state tests pass; tiny-subset overfit works; held-out metrics and local generation available |
| M3: Routing | Selective module computation | Quality and actual end-to-end costs compared against M2 |
| M4: Rewiring | Budgeted topology updates | Learned and appropriate control variants compared at declared budgets |
| M5: Growth | Bounded capacity expansion | Fixed-initial, fixed-final, and matched-compute comparisons available; resume after growth verified |
| M6: Adaptation | Continual-learning evaluation | Retention, transfer, adaptation cost, and reset semantics reported |
| M7: Optimization | Measured runtime improvement | Reference agreement and end-to-end improvement on a named device |

A milestone can end in a documented negative research result. It does not automatically justify adding the next mechanism.

## 11. Success criteria

### Engineering success

A contributor can reproduce a documented run, train the fixed graph, evaluate it, resume it, and generate locally without manual changes to the model implementation.

### Research success

At least one proposed mechanism demonstrates a repeatable benefit under a declared comparison:

- Better held-out quality under comparable training and inference resource budgets; or
- Lower measured runtime or memory at comparable quality; or
- Better new-domain adaptation and old-domain retention at a comparable adaptation budget.

Set exact numerical thresholds and quality tolerances after baseline variance is measured and before the main comparison. Preserve all model-selection and tuning costs in the report. TinyStories-only success must be described as restricted-distribution evidence.

### Evidence insufficient for a success claim

An appealing graph visualization, more nodes after training, fewer nominal FLOPs without lower runtime, one favorable seed, or an unmatched comparison against an undersized or poorly trained baseline.

## 12. Verification requirements

Meaningful checks must cover causal leakage, state isolation, graph invariants, gradient flow, tiny-data learnability, checkpoint resume, growth-state migration, and consistency of metric aggregation. Later custom kernels require forward agreement and gradient checks where applicable.

Performance runs require warm-up, synchronized timing, repeated measurements, and a declared device profile. Broader capability and device claims require separate evaluations; they cannot be inferred from the first prototype.

## 13. Dependencies and risks

| Dependency / risk | Response |
|---|---|
| Hardware budget unknown | Establish before selecting final model size and run duration |
| Fixed shapes conflict with physical growth | Capacity stages, explicit reallocation, and recompilation accounting |
| Hard routing complicates gradients | Design and validate selection training before headline experiments |
| Sparse kernels underperform | Measure and retain dense batched controls |
| Growth breaks optimizer/checkpoint state | Explicit migration and post-event resume checks |
| Online learning causes forgetting | Stage-wise retention tests and explicit persistence policies |
| Dataset artifact or overlap distorts results | Versioned manifests and split checks |
| Multiple frameworks expand scope | Select one primary framework; require evidence before a port |
| Novelty unclear | Review closest work after equations and growth rules stabilize |

## 14. Open decisions before implementation

1. Confirm the primary framework: working default JAX + Flax, alternative Burn.
2. Record available CPU/GPU, RAM/VRAM, operating system, and training budget.
3. Choose the first inference device and measurable resource target.
4. Select the initial corpus subset, tokenizer, context length, and token budget.
5. Specify the first graph update equations and baseline scales.
6. Decide which later adaptation goal matters first: efficient specialization, memory, or durable continual learning.

These open items do not prevent preserving the idea or preparing the experiment design. They prevent treating an unmeasured design as a finished product specification.

## 15. Change policy

Update these documents when the architecture, framework, budgets, or evidence changes. Preserve the distinction between founder intent, working proposals, measured results, and rejected hypotheses. Keep negative results with their conditions. Do not silently replace the original growing-spiderweb vision with an ordinary static model; describe simplifications as staged experiments.


## 16. Implementation progress: 2026-10-23

A fixed-graph CPU smoke prototype is implemented and has completed a 300-update synthetic-text run. See [results](experiments/smoke/report.md) and [usage](usage.md). This is partial delivery of the research milestones: baseline models, full-document streaming, real-corpus evaluation, growth, routing, rewiring, and continual-learning experiments are not complete. The current CLI uses independent prefixes with truncation explicitly reported. Preserve those distinctions when assessing requirements and success claims.


## 17. Full-document and real-data progress: 2026-10-23

Document streaming, bounded public English/conversation downloads, separate development/test evaluation, and a parameter-initialized dialogue stage are implemented. See [the real-data report](experiments/english-chat/report.md). There are now measured byte-prediction improvements on full held-out documents, plus a working experimental chat wrapper. Simple comprehension probes still fail; reliable chat/understanding requirements are not met. Matched baselines, assistant-only loss, multi-seed comparisons, growth, and rewiring remain outstanding.


## 18. Baseline and comprehension progress: 2026-10-23

Plain-GRU and message-free graph comparisons are complete for one seed and a matched 4,000-update English protocol. The plain GRU slightly outperformed the graph on story prediction; the graph outperformed its message-free ablation. See [comparison evidence](experiments/comparison/report.md). A paired reading diagnostic found fixed answer preferences rather than successful counterfactual answers in every tested checkpoint. Reliable understanding is not met. Multi-seed comparisons, Transformer baseline, assistant-only training, learned edges, rewiring, and node growth remain outstanding.


## 19. Assistant-only learning progress: 2026-10-23

Assistant-span supervision, masked full-document training/evaluation, exact objective-aware resume, and matched 2,000-update graph/GRU adaptation are complete. All 33 tests pass. See [experiment evidence](experiments/assistant-learning/report.md). Reading remains unsuccessful: neither architecture solves any of 64 held-out context-swapping pairs. Lower supervised loss does not satisfy understanding requirements, and English retention worsened. Keep earlier English checkpoints available.

Next measurable experiment: test whether a memory/readout change increases complete-pair accuracy and context sensitivity while preserving story prediction, against the unchanged recurrent baselines. Freeze a development protocol before tuning; use fresh test data after repeated experiments. Growth, rewiring, learned routing, device benchmarks, Transformer comparison, and multi-seed evidence remain undelivered.


## 20. Retention mechanism experiment: 2026-10-23

Implemented an optional offline retention-gate bias intervention without changing runtime architecture or parameter/state budget. Both models completed matched 2,000-update experiments. The graph becomes context-sensitive but remains near chance on individual reading (66/128); 19/64 complete pairs include a corresponding 17/64 both-wrong pairs. English quality worsens substantially. This does not meet reliable understanding, English retention, or product-upgrade requirements. See [evidence](experiments/memory-retention/report.md).

All 36 tests pass. Continue with a controlled entity–fact retrieval experiment and new confirmation splits before claiming comprehension improvements. Growing topology and device-efficiency requirements remain unmet.


## 21. Controlled retrieval progress: 2026-10-23

A two-stage retrieval curriculum and counterfactual binding evaluator are implemented and tested (39 tests pass). Both architectures complete 500 basic and 1,500 binding/rehearsal updates. Both reach 100% neural candidate-ranking accuracy on 96 fresh single-fact questions, including 128 filler bytes, but solve zero of 96 complete multi-entity binding groups. Multiple-fact accuracy remains around chance; reliable understanding requirements remain unmet.

English retention remains worse than the preserved original English checkpoints. The experimental models are not promoted to default chat/English models. [Evidence](experiments/retrieval-curriculum/report.md) includes pre-test checkpoint locks and actual supervision exposure. The next focused requirement is correct fixed-context query selection with one distractor, then fresh confirmation and broader language transfer. Growing topology and efficiency requirements remain outstanding.


## 22. Focused binding and optional readout: 2026-10-23

Complete-group training, varied entity pairings, development-group-based checkpoint selection, and an optional learned node readout are implemented (44 tests pass). Three 2,000-update experiments yield 0/56 graph, 1/56 GRU, and 0/56 learned-readout complete test groups. English retention worsens substantially. Neither reliable entity binding nor an accepted language-model upgrade is delivered. [Measured evidence](experiments/focused-binding/report.md).

A training-only sanity check fits one four-question group with 4/4 correct raw answers; this is memorization, not satisfaction of generalization requirements. The next research requirement is gradual curriculum/optimization with separate known-name novel-pair and held-out-name evaluation. Default readout, source checkpoints, and the original growing-graph vision are preserved. Node growth, learned edges and online parameter learning remain outstanding.


## 23. Gradual curriculum progress: 2026-10-23

The unchanged graph has completed matched 3,000-update curriculum/flat experiments. Curriculum fits up to 16 training groups completely before declining as cases expand. Fresh tests requiring both fact orders show partial transfer: 18/24 familiar-name and 1/24 new-name cases for curriculum, versus 11/24 and 9/24 flat. Source scores zero in both pools. Reliable name-independent binding and broad understanding requirements remain unmet.

The familiar-name split has a discovered early-role imbalance, so high familiar-name test accuracy needs role-balanced independent confirmation. English loss regresses substantially; no model is adopted as a general English/chat upgrade. All 47 tests pass. [Evidence and audit](experiments/gradual-binding/report.md). Next requirements are balanced confirmation, expanded name diversity and explicit retention measurement. Dynamic graph growth and device-efficiency goals remain unfinished.


## 24. Balanced binding and retention study: 2026-10-24

A fresh 32-name role-balanced benchmark and explicit alternating QA/English trainer are implemented (50 tests pass). Two arms share 2,000 QA updates; rehearsal adds 2,000 English updates. All selected models fail all 96 familiar-name and 48 new-name complete test cases. Reliable binding and broad comprehension requirements remain unmet.

English retention is successful under the tested rehearsal schedule: test loss 1.087839 versus source 1.121557. The unconstrained task-only candidate reaches 3.381755 and is rejected by the protected policy, which falls back to the unchanged source. Protected/unconstrained checkpoints and extra rehearsal cost are reported explicitly. [Evidence](experiments/balanced-confirmation/report.md). This is not a source-matched replication of the earlier toy transfer. Next requirement: source-controlled balanced learning progression with explicit retention and fresh confirmation. Architecture/growth/efficiency goals remain outstanding.


## 25. Hierarchical routing prototype: 2026-10-24

Two-level recurrent node-update routing is implemented with legacy dense defaults and fixed-slow control (54 tests pass). Four groups of two GRUs receive learned update probabilities; grouping has no assigned grammar labels, growth or rewiring. All three pilot models complete 1,000 QA plus 1,000 English updates and score 0/144 complete fresh binding cases. Hierarchical English loss improves over fixed slowing but remains worse than dense updates. [Evidence](experiments/hierarchical-routing/report.md).

The prototype establishes implementation feasibility, not useful specialization or a language-model upgrade. Soft routing is dense execution; Transformer replacement, sparse runtime efficiency, device portability and broad comprehension requirements remain unmet. Initialization mismatch and lack of a flat learned-gate control limit causal architecture claims. Preserve source/default models and test these factors before increasing hierarchy depth or expert count.


## 26. Matched fresh English pretraining

Both dense and hierarchical graphs now have independently trained English checkpoints:4,000 updates each, identical shared initialization and 1,916,320 training payload bytes. Independent memorization probes fit one training group in both modes; four-group ranking is 62.5% dense and 100% hierarchical. Probe weights never seed English training. Each architecture then adapts its own English checkpoint with 1,000 QA plus 1,000 English rehearsal updates on fresh balanced data.

dense: English test loss 1.121557, adapted 1.126128; complete binding cases 0/144. hierarchical: English test loss 1.323535, adapted 1.321262; complete binding cases 0/144.

[Full protocol, locked tests, delayed probes and CPU measurements](experiments/from-scratch/report.md). This removes the prior inherited-weight mismatch without proving specialization, growth, arbitrary-device portability or Transformer replacement. Soft routing still computes all nodes. Narrow task acquisition remains the next gate; preserve these negative outcomes and use fresh confirmation data after changes. All 56 tests pass.


## 27. Acquisition versus transfer diagnostic

Completed independent 1/4/16/64-group, 500-update task-only trials for dense and hierarchical models, resetting each to its own English checkpoint with fresh optimizer. Same paired subsets, traces and bytes; no test data used. Fixed endpoints, LR 0.003, seed 42.

dense: 1 groups: 100.0% complete training groups; 4 groups: 100.0% complete training groups; 16 groups: 0.0% complete training groups; 64 groups: 0.0% complete training groups.

hierarchical: 1 groups: 100.0% complete training groups; 4 groups: 100.0% complete training groups; 16 groups: 6.2% complete training groups; 64 groups: 0.0% complete training groups.

[Full training/development results, name coverage and English forgetting](experiments/acquisition/report.md). Memorization and held-out transfer are distinct requirements. Fixed budgets reduce presentations per example at larger sizes, so outcomes do not prove a capacity limit. These task-only checkpoints are diagnostic and do not replace English models. Next control exposure per example and compare any curriculum with matched flat training; use fresh confirmation after changes.


## 28. Equal-exposure diagnostic

The unchanged acquisition runner completed paired 16-group/2,000-update task-only trials from each own English checkpoint, matching the four-group study's 125 presentations per group. Same config, initial parameters and fresh optimizer as the previous 16-group trials; update traces share the first 500 group IDs and paired byte counters match. No test set or checkpoint selection was used. Four times the original update budget is explicitly counted.

dense: 51.56% training answer ranking, 0/16 complete training groups, 0/144 complete development cases; English development loss 9.424430.

hierarchical: 95.31% training answer ranking, 14/16 complete training groups, 0/144 complete development cases; English development loss 9.336111.

[Protocol, exposure controls, name coverage and measured results](experiments/exposure/report.md). Fixed-endpoint exploratory results are not a capacity proof or independent final confirmation. Task-only English forgetting prevents promotion. Next isolate shared/node learning rates and rehearsal before any growth or deeper hierarchy.


## 29. Opt-in role rates and controlled rehearsal

Implemented opt-in shared/node/router learning-rate multipliers on completed AdamW updates, preserving default behavior, optimizer-state structure and legacy checkpoint loading. Six fixed-endpoint hierarchical trials compare high uniform, low uniform and split representation rates, each with/without English rehearsal. Same own English source, 16 groups and 2,000 QA updates; rehearsal adds 2,000 English updates. No test data or endpoint selection.

high-task: 14/16 complete training groups; 0/144 complete development cases; English development loss 9.336111.

high-rehearsal: 16/16 complete training groups; 0/144 complete development cases; English development loss 1.319806.

low-task: 12/16 complete training groups; 0/144 complete development cases; English development loss 3.537662.

low-rehearsal: 5/16 complete training groups; 0/144 complete development cases; English development loss 1.307681.

split-task: 16/16 complete training groups; 0/144 complete development cases; English development loss 6.268202.

split-rehearsal: 16/16 complete training groups; 0/144 complete development cases; English development loss 1.305334.

[Frozen protocol, costs, optimizer/default reproduction and full results](experiments/plasticity/report.md). This is an exploratory fixed-rate/data-schedule study, not an adaptive growth mechanism. Acquisition/retention and held-out transfer remain separate gates. No model is promoted without fresh confirmation.


## 30. Fresh pairing/name/wording transfer study

Completed three matched update-budget rehearsal trials from the same English-only source: fixed 16, diverse 128 and mixed-wording 128 training groups. New seed 904217 task pools reserve familiar-name pairings and new-name labels for development/test; every arm covers the same 16 training labels. Canonical/mixed underlying episodes and sample traces match; wording differs. No architecture/optimizer/trainer changes or old QA initializers.

fixed: 0/16 complete training groups; 0/288 complete fresh test cases; English development loss 1.303425.

diverse: 0/128 complete training groups; 0/288 complete fresh test cases; English development loss 1.296879.

mixed: 0/128 complete training groups; 0/288 complete fresh test cases; English development loss 1.301324.

[Full test buckets, source baseline, locked checkpoints, raw samples and exposure limits](experiments/transfer/report.md). Fixed versus diverse changes partner/color diversity and repetitions per example at a fixed budget; it is not a capacity proof. Transfer requires at least 90% complete cases in every fresh name/wording bucket, separately from acquisition/retention. No model is promoted without that gate and further confirmation.


## 31. Opt-in associative fast-state memory

Implemented optional memory=delta on the fixed sum-readout graph, using learned key/value/query/write projections and a document-local delta-rule matrix. Read-before-write, reset/padding/chunk contracts and checkpoint continuation are tested. Zero output gain preserves source predictions mathematically; packed CPU scan roundoff is bounded and real initial English loss differs by 7.01e-9. Default memory=none preserves legacy graph behavior. Extra 16,514 parameters and state 2,048→6,144 bytes are explicit.

none: 0/128 complete training groups, 0/288 complete new test cases; English development loss 1.296104.

delta: 0/128 complete training groups, 0/288 complete new test cases; English development loss 1.291094.

[Full mechanism, primary-paper links, source/trace/lock audits and measured state/CPU costs](experiments/associative/report.md). Both arms use the same English source and 2,000 QA + 2,000 English updates on new seed 638101 diverse data. This is an adaptation pilot with extra parameters/state/compute, not a new delta-rule invention, a parameter-matched proof or an optimized parallel implementation. Learning useful language addresses remains distinct from toy memory algebra. No model is promoted; 68 tests and review passed.


## 32. Learned-memory addressing diagnostic

Implemented read-only actual-forward tracing, overwrite-aware signed read decomposition, full-prompt reconstruction checks and frozen memory-output ablation. Region metadata is offline only; no inference rules or neural/training changes. Four independent source-initialized500QA+500English probes compare graph/delta on1 and4 groups. Both memorize1/1 but complete0/4 at the larger subset; per-group exposure differs, so no causal diversity or capacity claim. All tiny development complete-group scores are zero. No new test data opened or checkpoint promoted.

Failed pilot readout removal changes9/512 training predictions, accuracy51.76%→50%, complete groups remain0/128. Memory influences likelihood without reliable binding. Preserve source models, raw scores, counters, signed/projection traces and input hashes.71 tests passed with static checks and independent review. Next gate: disentangle exposure and diversity before selecting a learned addressing intervention; successful training acquisition must precede fresh transfer evaluation. See [diagnostic report](experiments/addressing/report.md).


## 33. Four-group acquisition at matched per-group exposure

Completed source-initialized graph/delta controls at2,000QA+2,000English updates each:500 presentations per group. Delta fits4/4 complete training groups and16/16 rankings, graph0/4 and8/16; fixed raw training samples4/4 versus2/4. A post-hoc frozen readout ablation drops delta to0/4 and8/16, supporting learned output-channel dependence. Both complete0/2 reused tiny development groups. No new test opened or checkpoint promoted.

All72 frozen inputs and budget/sampler/source invariants audited. Neural/training code unchanged; prior71-test result remains the verification basis, with analysis lint/format and independent review. Added compute/rehearsal and extra delta parameters limit causal interpretation. Next gate is multi-seed acquisition replication, then separately frozen transfer testing. See [report](experiments/addressing-exposure/report.md).


## 34. Acquisition replication gate failed

Completed all four predefined adaptation-seed43/44 trials (graph/delta,2,000QA+2,000English each). Delta scores8/16 and9/16 training rankings,0/4 complete groups each; graph8/16,0/4 both. Acquisition gate0/2, retention2/2 per model. Every tiny development complete-group result0/2. Preserve selected seed42 success as prior reference, not robust model qualification. No new test or promotion; no architectural changes.

All74 frozen inputs and actual checkpoint step/config/source/sampler/budget invariants audited. Reporting scripts independently reviewed; annotation-only type amendment preserves normalized executable AST. Prior71-test result is explicitly prior, with fresh experiment audits and script lint/type checks. Next gate: decouple memory initialization and data/rehearsal order to diagnose inconsistent acquisition before transfer testing. See [replication report](experiments/addressing-replication/report.md).


## 35. Memory initialization crossed with combined schedule

Two fixed-budget crossed trials complete an exploratory post-hoc2×2 grid. Both memory initializations42/43 fit16/16 rankings and4/4 groups under schedule42; both score8/16 and0/4 under43. New crossed raw samples2/4 and4/4, English retention passes, tiny development complete groups remain0/2 each. Reused selected historical diagonals and unrecorded old numerical flags limit causal interpretation; QA order and English document exposure remain combined. This does not reverse the failed independent-seed acquisition gate.

No model/training changes. Step-zero source tensors and fresh optimizer preflight, enforced child numerical settings, checkpoint schedule/stream/counter verification and score-derived artifact auditing passed203 checks. Prior71-test result labeled prior; scripts pass Ruff/ty. Next gate is explicit legacy-compatible QA/replay seed controls and a fixed-initializer schedule-component experiment. See [report](experiments/addressing-factorial/report.md).


## 36. Independent samplers and English replay lead

Implemented --qa-seed/--english-seed on plasticity_run with legacy defaults, zero/invalid seed handling, old Namespace support and saved actual seed policy. Config seed still sets initialization/subset; neural/training-step/optimizer/scorer code unchanged.75 tests pass; actual legacy/default compatibility13 checks include bitwise weights/moments.

Two new crossings with fixed initializer42 complete an exploratory selected QA×English grid: English43 gives16/16 and4/4 at both QA seeds, English44 gives7/16 and8/16 and0/4. All retain English and fail complete tiny development groups.219 endpoint/input checks pass; source changes and historical limits explicit. No new test or promotion, earlier replication gate unchanged. Next gate: frozen training-only gradient plus copied actual optimizer-step interference diagnostic before intervention. See [report](experiments/schedule-components/report.md).


## 37. Frozen replay-interference diagnostic completed

Added separate training-only diagnostic helper/CLI: exact frozen replay-prefix reconstruction,80 independent copied Adam/zero-gradient comparisons over5checkpoints, all16 QA target-weighted CE, raw gradient/actual-update role accounting and source immutability. No existing model/training math, saved checkpoints or release version changed.78 tests and504 audits pass.

Local QA-loss interference is supported at some snapshots, but replay44 is not consistently worse than43; optimizer history/decay changes the interpretation. ExtraEnglish steps after end-of-English snapshots and frozen-context reconstruction do not reproduce normal QA→English training. No new ranking/test/generalization claim or intervention. Next gate is an actual-next-phase copied QA→English pair using saved stream/carry, followed by a separately frozen intervention only if supported. See [report](experiments/replay-interference/report.md).


## 38. Restored next-pair diagnostic completed

Five restored next-loop pairs use complete QA history, actual English stream/carry and loaded optimizer states, with matched English-only and zero-gradient controls:25 copied transitions. QA raises aggregate CE at3/5 snapshots; English after QA raises it at2/5, while control-adjusted effects have different membership. Selected-group and aggregate changes can disagree. These results do not isolate Adam or show universal replay erasure.

All80 regression tests and191 runtime audits pass;78 frozen hashes including72 prior inputs remain unchanged. No checkpoint, neural/training math, ranking, held-out evaluation or release changed. Next isolate optimizer-history and router-balance contributions to the QA step before a separately frozen intervention. Earlier failed replication/generalization gates remain unchanged. See [report](experiments/qa-replay-pair/report.md).


## 39. QA history/regularizer control gate completed

Five prior snapshots, identical actual-next QA and six copied branches/source cross loaded versus fresh Adam (moments plus internal clock) with configured versus zero router coefficient, plus both zero-gradient controls. All30 transitions and391 audits complete;88 frozen inputs including78 prior pair inputs remain unchanged. Final81-test suite passes. No checkpoint, model/training algorithm, rankings, held-out data or release changed.

At all four fitted snapshots, unchanged-rate fresh Adam worsens selected and aggregate CE and its active objective; configured update norms rise about6.7–7.1 times. Loaded regularizer removal has mixed aggregate effects. At q42_e43 selected language worsens while total objective improves; at q42_e44 both worsen. All loaded gradient branches improve aggregate CE relative to their zero controls. These conditional results do not isolate moments, scalar update size or a long-run fix. Preserve loaded history; next freeze a bounded update-size control before longer intervention. Earlier replication/generalization failures remain unchanged. See [report](experiments/qa-step-controls/report.md).


## 40. Bounded loaded-update diagnostic completed

Five fixed sources, exact next QA, loaded optimizer and configured regularizer:10 independent optimizer computations and50 parameter-displacement evaluations at0,1/8,1/4,1/2,1. Smaller fractions reduce raw aggregate-loss harm at three fitted snapshots but do not turn it into clear improvement; q43_e43 at1/8 is unresolved(+8.74e-7). Smaller fractions also reduce helpful changes at initial andq43_e44. Do not promote a universal smaller-update rule.

All82 tests and653 runtime audits pass;98 hashes including88 prior inputs preserved. No checkpoints, prior core math, rankings, held-out evaluation, release or model-quality claim changed. See [report](experiments/qa-update-scale/report.md). [DeepSeek research](research/deepseek-inspirations.md) records shared/routed specialists, balancing, MTP, mHC, Engram and optimizer ideas as separate hypotheses. A bounded future-byte objective pilot is a next experimental comparison, not implemented or a proven fix; current instability and failed replication/generalization gates remain explicit.


## 41. Future-byte feasibility pilot implemented, not promoted

Implemented a disposable rank16 future-byte head in separate modules, preserving core model/Config/trainer/scorer. Two fixed arms(weight0/0.1),128QA+128English each,32visits/group, matched source/head/samplers/exposure:512actualbackboneupdates. Main QA training CE0.165443→0.156376 versus control; fixed English training-probe CE1.121980→1.121316. Narrow selected-schedule training evidence only; no ranks, generation, held-out retention or generalization measured.

Full84-test suite passed beforetraining;3focused tests afterauditfix. Control training completed before a head-serialization byte-order audit stopped execution. A frozen audit-only continuation validated/reused control without retraining and trained only the unstarted active arm. All134finalaudits pass;111continuationinputs,102originalnon-driverinputs and98priorboundedinputs preserved, explicit driver amendment recorded. Source checkpoints and release unchanged; new final checkpoints local-only, no jointresume or promotion. Next freeze independent-seed replication then held-out qualification. Earlier failed gates remain failed. See [report](experiments/future-byte/report.md).


## 42. Future-byte replication gate failed

Four new arms repeat weight0/0.1 atQA43/E44 andQA44/E45, same original initializer/head9042 and128QA+128English each(1,024newupdates,32visits/group). Active−control QA CE changes+.004469320 and−.000707626:primarygate1/2fails. Fixed English corpus-training-probe changes−.000130892 and−.002619267:secondarydirection2/2passes; jointgatefails. AllnewEnglish endpoints exceed initial1.157155037, so relative benefit is not retention. Historical favorable pilot excluded fromnewdenominator.

All89tests and212runtimeaudits pass;140inputhashes including111priorinputs preserved,30historicalartifacts validated. Existing model/trainer/helper/driver/source checkpoints unchanged; new checkpoints local-only and no promotion. No ranks, generation, development/fresh test or held-out quality measured. Next inspect answer-level/per-position tradeoffs on existing fixtures before more training; keep fresh tests closed and earlier failed gates unchanged. See [report](experiments/future-byte-replication/report.md).


## 43. Answer-level analysis completed, acquisition not qualified

Read-only scoring of initializer and all six128QA+128English endpoints:all8/16training rankings,0/4complete groups,0/8completecontext pairs andzero context-prediction flips. No near ties. Query sensitivity occurs in rep44auxiliary(2/8queryflips,1/8whollycorrectquerypairs), without binding qualification. Predictions differ across arms but all paired net-correct changes arezero; pilot:0flips; replication43:8flips; replication44:14flips with balanced gains/losses.

Teacher partitions(first16/rest48/newline16/EOS16) reproduce prior mainCE and canonical goldscores. PilotauxiliaryCE benefit is mainly first-byte probability improvement, not formatting, but rankings unchanged. No causal mechanism or universal capacity limit identified. All4focused tests and258audits pass;188hashes including140priorinputs preserved, zero optimizer updates or model writes. Prior89fullsuite explicitlyprior; original model/trainer/scorer unchanged. Earlier longer-budget acquisition results and failed replication/generalization gates unchanged. Use answer/group acquisition gates for next predefined matched-budget experiment. See [report](experiments/future-byte-answers/report.md).

## 44. Context objective: acquisition candidate, no promotion

A bounded same-answer context-contrastive training objective is implemented without changing inference parameters. Stronger weighting learned16/16training rankings and4/4binding groups at128QA+128English on one schedule; the matched control remained8/16 and0/4. English training-probe CE worsened by0.041813, failing the joint gate. Require independent-schedule replication and a separate English-interference experiment before broader evaluation or promotion. No held-out/chat/device-efficiency claim follows. [Experiment and audit limits](experiments/context-contrastive/report.md).
