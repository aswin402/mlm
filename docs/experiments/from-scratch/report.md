# From-scratch dense versus hierarchical study

This study removes inherited-weight mismatch from the previous routing pilot. Both architectures are independently initialized for English training rather than loading a dense-trained English checkpoint into the altered recurrent update rule. The graph, sum readout, byte vocabulary, eight width64 GRUs and base initialization seed remain shared. Hierarchical routing retains its additional 1,548 parameters and auxiliary balance loss; the comparison is token/update matched, not exactly parameter/compute matched.

## Phases and boundaries

1. A separate training-only learnability probe initializes each architecture randomly at seed 42, LR 0.003, batch 4/unroll 128. It trains one existing four-question group for 500 updates, then four groups for 500 further updates with continuous optimizer state. These are memorization diagnostics; their weights never seed English pretraining or the final evaluation models.
2. Each English model is independently reset to random seed 42 and trained for 4,000 updates on the same cached 1,000 English training stories with 32 development stories, LR 0.001, batch 4/unroll 128. Full-document state carries across chunks with truncated backpropagation, and padding/document resets follow the tested streaming implementation. Minimum pure English development loss selects initial or 500-step endpoints. Only hierarchical gradients include the existing routing auxiliary; selection/evaluation losses exclude it.
3. Each architecture adapts its own development-selected English checkpoint using fresh optimizer,1,000 complete-group QA updates interleaved with 1,000 English-stream updates, LR 0.001, batch 4/unroll 128. The fresh balanced task uses seed 631204, 32 training labels, disjoint familiar-name held-out pairings, and separate 16-name new development/test pools. Both arms receive identical streams and the same adaptation budget. Primary selection maximizes macro eight-condition development correctness across known/new-name buckets; QA loss breaks ties, initial eligible. English is monitored separately, without a retention feasibility constraint in this study.
4. Both English-selected and task-selected checkpoints are locked before the fresh task and reused English tests. Additional neutral-filler test probes insert exactly 64 or 128 bytes between facts and question, without delayed-example training. They retain the same ground-truth facts and all eight assignment/query/sentence-order conditions. Zero-delay failure limits interpretation of delayed failure; it cannot by itself isolate forgetting.
5. After all training/evaluation workers finish, selected English checkpoints are measured sequentially in fresh CPU processes. Timings cover warmed JIT full forwards at batch 1 and lengths 32/128/512 with fixed seeded inputs, 5 warmups and 30 repetitions. Compilation/first execution is separate. These are not text-processing/generation latency, sparse kernels, Transformer or device-portability benchmarks. Tensor sizes and peak RSS are recorded, with RSS explicitly including checkpoint optimizer loading, JAX and compilation overhead.

All dataset/config/core-code hashes are frozen before training. No probe weights, QA data or test outcomes influence English initialization/selection. Shared base tensors must match between the two random initial checkpoints, with only router extras in hierarchical. Full stream traces and input-byte counts must match at the end of each paired training phase. Offered budgets match; development-selected step counts can differ and are reported.

## Scope

The English source is a small bounded public-data snapshot, not a full language-model corpus. Binding uses one sentence grammar and four colors, with role-balanced partitions and artificial names. Names are held out within this task, not guaranteed globally unseen. This is one seed. Source architecture pretraining mismatch is addressed, but learning rate, balance loss, model size and limited optimization budgets still restrict conclusions about architecture potential. No growing categories, verb/pronoun specialization or online inference-time weight learning is implemented.

A Transformer benchmark is a later conditional milestone. This internal comparison cannot establish replacement, superiority, broad English conversation or scaling. Useful language/retrieval performance must first be demonstrated, and later direct matched comparisons would need their own protocol and untouched confirmation sets.

## Reproduction

Use fresh output paths. The learnability probe uses the existing gradual-v1 training fixture; it never supplies pretraining weights.

```sh
uv run python -m mlm.learnability_run --config configs/scratch-probe-hierarchical.json --out runs/scratch-probe-hierarchical-new
uv run python -m mlm.pretrain_run --config configs/scratch-hierarchical.json --out runs/scratch-english-hierarchical-new --steps 4000 --eval-every 500
```

Repeat with the dense configs and fresh paths. Adapt each English best checkpoint using hierarchy_run with the fresh scratch-binding-v1 data, not another architecture's weights. Freeze both selections before tests. Full measured commands and artifact pointers accompany the results below.

## Measured results

Both pretraining runs completed 4,000 updates and 1,916,320 payload bytes. Checkpoint selection chose step 4,000 for both. Shared base initial tensors match exactly; hierarchical adds1,548 router parameters. Both adaptation runs completed 1,000 QA and 1,000 English updates with a fresh optimizer. Full sampler snapshots and update/byte accounting match between arms; checkpoint locks precede test evaluation. See `initialization-audit.json`, `stream-audit.json`, `checkpoint-lock.json`, `protocol.json` and `summary.json`.

| Model | Parameters | English selected step | English test loss | Adapt selected QA step | Adapt English test loss | Complete binding cases | Answer ranking accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense | 366,722 | 4000 | 1.121557 | 1000 | 1.126128 | 0/144 | 50.347% |
| hierarchical | 368,270 | 4000 | 1.323535 | 750 | 1.321262 | 0/144 | 50.000% |

Lower English loss is better. These losses are pure next-byte cross-entropy including document EOS; no routing auxiliary is included. The repeatedly used 32-story English test is a retention diagnostic, not an untouched broad-language confirmation. Adaptation selection has no English-retention constraint.

### Memorization sanity

| Model | One training group,500 updates | Four groups, 500 further updates | Fixed first-group greedy samples after four-group stage |
|---|---:|---:|---:|
| dense | 100.0% | 62.5% | 50.0% |
| hierarchical | 100.0% | 100.0% | 100.0% |

Ranking covers 4/16 training questions respectively; greedy samples cover the fixed first four questions only, without extraction or answer repair. Probe weights never initialize either English run. This establishes finite-budget memorization, not generalization or broad language understanding. Probe random initialization matches the separately saved English initial base tensors at the same seed. No initial probe ranking was recorded; this reporting omission does not change pretraining initialization or selection.

### Fresh binding and neutral delays

| Model | Delay bytes | Familiar names, new pairs | New names | Complete total | Answer ranking accuracy |
|---|---:|---:|---:|---:|---:|
| dense | 0 | 0/96 | 0/48 | 0/144 | 50.347% |
| dense | 64 | 0/96 | 0/48 | 0/144 | 50.000% |
| dense | 128 | 0/96 | 0/48 | 0/144 | 50.000% |
| hierarchical | 0 | 0/96 | 0/48 | 0/144 | 50.000% |
| hierarchical | 64 | 0/96 | 0/48 | 0/144 | 50.000% |
| hierarchical | 128 | 0/96 | 0/48 | 0/144 | 49.913% |

Each complete case requires all eight color-assignment, query and sentence-order conditions. Familiar/new buckets contain 96/48 cases; within/cross role strata are balanced and reported in each test artifact. Candidate scoring uses only prompts and candidate strings; correct indices and fact metadata grade outputs offline. Greedy answers from fixed zero-delay groups are also preserved. Delays were never trained or used for selection. If zero-delay binding fails, delayed failures cannot isolate additional forgetting. All conditions share a narrow grammar and four colors; this is one seed.

### CPU forward timing and storage

| Model | Length | Warmed median ms | p95 ms | Compile + first forward seconds |
|---|---:|---:|---:|---:|
| dense | 32 | 0.817 | 1.047 | 0.132 |
| dense | 128 | 3.321 | 4.116 | 0.147 |
| dense | 512 | 13.054 | 14.665 | 0.178 |
| hierarchical | 32 | 0.823 | 1.047 | 0.169 |
| hierarchical | 128 | 3.338 | 3.708 | 0.166 |
| hierarchical | 512 | 13.557 | 14.461 | 0.181 |

| Model | Float parameter bytes | Optimizer tensor bytes | Persistent state bytes/sequence | Peak process RSS KiB |
|---|---:|---:|---:|---:|
| dense | 1,466,888 | 2,933,780 | 2,048 | 349,832 |
| hierarchical | 1,473,080 | 2,946,164 | 2,048 | 363,272 |

Separate processes ran sequentially after training/evaluation, using the same fixed random byte inputs, batch 1, five warmups and 30 timed forwards. Peak RSS includes optimizer checkpoint loading, JAX and compilation; it is not isolated inference RAM. Dense execution computes all nodes in both models. These timings do not measure generation loops, sparse dispatch, a Transformer comparison or arbitrary-device deployment. Training wall times were concurrent and are not a controlled training-speed comparison.

### Interpretation and next gate

Neither selected model demonstrates robust binding on this fresh test. Removing inherited-weight mismatch allowed independent English learning, but does not establish that hierarchical routing improves language modeling or solves binding. The hierarchical model can memorize a small training set, so a categorical claim that the architecture cannot learn is also unsupported. The next controlled step is to locate acquisition failure with nested training-set sizes and training-versus-held-out swap scores under a fixed own-architecture source, before adding categories, growth or expert depth. Reserve fresh confirmation data for any revised algorithm. No Transformer replacement, chatbot readiness or general efficiency claim follows from this study.

All 56 regression tests passed (167.76 seconds); Ruff, formatting and source type checks passed. Current source/tests/configuration/lockfile are snapshotted under `code/` with SHA256 hashes. `execution-script.py.txt` records the local orchestration; direct CLI commands above are preferable for fresh output paths. No architecture or default-checkpoint changes were made in this study.
