# Learned delta-memory architecture pilot

An opt-in key/value fast-state matrix is added to the fixed graph. This is inspired by [Linear Transformers Are Secretly Fast Weight Programmers](https://arxiv.org/abs/2102.11174) and the [DeltaNet parallelization paper](https://arxiv.org/abs/2406.06484); it is not a new delta-rule invention, a complete paper reproduction or an optimized parallel kernel. The reference implementation is a dense sequential JAX scan.

## Mechanism and state contract

At each input byte, the existing graph produces pooled features. Learned projections of the current embedding and pooled graph features produce normalized key k, query q, tanh value v and sigmoid write gate beta. Read r=qᵀM_old; then correct the matrix with M_new=M_old+beta*k*(v−kᵀM_old)ᵀ. The old-matrix read contributes gain*(r@W_out) before the existing output normalization/head. gain initializes to zero. Querying before writing excludes the current write; predictions still use the current byte through graph features and predict the following byte. No future input, entity parser, factual metadata, string answer lookup or label-driven gate is used in inference.

The matrix is transient document-local state, not persistent parameter learning during chat. It resets with the GRU state at document boundaries. Invalid padding freezes the complete previous packed state, even when reset is true. State carries across chunks; graph routing sees only actual GRU nodes, not memory rows. Overlapping learned addresses can interfere, so the matrix is not an exact symbolic dictionary.

| Variant | GRU nodes | Width | Matrix rows | Parameters | Persistent state bytes/sequence |
|---|---:|---:|---:|---:|---:|
| graph (memory=none) | 8 | 64 | 0 | 368,270 | 2,048 |
| graph + delta | 8 | 64 | 16 | 384,784 | 6,144 |

Delta adds 16,514 parameters and triples persistent recurrent state. The packed layout is [B,nodes+memory_keys,width], with graph rows first and matrix rows last. The 16 matrix rows are not 16 extra GRUs or named linguistic experts. New memory tensors have their own optimizer role/rate. Config.memory defaults to none, preserving default parameter/state shapes; memory_keys defaults to 16 and memory_lr_scale to 1. Graph/sum readout is required for the delta option. Standard stream warm starts require matching memory settings; the dedicated experimental initializer may add source-preserving zero-gated parameters.

## Preflight and tests

Shared base tensors match exactly. Zero-gate eager predictions and graph state are bitwise identical. CPU packed-scan compilation changes rounding: the tiny fixture's maximum logit difference is 5.96e-7 and graph-state difference 3.73e-9. Compiled comparisons use 1e-6 tolerance; full-English initial loss difference must be ≤1e-5. Do not claim bitwise compiled equivalence between different state layouts.

Real-source initial English development loss: graph 1.352696033935, delta 1.352696040947; absolute difference 7.01e-09.


All 68 regression tests passed (178.86 seconds); 16 targeted checks passed (69.22 seconds). Tests cover orthogonal overwrite/read, overlapping addresses and fractional writes, finite/open-gate gradients into every memory tensor, nonzero first gradient at the zero output gate, prefix causality, chunk equivalence, reset/padding, true-node route shape, effective memory update scaling including decay, and exact checkpoint/CLI stream continuation. A populated legacy checkpoint omits the new settings and continues with default behavior. Separate review found three test gaps; all were addressed and the follow-up review has no unresolved findings. Ruff lint/format and source type checks passed.

## Frozen architecture comparison

Both variants independently adapt the same preserved English-only hierarchical checkpoint, with fresh optimizers, seed 42, batch 4/unroll 128 and 2,000 QA + 2,000 English rehearsal updates. Shared representation rate is 0.0003; node/router/memory rates are 0.003. A new transfer_data seed 638101 snapshot provides 128 diverse training groups on 16 labels. Both arms receive exactly the same facts, queries, sampling trace, supervision and English stream. The generator also creates fixed/mixed pools, but neither is trained in this pilot.

These are fixed endpoints, not development-selected checkpoints. Initial functions match numerically due to the zero output gate, avoiding a large inherited-function mismatch. The memory model still has extra parameters, state and compute: this is a source-preserving adaptation pilot, not a parameter/state-matched proof, matched English pretraining from scratch, an inference-time online learning result or an architectural superiority claim. All Adam moments are deliberately reset in both arms.

Training acquisition/retention requires ≥87.5% complete training groups and English-dev loss ≤1.25×source. A training group has four assignment/query conditions. Fresh-test reliable transfer requires ≥90% complete cases in every bucket, including both fact orders (eight conditions). Six test buckets cross familiar-name/new-pair versus new-name labels with base, alternate and heldout wording. Each has 48 cases; total 288 cases/2,304 questions. Only base rendering is trained in this pilot. Task-level novelty is not global novelty relative to English pretraining. Test scoring starts only after all final hashes are locked; the repeatedly used English test is not reopened.

## Acquisition and English retention

| Variant | Training answers correct | Complete training groups | English dev loss | Joint diagnostic gate |
|---|---:|---:|---:|---|
| none | 255/512 | 0/128 | 1.296104 | fails |
| delta | 265/512 | 0/128 | 1.291094 | fails |

Final learned memory output gain: 0.02116. A nonzero gain or changed memory parameters is an optimization diagnostic, not proof of effective memory use or correct entity binding. Raw fixed training samples and per-role parameter-change norms are preserved.

## Fresh locked test

| Checkpoint | Familiar/base | Familiar/alternate | Familiar/heldout | New/base | New/alternate | New/heldout | Complete total | Answer ranking accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| source | 0/48 | 0/48 | 0/48 | 0/48 | 0/48 | 0/48 | 0/288 | 50.043% |
| none | 0/48 | 0/48 | 0/48 | 0/48 | 0/48 | 0/48 | 0/288 | 50.608% |
| delta | 0/48 | 0/48 | 0/48 | 0/48 | 0/48 | 0/48 | 0/288 | 50.868% |

These are neural likelihood rankings over two candidate colors. Free generation is additionally preserved for the preselected first four questions per bucket (24 examples per trained endpoint), at most 12 bytes each, using strip/casefold grading without extraction/repair. This is not a broad chat or complete free-generation benchmark. Within/cross-role test strata and raw scores are included. Offline roles, facts and correct indices never act as an inference solver.

## Cost measurement

Every arm uses 2,000 QA + 2,000 English updates, 40,000 supervised QA payload bytes and the same English-stream coverage. Matched update count does not imply equal FLOPs or wall time. Both endpoints are timed sequentially in fresh CPU processes after all training/evaluation workers finish: batch 1, fixed identical random byte inputs, 5 warmups and 30 repetitions, with compile/first forward separate.

| Variant | Forward length | Median ms | p95 ms | Compile + first forward seconds |
|---|---:|---:|---:|---:|
| none | 32 | 0.954 | 1.534 | 0.155 |
| none | 128 | 3.684 | 4.133 | 0.153 |
| none | 512 | 13.561 | 18.428 | 0.169 |
| delta | 32 | 1.023 | 1.378 | 0.209 |
| delta | 128 | 3.700 | 3.977 | 0.226 |
| delta | 512 | 15.700 | 16.558 | 0.227 |

| Variant | Parameter bytes | Optimizer tensor bytes | State bytes/sequence | Peak process RSS KiB |
|---|---:|---:|---:|---:|
| none | 1,473,080 | 2,946,164 | 2,048 | 377,024 |
| delta | 1,539,136 | 3,078,276 | 6,144 | 402,572 |

Timing is a warmed full forward, excluding text I/O and generation-loop costs. RSS includes checkpoint optimizer loading, JAX and compilation; it is not isolated inference RAM. The matrix has fixed size and uses dense operations. Neither these CPU figures nor recurrent state size establish any-device portability, sparse efficiency, Transformer replacement or scaling advantages.

## Interpretation and next gate

none: 0/288 complete fresh test cases; reliable-transfer gate not met.

delta: 0/288 complete fresh test cases; reliable-transfer gate not met.

The matrix mechanism is implemented and differentiable, but it does not demonstrate reliable binding at this budget. Algebraic storage tests assume supplied keys/values; the language model must still learn suitable encodings and write/read timing from byte-level loss. Added memory does not by itself teach that procedure. With 128 groups and only 15–16 presentations per group, exposure and optimization remain explanations. Preserve this negative result, diagnose learned addressing/write behavior and acquisition on small controlled sets before increasing memory size or depth.

No endpoint is promoted. Use fresh confirmation for any changes chosen after this test. The prototype still has fixed graph nodes and fixed memory dimensions; there is no growing expert web, assigned grammar categories, permanent self-learning during generation or proven Transformer replacement.

## Reproduction and artifacts

```sh
uv run python -m mlm.transfer_data --out data/kv-binding-new --seed 638101
uv run python -m mlm.plasticity_run --data data/kv-binding-new/diverse --config configs/associative-delta.json --init runs/scratch-english-hierarchical-v1/best.mlm --out runs/associative-delta-new --groups 128 --steps 2000 --rehearsal
```

Repeat with associative-none config and a fresh output path. Finish and lock all endpoints before model test scoring. protocol.json stores commands and frozen file hashes; preflight.json records source equivalence/state cost; checkpoint-lock.json precedes fresh tests; summary.json contains audits/outcomes; runtime-*.json contains raw timings/environment. Source/tests/config/lock snapshots have SHA256 hashes. Initial/final scored items, raw generation and full optimizer checkpoints remain archived. The runner has no resume CLI; generic checkpoint loading and standard full-stream continuation are tested. Standard stream warm starts reject mismatched memory settings.
