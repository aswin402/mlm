# Binding transfer through pairing, color and wording diversity

Three hierarchical-model trials compare fixed data, a larger canonical pool, and that same larger pool with varied wording. Every arm independently starts from the preserved English-only checkpoint, not a previous QA endpoint. The graph, sum readout, split learning rates, seed 42 and batch 4/unroll 128 stay fixed. Every arm receives 2,000 QA updates plus 2,000 English rehearsal updates; all endpoints are fixed with fresh optimizers. No development-based checkpoint selection occurs.

## Data and comparison boundaries

A new seed 904217 snapshot has 16 training labels and disjoint 16-label new development/test pools. Familiar-name development and test reserve disjoint balanced matchings of 8 pairs each. A third matching covers all training names for the fixed control. Larger pools contain 32 training pairs, balanced 16 within-role/16 cross-role, and include that fixed matching. These roles are offline strata, not model inputs or neural expert labels.

Fixed uses 8 pairs × 1 color pair × 2 fact orders = 16 complete four-question groups. Diverse uses 32 pairs × 2 complementary color pairs × 2 orders = 128 groups. Mixed preserves every diverse fact, query entity, candidate list, label and group ID, but uses two fact renderers and two question renderers. The four combinations are balanced across pair/color cases, with both fact orders using the same renderer choice. No model or inference implementation changed.

| Arm | Groups | QA documents | Training names | Presentations/group | Fact/query rendering |
|---|---:|---:|---:|---:|---|
| fixed | 16 | 64 | 16 | 125 | canonical |
| diverse | 128 | 512 | 16 | 15–16 | canonical |
| mixed | 128 | 512 | 16 | 15–16 | four combinations of two fact/two query forms |

Fixed versus diverse tests the combined addition of partners and color coverage at a fixed update/supervision budget. It does not isolate these two diversity factors or match repetitions per example. Diverse versus mixed is a stronger paired comparison: identical sampled underlying episodes, with only rendered wording changing. Wording changes input bytes; padded batch/unroll and answer supervision remain shared. Larger-pool failure at this budget cannot establish a capacity limit. Training partner counts and sampled query frequencies are included per name in summary.json; diverse degrees/frequencies vary, whereas fixed uses one partner per name.

Development and test each contain 2,304 questions / 288 eight-condition cases in six buckets: familiar pairs versus new names, crossed with base, alternate and heldout wording. Every bucket has 48 cases across 8 pairs and six color combinations. Base rendering is trained in every arm; alternate rendering is trained only by mixed. Heldout uses a third fact and question form absent from all task-training pools. Task-level name/wording absence does not guarantee absence from English pretraining. This is narrow grammar transfer, not broad English understanding.

## Training acquisition and English retention

The joint diagnostic gate requires at least 87.5% complete training groups and English development loss ≤1.25× source. Source English development loss is 1.352696. A complete training group requires all four assignment/query conditions; test cases additionally require both sentence orders. Answer rankings use two candidate colors rather than unrestricted generation.

| Arm | Train answers correct | Complete training groups | English development loss | Source multiple | Acquisition/retention gate |
|---|---:|---:|---:|---:|---|
| fixed | 35/64 | 0/16 | 1.303425 | 0.964× | fails |
| diverse | 256/512 | 0/128 | 1.296879 | 0.959× | fails |
| mixed | 256/512 | 0/128 | 1.301324 | 0.962× | fails |

## Fresh locked test

All three endpoint hashes and the original English source hash were saved before any model scored the test questions. The reliable-transfer gate is at least 90% complete cases in every bucket, not merely above-chance answer accuracy. With 48 cases per bucket this requires at least 44 complete cases in each. No endpoint is promoted on this diagnostic alone.

| Checkpoint | Familiar/base | Familiar/alternate | Familiar/heldout | New/base | New/alternate | New/heldout | Complete total |
|---|---:|---:|---:|---:|---:|---:|---:|
| source | 0/48 | 0/48 | 0/48 | 0/48 | 0/48 | 0/48 | 0/288 |
| fixed | 0/48 | 0/48 | 0/48 | 0/48 | 0/48 | 0/48 | 0/288 |
| diverse | 0/48 | 0/48 | 0/48 | 0/48 | 0/48 | 0/48 | 0/288 |
| mixed | 0/48 | 0/48 | 0/48 | 0/48 | 0/48 | 0/48 | 0/288 |

| Checkpoint | Individual answer ranking accuracy |
|---|---:|
| source | 50.043% |
| fixed | 50.087% |
| diverse | 49.566% |
| mixed | 50.000% |

Complete eight-condition cases expose shortcuts masked by individual-answer scores. Both within/cross strata, raw likelihoods, initial/final development results and selected raw text are preserved. Raw generation covers the fixed first four questions in each of six test buckets (24 examples per trained arm), limited to 12 output bytes with no extraction or answer repair. This is not a comprehensive free-generation benchmark. Source test ranking is included without raw sampling.

## Exposure, costs and integrity

| Arm | QA payload bytes | QA supervised bytes | English payload/supervised bytes | Optimizer updates |
|---|---:|---:|---:|---:|
| fixed | 776,000 | 40,000 | 958,088 | 4000 |
| diverse | 776,000 | 40,000 | 958,088 | 4000 |
| mixed | 768,008 | 40,000 | 958,088 | 4000 |

Each arm uses the same source parameters, all 16 training names, QA update count, answer-byte supervision and English stream. Canonical/mixed QA traces match exactly at the fact/query level. Fixed-pool exposure per example is much higher. Rehearsal is explicit extra work and the corpus is the same bounded English snapshot; improved development loss is not a claim about broad chat competence. The repeatedly used English test was not reopened.

Tests independently validate rendered context/questions, queried entities, correct answer labels, complete chat text, supervised byte spans and EOS; actual per-bucket name coverage; reserved pair/name/text separation; every assignment/query/order condition; balanced roles/renderers; deterministic hashes and 128-byte input bounds. The generator is offline only. Neural inference receives textual prompts and candidates; factual metadata is used for grading, not solving. Source/training inputs and all endpoint hashes are rechecked. Architecture, optimizer and training runner were reused unchanged.

One seed, a small vocabulary, finite budgets and reused source/development knowledge restrict conclusions. The three arms were fixed before these fresh model-test results, with no sweeps. Any revised method chosen after this comparison needs another confirmation snapshot. The old QA checkpoints remain historical artifacts and were never initializers here.

## Interpretation and next gate

fixed: 0/288 complete fresh test cases; reliable-transfer criterion not met.

diverse: 0/288 complete fresh test cases; reliable-transfer criterion not met.

mixed: 0/288 complete fresh test cases; reliable-transfer criterion not met.

The diversity and wording changes did not produce complete-case transfer at this offered budget. This negative result does not show that diversity cannot help or that the graph lacks capacity: the larger pools received fewer repeats, and a longer-budget learning curve would distinguish exposure limits. Before scaling a hierarchy or asserting general comprehension, compare an explicit learned memory/readout mechanism against the preserved recurrent baseline under matched training and fresh confirmation. Separate acquisition, retention and transfer gates remain necessary.

The fresh fixed control itself fails the acquisition threshold. The previous 16-group success therefore must not be read as mastery of arbitrary 16-group tasks. This dataset changes names, pair layout, color allocation and full paired sentence-order coverage relative to that earlier subset. The mismatch cannot be attributed to any one factor from this study, and failure is not solely about transferring an already learned task.

No growing nodes, named linguistic categories, Transformer replacement, sparse efficiency or arbitrary-device result is established by this study.

## Reproduction and evidence

```sh
uv run python -m mlm.transfer_data --out data/transfer-new --seed 904217
uv run python -m mlm.plasticity_run --data data/transfer-new/mixed --config configs/plasticity-split.json --init runs/scratch-english-hierarchical-v1/best.mlm --out runs/transfer-mixed-new --groups 128 --steps 2000 --rehearsal
```

Use --groups 16 with the fixed arm, and 128 with diverse/mixed, always fresh paths. Finish and lock every endpoint before model test scoring. protocol.json includes all commands and pre-run file hashes; binding-manifest.json records pools, pairs, templates and data hashes; checkpoint-lock.json precedes test evaluation; summary.json records the audits and outcomes. Source/tests/config/lock snapshots have SHA256 hashes. Weights remain under runs/transfer-{arm}-v1/final.mlm. verification.json records the 64-test suite, strengthened data check and resolved code review.

Verification: the full 64-test regression suite passed in 159.80 seconds; the strengthened transfer-data check passed separately in 1.65 seconds. Ruff lint/format and source type checks passed after archiving. The code review has no unresolved findings.
