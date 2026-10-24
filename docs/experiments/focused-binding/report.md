# Focused two-person binding

This experiment tests the smallest failed retrieval case: two people, two distinct colors, no filler. All unordered training-name pairs are represented, rather than the previous fixed query partners. Fact order, color assignment, and queried person are balanced. Each complete group contains both queried people crossed with both color assignments, so preferring the first/last fact cannot solve a group.

## Protocol

The task has 16 training names (120 entity pairs / 240 groups / 960 documents), eight new development names (28 pairs / 56 groups / 224 documents), and eight separate test names with the same counts. Each entity pair receives one cyclically assigned color pair from six combinations in a seeded shuffled entity-pair order, with both fact orders. This covers varied partners but not every possible name/color combination. Grammar and the four color words are shared across splits; names are disjoint within this task, not necessarily globally novel in earlier training.

One full four-question group forms each batch. Every synthetic document is 98 input bytes including BOS and fits the 128-byte unroll. Assistant answer/newline and EOS alone receive cross-entropy loss; prompt bytes update neural state. State resets between independent examples and groups. The model has no access to grading labels or factual metadata during inference. Offline label construction is not an answer solver.

Graph and GRU warm-start from their respective single-fact basic checkpoints with fresh optimizers. Both use seed 42, 2,000 updates, batch four, learning rate 0.0005, no story rehearsal, and evaluation at zero and every 250 updates. Selection maximizes complete development binding-group accuracy; target loss only breaks ties, and exact ties preserve the earlier step. This dedicated runner does not implement resume and refuses nonempty output directories.

The source/config/data hashes, group identities, and conditional readout policy are recorded before training. The conditional architecture experiment is permitted only if both controls remain below 90% development group accuracy. Test data is not used to choose this branch. All selected checkpoint hashes must be locked before first focused-test scoring.

Two-choice likelihood ranking and raw greedy generation are distinct measures. Independent random two-choice predictions average 50% individual, 25% complete context pairs, and 6.25% complete four-question groups; these are reference rates, not a significance test. Report wrong flips as well as correct groups. English retention uses the repeatedly inspected story diagnostic and is not a fresh generalization benchmark.

## Reproduction

Use fresh output paths and preserve source checkpoints.

```sh
uv run python -m mlm.binding_data --out data/binding-new
uv run python -m mlm.binding_run --data data/binding-new --config configs/binding-graph.json --init runs/retrieval-basic-graph-v1/best.mlm --out runs/focused-binding-graph-new --steps 2000 --eval-every 250
uv run python -m mlm.binding_run --data data/binding-new --config configs/binding-gru.json --init runs/retrieval-basic-gru-v1/best.mlm --out runs/focused-binding-gru-new --steps 2000 --eval-every 250
uv run python -m mlm.retrieval_eval runs/focused-binding-graph-new/best.mlm --questions data/binding-new/test-questions.json --out results/focused-binding-graph-new.json
```

Complete all planned development experiments and lock the checkpoints before running the last command. The dataset generator defaults to seed 91827. Repeated tuning after viewing this test requires another confirmation split. This is a one-seed, bounded synthetic task; it does not establish natural-language comprehension or Transformer superiority.

## Conditional learned readout

Both controls scored zero complete development groups after 2,000 updates, triggering the single predeclared readout variant before any focused-test scoring. Its key and query transforms score the eight current recurrent node states:

q = h_0 Q; e_i = v^T tanh(h_i K + q); alpha = softmax(e); readout = sqrt(N) sum_i alpha_i (h_i R_i).

Node zero supplies the conditioning state because it started with the shortest nominal retention prior. It is not an explicit question encoder, and its state is not guaranteed to represent only the query. This builds on established [additive attention](https://arxiv.org/abs/1409.0473); it is a research adaptation, not a new attention principle. Graph message aggregation remains fixed. No stored token-history attention, growing nodes, or structural routing was added.

The optional configuration is `node_readout: "attention"`; default `"sum"` retains the original path and legacy checkpoint behavior. A zero initial score vector yields uniform alpha and reproduces the old sum/sqrt(N) pooling when shared parameters are copied. The variant adds 8,256 learned parameters: 374,978 total against graph's 366,722 and GRU's 368,210. Its recurrent state remains 512 scalars per example; GRU uses 208. It is not a parameter- or compute-matched superiority test.

The dedicated grouped runner supports copying shared source tensors into a model with new readout parameters. Ordinary stream initialization requires matching readout configurations and rejects incompatible parameter trees. The new tests verify uniform equivalence, nonzero gradients, source integrity, padding, resets, causality, chunk state, config validation, and checkpoint roundtrip. Readout activity auditing confirms all three new parameter tensors changed during training. On the first test group weights remain fairly diffuse, with query-swap weight differences about 0.005 in L2 norm; that is a narrow diagnostic, not an established explanation of failure.

## Results

All three models completed 2,000 updates on the same complete-group order, consuming 776,000 input payload bytes and 40,000 supervised payload bytes each. Final sampler order, cursor, epochs, RNG, group identities, and supervision hashes match. Config, data, source and selected checkpoint hashes were checked. The graph selects step 1,750; GRU and readout variant select step 2,000. Every development group score remains zero, so the secondary loss criterion decides among eligible checkpoints. There was no test-based checkpoint choice.

| Model | Individual test answers /224 | Complete context pairs /112 | Complete binding groups /56 | Complete training groups /240 | Story test loss |
|---|---:|---:|---:|---:|---:|
| Graph, sum readout | 115 | 12 | 0 | 0 | 7.439665 |
| Plain GRU | 121 | 18 | 1 | 7 | 8.119605 |
| Graph, learned readout | 111 | 10 | 0 | 1 | 7.668399 |

Individual accuracy is 51.34%, 54.02%, and 49.55%. These results do not demonstrate reliable binding or a readout improvement. The training diagnostics evaluate the selected models on all training questions; they are not generalization results. Poor training-group performance shows that novel held-out names alone do not explain the failure. This does not prove the architectures incapable of learning binding under other optimization or data conditions.

Raw greedy generation is preserved for the fixed first complete test group (four samples per model). That tiny sample must not be presented as a full-test generation score. The complete neural likelihood scores and all correct/incorrect query/context swaps are saved in the test JSON files. No answer extraction or repair occurs.

Held-out assistant-target losses are 0.117465, 0.116768, and 0.116877. Low byte loss is misleading here: for illustration, uncertainty between two color words with distinct first bytes can contribute log(2)/6 = 0.1155 average loss over four color bytes, newline and EOS, while all later teacher-forced targets are easy. This is an explanatory reference calculation, not a fitted model of these networks' internals.

Removing rehearsal deliberately made the task objective explicit, but English prediction worsened sharply. Earlier basic-stage story losses were 3.487491 graph and 4.133074 GRU, and original English checkpoints were 1.121557 and 1.098890. None of these focused checkpoints replaces a preserved English/chat reference. The story set is a reused retention monitor. This one-seed experiment changes data pairing, batching, objective mix and selection relative to the previous curriculum, so it cannot attribute failure or improvement to one factor alone. Timing is uncontrolled and does not support speed claims.

## Training-only learnability check

After the main test was recorded, a separate post-hoc diagnostic trained the original sum-readout graph on just one four-question training group for 500 updates with learning rate 0.003 and fresh optimizer, starting from the same basic graph source. It reaches all four correct neural rankings and one complete correct binding group. This is a fixed single diagnostic, not a hyperparameter sweep or an additional model selected on the held-out test. Its scope is memorization, not generalization. The smaller dataset and higher learning rate changed together, so the result does not identify the sole cause of earlier failure.

Raw greedy generation on those same four memorized training questions scores 100%; unedited outputs are in single-group-generation.json. The sanity checkpoint is stored separately under runs/binding-overfit-sanity-v1 and is not promoted as a language model upgrade.

## Decision and verification

Keep the learned readout as an optional experimental feature, not the default. The graph can fit a tiny binding case, but the broader training protocol does not learn a useful rule. Next use a gradual curriculum from one group to multiple names/pairings, with known-name novel-pair evaluation separated from held-out-name evaluation. Fix optimization and supervision accounting before increasing capacity, and freeze new confirmation data before tuning. More graph nodes or growing topology are not yet justified by these results.

All 44 tests pass, plus Ruff lint/format and source type checks. Source checkpoints remain intact; no answers are hardcoded into inference. Offline data construction, grouping and answer keys are training/evaluation infrastructure. The optional input-dependent readout changes activations, while learned parameters remain fixed during ordinary inference; it does not implement online weight updates or node growth.
