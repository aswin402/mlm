# Gradual binding versus a matched flat control

This experiment keeps the original eight-node sum-readout graph unchanged. It starts from the separate checkpoint that memorized one four-question Dan/Ned gold/blue binding group. The source already received 500 updates on that group after its earlier English and basic-stage training. Both arms start with identical source parameters and a fresh optimizer, then receive 3,000 additional updates. The experiment does not count source-history updates as part of that new budget.

## Data and questions

The nested training pool has 64 groups / 256 questions. All groups cross two color assignments and two queried people, and each group occupies one four-example batch. First comes the exact source memorization group. The first four groups introduce eight training names; the next four reverse those fact orders. Further groups add color combinations and other entity partners. By 32 groups all 20 allowed training entity pairs appear; four other pairs are reserved for development and four for test. Exact training texts cannot appear in either evaluation split.

Evaluation has two distinct buckets, each with 48 four-question groups / 192 questions:

- New pairings of familiar training-pool names, with pair identities absent from training.
- A separate eight-name pool that does not occur in this task's training names.

Each bucket's four entity pairs cover all eight of its names, all six color pairs, and both fact orders. Development and test have different reserved familiar-name pairs and different new-name pools. Shared grammar and four color words intentionally isolate binding rather than unfamiliar wording. Names are held out within this task, not guaranteed globally unseen in English pretraining. This is a bounded synthetic task, not a broad chat benchmark.

Early familiar-name-pool development rows may include names not yet introduced for binding in the current run. Logs therefore include names sampled in the run and an additional metric restricted to genuinely sampled-name pairs. Source training history is excluded from that coverage counter. Selection only begins once all eight training names have been sampled.

## Fixed comparison

Curriculum adds groups at 500-update stages: 2, 4, 8, 16, 32, then 64. The already memorized single-group checkpoint supplies the initial one-group reference. Adam state and optimizer step count continue across stages. The flat control sees all 64 groups from the start for the same total 3,000 updates, with continuous shuffling. Both use the same source, seed 42, batch four, unroll 128, learning rate 0.003, architecture, and parameter count. Learning rate matches the prior successful sanity run; it is not tuned on this test.

All examples fit one unroll, so answer loss can backpropagate to all input facts. Prompt bytes update neural state but receive no target loss. Assistant answer/newline and EOS are supervised. Recurrent state resets between independent examples/groups. There is no story rehearsal; English prediction is monitored separately and does not choose checkpoints.

Initial and 500-update scores are diagnostic only. Both arms select among update endpoints 1,000–3,000 after actual eight-name coverage, maximizing complete-group development accuracy over the fixed equal-sized familiar/new-name buckets, then minimizing target loss. Exact ties retain the earlier endpoint. The global development score therefore equals the mean of the two bucket group accuracies. Training accuracy and the reused English story diagnostic do not control selection. The runner does not support resume and refuses nonempty outputs.

Data, config, source, runner and generator hashes are frozen in protocol.json. Both selected checkpoint hashes are locked before the new test is evaluated. There is no hyperparameter sweep. Repeated experiments informed this task design, and the source is a post-hoc memorization checkpoint; findings require independent confirmation before claims of general language understanding or a robust curriculum advantage.

## Reproduction

Use fresh output paths. The seed-group artifact and sanity source checkpoint must exist.

```sh
uv run python -m mlm.gradual_data --out data/gradual-new
uv run python -m mlm.gradual_run --data data/gradual-new --config configs/gradual-graph.json --init runs/binding-overfit-sanity-v1/final.mlm --out runs/gradual-curriculum-new --strategy curriculum --stage-steps 500 --stories data/hf/stories/test.jsonl
uv run python -m mlm.gradual_run --data data/gradual-new --config configs/gradual-graph.json --init runs/binding-overfit-sanity-v1/final.mlm --out runs/gradual-flat-new --strategy flat --stage-steps 500 --stories data/hf/stories/test.jsonl
uv run python -m mlm.gradual_eval runs/gradual-curriculum-new/best.mlm --questions data/gradual-new/test-questions.json --out results/gradual-curriculum-new.json
```

Lock both development-selected checkpoints before the last command. Neural candidate ranking uses only textual prompts and learned parameters. Labels and factual metadata only construct/grade the offline dataset; no symbolic answer lookup or regex solver participates in inference. The evaluator preserves raw greedy samples from the first complete group of each bucket, with whitespace/case normalization and no extraction or repair. Those samples are not a full-test generation benchmark.

## Training progression

| Additional updates | Curriculum groups | Complete active training groups | Original group retained | Familiar-name dev groups | New-name dev groups |
|---|---:|---:|---:|---:|---:|
| 0 | 1 | 1/1 | 100% | 0.0% | 0.0% |
| 500 | 2 | 2/2 | 100% | 0.0% | 2.1% |
| 1000 | 4 | 4/4 | 100% | 6.2% | 16.7% |
| 1500 | 8 | 8/8 | 100% | 2.1% | 16.7% |
| 2000 | 16 | 16/16 | 100% | 0.0% | 20.8% |
| 2500 | 32 | 29/32 | 100% | 0.0% | 41.7% |
| 3000 | 64 | 50/64 | 100% | 2.1% | 29.2% |

The source fits one group; the curriculum preserves that group and fits 2, 4, 8 and 16 groups completely at the recorded endpoints. It drops to 29/32, then 50/64 as new pairings and color assignments are introduced. This is a measured training failure point under this schedule, not proof of a fixed model capacity limit. The flat control fits all 64 training groups by update 2,000 and again at 3,000. Early flat training forgets the seed group, then recovers it.

Both runs finish the full 3,000-update budget. Development selection chooses curriculum step 2,500 (32-group stage) and flat step 2,000 (64 groups). **The reported curriculum test checkpoint therefore predates its final 64-group stage.** The selected curriculum has sampled all eight names and all 20 allowed training entity pairs, but not all 64 eventual color/order groups. Both runs are offered the same budget; their selected update counts and exposure distributions differ as specified by the protocol.

Each arm consumes exactly 1,164,000 input payload bytes and 60,000 supervised payload bytes during the new run. Source, architecture, parameter count (366,722), optimizer settings and budget match; sampling schedules deliberately differ. Adam state continues across stages. The baseline initial source predictions match across both arms. Data/config/source hashes and selection eligibility were verified, then both selected hashes locked before fresh test scoring. Timing is uncontrolled and not a speed benchmark.

## Fresh test results

Two-choice neural likelihood ranking is the primary scoring method. A four-question group requires both color assignments and both queried people for one sentence order. A stricter secondary case requires all eight conditions including both sentence orders. The secondary metric was added after initial development observations, before test scoring; it did not change checkpoint selection. This avoids counting one-order correctness alone as fully order-robust binding.

| Checkpoint | Pool | Individual correct /192 | Complete four-question groups /48 | Complete eight-condition cases /24 |
|---|---|---:|---:|---:|
| Source memorization checkpoint | Familiar names, new pairs | 95 | 0 | 0 |
| Source memorization checkpoint | New names | 95 | 0 | 0 |
| Curriculum selected | Familiar names, new pairs | 186 | 42 | 18 |
| Curriculum selected | New names | 122 | 10 | 1 |
| Flat selected | Familiar names, new pairs | 145 | 23 | 11 |
| Flat selected | New names | 150 | 21 | 9 |

Thus the curriculum reaches 96.88% individual accuracy and 18/24 complete order-robust familiar-name cases, but only 63.54% individual and 1/24 robust new-name cases. Flat reaches 75.52% and 11/24 for familiar names, versus 78.13% and 9/24 for new names. The original source has zero robust cases in both pools. There is partial transfer on this controlled task, not just recall of exact training questions, but it is far from reliable name-independent retrieval.

Overall primary four-question test-group correctness is 52/96 curriculum versus 44/96 flat. Overall secondary eight-condition correctness is 19/48 versus 20/48. These measures emphasize different error patterns: there is no clean overall winner. Independent fair random answers have reference probabilities 1/16 for four conditions and 1/256 for eight, but model predictions and examples are dependent; these are not statistical null tests or proof of broad semantic understanding.

Raw greedy samples are fixed before output inspection: the first complete group from each pool (eight total per checkpoint). They score 5/8 for source, 7/8 curriculum and 6/8 flat. This small sample is saved unedited and is not a full-test generation score. Normalization only strips whitespace and case-folds; there is no answer extraction or repair.

## Important split limitation discovered in the audit

A post-hoc construction audit found an avoidable imbalance in the familiar-name split. The initial four training pairs partition people into early pair roles: Dan/Ada/Cal/Fay versus Ned/Ben/Eli/Gus. All four familiar-name development pairs join people from the same early role; all four familiar-name test pairs join opposite early roles. Pair identities are disjoint and texts do not leak, but the two pools are structurally different.

This is particularly relevant because curriculum familiar-name development accuracy is near chance while familiar-name test accuracy is very high. A coarse person-role shortcut could benefit from this structure. The audit does not establish what representation the network actually learned, but it prevents presenting the 18/24 test result as clean evidence of a general entity-matching rule or a robust curriculum advantage. The split I constructed needs a balanced replication across role types and multiple name assignments. Existing results remain preserved rather than silently replacing the test after seeing outcomes.

## English retention and limits

The reused English story monitor starts at loss 4.743716 for the memorization source, already worse than the preserved English checkpoint's 1.121557. Selected curriculum loss is 12.834514 and selected flat loss 10.170839; final endpoints are 13.096501 and 11.697480. This task-only training severely degrades English prediction. Neither model replaces an English/chat checkpoint. No story data enters the new training objective or the selection score.

The current comparison controls architecture, source, learning rate and total offered updates; it does not establish that all curricula are better or worse. Under this schedule, the curriculum gets only 500 updates with all 64 groups available, versus 3,000 in flat training. Longer consolidation or mastery-based progression might change results, but neither was tested. Compared with prior experiments, learning rate, source, name pool and task size changed, so improvements cannot be attributed to one of those factors alone. This is one seed, eight training names, four colors and one shared grammar. It does not establish reliable English chat, broad comprehension, dynamic topology, online inference-time learning or Transformer superiority.

## Decision and verification

Keep the models as research artifacts. The unchanged graph can learn multiple binding cases and transfer partially, but name/pair robustness, benchmark balance and English retention remain unresolved. Before further architecture work, build a fresh role-balanced confirmation split with more varied names and evaluate both known-name new pairings and new-name transfer. Preserve independent development/test pools and monitor English with an explicit retention strategy. Do not retune using this test.

All 47 tests pass, including nested data/split/source checks, persistent optimizer steps, selection eligibility, output protection, and eight-condition grading that rejects one-order luck. Ruff lint/format and source type checks pass. Inference remains entirely neural; labels, pair-role auditing and grouping are offline analysis. No response rules, growing nodes or architecture changes were added in this experiment.
