# Balanced confirmation with explicit English rehearsal

This experiment addresses the previous role-imbalanced split and increases training-name diversity while keeping the original sum-readout graph unchanged. It is a new snapshot and does not replace historical tests or outcomes.

## Dataset and balance

A fixed seed samples 64 artificial three-letter labels: 32 training, 16 new development and 16 new test names. Offline role classes are random strata and never appear in model prompts. Training uses 64 entity pairs: two within-role cycles and two cross-role matchings. Every training name has exactly four partners. Half of the training pairs are within-role and half cross-role; known held-out development/test pair identities are excluded.

Each training pair receives two disjoint color pairs covering all four colors, both color assignments, both queries and both sentence orders. There are 256 four-question groups / 1,024 training documents. Every document fits the 128-byte unroll. Source words and one English sentence grammar are shared across splits; names are held out within this binding task, not guaranteed globally new in English pretraining.

Development and test each contain 1,152 questions. The familiar-name bucket uses 16 disjoint pairs covering all 32 training names, half within-role and half cross-role, evaluated over all six color pairs and both fact orders: 96 eight-condition cases. The new-name bucket uses eight pairs covering all 16 split-specific names, also half within-role and half cross-role: 48 eight-condition cases. New-name pools, familiar pair identities and exact texts are disjoint where required. Unit tests verify these properties, partner degree, labels, spans, eight-condition completeness, and truncation bounds.

## Source, objectives and cost

Both arms start from `runs/english-v1/best.mlm`, the preserved 4,000-update English checkpoint, with a fresh optimizer. This differs from the previous memorization source: the current question is whether a clean English model can acquire binding while retaining English. Differences from earlier experiments cannot be attributed to balancing alone.

Both use seed 42, learning rate 0.001, batch four, unroll 128 and the same 2,000 complete-group QA updates in the same sampler order. QA loss supervises assistant color/newline and EOS; facts and question still update recurrent state. Task-only receives 2,000 optimizer updates. Rehearsal performs one full-document English training-stream chunk after each QA update, receiving 4,000 optimizer updates total. Rehearsal uses the cached 1,000 training stories, not development/test stories. QA state resets per example; English has its own persistent stream carry with truncated backpropagation. Adam state is shared within the rehearsal arm across both objectives.

Each batch loss is mean target cross-entropy. Alternating steps do not imply equal token counts or gradient contribution. Actual input/supervised-byte and update counts are recorded separately. QA exposure is matched; total compute, optimizer steps and input exposure are not. Checkpoints in the rehearsal arm are evaluated at the end of a full QA+English cycle. No speed, compute-efficiency or architecture-superiority claim follows from this comparison.

## Selection and test policy

Initial state and every 500 QA updates are eligible. Primary binding score is the macro-average complete eight-condition correctness of the familiar-name and new-name development buckets; their different sizes do not give the larger bucket extra weight. Both sentence orders, both assignments and both queried people must be correct. QA development loss breaks ties; exact ties preserve earlier weights.

Two policies are saved per arm:

- `best-task.mlm`: strongest development binding score, unconstrained by English.
- `best.mlm`: strongest binding score among checkpoints whose English development loss is at most 1.25 times the initial source loss. Step zero is eligible, so this may be a source fallback rather than a newly learned model.

English development is explicitly a selection constraint. English test remains unopened until all four selected checkpoints are locked, and is a reused retention diagnostic rather than fresh independent generalization evidence. Task test is newly generated and never chooses a checkpoint or hyperparameter. The budget, source, data, config and runner hashes are frozen in protocol.json. There is no sweep or resume feature. The runner rejects nonempty output paths.

Primary task results use two-choice frozen neural likelihood ranking. Raw greedy answers are preserved separately for fixed samples and must not be confused with full-test ranking. Offline labels, role tags and grouping only construct/grade data; no inference answer lookup or regex solver is added. Names are synthetic and grammar is shared, so success would not establish broad conversation or natural-language comprehension. One seed does not establish a robust general advantage.

## Reproduction

Use fresh paths and preserve source/cache files.

```sh
uv run python -m mlm.confirmation_data --out data/confirmation-new
uv run python -m mlm.confirmation_run --data data/confirmation-new --config configs/confirmation-graph.json --init runs/english-v1/best.mlm --out runs/confirmation-task-new --steps 2000 --eval-every 500
uv run python -m mlm.confirmation_run --data data/confirmation-new --config configs/confirmation-graph.json --init runs/english-v1/best.mlm --out runs/confirmation-rehearsal-new --steps 2000 --eval-every 500 --rehearsal
uv run python -m mlm.gradual_eval runs/confirmation-rehearsal-new/best.mlm --questions data/confirmation-new/test-questions.json --out results/confirmation-rehearsal-new.json
```

Run the last command only after locking both policies from both completed arms. Repeat for unconstrained best-task checkpoints and the source reference as separately labelled results. The original graph topology, parameter/state budget, and inference algorithm are unchanged.

## Verified selection and exposure

Both arms completed all 2,000 QA updates. QA sampler order, cursor, RNG and group identities match exactly, and both consume 776,000 QA input bytes and 40,000 supervised QA payload bytes. The rehearsal arm additionally performs 2,000 English updates, consuming and supervising 958,088 English payload bytes. Thus total optimizer calls are 2,000 versus 4,000, and input exposure is 776,000 versus 1,734,088 bytes. Rehearsal's added resource cost is explicit; these results do not establish a cheaper training method.

The initial English development loss is 1.155704; the predeclared constraint is 1.444630. Task-only development loss rises above this limit at every trained checkpoint. Its protected selection is therefore QA step zero, with parameter hash verified identical to the original English source; this is a fallback, not a successful new binding model. Its unconstrained task selection is QA step 1,500. Rehearsal selects QA step 2,000 / optimizer step 4,000 under both policies. Those two slots have identical parameter hashes and are the same trained model, not independent successes.

All development eight-condition binding scores remain zero, so QA loss is the tiebreaker. Rehearsal's lower QA loss is therefore not evidence of semantic success. Protected selection enforces only an English prediction constraint; it does not guarantee successful adaptation. All four selected checkpoint hashes and source identities were locked before task or English test scoring.

## Test results

| Arm / policy | Selected QA / optimizer updates | Familiar-name cases correct /96 | New-name cases correct /48 | English test loss |
|---|---:|---:|---:|---:|
| Task-only, protected source fallback | 0 / 0 | 0 | 0 | 1.121557 |
| Task-only, unconstrained | 1,500 / 1,500 | 0 | 0 | 3.381755 |
| Rehearsal, protected and unconstrained (same model) | 2,000 / 4,000 | 0 | 0 | 1.087839 |

A case requires all eight color-assignment/query/sentence-order conditions correct. All within-role and cross-role strata also score zero complete cases. Individual candidate-ranking accuracy is exactly 50% and preferred answers do not change with context/query swaps for these selected models. This is consistent with fixed answer preferences, not useful binding. Full item scores and histograms are preserved; the larger balanced test does not demonstrate name-independent retrieval under this acquisition protocol.

Rehearsal preserves and slightly improves held-out English byte prediction relative to the source, while unconstrained task-only adaptation harms it substantially. English test is a repeatedly inspected retention diagnostic, so this is one measured run rather than independent broad generalization evidence. The preserved English source remains available. The rehearsal model is a retention-compatible research candidate; it is not a capable chatbot or an accepted binding upgrade.

The protected task-only selection doubles as the source reference on this fresh task. The duplicate rehearsal policy result is explicitly labelled as equivalent checkpoint reuse; only three unique parameter sets were evaluated. Eight raw greedy samples per unique checkpoint (first full group per bucket) are saved separately. Those samples are not a full-test generation benchmark, and no answer extraction/repair is applied.

## Interpretation

The balanced benchmark removes the previously discovered within/cross-role split imbalance and increases the training pool from eight to 32 names. It does not by itself resolve binding learning. Compared with the earlier partial-transfer study, source weights, learning rate, name count, task size and training schedule also changed. This is a new acquisition-and-retention setting, not a source-matched replication proving that all earlier transfer was invalid or that balancing alone caused failure.

The original English model can learn short answer formatting and maintain English with explicit replay, but no reusable entity–fact retrieval rule was demonstrated within this budget. This does not establish an architecture impossibility: insufficient task progression, optimization, memory initialization and data exposure remain possible contributors. Further diagnosis should hold source/learning rate fixed while scaling a balanced task from small memorized cases to more names, and distinguish failure to fit training from failure to transfer. More architecture or capacity is not yet supported as a proven remedy.

No graph growth, rewiring, learned edges, online inference-time weight updates, or new readout was introduced. Parameters remain 366,722 with the original recurrent state budget. Inference stays entirely neural. Offline labels, matching/cycle construction and role strata are dataset/evaluation logic, never an answer solver.

## Verification and decision

All 50 regression tests pass. Ruff lint/format and source type checks pass. Tests cover balanced partitions, partner degree, exact-text/name/pair disjointness, labels/spans and eight-condition completeness, objective counters, protected selection, output protection and source immutability, alongside prior model/state/checkpoint tests. Dataset/config/source/code hashes and matched QA exposure were verified before final scoring.

Keep all checkpoints and negative results as research evidence. Do not promote a binding-capability claim from low answer loss or English retention alone. The next focused study needs a source-controlled balanced learning ladder, with retention evaluated explicitly and a fresh independent confirmation set after tuning.
