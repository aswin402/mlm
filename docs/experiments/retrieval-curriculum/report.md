# Controlled entity–fact retrieval curriculum

This experiment isolates a language model's ability to retrieve the color assigned to a queried person. It uses synthetic English sentences with the same grammar across splits. It does not test broad chat or transfer to unfamiliar wording. The graph and GRU architectures, learned update equations, parameter counts, and state sizes remain unchanged.

## Design

The offline generator writes sentences such as `Ada's color is blue. Ben's color is pink.` and asks `What color belongs to Ada?`. All four answer colors have four ASCII bytes: blue, gold, lime, pink. There are 16 training names, eight development names, and eight test names, disjoint within this task. They are not guaranteed unseen in earlier English training or earlier experiments.

There are nine difficulty cells: zero, one, or three distractors, crossed with 0, 64, or 128 bytes of neutral filler. Each document assigns a unique color to each person. A context-swap pair exchanges the queried person's color and an alternative, keeping the question and candidate choices fixed. Binding cells additionally ask about the other person with exactly the same context, producing four-question groups. Correctness on the whole group requires both context swaps and query swaps to be answered correctly. The generator balances query position by reversing fact order, and records the factual assignments for offline validation only.

Two-choice candidate ranking uses the frozen neural model's answer-byte log probabilities. Labels and factual metadata grade predictions; they never enter neural scoring. Candidates are not inserted into the prompt. Raw greedy decoding is also saved for the first complete group from each cell, selected without inspecting outputs. It uses whitespace/case normalization only, with no answer extraction or repair. Two-choice ranking and unrestricted generation are different measures.

Independent fair two-choice guesses average 50% individual accuracy, 25% complete context pairs, and 6.25% complete four-question groups. A fixed answer can achieve 50% individual accuracy while solving no complete pairs. A first/last-fact preference can respond to a context swap without correctly using the query. These reference rates are not significance tests, and examples within a group are dependent.

## Training and selection

Both models start from their respective English-derived retention-bias initializations. No weights selected using previous reading-test results are used. Each receives a basic single-fact/no-delay stage of 500 updates, then a binding/delay stage of 1,500 updates initialized from the basic development-selected checkpoint. Each stage creates a fresh optimizer. Basic validation occurs every 250 updates; binding validation every 500. Minimum stage-specific development target loss selects checkpoints, including step zero.

Both use seed 42, batch size four, unroll 384, and learning rate 0.0005. Every synthetic example is at most 268 input bytes including BOS and fits in one unroll, so its answer loss can backpropagate to all of its facts. The binding training set includes 32 cached public training stories with explicit full-text loss as rehearsal. The synthetic records supervise only assistant answer/newline and EOS; prompt bytes still update neural state.

The basic training set has 64 documents; the full training set has 960 synthetic documents plus 32 stories. Development and test each contain 480 synthetic documents spanning all cells; their basic subsets each contain 32. Synthetic example selection, names, data files, configurations, and source checkpoint hashes are frozen in protocol.json. Final selected checkpoint hashes are recorded before opening test questions. This is one seed and a bounded synthetic task; positive findings require further independent confirmation.

## Reproduction

Reuse the existing public cache and English-derived retention initializations. Choose fresh output paths.

```sh
uv run python -m mlm.retrieval_data --out data/retrieval-new --source data/hf
uv run mlm stream-train --train data/retrieval-new/basic-train.jsonl --validation data/retrieval-new/basic-validation.jsonl --config configs/retrieval-graph.json --init runs/memory-init-graph/initial.mlm --out runs/retrieval-basic-graph-new --steps 500 --eval-every 250
uv run mlm stream-train --train data/retrieval-new/train.jsonl --validation data/retrieval-new/validation.jsonl --config configs/retrieval-graph.json --init runs/retrieval-basic-graph-new/best.mlm --out runs/retrieval-binding-graph-new --steps 1500 --eval-every 500
uv run python -m mlm.retrieval_eval runs/retrieval-binding-graph-new/best.mlm --questions data/retrieval-new/test-questions.json --out results/retrieval-graph-new.json
uv run mlm evaluate runs/retrieval-binding-graph-new/best.mlm --data data/hf/stories/test.jsonl --full-documents
```

For the GRU substitute configs/retrieval-gru.json, runs/memory-init-gru/initial.mlm, and fresh GRU output paths. Run and lock both models before examining test outcomes. Repeated tuning against these test questions would make them development data.

## Results

Both basic stages completed 500 updates and selected step 500. Both binding stages completed 1,500 updates and selected step 1,500. The selection lock was written and checkpoint/data hashes verified before retrieval test evaluation. Training/development identities, supervision masks, final sampler snapshots, input exposure, and supervised-byte exposure matched between architectures in each stage. The basic stage consumed 152,000 payload bytes and supervised 10,000; the binding stage consumed 1,091,028 and supervised 152,091.

Both basic-stage checkpoints answered all 32 fresh single-fact/no-delay test questions correctly by candidate likelihood, including all 16 counterfactual pairs. This is successful narrow recall on held-out task names with shared grammar, not evidence of query binding: there is only one person's fact to choose from. Raw greedy examples are saved separately.

After binding adaptation, graph answers 281/480 individual test questions correctly (58.54%) and GRU 291/480 (60.63%). Complete context-pair accuracy is 68/240 for graph and 66/240 for GRU. **Neither solves any of the 96 complete four-question binding groups.** Only 6/192 fixed-context query pairs are both correct for graph, and 8/192 for GRU. Preferred answers change with the query in 19/192 and 13/192 cases respectively; many changes are wrong. Individual accuracy is not a reliable substitute for the binding measure.

| Difficulty cell | Graph individual | GRU individual | Graph complete binding groups | GRU complete binding groups |
|---|---:|---:|---:|---:|
| d0-delay0 | 100.0% | 100.0% | — | — |
| d0-delay128 | 100.0% | 100.0% | — | — |
| d0-delay64 | 100.0% | 100.0% | — | — |
| d1-delay0 | 50.0% | 50.0% | 0.0% | 0.0% |
| d1-delay128 | 46.9% | 50.0% | 0.0% | 0.0% |
| d1-delay64 | 45.3% | 50.0% | 0.0% | 0.0% |
| d3-delay0 | 51.6% | 46.9% | 0.0% | 0.0% |
| d3-delay128 | 50.0% | 53.1% | 0.0% | 0.0% |
| d3-delay64 | 45.3% | 54.7% | 0.0% | 0.0% |

Each single-fact cell contains 32 questions / 16 context pairs. Each binding cell contains 64 questions / 32 context pairs / 16 four-question groups. The gap between single-fact and multiple-fact performance is the important diagnostic; aggregate accuracy blends them. All group failures are preserved, including contradictory query/context preferences.

Graph: across all delays, single-fact individual accuracy is 100.00%; multiple-fact individual accuracy is 48.18%. The fixed 30-example raw greedy sample scores 50.00%. Greedy outputs are unedited and this small sample is not a full-test generation score.

Gru: across all delays, single-fact individual accuracy is 100.00%; multiple-fact individual accuracy is 50.78%. The fixed 30-example raw greedy sample scores 50.00%. Greedy outputs are unedited and this small sample is not a full-test generation score.

## English retention and supervision exposure

Story test loss after the basic stage is 3.487491 for graph and 4.133074 for GRU. After the binding/rehearsal stage it is 1.577292 and 1.264509. Recovery coincides with story rehearsal but no no-rehearsal control was run, so rehearsal is not an isolated causal explanation. Both remain worse than their English source checkpoints (1.121557 and 1.098890). Preserve the original English checkpoints; these are task experiments, not accepted English/chat upgrades.

The 32 stories are only about 3.2% of training documents, but a deterministic stream replay verifies 124,221 of 152,091 supervised payload bytes come from stories, versus 27,870 from synthetic retrieval. Thus roughly 81.7% of scored payload bytes are rehearsal. Per-update masking and normalization mean this is not an exact fraction of gradient contribution. It shows why document-count ratios alone are insufficient. The exposure audit and matched-stream checks are saved with raw evidence.

Final held-out assistant-target loss is 0.147348 for graph and 0.146569 for GRU. These losses include answer spelling, newline, and EOS; they must not be presented as comprehension accuracy or compared directly with full-text story loss.

## Interpretation and next experiment

The current networks can learn single-fact recall with this grammar and new task names. They do not demonstrate reliable entity–fact binding under the tested curriculum, budget, and selection rule. Longer-lived state and a lower supervised loss are insufficient evidence of successful query-dependent retrieval. This does not prove that the architecture cannot learn binding: optimization, curriculum, data coverage, and supervision balance remain confounded.

The training data uses a bounded set of paired query entities, one shared sentence template, four colors, and one seed. Compared with earlier runs, data, curriculum, and unroll changed together. There is no unshifted-gate control on this new task, so these results cannot isolate a benefit of the retention intervention. Runtime contention also rules out a controlled speed comparison.

Before increasing graph size, isolate the smallest failed case: one distractor, zero filler, varied entity pairings, and an explicitly accounted task/rehearsal objective. Use complete fixed-context query swaps as the primary development metric, with target loss as a secondary measure. A query-conditioned node readout or associative memory is a subsequent architecture hypothesis, not an already-established cure. Any tuning after viewing these outcomes must use a new confirmation test set.

All 39 regression tests passed, and lint, formatting, and source type checks passed. New tests cover offline labels/spans, split identity, exact delays, unroll bounds, swaps, grouping, and aggregation that distinguishes wrong query flips from correct answers. Inference still consists only of learned neural computation. Dataset label generation and evaluation keys never serve as an answer solver; no regex or response lookup was added. Node growth and learned topology remain unimplemented.

English story scores reuse a previously inspected retention diagnostic; they are not fresh independent generalization evidence.
