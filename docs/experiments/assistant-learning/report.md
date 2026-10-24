# Assistant-only learning experiment

Date: 2026-10-23. Outcome: the training objective and answer format improved; the reading-comprehension objective was not achieved.

## Implemented behavior

Input validity and target supervision are separate. Prompt bytes update recurrent memory; their next-byte targets do not contribute to chat/QA loss. Assistant spans are computed offline from structured roles using UTF-8 byte offsets. Newline and EOS targets are explicit. Story rehearsal retains the ordinary full-text language-model loss.

A chunk with zero supervised targets advances recurrent state while preserving parameters, optimizer moments, and optimizer step. Checkpoints preserve data and supervision hashes, masks, per-lane offsets, shuffle RNG, optimizer, and recurrent carry. Changing masks on unchanged text is rejected on resume.

Validation selection now uses global optimizer-step multiples. Off-grid endpoint evaluations are logged without competing for best. Relocating resume preserves the source best checkpoint. Incompatible legacy off-grid best states are rejected. Prefix training/evaluation rejects explicit supervision; use stream-train and full-document evaluation.

## Data and fixed protocol

The generated reading task covers color ownership and object locations with a distractor entity and varying fact order. There are 800 unique balanced training pairs / 1,600 reading examples, and 64 pairs / 128 examples each for development and test. Names and wording templates are disjoint within this supervised stage. This does not mean the names were absent from earlier English pretraining.

Short QA examples are repeated eight times explicitly. The training mix contains 12,800 QA records, 64 cached public conversations with assistant-only spans, and 64 English stories for full-text rehearsal. The manifest records counts, source hashes, and frozen suites. The preliminary assistant-v1 preparation contained duplicate pairs; it was corrected before any training. Only assistant-v2 was trained.

Offline generation rules create supervised labels. Those rules are never called by model inference. Frozen candidate scoring receives prompts and all choices, not correct-answer indices. Greedy generation receives no choices or answer key. The original reading-v1 suite was not used for training or selection and its hash remained unchanged.

Both graph and parameter-matched GRU start from their respective 4,000-update English checkpoints with fresh optimizers. Both receive 2,000 additional supervised updates, seed 42, batch size four, unroll 256, learning rate 0.0005, the same optimizer settings, and selection every 500 updates. Final sampler snapshots, corpus hashes, consumed bytes, and supervision hashes/counts match. This is one seed, not a statistical architecture ranking.

## Results

Answer loss scores assistant-target bytes and EOS. It is not comparable to the earlier full-text NLL as an improvement percentage.

| Model | Selected update | Held-out answer loss, nats | Two-choice accuracy | Both-in-pair accuracy | Greedy exact accuracy, first 16 |
|---|---:|---:|---:|---:|---:|
| graph | 1000 | 0.234342 | 50% | 0% | 12.5% |
| gru | 1500 | 0.244349 | 50% | 0% | 37.5% |

Before this stage, both English checkpoints scored 50% two-choice accuracy, 0% pair accuracy, and 0/16 exact greedy answers on the new task. After training, both still ranked the same candidate within every pair. Both also remained at 50% / 0% on the unchanged original reading suite. The familiar training-style subset also scored 50% / 0%; this is a training-data diagnostic, not a generalization result.

Raw generation became short task words. On the fixed 16-question subset, graph outputs were eight `red` and eight `bag` replies; the GRU produced category-specific `green`, `white`, and `box` defaults. The apparent 2/16 and 6/16 exact matches do not establish context use. No regex answer extraction, favorable stochastic sampling, or answer-key response rule was used. Surrounding whitespace and case are normalized only for grading.

See the complete before/after evaluation JSON files and raw greedy samples in this directory. Candidate ranking uses scorer version 2, scores answer bytes only, and uses identical chat formatting across models and stages.

## English retention

| Model | Before story test loss | After story test loss |
|---|---:|---:|
| graph | 1.121557 | 1.267662 |
| gru | 1.098890 | 1.169831 |

Story prediction worsened despite bounded rehearsal. The new checkpoints should be treated as reading experiments, not replacements for the earlier general-English checkpoints. No timing advantage is claimed because verification/evaluation ran concurrently with parts of training.

## Memory diagnostic

A controlled, equal-length substitution changed an earlier hat color between green and white while keeping the question fixed. Neutral filler increased the distance to the answer. For graph MLM, relative final-state difference fell from about 0.00262 with no filler to 0.0000458 with 34 filler bytes and about 0.00000020 with 136 bytes. Answer preference changed by only about 0.0000049 nats/byte initially and was near numerical noise with more filler. The GRU also lost sensitivity rapidly.

This is a diagnostic observation from one construction. It suggests a retention/readout problem worth isolating; it does not establish the sole cause of reading failure. Optimization, task design, finite training budget, and architectural capacity remain possible contributors. The raw measurements and in-domain subset scores are saved in memory-diagnostic.json.

## Reproduce

```sh
uv run python -m mlm.supervised_data --source data/hf --out data/assistant-v2
uv run mlm stream-train --train data/assistant-v2/train.jsonl --validation data/assistant-v2/validation.jsonl --test data/assistant-v2/test.jsonl --config configs/assistant-graph.json --init runs/english-v1/best.mlm --out runs/assistant-graph-new --steps 2000 --eval-every 500
uv run mlm stream-train --train data/assistant-v2/train.jsonl --validation data/assistant-v2/validation.jsonl --test data/assistant-v2/test.jsonl --config configs/assistant-gru.json --init runs/baseline-gru/best.mlm --out runs/assistant-gru-new --steps 2000 --eval-every 500
uv run python -m mlm.supervised_eval runs/assistant-graph-new/best.mlm --out results/assistant-graph-new-evaluation.json
```

Use fresh output directories. Prepared data is reused rather than silently overwritten. Public background data attribution and license links are in [data/README](../../data/README.md). Raw configs, environment/source fingerprints, metrics, masks/protocol identities, and checkpoints are preserved.

## Decision

Do not infer understanding from the lower answer loss. The immediate next research target is controlled delayed recall and a memory-timescale experiment, with fact sensitivity and English retention as explicit criteria. Compare that change against the unchanged graph and GRU before adding routing, rewiring, or growth. Assistant-only infrastructure is ready; reliable reading and general chat remain unachieved.
