# Shared/node learning rates and English rehearsal

Six fresh trials test three learning-rate policies with and without English rehearsal. Every trial starts from the same hierarchical model's frozen English checkpoint, with fresh optimizer state, the same 16 complete training groups, seed 42, batch 4/unroll 128 and 2,000 QA updates. Each group receives exactly 125 presentations. Rehearsal adds 2,000 full-document English updates, with separate streaming carry and accounting. Endpoints are fixed; no test files are opened and no checkpoint is selected by scores.

The architecture is unchanged: fixed eight-node graph, width 64, hierarchical soft updates and sum readout. This is an optimizer/data-schedule experiment, not growth, sparse dispatch or an adaptive learning-rate controller.

## Rate policies

| Policy | Shared representation rate | Node-specific rate | Router rate |
|---|---:|---:|---:|
| high | 0.003 | 0.003 | 0.003 |
| low | 0.0003 | 0.0003 | 0.0003 |
| split | 0.0003 | 0.003 | 0.003 |

Node parameters are input_kernel, hidden_kernel, gru_bias and read_kernel. Shared representations comprise embedding, message_kernel, normalization and vocabulary head. Group/child router kernels and biases have their own explicit role; they stay fast in the split arm. This does not test a slower-router alternative. The role registry rejects unknown parameters when applying non-default scales. Completed AdamW updates are scaled after global clipping and moment estimation; decoupled decay follows the effective learning rate. Defaults remain identical to the old optimizer, including state structure. Scale settings are serialized in Config and restored by checkpoint loading.

## Measured outcomes

Training answers are two-candidate neural likelihood rankings. A complete training group requires four assignment/query conditions; development requires eight including both fact orders. Source English development loss is 1.352696. The predeclared diagnostic gate requires at least 14/16 complete training groups and English development loss ≤1.25× source. It tests acquisition plus retention only and neither selects nor promotes a model.

| Arm | Optimizer updates | Train answers correct | Complete train groups | Familiar-pool dev cases | New-pool dev cases | English dev loss | Source multiple | Joint diagnostic gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| high-task | 2000 | 61/64 | 14/16 | 0/96 | 0/48 | 9.336111 | 6.902× | fails |
| high-rehearsal | 4000 | 64/64 | 16/16 | 0/96 | 0/48 | 1.319806 | 0.976× | meets |
| low-task | 2000 | 56/64 | 12/16 | 0/96 | 0/48 | 3.537662 | 2.615× | fails |
| low-rehearsal | 4000 | 45/64 | 5/16 | 0/96 | 0/48 | 1.307681 | 0.967× | fails |
| split-task | 2000 | 64/64 | 16/16 | 0/96 | 0/48 | 6.268202 | 4.634× | fails |
| split-rehearsal | 4000 | 64/64 | 16/16 | 0/96 | 0/48 | 1.305334 | 0.965× | meets |

## Controlled contrasts

high: adding rehearsal changes complete training groups 14/16 → 16/16, and English development loss 9.336111 → 1.319806. Rehearsal adds 2,000 optimizer updates.

low: adding rehearsal changes complete training groups 12/16 → 5/16, and English development loss 3.537662 → 1.307681. Rehearsal adds 2,000 optimizer updates.

split: adding rehearsal changes complete training groups 16/16 → 16/16, and English development loss 6.268202 → 1.305334. Rehearsal adds 2,000 optimizer updates.

Within task, split versus low uniform changes complete training groups 12/16 → 16/16 and English loss 3.537662 → 6.268202. Shared representation rates match, but both node and router rates are 10× higher in split.

Within rehearsal, split versus low uniform changes complete training groups 5/16 → 16/16 and English loss 1.307681 → 1.305334. Shared representation rates match, but both node and router rates are 10× higher in split.


The high-versus-split comparison changes only the shared representation rate within each schedule. Low uniform controls for slowing the whole network. Rehearsal versus task-only is QA-exposure matched but not compute matched: it doubles optimizer updates. No causal claim that a particular tensor stores English follows from parameter drift or this one-seed study.

## Coverage, costs and evidence

The subset contains 26 of 32 task-training names. Familiar-name development is defined relative to the full 32-name pool. Name-coverage artifacts separately count cases with zero, one or two names present in this actual subset; no development pair is a training pair. Labels may have occurred in English pretraining. Both development categories remain narrow shared-grammar diagnostics, not independent final confirmation.

| Arm | QA payload bytes | QA supervised bytes | English payload/supervised bytes |
|---|---:|---:|---:|
| high-task | 776,000 | 40,000 | 0 |
| high-rehearsal | 776,000 | 40,000 | 958,088 |
| low-task | 776,000 | 40,000 | 0 |
| low-rehearsal | 776,000 | 40,000 | 958,088 |
| split-task | 776,000 | 40,000 | 0 |
| split-rehearsal | 776,000 | 40,000 | 958,088 |

All QA traces, initial parameters and QA byte counters match. Rehearsal arms also match English sampler snapshots and byte counters. The high/task endpoint exactly reproduces the earlier equal-exposure model's parameters and full optimizer state, auditing that defaults and the new runner preserved the original training behavior. Source/code/config/data/baseline hashes remain frozen. Full initial/final scored items, fixed first-group raw samples without answer repair, optimizer policies, checkpoints, stream state and parameter-change norms are saved. Grading labels never enter the neural scorer as expected answers.

The full regression suite and separate code review validate the implementation, including completed-update scaling, populated optimizer resets, carried English state, exact QA sampling, legacy checkpoints without new fields, and split-rate checkpoint continuation. Source/test/config/lock snapshots have hashes. Only initial/final scores are evaluated; intermediate acquisition or instability cannot be ruled out. All arms deliberately reset Adam moments; this is adaptation from pretrained weights, not uninterrupted continual training. A warm-optimizer control would be required before generalizing to continuous training. One seed, a small synthetic subset and reused development sets limit generalization.

## Next gate

Arms meeting the predeclared acquisition/retention diagnostic: high-rehearsal, split-rehearsal. No arm solves a complete development binding case. No endpoint is promoted. Any next curriculum or replay adjustment must retain matched controls and an explicit compute budget. If joint acquisition/retention improves, the remaining gate is transfer to unseen pairings, names and wording, evaluated on fresh confirmation after development choices. Otherwise diagnose optimization and interference before adding capacity. These outcomes do not establish Transformer replacement, dynamic growth, sparse speedups or arbitrary-device deployment.

## Reproduction

```sh
uv run python -m mlm.plasticity_run --config configs/plasticity-split.json --init runs/scratch-english-hierarchical-v1/best.mlm --out runs/plasticity-split-new --groups 16 --steps 2000 --rehearsal
```

Use high/low configs and omit --rehearsal for task-only arms; every output path must be fresh. The driver has no resume feature, but standard checkpoint loading restores its rate settings and Adam state. Stream resume rejects the experimental checkpoint mode. Commands for all six arms and frozen hashes are in protocol.json. Weights remain under runs/plasticity-{arm}-v1/final.mlm. Historical models and previous result snapshots are preserved.

Verification: all 63 regression tests passed in 161.72 seconds; the five strengthened split-rate tests passed separately. Ruff lint/format and source type checks passed after archiving. The code review has no unresolved findings. See verification.json.
