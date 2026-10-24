# Equal-exposure acquisition control

Both modes independently trained the same 16-group subset from their own frozen English checkpoint for 2,000 updates. This gives exactly 125 complete-group presentations per group, matching the successful four-group/500-update trials. The old 16-group trials received only 31–32 presentations per group in 500 updates. Learning rate 0.003, seed 42, batch 4/unroll 128, architecture, routing auxiliary and task-only schedule remain unchanged. There is no English rehearsal, curriculum or split-rate update in this study. No test data was opened and no checkpoint was selected by development score: all results describe fixed endpoints.

## Results

| Model | Groups | Updates | Presentations/group | Training answers correct | Complete training groups | Familiar-pool dev cases | New-pool dev cases | English dev loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 4 | 500 | 125 | 16/16 | 4/4 | 0/96 | 0/48 | 5.586892 |
| dense | 16 | 500 | 31–32 | 32/64 | 0/16 | 0/96 | 0/48 | 5.210094 |
| dense | 16 | 2000 | 125 | 33/64 | 0/16 | 0/96 | 0/48 | 9.424430 |
| hierarchical | 4 | 500 | 125 | 16/16 | 4/4 | 0/96 | 0/48 | 6.798242 |
| hierarchical | 16 | 500 | 31–32 | 36/64 | 1/16 | 0/96 | 0/48 | 4.981437 |
| hierarchical | 16 | 2000 | 125 | 61/64 | 14/16 | 0/96 | 0/48 | 9.336111 |

Answer accuracy ranks two candidate colors by neural likelihood. A training group requires all four fact-assignment/query conditions; a development case requires all eight including both fact orders. These definitions differ deliberately and are not interchangeable. Source English development losses are 1.155704 dense and 1.352696 hierarchical. All task-only checkpoints remain diagnostics; no English/chat model is promoted.

## What the extra exposure changes

dense: training answer accuracy 50.00% → 51.56%; complete training groups 0/16 → 0/16. English development loss 5.210094 → 9.424430.

hierarchical: training answer accuracy 56.25% → 95.31%; complete training groups 1/16 → 14/16. English development loss 4.981437 → 9.336111.


This is an exposure control, not a compute-matched comparison: the new 16-group endpoint costs four times as many updates as either archived 500-update endpoint. Larger subsets also contain more names, pairs and color combinations. Matching repetitions does not remove task-diversity or optimization effects. Architectures use their own independently pretrained sources, so this compares complete pipelines rather than changing routing on one identical learned tensor set. One seed, narrow grammar, four color words and reused exploratory development data limit inference. Only initial and final states are scored; intermediate success or late instability cannot be ruled out. Better training fit demonstrates acquisition for these examples, not a general binding algorithm; failure at one finite endpoint cannot prove a capacity limit.

## Name coverage

The 16-group subset contains 26 of the full 32 task-training names. The old familiar-name development category is defined relative to that full pool; its names are not all present in this smaller trial. Below, each case is stratified by how many of its two entity names actually appeared in these training groups. Pair identities are disjoint from development. Absence from task training does not imply absence from English pretraining.

| Model | Zero seen names | One seen name | Two seen names |
|---|---:|---:|---:|
| dense | 0/60 | 0/12 | 0/72 |
| hierarchical | 0/60 | 0/12 | 0/72 |

Counts combine the two development categories and require all eight conditions. They avoid labeling names familiar merely because they belong to the full source task pool.

## Integrity and reproducibility

The paired update traces, source fingerprints, subset IDs and byte counters are checked in summary.json. Each group appears exactly 125 times. The first 500 sampled group IDs exactly match the archived 16-group trial; the initial parameter fingerprint and fresh optimizer also match. The original source/config/data/code/baseline hashes remain unchanged. Original weights are preserved. Fixed initial/final scores and raw generations are archived; correct answers/fact metadata grade predictions offline and are not an inference solver. Scheduler rules and experimental checks do not replace the neural answer computation.

```sh
uv run python -m mlm.acquisition_run --config configs/acquisition-dense.json --init runs/scratch-english-dense-v1/best.mlm --out runs/exposure-dense-new --sizes 16 --steps 2000
uv run python -m mlm.acquisition_run --config configs/acquisition-hierarchical.json --init runs/scratch-english-hierarchical-v1/best.mlm --out runs/exposure-hierarchical-new --sizes 16 --steps 2000
```

The unchanged fresh-only runner and existing regression tests are reused; no model or training implementation changed. Final checkpoints remain in runs/exposure-{mode}-v1/groups-16/final.mlm. Source/tests/config/lockfile snapshots and their SHA256 hashes accompany this report. protocol.json records the pre-run frozen inputs. execution-status.json records completed checks separately from the frozen plan.

## Next decision

The next isolated experiment is a shared-versus-node learning-rate control with explicit uniform-rate and English-rehearsal baselines. It should compare acquisition and retention under declared budgets, with router rates specified separately. Do not stack growth, deeper processing and adaptive curricula into the same change. No claim of Transformer replacement or general chat competence follows from these trials. Fresh confirmation data is required after choosing training changes from these exploratory results.
