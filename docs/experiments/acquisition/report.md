# Binding acquisition at increasing dataset sizes

This exploratory diagnostic keeps the recurrent graph and routing unchanged. Each mode independently starts four trials from its own frozen from-scratch English best checkpoint, using fresh optimizer state, learning rate 0.003, seed 42 and exactly 500 task updates. Nested 1/4/16/64-group subsets are prefixes of one seeded permutation of the 256-group training pool. A group contains all four assignment/query swaps for one fact order. Groups are sampled in shuffled cycles; endpoints are fixed and no best checkpoint is selected. No held-out test data is opened.

This is task-only training, with no English rehearsal, to isolate acquisition. Learning rate and rehearsal differ from the previous joint-training study, so cross-study changes cannot be attributed to subset size alone. Within this study, paired modes have identical subset identities, update traces, input bytes and supervision bytes. Each mode's starting parameter hash is identical across its four trials, optimizer step starts at zero, and source checkpoint hashes remain unchanged.

## Results

Train questions are ranked by neural likelihood over two candidate colors. Complete training groups require all four answers correct; development cases require all eight conditions including both fact orders. English loss is measured on the same 32 development stories before and after each trial. This is a diagnostic development set, not a final confirmation.

| Model | Training groups | Unique names | Train answers correct | Complete train groups | Familiar-pool dev cases | New-pool dev cases | English dev loss before → after |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense | 1 | 2 | 4/4 | 1/1 | 0/96 | 0/48 | 1.156 → 6.499 |
| dense | 4 | 8 | 16/16 | 4/4 | 0/96 | 0/48 | 1.156 → 5.587 |
| dense | 16 | 26 | 32/64 | 0/16 | 0/96 | 0/48 | 1.156 → 5.210 |
| dense | 64 | 32 | 128/256 | 0/64 | 0/96 | 0/48 | 1.156 → 4.414 |
| hierarchical | 1 | 2 | 4/4 | 1/1 | 0/96 | 0/48 | 1.353 → 5.319 |
| hierarchical | 4 | 8 | 16/16 | 4/4 | 0/96 | 0/48 | 1.353 → 6.798 |
| hierarchical | 16 | 26 | 36/64 | 1/16 | 0/96 | 0/48 | 1.353 → 4.981 |
| hierarchical | 64 | 32 | 128/256 | 0/64 | 0/96 | 0/48 | 1.353 → 5.591 |

## Name coverage and interpretation

The original familiar-name bucket refers to the full 32-name training pool. Small subsets contain fewer names, so many familiar-pool development questions actually contain unseen task labels for that trial. No development pair is a training pair. The following breakdown counts how many of each case's two names occur in the actual trial subset; all conditions for each case stay together. Zero means both names absent, one means mixed coverage, and two means both seen in some training groups. Labels may have occurred in English pretraining; task absence does not guarantee global novelty.

| Model | Training groups | Zero seen names, complete dev cases | One seen name | Two seen names |
|---|---:|---:|---:|---:|
| dense | 1 | 0/132 | 0/12 | 0/0 |
| dense | 4 | 0/102 | 0/36 | 0/6 |
| dense | 16 | 0/60 | 0/12 | 0/72 |
| dense | 64 | 0/48 | 0/0 | 0/96 |
| hierarchical | 1 | 0/132 | 0/12 | 0/0 |
| hierarchical | 4 | 0/102 | 0/36 | 0/6 |
| hierarchical | 16 | 0/60 | 0/12 | 0/72 |
| hierarchical | 64 | 0/48 | 0/0 | 0/96 |

Four independent starting points prevent earlier small-set optimization from receiving extra credit at larger sizes. Fixed 500-update budgets imply approximately 500, 125, 31 or 8 presentations per group respectively. Declining training fit can reflect optimization/sample exposure as well as interference or capacity; this study cannot distinguish these causes or prove an architectural limit. Shared grammar, four color words, one seed and the same exploratory development data limit generalization claims. Eight-condition success is stricter than a chance-level single-answer score; both metrics are retained in the raw artifacts.

Each summary includes initial/final training and development metrics, fixed first-training-group raw generations without answer repair, unique name/pair/color coverage, byte counters and all 500 sampled group IDs. Correct labels and fact metadata are offline graders; the network receives textual prompts and candidate answer strings. No category solver, regex answer extraction, growth or extra expert machinery was added.

## Next decision

dense: completely fits training groups at tested sizes [1, 4]. hierarchical: completely fits training groups at tested sizes [1, 4].

Training fit and development transfer are separate gates. Before increasing model complexity, the next comparison should control presentations per example and test whether learning interference or insufficient repetitions explains larger-set failures. Any curriculum should be compared with a flat control at matched total updates. English retention must be restored explicitly before promoting a checkpoint; these task-only endpoints do not replace the preserved English models. Revised training choices require fresh confirmation data. This study provides no Transformer superiority, dynamic-growth or chatbot-readiness evidence.

## Reproduction and evidence

```sh
uv run python -m mlm.acquisition_run --config configs/acquisition-dense.json --init runs/scratch-english-dense-v1/best.mlm --out runs/acquisition-dense-new
uv run python -m mlm.acquisition_run --config configs/acquisition-hierarchical.json --init runs/scratch-english-hierarchical-v1/best.mlm --out runs/acquisition-hierarchical-new
```

Use fresh output paths. The runner is fresh-only, with no resume support. `protocol.json` freezes code/config/data/source hashes; `summary.json` includes paired audit results and endpoint checkpoint hashes. Architecture-specific sources differ because each was trained in its own mode; this is a paired pipeline comparison, not a single-tensor gate ablation. Raw scores are archived per mode/subset, and source/tests/configs/lockfile snapshots have separate SHA256 hashes. Local weights remain under runs/acquisition-{mode}-v1/groups-{N}/final.mlm.

Verification: all 58 regression tests passed in 176.44 seconds. Ruff lint/format and source type checks passed. The two new safeguard tests first failed because the diagnostic module was absent, then passed with the implementation.
