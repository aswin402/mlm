# Mini Learning Model (MLM)

Version **0.0.1** — an experimental byte-level language model built from small GRU networks connected by a directed graph, with optional hierarchical routing and learned associative memory.

The prototype trains and can memorize tiny binding tasks. Learning remains sensitive to English replay sampling, and reliable generalization and chat are not established. Automatic node growth, graph rewiring, sparse execution and a Transformer-replacement claim remain future research.

## Quick start

Tested on Linux x86-64, Python 3.12 and CPU. GPU and other-device support have not been validated. Install [uv](https://docs.astral.sh/uv/), then run from the repository root:

```sh
uv sync --locked
uv run mlm train --train data/smoke/train.txt --validation data/smoke/validation.txt --config configs/smoke.json --out runs/my-first-mlm --steps 300 --eval-every 50
uv run mlm generate runs/my-first-mlm/best.mlm --prompt 'The red ' --max-bytes 64
uv run mlm evaluate runs/my-first-mlm/best.mlm --data data/smoke/validation.txt
```

Use a fresh output directory. The bundled smoke data is a small original fixture, not a language-quality benchmark. Public English datasets are downloaded separately using the [data guide](docs/data/README.md). Trained checkpoints and cached datasets are not included in Git.

## Architecture

- Fixed directed graph of independent GRU nodes with synchronous message passing.
- Optional learned hierarchical update routing; current implementation still computes all nodes.
- Optional delta-rule memory with learned keys, queries, values and write gates. It is document-local state, separate from weight training.
- JAX/Flax execution, Optax AdamW training, byte-level vocabulary and atomic resumable checkpoints.

There is no inference-time entity parser, regex answer generator or symbolic answer lookup. Synthetic task labels are used for training and grading.

## Current evidence

The latest selected experiment held memory initialization fixed and varied QA ordering and English replay sampling:

| English replay seed | QA order 42 | QA order 43 |
|---|---:|---:|
| 43 | 16/16 training answers | 16/16 |
| 44 | 7/16 | 8/16 |

Both replay 43 runs completed all four training groups; replay 44 runs completed none. All four completed zero groups on the small reused development set. Two cells were historical, and this is exploratory acquisition evidence. It does not reverse the failed independent-seed replication result. See the [full QA/replay report](docs/experiments/schedule-components/report.md).

The [frozen replay diagnostic](docs/experiments/replay-interference/report.md) found snapshot-dependent QA-loss interference and substantial optimizer-history effects. It did not consistently show replay 44 to be more harmful than 43, and no training intervention has been applied.

The [actual-next QA→English diagnostic](docs/experiments/qa-replay-pair/report.md) restores saved sampler/carry state for five copied pairs. QA and English have mixed local effects;191 audits pass, and checkpoint files remain unchanged. Next isolate optimizer-history and router-balance effects before a training intervention.

The [bounded QA-update diagnostic](docs/experiments/qa-update-scale/report.md) evaluated50 copied parameter candidates: smaller fractions reduce some local harm but do not establish a universal fix. All653 audits pass. [DeepSeek-inspired research](docs/research/deepseek-inspirations.md) records possible next experiments; no new architecture or training intervention has been applied.

## Development

```sh
uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
uv run ty check src
```

The last full regression run passed 82 tests. Separate QA/replay controls are available through `python -m mlm.plasticity_run --qa-seed ... --english-seed ...`; omitted values retain config.seed and config.seed+1. Config.seed still controls initialization and subset selection.

## Documentation

- [Documentation index](docs/README.md)
- [Original idea and research](docs/research.md)
- [Product requirements](docs/prd.md)
- [Detailed usage and experiment history](docs/usage.md)
- [Dataset preparation and provenance](docs/data/README.md)
- [Release changelog](CHANGELOG.md)

Source lives in `src/mlm/`, tests in `tests/`, configurations in `configs/`, and original smoke/reading fixtures in `data/`. Compact experiment evidence is under `docs/evidence/`; full local runs and artifacts are ignored. Historical plan text and documentation migration records are preserved in `docs/archive/`.
