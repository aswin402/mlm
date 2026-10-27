# Documentation

Start with the [release overview](../README.md) and [changelog](../CHANGELOG.md).

## Project

- [Original idea, research and measured progress](research.md)
- [DeepSeek papers and proposed MLM adaptations](research/deepseek-inspirations.md)
- [Product requirements and capability gates](prd.md)
- [Detailed usage and historical experiment log](usage.md)
- [Data sources, preparation and split rules](data/README.md)
- [Smoke fixture](data/smoke.md)
- [Reading evaluation fixture](data/evaluation.md)

## Current research

[Frozen answer/position analysis](experiments/future-byte-answers/report.md) finds8/16training rankings and0/4complete groups for the initializer and all six128-pair endpoints. CE gains do not yield answer-count gains; context swaps never change predictions. All258audits and4focused tests pass, with zero optimizer updates. Prior89-test full suite remains the last full regression. Head unpromoted; fresh tests closed.

## Experiment reports

- [Future-byte answer and target-position analysis](experiments/future-byte-answers/report.md)

- [Future-byte sampling-seed replication](experiments/future-byte-replication/report.md)

- [Future-byte objective feasibility pilot](experiments/future-byte/report.md)

- [Bounded loaded-QA update fractions](experiments/qa-update-scale/report.md)

- [QA optimizer-history and router-regularizer controls](experiments/qa-step-controls/report.md)

- [Actual-next QA→English pair diagnostic](experiments/qa-replay-pair/report.md)

- [Binding acquisition at increasing dataset sizes](experiments/acquisition/report.md)
- [Learned-memory addressing diagnostic](experiments/addressing/report.md)
- [Four-group exposure control](experiments/addressing-exposure/report.md)
- [Memory initialization versus combined training schedule](experiments/addressing-factorial/report.md)
- [Fixed-source adaptation-seed replication](experiments/addressing-replication/report.md)
- [Assistant-only learning experiment](experiments/assistant-learning/report.md)
- [Learned delta-memory architecture pilot](experiments/associative/report.md)
- [Balanced confirmation with explicit English rehearsal](experiments/balanced-confirmation/report.md)
- [MLM baseline and comprehension comparison](experiments/comparison/report.md)
- [First real-English and conversation experiments](experiments/english-chat/report.md)
- [Equal-exposure acquisition control](experiments/exposure/report.md)
- [Focused two-person binding](experiments/focused-binding/report.md)
- [From-scratch dense versus hierarchical study](experiments/from-scratch/report.md)
- [Gradual binding versus a matched flat control](experiments/gradual-binding/report.md)
- [Hierarchical recurrent update routing](experiments/hierarchical-routing/report.md)
- [Retention-gate intervention](experiments/memory-retention/report.md)
- [Shared/node learning rates and English rehearsal](experiments/plasticity/report.md)
- [Controlled entity–fact retrieval curriculum](experiments/retrieval-curriculum/report.md)
- [QA ordering versus English replay sampling](experiments/schedule-components/report.md)
- [MLM v0.1 CPU smoke result](experiments/smoke/report.md)
- [Binding transfer through pairing, color and wording diversity](experiments/transfer/report.md)

- [Frozen replay/optimizer interference](experiments/replay-interference/report.md)

## Evidence and archives

Compact numerical summaries, protocols and frozen recipes are stored in `evidence/`. Full artifacts and model weights remain local-only. [Archive notes and migration records](archive/README.md) preserve the original text and hashes behind the removed one-off Markdown plans. Historical commands run from the repository root and may require the original archived document layout and locally generated artifacts.
