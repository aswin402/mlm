# Context-contrastive supervision: acquisition with an English tradeoff

Implemented first, then tested, trained, inspected and improved, following the requested order. This is a training-objective adaptation inspired by context-sensitive preference learning; it adds no model parameters or inference rules. It is not an exact DeepSeek algorithm reproduction or a new Transformer replacement.

The loss compares the same answer's mean neural byte log probability under supporting and swapped contexts. The original supervised language/router objective remains as an anchor. Both gradients update the existing graph/delta-memory model through its original optimizer. English replay uses the unchanged trainer. No answer lookup, regex or semantic routing rules were added to inference.

## Matched training results

All three arms started from the same original step-zero checkpoint, including optimizer state. Each received 128 QA and 128 English updates with QA seed 43 and English seed 44: 32 visits per group. Candidate scoring makes active arms more expensive; compute was not matched. Initial English probe CE was 1.157155.

| Contrast weight | Correct training answers | Complete binding groups | QA CE | English training-probe CE | Context loss |
|---|---:|---:|---:|---:|---:|
| 0 control | 8/16 | 0/4 | 0.127620 | 1.191786 | 0.693146 |
| 1 first implementation | 8/16 | 0/4 | 0.134767 | 1.187771 | 0.693149 |
| 100 bounded improvement | 16/16 | 4/4 | 0.022787 | 1.233599 | 0.001523 |

The initial context/main gradient-norm ratio was 0.000309562. The protocol allowed one weight-100 rescaling only if weight 1 failed acquisition and this ratio was below 0.01. That condition held. The stronger arm restarted from the original source; it did not continue the weight-1 checkpoint. This is engineering iteration on one schedule, not independent replication.

Weight 100 passed the training-answer gate, including context/query swaps in this fixture. Its English probe CE increased by 0.041813 versus control, about 3.51%. The combined acquisition/English gate failed for both active arms. All English endpoints also exceed the initial probe loss. The result supports acquisition on this tiny repeated training fixture, not retention, chat quality, held-out comprehension or architectural superiority. Prior longer-budget results remain unchanged.

## Verification and review limits

Three new tests passed in 13.75 seconds before training; nine related regressions passed in 44.24 seconds. Candidate likelihoods agree with the unchanged canonical evaluator, labels do not affect neural scores, gradients are finite/nonzero, zero weight reproduces the legacy optimizer state exactly, and answer gates reject loss-only improvement. Ruff and ty passed. The three focused tests passed again after the audit fixes in 12.31 seconds. The independent final artifact check passed 196 checks. The prior 89-test full suite is historical evidence, not a new full-suite run.

An independent audit confirmed 384,784 unchanged model parameters, 96 main QA targets, candidate masks totaling 32 answer-byte positions per selected group, and exact equality between the new control and the previous same-schedule control checkpoint. All three saved final checkpoints were independently rescored; state hashes, 256 optimizer steps and 32 visits per group were checked.

Code review found two audit gaps in the executed runner: candidate exposure totals were declared rather than accumulated from runtime metrics, and the run did not check its own protocol digest/settings again between arms. These were fixed in the current runner after the completed experiment, without retraining. The executed source is preserved as `docs/evidence/context-contrastive/executed-runner.py`; the original frozen manifest resolves that one path through this archive. Declared 4,096 candidate positions and 512 context terms per active arm were independently reconstructed, but missing runtime counter checks cannot be supplied retroactively. The original protocol, results and checkpoints remain unchanged.

The requested `graphify update .` failed because the installed command cannot import `graphify`. This does not affect numerical model execution.

## Next experiment

Keep this result as an unpromoted acquisition candidate. Next freeze an independent-schedule replication of weight 100 before changing any other idea. If acquisition repeats, evaluate the English tradeoff with a separately bounded rehearsal or gradient-balancing experiment. Keep fresh evaluation data closed until the recipe is fixed. Do not combine another architecture change with this objective yet.

[Protocol](protocol.md) · [Compact evidence](../../evidence/context-contrastive/verification.json). Full checkpoints remain local under `results/context-contrastive/`; version 0.0.1 and the release are unchanged.
