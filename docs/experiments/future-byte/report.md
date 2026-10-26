# Future-byte auxiliary objective: first feasibility pilot

Completed2026-10-27 on `research/future-byte-pilot`. The optional future-byte objective is implemented and trained. At the fixed128QA+128English budget, the auxiliary arm has lower main training loss than the matched control on the16-question QA fixture and one fixed English training batch. This is a narrow positive pilot, not a replicated or held-out model-quality result.

## Implementation

A separate learned rank16 tanh head maps current causal main log-probabilities to a prediction for the second upcoming byte. It adds8,514 training-only parameters to the384,784-parameter backbone. Neither future input tokens nor labels enter this predictor. Labels enter only the loss. The main backbone architecture, Config, legacy trainer, checkpoint loader and scorer files remain unchanged. Head removal leaves ordinary backbone inference unchanged in structure; the two trained backbones have different learned weights.

Auxiliary masks require both adjacent positions to be valid supervised byte targets(<256), with no reset at the next position. They exclude prompt-only spans, padding, EOS/BOS and batch/document crossings. Main next-byte masking remains unchanged. This is a short-horizon adaptation inspired by [DeepSeek-V3 MTP](https://arxiv.org/html/2412.19437v2), not its sequential prediction-module design or speculative decoding.

Weight0 uses the exact existing routed trainer for backbone updates and zero auxiliary-head gradients. Weight0.1 differentiates the existing language/router objective plus0.1 times mean auxiliary CE. The backbone retains its actual optimizer; the head has separate clipped AdamW at the shared learning rate. Clipping groups are separate, not a single global norm over both networks. Both arms have the same head initialization, but control head parameters are not supervised and only undergo decay. This matches architecture and update/data exposure, not training FLOPs.

## Predefined comparison

[Protocol](protocol.md): common step0 source, fixed head seed9042, QA42/English43 and128QA+128English updates per arm. Every QA group appears32times; no additional seeds, rates, early stopping or best-checkpoint selection. All512 backbone updates across both arms were actually performed, with no repeated control training. Each arm maintains its own changing-weight English recurrent carry; QA carry always resets and never replaces English carry.

Main supervised positions per arm:3,072QA and61,181English. Eligible auxiliary positions:2,048QA and60,604English. Both arms consume49,664QA and61,116English byte positions under the existing consumed-byte metric; QA supervised bytes2,560. Expected exposure is independently reconstructed with NumPy and checked against observed counters, rather than just comparing the arms to each other.

Both evaluations are training-only. QA covers all16questions/96main targets/64auxiliary targets. English uses the first predefined training batch,512main targets/508auxiliary targets, with fresh evaluation carry. That batch is in the training schedule; it is not an English retention or generalization benchmark.

## Main prediction results

Lower CE is better. Main CE excludes the router penalty and auxiliary loss.

| Training measure | Common initial | Weight0 control final | Weight0.1 final | Auxiliary minus control |
|---|---:|---:|---:|---:|
| QA language CE | 6.866767883 | 0.165443063 | 0.156375960 | -0.009067103 |
| English fixed training-probe CE | 1.157155037 | 1.121979952 | 1.121316195 | -0.000663757 |

QA CE is about5.480% lower relative to control; the English training-probe difference is about0.059%. These are conditional descriptive differences from one short selected schedule, not effect-size estimates across independent runs or answer-accuracy improvements. No success threshold for general capability was tested.

| Training measure | Control final | Auxiliary final |
|---|---:|---:|
| QA main objective | 0.181090370 | 0.172145292 |
| English main objective | 1.151257515 | 1.150587797 |
| QA auxiliary CE | 5.556950569 | 3.829963684 |
| English auxiliary CE | 5.544595718 | 4.092679977 |

Main objective includes the configured router penalty but excludes auxiliary loss in this evaluation. Auxiliary-head CE falls substantially only in the supervised arm. That demonstrates auxiliary-task learning but is not a fair head-quality comparison against an unsupervised head and cannot alone establish backbone improvement. The main-CE table provides the separate backbone comparison.

## Audit interruption and recovery

The original freeze captured103input hashes and the original execution settings. Control completed all256updates and saved its final artifacts. The run then stopped before starting the auxiliary arm because a head serialization audit compared raw bytes against bytes restored into a fresh template. Training changes dictionary insertion order; named tensor values, dtypes, shapes and counters were unchanged even though serialized byte order differed. Backbone checkpoint hash roundtrip already passed.

The audit was corrected to compare named state trees exactly, with tests that reproduce the order difference and reject mutated parameters/counters. Existing helper/training code, original tests, source, data, settings and experiment choices were not changed. Original protocol/execution, control artifacts and driver source were preserved; [original driver archive](../../evidence/future-byte/original-driver.json) retains the source text and hash.

A separate frozen [continuation protocol](../../evidence/future-byte/continuation-protocol.json) binds111inputs, explicitly accounts for old/new driver hashes and freezes all four completed-control artifacts. All102non-driver original input hashes remain unchanged; all98 prior bounded-update inputs also remain unchanged. Numerical settings are identical; the aggregate code SHA changes only because the audit driver changes. The continuation validates source/head values,256steps, histories, every counter,32visits/group, stream snapshot, checkpoint hashes and recomputed scores, then reuses the control without updating it. It rejects active-arm/partial outputs and changed experimental inputs. Only the unstarted auxiliary arm trains.

Final134auditflags pass. This was an interrupted execution followed by an audit-only continuation, not one uninterrupted successful run. No outcome-driven selection or budget extension occurred.

## Tests and runtime

Before original training, the full84-test suite passed in314.29s. Original focused tests passed2/2 in33.41s, covering exact legacy parameters/moments/carry for two updates, explicit masks and shifted target loss, causality, direct joint gradient/update comparison, active backbone gradient, empty auxiliary targets, finite/no-supervision rejection, source immutability and serialization.

After the audit fix, all3focused tests passed in37.84s, including the new mapping-order regression. Subsequent audit-only continuation guards passed static checks and the actual continuation; the full suite was not repeated after this driver-only correction. Ruff, new-file formatting and ty pass. Independent code/protocol review cleared original training and the recovery; final publication verification is in [verification](../../evidence/future-byte/verification.json).

Recorded training-loop times including compilation and synchronization are74.73s control and77.49s auxiliary. Evaluation/checkpoint time is excluded. Different processes, restart and compiler/cache behavior prevent interpreting their ratio as a speed benchmark. No inference latency, device portability, sparse execution or energy advantage was tested.

## Artifacts and use

Full new checkpoints remain local and ignored:

- `results/future-byte/control/final.mlm`
- `results/future-byte/future_byte/final.mlm`

Each is a standard backbone checkpoint loadable by existing code, with the auxiliary optimizer state separately stored as `head.msgpack`. Backbone alone is insufficient to resume joint training; joint resume is unsupported. No original source checkpoint was overwritten, and no checkpoint was promoted.

For a fresh repeat using the existing local source and cached corpus:

```sh
uv run python -m mlm.future_byte_run freeze --out results/future-byte-repeat
uv run python -m mlm.future_byte_run run --out results/future-byte-repeat
```

Use a new output directory. This fixed pilot driver assumes the prior local checkpoint/data/evidence chain; it is not the standalone public smoke-training command. `amend`/`continue` were recovery operations for the documented artifact audit, not joint-training resume.

## Decision and limits

Keep the objective optional and experimental. The observed main training-loss reduction is worth a separately frozen seed-replication study before promotion, followed by predefined held-out binding and English retention checks. Do not increase this run's budget or choose another coefficient based on these outcomes. The source and QA42/English43 schedule are historically selected; this pilot is not an independent seed replication.

No post-training answer ranking, raw generation, development or fresh test was evaluated. Reliable binding, chat, dynamic graph growth, categories, sparse execution and Transformer superiority are unproven. Earlier failed replication/generalization gates remain failed. Main, v0.0.1 tag and package version remain unchanged. Local research commit only; no push.
