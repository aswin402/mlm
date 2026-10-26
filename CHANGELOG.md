# Changelog

## [Unreleased]

### Added

- Optional rank16 causal future-byte auxiliary head and fixed two-arm training pilot, preserving existing core model/trainer/checkpoint behavior.
- Narrow training-loss result after128QA+128English updates/arm; no held-out quality or promotion claim.
- Original84-test regression run,3focused post-fix checks,134finalaudits and documented audit-only continuation preserving completed control training.

- Bounded loaded-QA update diagnostic:10 optimizer computations,50 independent parameter evaluations,653 passing audits and82 passing regression tests; no universal step-size fix established.
- Sourced DeepSeek research note and staged MLM adaptation hypotheses; no architectural changes or model-quality claims.

- Frozen QA optimizer-history/router-regularizer controls:30 independent copied transitions,391 passing audits and81 passing regression tests.
- Objective and update-size evidence showing the tested unchanged-rate Adam reset worsens local QA losses; no training fix or release change.

- Restored actual-next QA→English pair probes with matched-input English-only and zero-gradient controls:25 copied transitions,191 passing audits and80 passing regression tests.
- Pair diagnostic report and immutable evidence; mixed phase effects do not establish a replay or optimizer fix.

- Frozen training-only replay/optimizer diagnostics with document-prefix context, loaded-Adam and zero-gradient controls, role accounting and checkpoint immutability audits.
- Predefined 80-probe report: local QA-loss effects are snapshot-dependent; a consistently worse replay44 effect is not established.
- 78-test regression suite and504 completed diagnostic audit checks.

## [0.0.1] - 2026-10-25

### Added

- Initial experimental graph-GRU language model with hierarchical routing and optional learned delta-rule memory.
- Byte-level English training, full-document streams, assistant-target masking, checkpoint/resume and generation/evaluation commands.
- Independent QA and English replay sampler controls with legacy-compatible defaults and explicit seed provenance.
- Controlled acquisition, retention, adaptation-seed and schedule-component experiments, with 75 regression tests passing before this release cleanup.
- Organized documentation, compact experiment evidence and original-document provenance archives under `docs/`.

### Changed

- Set package and lockfile version to 0.0.1.
- Replaced the long root README with a concise setup, architecture, evidence and documentation overview.
- Moved important Markdown documentation to `docs/`; removed obsolete working-plan Markdown files after preserving their original text and hashes.
- Excluded generated datasets, checkpoints, caches, build outputs and full local artifacts from publication.

### Research limitations

- Tiny-task memorization depends on replay sampling; reliable held-out binding and chat are not established.
- Dynamic node growth, rewiring, sparse execution, broad device support and Transformer superiority are not demonstrated.
