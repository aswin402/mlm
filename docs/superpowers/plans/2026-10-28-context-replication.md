# Context Replication Implementation Plan

> Execute inline; user-authorized order is implementation before tests.

**Goal:** Check whether fixed weight-100 acquisition repeats on a new sampling schedule.
**Architecture:** Separate driver reuses the existing objective, optimizer and scorer; preserve prior executed files and evidence. New accounting compares independent batch-derived totals with runtime metrics.
**Tech Stack:** Existing Python, JAX/Flax/Optax, uv, Ruff, ty and pytest.
**Spec:** docs/experiments/context-replication/protocol.md

- [x] Write src/mlm/context_replication_run.py with fixed seeds44/45 and weights0/100, prior-evidence validation, immutable protocol and runtime exposure checks.
- [x] Add tests/test_context_replication.py covering audit rejection of tampering/count mismatches and gate behavior; run focused tests/static checks after implementation. Obtain independent review.
- [x] Freeze all inputs and expected counts; train each arm once for128QA+128English; verify checkpoints, ranks, counts and prior hashes.
- [x] Publish report and evidence, update research/PRD/changelog, record graphify status and save a local commit. No push or promotion.

Review focus: preserve the original step-zero optimizer; no pilot in new comparison denominator; verify the protocol itself between arms; distinguish expected from observed counts; do not equate fixture acquisition with generalization or English retention.
