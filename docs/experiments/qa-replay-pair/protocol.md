# Copied actual-next QA and replay pair

## Predefined scope

Use the same common initializer and four fixed-init42 endpoints as the frozen replay diagnostic, with their full saved optimizer states. Every model file must match the prior diagnostic's recorded file and serialized-state hashes. No persistent training, architecture/optimizer change, new test or validation data, favorable-seed choice, checkpoint selection, release or push. Work is isolated on research/qa-replay-pair.

This follow-up is motivated by the earlier extra-English-step phase limitation, and is exploratory. It tests only the actual next loop pair at five selected snapshots, not a training trajectory, population effect or generalization.

## Select actual next data before measuring

Use all16 original training questions/four groups for global supervised-token-weighted assistant CE including newline/EOS. Reconstruct the next QA minibatch from the saved QA seed and complete group history, checking every prior group ID against the original permutation cycle, including epoch-boundary reshuffles. Preserve config.seed as the subset seed; use explicit saved sampler policy when available and documented legacy defaults otherwise.

Restore DocumentStream from saved English state; independently rebuild from seed/count to validate its full snapshot and next batch. Use the exact saved English recurrent carry, cast back to float32 and shape/finite checked. Do not reconstruct a frozen prefix or replace it with QA carry. Saved QA/English update counts must agree, trace length must match, and model step must equal their sum. Capture immutable metadata and sampler-next fingerprints before any update computation.

## Independent branches per checkpoint

1. From the loaded source, take one actual next QA update using the unchanged routed training step, fresh QA carry and loaded Adam moments, clipping, decay and per-role scaling.
2. From that QA candidate, take one actual next English update using the restored batch and unchanged saved English carry. Independently take a zero-gradient optimizer step from the same QA candidate as a momentum/decay control.
3. From the original source, take an English-only step on the exact same English input/carry and its own zero-gradient control. This matched-input comparator separates preceding-QA state effects from a change of replay batch/context.

Never feed any branch into another checkpoint or persist model weights. Five snapshots produce five QA steps, ten English steps and ten zero-gradient steps:25 copied state transitions total. Baseline source steps0/4000 become1/4001 after QA and2/4002 after the pair; English-only branches end1/4001.

## Measurements and interpretation

Record aggregate QA CE before QA, after QA, after QA→English, after QA→zero, after matched English-only and after source→zero. Primary descriptive changes: QA effect, English effect after QA, net pair effect, English-minus-zero after QA, matched English-only effect, and the difference between English effects with/without preceding QA. Positive QA CE change means worse; classify absolute changes≤1e-6 as unresolved and report exact values. Do not pool the much larger initializer loss with fitted endpoints or attach statistical population claims to five selected snapshots.

Also record selected QA-group CE before/after its QA step, English objective/language CE before/after, and reuse the tested gradient/actual-update role metrics for both English branches. QA→English is the real loop ordering, but only one next pair is probed; no post-step answer ranking, raw generation or generalization is measured. A group-specific QA step can change full-task CE differently from its own minibatch CE.

## Implementation and gates

New pair_probe helper/runner and focused tests; existing model, trainer, optimizer, interference helpers, scorer and Config code stay unchanged. Tests first: real changing-weight saved carry and sampler restoration at startup, epoch boundary and partial cycle; exact copied-branch match against sequential routed trainer and zero controls; no QA-carry substitution; metadata/state immutability; legacy/default seed restoration and corruption rejection. Run full regression, static checks and independent protocol/code/report review before claims.

Freeze original source/data/code/test/protocol hashes, state hashes, numerical environment, next batch/carry/snapshot fingerprints before computation. Independently validate all five branches/counters, metrics, zero controls, next data, finite values and before/after file/state integrity. Store raw outputs under ignored results/qa-replay-pair and compact evidence/report under docs. Update research/requirements/overview and Unreleased changelog; keep v0.0.1/main untouched.
