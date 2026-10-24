# QA ordering versus English replay sampling

In this selected, fixed-initialization comparison, acquisition follows the English replay stream. Both QA orderings learn16/16 training rankings and4/4 groups with English seed43. With English seed44, they learn7/16 and8/16, with0/4 complete groups. The new crossed runs narrow the earlier combined-schedule finding to English replay sampling for these tested levels.

This does not identify a general mechanism or establish reliable generalization. English seed changes document exposure/order and byte budget together. Two cells are historical, levels were outcome-selected, and the sampler-control driver has a reviewed, compatibility-tested source change. All cells still complete zero tiny development groups; the earlier independent adaptation-seed replication gate remains failed.

## Results

Memory initialization is fixed to the saved step-zero initializer42 throughout. All non-memory source tensors and every memory tensor match across initial states.

| QA seed | English seed | Provenance | Training rankings | Complete training groups | Fixed raw training samples | English dev loss |
|---|---|---|---:|---:|---:|---:|---:|
| 42 | 43 | reused historical | 16/16 | 4/4 | 4/4 | 1.286896 |
| 42 | 44 | new crossed run | 7/16 | 0/4 | 2/4 | 1.265234 |
| 43 | 43 | new crossed run | 16/16 | 4/4 | 4/4 | 1.287316 |
| 43 | 44 | reused historical | 8/16 | 0/4 | 2/4 | 1.270391 |

Every cell scores4/8 questions,0/2 complete groups and0/1 complete eight-condition case on the reused tiny development set. All pass the predefined retention threshold1.6908700424 (1.25×source loss1.3526960339). The English44 cells have lower English loss while failing binding acquisition; lower language loss alone does not establish the intended binding behavior.

Raw samples are the fixed first four training prompts, greedy max12 bytes, graded only by strip/casefold. E43 yields4/4 raw answers and E44 yields2/4. Candidate rankings cover all16 questions; raw generation does not. No parser, extraction, label-driven routing or answer repair was added.

Each new trial receives2,000QA+2,000English updates,4000 optimizer steps,500 presentations per group. New trial work totals8000 updates. The separate old/new-default compatibility preflight uses32 optimizer updates total. No budget extension, replacement cell, best-checkpoint selection, new test data, trained ablation or model promotion occurred.

## What changed in the runner

`mlm.plasticity_run` now accepts `--qa-seed` and `--english-seed`. QA seed changes only complete-group ordering; English seed changes only `DocumentStream` sampling. Config.seed still controls initialization and subset selection. Omitted overrides preserve QA=config.seed and English=config.seed+1. Explicit zero is supported; invalid seeds fail before output creation. Old programmatic Namespaces without the new fields continue to work. The resolved subset/QA/English policy is saved in protocol, summary and checkpoint metadata.

Both new runs use config.seed42 and the same complete four-group pool, with QA42/English44 and QA43/English43. The historical QA43/English44 cell used config.seed43, but its memory initialization is also42 and all four groups were selected. Full tensor identity, membership, natural batch order/actual QA trace and stream audits establish the intended separation. No Config schema, topology, model, gradient, optimizer, loss, scorer or data-generation code changed.

The only source/test changes are the reviewed runner and its test. Their old versions are preserved under `legacy/`. Dynamic invalid-config test construction also uses Pydantic's model_validate API to avoid a type-check ambiguity; validation semantics are preserved. A reconstructed legacy source digest, substituting the archived driver into otherwise unchanged model sources, exactly matches both historical digests.

## Descriptive contrasts and limits

For complete-group fraction, English44 minus43 is−1.0 at each QA seed:−100 percentage points. The averaged QA43-minus42 contrast and interaction are0. For question accuracy, English44-minus43 averages−0.53125 (−53.125pp), QA43-minus42 averages+0.03125 (+3.125pp), and interaction is+0.0625 (+6.25pp). These are descriptive differences over the selected table, not statistical population effects.

Replay43 consumes958,088 English bytes and replay44 consumes958,288 bytes per run; each cell consumes776,000 QA input bytes with40,000 supervised QA targets. Replay changes document content/position/exposure as well as order and has a200-byte total difference. The study cannot separate these components, attribute failure to particular stories, or prove gradient interference. Objective interference and replay-content sensitivity remain hypotheses, not measured mechanisms.

There is one run per cell and one memory initializer. Two reference cells were already observed; no fresh four-cell replication or new graph-architecture comparison occurred. Defaults were checked on a bounded preflight, not by retraining the historical4000-step endpoints. Recorded Python/dependency/platform/backend fields match, and the known historical QA43/English44 numerical flags match current settings. QA42/English43 historical numerical/thread flags were not recorded, so full historical/current numerical equivalence cannot be proven. The implementation-version and historical reuse limitations remain explicit.

## Verification

The new QA-override test first failed on the actual wrong sampler trace before implementation. All9 targeted plasticity tests then passed in59.02s, followed by the complete suite:75 tests in234.90s. Coverage includes independently overridden QA and English streams, actual English input batches, zero seeds, legacy Namespace/default behavior, source preservation, fresh optimizer, continued English carry and exact default-versus-explicit legacy parameter/moment equality. Invalid seeds are tested.

The actual preserved old driver and updated default path also match bitwise for all parameters/Adam moments, metrics, QA trace, counters, English stream/carry and initial fingerprints after eight QA/rehearsal pairs. All13 compatibility checks pass. The environment/source preflight passes9 checks and identifies exactly the two intentional changed files. Ruff lint/format and ty checks passed. Independent review approved implementation, audit logic, frozen cell mappings and all115 input hashes before launch.

Both new processes completed normally. All219 input/endpoint audit checks pass: fixed initializer tensors, fresh optimizer, actual final steps/config/counters/finite parameters, exactly500/group, QA trace and byte counts against the same-QA reference, English snapshot/bytes against the same-English reference, actual seed policies in all metadata, frozen child numerical/environment settings, full train/dev question evidence, score-derived argmax/correctness/metrics and fixed raw prompts.

See [protocol](../../evidence/schedule-components/protocol.json), [summary](../../evidence/schedule-components/summary.json), [audit](../../evidence/schedule-components/audit.json), [compatibility evidence](../../evidence/schedule-components/compatibility.json), [source/environment audit](../../evidence/schedule-components/environment-preflight.json) and logs. Reproduce with the protocol launcher in fresh output directories; existing checkpoints must be preserved. New checkpoints reside in `runs/schedule-components-q42-e44-v1` and `runs/schedule-components-q43-e43-v1`.

## Next step

Measure whether English replay updates oppose QA learning on predefined training batches and frozen checkpoints. Report objective-gradient alignment and the actual QA-loss change from a copied English optimizer step; a gradient cosine alone does not capture Adam's moments, clipping and per-role scaling. This would test a possible mechanism without changing model weights on disk or opening test data. Then select one controlled training intervention if supported, with fixed English exposure/budgets and acquisition/retention gates. Do not hardcode a favorable replay seed as a solution or claim general understanding from this tiny fit.
