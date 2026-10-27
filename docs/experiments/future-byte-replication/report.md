# Future-byte sampling-seed replication: QA gate failed

Completed2026-10-27 on `research/future-byte-replication`. The earlier QA training-loss benefit repeats in1/2 new comparisons, so the predefined primary and joint gates fail. The auxiliary arm has lower fixed English corpus-training-probe loss than control in2/2, passing the separate secondary gate. All new arms remain worse than their common initial English-probe loss. Keep the head optional and experimental; no model is promoted.

## Frozen question and comparison

[Protocol](protocol.md): compare weight0 versus0.1 at QA43/English44 andQA44/English45, with unchanged source initialization, loaded optimizer, head seed9042, rank16 architecture and128QA+128English updates/arm. All four independent arms complete256steps,32presentations/group:1,024new backbone updates. No replacements, extra budgets, early stopping or coefficient search.

This is sampling-schedule replication conditional on one fixed model/head initialization. These sampler levels have historical results in other experiments and are not randomly sampled population representatives; they are new comparisons for this objective. OriginalQA42/E43 pilot is a selected historical reference, excluded from the new2-seed denominator.

The main backbone remains384,784parameters with8,514discardable auxiliary parameters. Existing future-byte helper, fixed pilot driver, model, Config, trainer, scorer and checkpoints are unchanged. The new arm module takes explicit seeds without monkeypatching the old runner. Weight0 uses the actual legacy backbone trainer; auxiliary code and clipping groups are unchanged from the pilot.

## Main training losses

Lower CE is better. Deltas are active minus matched control. QA evaluation uses the original16training questions/96main targets. English uses the same fixed512target corpus-training probe from seed43 as the pilot, not a new probe chosen by each replay seed.

| QA/English seeds | QA control | QA auxiliary | QA delta | English control | English auxiliary | English delta |
|---|---:|---:|---:|---:|---:|---:|
| 43/44 | 0.127619997 | 0.132089317 | +0.004469320 | 1.191785932 | 1.191655040 | -0.000130892 |
| 44/45 | 0.126891673 | 0.126184046 | -0.000707626 | 1.206517696 | 1.203898430 | -0.002619267 |

All initial QA CE values are6.866767883 and initial English probe CE1.157155037. Both objectives acquire much lower QA training CE, but neither answer accuracy nor held-out binding was measured. Relative English benefit does not establish retention: all four new English-probe endpoints exceed the initial value. The probe belongs to the fixed training corpus and may not appear in a particular short replay stream; it is selected/reused material, not a held-out benchmark.

## Predefined gates

| Gate | Fixed requirement | New results | Verdict |
|---|---|---|---|
| Primary QA | Active−control <−1e-6 in BOTH comparisons |1/2| Fail |
| Secondary English direction | Active−control <=+1e-6 in BOTH |2/2| Pass |
| Joint | Both gates pass |Primary fails| Fail |

The1e-6 thresholds are descriptive numerical tolerances, not statistical significance. We do not replace the primary verdict with a pooled2/3 count that includes the historically favorable pilot. That earlier pilot remains a true conditional training result; it is not robust QA qualification.

## Other training measurements

Main objective includes the configured router penalty but excludes auxiliary loss in evaluation. Auxiliary-head loss is informative about auxiliary learning; the control head receives no auxiliary label gradients, so its loss is not a trained-head quality baseline.

| Seeds | QA main objective control/auxiliary | QA auxiliary CE control/auxiliary | English auxiliary CE control/auxiliary |
|---|---:|---:|---:|
| 43/44 | 0.144815624 / 0.151294723 | 5.556009769 / 3.827276468 | 5.544422626 / 4.096251965 |
| 44/45 | 0.144953385 / 0.141380176 | 5.556884289 / 3.824144125 | 5.544353962 / 4.095093250 |

The auxiliary head learns its own task under both new schedules, but that does not consistently help main QA CE. Which supervised positions or gradient effects cause the mixed backbone outcome is not identified; no spelling-versus-binding mechanism is claimed.

## Matched exposure

Per arm, QA has128updates,3,072main targets,2,048eligible auxiliary positions,49,664consumed byte positions and2,560supervised bytes. Every four-question group is seen32times. Auxiliary positions are eligible positions recorded for both arms; control weight0 does not use auxiliary label gradients.

| Seeds | English main targets | Eligible auxiliary positions | English consumed/supervised bytes |
|---|---:|---:|---:|
| 43/44 | 61942 | 61369 | 61880 |
| 44/45 | 61440 | 60864 | 61376 |

Within each seed, batch fingerprints, visits, target/byte counts and initial scores match across arms. Across replay seeds, English exposure differs, so cross-seed final losses are not controlled effect estimates. English carry evolves separately inside each arm; fresh QA carry never replaces it.

## Verification, provenance and runtime

New focused tests first failed before implementation. Final4focused tests pass in14.33s, including exact42/43 schedule compatibility with the original runner, independently reconstructed seed permutations/replay snapshots, zero/invalid seed handling, budget divisibility, gate signs/equalities/both-seed requirements, a small real arm against the direct legacy trainer with distinct groups/manual zero head steps, exact carry/state/roundtrip checks, source immutability and rejection of existing output directories.

Independent review identified that a glob could silently accept missing/changed historical artifacts. Before freeze, it was replaced with explicit required files and publication-manifest validation, with rejection tests for missing/mutated active head and changed published history. The earlier full run was intentionally interrupted after37passing tests to apply these fixes. Final full regression:89passed in293.84s. Ruff, formatting and ty pass; independent code/protocol review cleared the fixes before freeze.

The single frozen replication run completes all four arms without interruption and passes212runtime audits. All140input hashes remain unchanged, including all111 original continuation inputs. All30required historical pilot artifacts are explicitly validated, including both final backbone/head pairs, histories/summaries, original and continuation execution/protocol hashes, archived artifact manifests and verification records. Pilot artifacts are preserved; its earlier audit-only continuation is historical and was not repeated here. New checkpoint/head roundtrips compare exact values and appropriate state hashes, avoiding the previous mapping-order byte comparison error.

Frozen schedules, independent target/byte expectations,256steps/arm,32visits/group, parameter counts, initial scores, fixed evaluation probes, sources, files and numerical settings are audited. See [full precision evidence](../../evidence/future-byte-replication/summary.json) and [verification](../../evidence/future-byte-replication/verification.json). No source/helper/trainer changes, old checkpoint overwrites, package changes or release/push occurred. Graphify still cannot run because its installed Python module is missing.

| Seeds | Control training-loop seconds | Auxiliary training-loop seconds |
|---|---:|---:|
| 43/44 | 69.68 | 88.56 |
| 44/45 | 69.80 | 90.06 |

Timing includes compilation and synchronization, excludes evaluation/checkpoint saving, and is not a controlled speed benchmark. Both arms match exposure and parameter count, not FLOPs. No inference speed, memory reduction, energy or portability improvement was tested.

## Artifacts and decision

New local-only checkpoints: `results/future-byte-replication/seed-{43,44}/{control,future_byte}/final.mlm`, with separate auxiliary state `head.msgpack`. Standard backbone loading is supported; joint training resume remains unsupported. All new files use fresh directories; none is promoted.

Do not enable the auxiliary objective as a general QA improvement or scale its budget based on the favorable pilot. Keep the baseline and optional experimental head. A useful next separately frozen analysis is answer-level/per-position evaluation on existing fixtures to distinguish loss tradeoffs before further training; do not infer a mechanism from these aggregates. Fresh held-out tests remain closed.

No ranking, generation, development/test opening or held-out retention evaluation occurred. Earlier acquisition-replication/generalization failures remain failures. Dynamic graph growth, category webs, sparse execution, reliable chat and Transformer superiority remain unproven. Main, v0.0.1 and package version are unchanged; research is local and unpushed.
