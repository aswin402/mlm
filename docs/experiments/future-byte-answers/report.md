# Future-byte answer and target-position analysis

Completed2026-10-28 on `research/future-byte-answer-analysis`. Every examined model—the initializer and all six128QA+128English endpoints—ranks8/16training answers correctly and completes0/4binding groups. The auxiliary objective supplies no net answer-ranking gain at this short budget, despite some lower cross-entropies. All models have zero context-swap prediction flips. These are training-only constrained rankings, not free generation, chat or held-out understanding.

## Frozen scope and unchanged scorer

The [protocol](protocol.md) fixes all7backbones: common initializer, original pilotcontrol/auxiliary and both replication comparisons. No parameter updates, auxiliary-head prediction, development/test access, model selection or new training occurs. The existing chat-formatted evaluate_questions scorer ranks the two choices by mean conditional answer-byte log probability; its existing argmax/tie behavior is unchanged. Labels grade scores afterward. Existing retrieval aggregation defines complete4question groups and context/query pairs.

Teacher-forced loss uses the exact original records, including gold answer/newline/EOS. Offline alignment verifies canonical prompt/text, byte offsets, loss spans, natural group order, actual tokens/masks and scoredEOS. The96supervised targets partition into16first answer bytes,48remaining answer bytes,16newlines and16EOS. Answer-only loss covers64bytes. Teacher-forced later-byte losses condition on earlier gold answer bytes; they cannot alone establish binding. First-byte loss is prompt-conditioned but is likewise not a complete binding test.

## Answer rankings

All16questions and4groups are reported for every model. Context pairs hold the question fixed and swap assignments; query pairs hold context fixed and swap the queried entity.

| Model | Correct /16 | Complete groups /4 | Complete context pairs /8 | Query flip fraction | Context flip fraction | Near ties |
|---|---:|---:|---:|---:|---:|---:|
| initial | 8 | 0 | 0 | 0.000 | 0.000 | 0 |
| pilot42_control | 8 | 0 | 0 | 0.000 | 0.000 | 0 |
| pilot42_future_byte | 8 | 0 | 0 | 0.000 | 0.000 | 0 |
| rep43_control | 8 | 0 | 0 | 0.000 | 0.000 | 0 |
| rep43_future_byte | 8 | 0 | 0 | 0.000 | 0.000 | 0 |
| rep44_control | 8 | 0 | 0 | 0.000 | 0.000 | 0 |
| rep44_future_byte | 8 | 0 | 0 | 0.250 | 0.000 | 0 |

The replication44 auxiliary model changes predictions on2/8fixed-context query pairs and gets1/8query pairs wholly correct, but still fails every context-swap pair and complete group. Thus limited query sensitivity is present there; it must not be described as successful context binding. All models score50%, the balanced fixed-choice baseline. This behavior does not prove the architecture is incapable of binding: these endpoints received only32presentations/group, and earlier selected longer-budget narrow acquisition results remain unchanged.

No model has an exact or near tie at the predefined1e-6 threshold. Mean signed gold-minus-other margin is close to zero because favorable and unfavorable margins cancel over balanced labels. It is not evidence that individual predictions are uncertain or numerically tied.

## Per-target teacher-forced CE

Per-role means use their own target count; total uses96targets. The parts have different weights, so their means must not simply be added.

| Model | First answer byte | Rest answer bytes | Newline | EOS | Total96-target CE |
|---|---:|---:|---:|---:|---:|
| initial | 9.180147201 | 2.666997234 | 12.401569128 | 11.617898941 | 6.866767829 |
| pilot42_control | 0.968022883 | 0.007502496 | 0.000616252 | 0.001511753 | 0.165443063 |
| pilot42_future_byte | 0.908978522 | 0.008949240 | 0.000667095 | 0.001762450 | 0.156375964 |
| rep43_control | 0.741393507 | 0.007238030 | 0.000998318 | 0.001614094 | 0.127620002 |
| rep43_future_byte | 0.764572620 | 0.008793910 | 0.000626147 | 0.000955403 | 0.132089317 |
| rep44_control | 0.734497488 | 0.007768194 | 0.002260149 | 0.001287818 | 0.126891673 |
| rep44_future_byte | 0.733894050 | 0.006804526 | 0.001529038 | 0.001267612 | 0.126184046 |

Training sharply reduces loss across these roles compared with the initializer, including spelling/continuation and formatting targets. That is real optimization progress even though rankings remain at8/16. The optional head's pilot gain specifically comes from lower first-byte NLL, not improved formatting: rest/newline/EOS losses are slightly worse in that comparison. First-byte probabilities can improve while both candidate ordering and context sensitivity fail to improve.

## Paired changes: auxiliary minus control

The following are contributions to the total CE difference, computed as each role's summed-NLL change divided by96. They sum to the total, but are arithmetic partitions rather than causal effects.

| Comparison | First contribution | Rest contribution | Newline contribution | EOS contribution | Total CE change |
|---|---:|---:|---:|---:|---:|
| pilot42 | -0.009840727 | +0.000723372 | +0.000008474 | +0.000041783 | -0.009067098 |
| rep43 | +0.003863186 | +0.000777940 | -0.000062029 | -0.000109782 | +0.004469315 |
| rep44 | -0.000100573 | -0.000481834 | -0.000121852 | -0.000003368 | -0.000707626 |

The failed replication43 comparison worsens first/rest answer-byte loss despite better newline/EOS loss. Replication44 improves all role means slightly, with the largest weighted contribution from later answer bytes. Neither change produces a net answer-count gain. No gradient, memory-write, spelling-versus-binding or capacity mechanism is identified by these aggregates.

| Comparison | Both correct | Control only correct | Auxiliary only correct | Both wrong | Prediction flips | Net correct change |
|---|---:|---:|---:|---:|---:|---:|
| pilot42 | 8 | 0 | 0 | 8 | 0 | 0 |
| rep43 | 4 | 4 | 4 | 4 | 8 | 0 |
| rep44 | 1 | 7 | 7 | 1 | 14 | 0 |

Identical8/16scores do not mean identical predictions: the replication comparisons exchange which questions are correct. The pilot's answer predictions do not change at all. Full per-question margins and target NLLs are retained in the [evidence](../../evidence/future-byte-answers/summary.json).

## Score margins and numerical checks

| Model | Minimum absolute margin | Maximum absolute margin | Teacher/candidate maximum error |
|---|---:|---:|---:|
| initial | 0.877117634 | 2.334364414 | 1.54972076e-06 |
| pilot42_control | 0.125238657 | 0.508381419 | 4.17232513e-07 |
| pilot42_future_byte | 0.061463371 | 0.450591754 | 6.10947609e-07 |
| rep43_control | 0.022049725 | 0.098652437 | 4.32133675e-07 |
| rep43_future_byte | 0.056212649 | 0.126761138 | 5.21540642e-07 |
| rep44_control | 0.042137817 | 0.134651110 | 4.61935997e-07 |
| rep44_future_byte | 0.000765979 | 0.071882591 | 8.34465027e-07 |

Gold answer-byte teacher NLL agrees with negative canonical gold-candidate score within1e-5. Global teacher CE reproduces each historical main-language CE within1e-6; small displayed differences from earlier summaries arise from host precision and weighted summation. No scorer, tie rule or mask was tuned after viewing outputs.

## Verification and provenance

Focused tests were written first and initially failed before the helper existed. All4focused tests pass in8.44s. They cover exact byte partitions and record/label/span/group corruption rejection, independent weighted summation, all correctness-transition categories/identity rejection, real-neural canonical-scorer/teacher/routed-CE agreement and source immutability, plus paired role-contribution arithmetic. Ruff, formatting and ty pass. Independent review checked the original16records and confirmed the16+48+16+16partition before freeze.

The previous full89-test result is explicitly prior evidence. Existing model/trainer/scorer/core tests are unchanged; the full suite was not repeated for these new read-only modules. Actual frozen execution covers all7sources and3comparisons, passing258audits with zero optimizer updates. All188input hashes remain unchanged, including all140replication inputs. Explicit raw/public artifact manifests validate all completed replication/pilot sources, histories, execution records, checkpoint/head hashes and required files. Existing numerical settings are preserved apart from aggregate source SHA reflecting added analysis modules.

Source file/state/metadata hashes and optimizer step counters remain unchanged after every analysis. Partition counts, question identities, all groups/pairs, finite likelihoods, canonical scorer reconciliation, historical CE, transition counts and weighted-delta algebra are audited. No model or optimizer files were written. [Verification](../../evidence/future-byte-answers/verification.json) records final publication integrity, independent review and the unavailable Graphify launcher (`ModuleNotFoundError: graphify`).

## Decision and limits

The objective remains optional and unpromoted. The failed primary/joint replication gates remain failed; these exploratory rankings do not replace them. Current evidence supports learning training-format/answer probabilities without reliable assignment-dependent answer selection at this short budget. It does not prove a universal architectural limitation or identify why binding fails.

The next training experiment should use an explicitly predefined answer/group acquisition gate and matched budgets, rather than treating lower aggregate CE as sufficient. Keep any change to objective, exposure or architecture separate and freeze it before measurement. Fresh held-out tests remain closed; stronger qualification still requires independent replication and held-out evidence. Earlier longer-budget narrow acquisition observations are neither rerun nor reversed here.

Dynamic growth, category webs, sparse execution, broad device support, reliable chat and Transformer superiority remain unproven. Main, v0.0.1 and package version are unchanged. Local research only, no push.
