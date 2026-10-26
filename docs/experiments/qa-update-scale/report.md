# Bounded loaded-QA update fractions

Completed2026-10-27 on `research/qa-update-scale`. Smaller final-update fractions reduce raw QA-loss increases at the three previously harmful fitted snapshots, but do not yield a clear aggregate-loss improvement at any of those sources. Smaller fractions also reduce gains at the initializer and the helpful fitted source. No universal update-size fix or model-quality improvement is established.

## Frozen design and semantics

The [protocol](protocol.md) fixes all five prior snapshots, their exact actual-next QA groups, loaded optimizer state and configured router penalty. Independently compute the actual routed-trainer QA update and zero-gradient optimizer update at each source, then evaluate final parameter displacement fractions0,1/8,1/4,1/2,1. This is10 copied optimizer computations and50 scaled parameter evaluations, not50 sequential training steps. Every branch starts independently from the same source.

Scale0 uses source parameters exactly; scale1 uses full endpoint parameters exactly. Intermediate parameters use float32 source + fraction*(endpoint-source). Every candidate retains the corresponding full endpoint's moments/internal optimizer state and public step+1. It is a counterfactual final-update multiplier including decay/role scaling, not gradient scaling, optimizer reset or a validated resumable training algorithm. No candidates are saved or rolled forward.

All16 training questions/96 assistant-newline-EOS targets define aggregate CE; each selected group has4questions/24targets. Report selected language CE, configured total objective and aggregate language CE relative to source. Positive change is worse. Absolute changes<=1e-6 are numerically unresolved under the predefined descriptive threshold, not statistical significance. Initial and fitted loss scales are not pooled.

## Aggregate QA CE changes

| Source | 0 | 1/8 | 1/4 | 1/2 | Full |
|---|---:|---:|---:|---:|---:|
| initial | +0 | -0.50772905 | -0.97187042 | -1.6906738 | -2.6904969 |
| q42_e43 | +0 | +1.4404068e-06 | +2.9901275e-06 | +6.1491737e-06 | +1.3709068e-05 |
| q42_e44 | +0 | +2.3320317e-06 | +5.312264e-06 | +1.2904406e-05 | +3.5107136e-05 |
| q43_e43 | +0 | +8.7416265e-07 | +1.7881393e-06 | +3.7152786e-06 | +8.225441e-06 |
| q43_e44 | +0 | -5.5588782e-05 | -0.00010885298 | -0.00020775944 | -0.00037556887 |

At q42_e43 and q42_e44, every nonzero fraction still raises aggregate CE beyond1e-6. At q43_e43,1/8 gives+8.7416e-7, which is unresolved, while larger fractions raise CE beyond threshold. Neither a clear sign reversal nor a universally improving nonzero fraction is observed. Zero displacement is the exact no-change baseline, not a learning improvement.

At initial and q43_e44, all nonzero fractions lower aggregate CE, with larger benefit for larger fractions over this bounded grid. These observations do not establish monotonic behavior outside the measured fractions or over training trajectories. No favorable fraction is selected or promoted.

## Selected group and actual training objective

| Source | Selected CE change at1/8 | Selected CE change full | Configured objective change at1/8 | At1/4 | At1/2 | Full |
|---|---:|---:|---:|---:|---:|---:|
| initial | -0.5768795 | -3.192992 | -0.57672501 | -1.113584 | -1.9785028 | -3.1918161 |
| q42_e43 | +1.5100231e-06 | +1.4464022e-05 | -1.5525147e-06 | -3.0581141e-06 | -5.7661673e-06 | -9.8189339e-06 |
| q42_e44 | +7.5250864e-07 | +3.2544136e-05 | -8.8661909e-07 | -8.6426735e-07 | +2.2649765e-06 | +1.9475818e-05 |
| q43_e43 | -5.9604645e-07 | -3.3378601e-06 | -1.4996622e-06 | -2.9980438e-06 | -5.5941055e-06 | -1.0536867e-05 |
| q43_e44 | -0.00015716255 | -0.0011649579 | -0.00017444789 | -0.00034575164 | -0.0006775111 | -0.0013011619 |

q42_e43 preserves the language/objective tradeoff: selected CE worsens while configured objective improves at every nonzero fraction. q43_e43 improves configured objective even though aggregate CE does not clearly improve; selected CE at1/8 is itself unresolved. At q42_e44, small-fraction objective changes are approximately−8.9e-7 and−8.6e-7, both unresolved; the larger fractions clearly worsen the objective. It would be inaccurate to call those small negative numbers a demonstrated objective improvement.

## Matching zero-gradient controls

| Source | QA-minus-zero aggregate change at1/8 | At1/4 | At1/2 | Full |
|---|---:|---:|---:|---:|
| initial | -0.50773048 | -0.97187281 | -1.6906776 | -2.690506 |
| q42_e43 | -1.2914534e-06 | -2.4835463e-06 | -5.1161041e-06 | -1.0162592e-05 |
| q42_e44 | -1.1473894e-06 | -2.30968e-06 | -5.3718686e-06 | -1.4185905e-05 |
| q43_e43 | -6.7555811e-07 | -1.4305115e-06 | -3.0498486e-06 | -6.0994644e-06 |
| q43_e44 | -3.5107136e-05 | -6.8195164e-05 | -0.00012818724 | -0.0002232939 |

Every numerical QA-minus-zero contrast is negative for nonzero fractions, but q43_e43 at1/8 is within1e-6 and therefore unresolved. A favorable conditional control contrast does not mean that loss improved relative to source. Zero-gradient candidates still apply loaded history/decay. Nonlinear optimizer behavior, displacement direction and magnitude are not individually identified by this scaling study.

## Verification and evidence

The focused test was first run before the helper existed and failed as intended. Its initial strict endpoint comparison then caught a2.9e-11 parameter difference from a separately compiled equivalent optimizer calculation. The helper was changed to reuse the exact existing routed trainer; the strict direct-trainer/zero parameter and moment identity test then passed. Source/moment immutability, exact zero baseline, independent intermediate displacement, distinct selected/aggregate targets, control algebra, supervision rejection and nonfinite rejection are covered.

Focused test:1 passed in19.38s. Full final regression:82 passed in327.39s. Ruff, new-file formatting and ty pass. Independent code/protocol review found no material issues before freeze. The actual freeze/run integration completed all five sources and passed653 runtime audits, including98 frozen hashes containing all88 prior QA-control inputs, file/state/metadata identity, restored next QA selection/counters,96 aggregate targets, exact scale0 baseline, scale1 prior-metric reproduction within1e-6, displacement/optimizer identity, objective/control algebra, finite outputs and source/settings immutability.

[Compact evidence](../../evidence/qa-update-scale/summary.json) preserves full precision and all50 rows; [verification](../../evidence/qa-update-scale/verification.json) records final integrity checks and the unavailable Graphify launcher (`ModuleNotFoundError: graphify`). Raw artifacts remain local. No checkpoint, prior diagnostic inputs, training/model/scorer math or release changed.

## Decision

Do not install a global smaller-update rule as a supposed solution, reset Adam, or claim improved binding. This diagnostic closes the planned local update-size check without identifying a universal fix. A separately predefined short-horizon future-byte/MTP pilot can now use this documented baseline, but it must be presented as an experimental objective comparison with existing instability, matched controls and independent-seed/held-out gates; it is not implemented by this task and must not be bundled with unrelated routing changes.

The [DeepSeek research note](../../research/deepseek-inspirations.md) records architectural inspirations and their prerequisites. No answer ranking, raw generation, English retention after branches, development or fresh test evaluation was performed. Previous failed replication and held-out binding gates remain failed. Version0.0.1, main and release tag are unchanged; local research only, no push.
