# Bounded loaded-QA update-size diagnostic

## Design

Use all five fixed snapshots and exact actual-next QA batches from qa-step-controls. Preserve source weights, loaded optimizer state, configured router coefficient and QA supervision. Compute one ordinary QA optimizer candidate and one zero-gradient optimizer candidate independently from each source. Evaluate final parameter displacement fractions [0,0.125,0.25,0.5,1] for each candidate. There are10 actual copied optimizer computations and50 scaled parameter evaluations, not50 sequential training steps. No branch feeds another.

A scaled candidate has parameters source + scale*(full_candidate-source), using float32 arithmetic. Scale0 uses exact source parameters; scale1 uses exact full candidate parameters. Each retains the full candidate optimizer state and incremented public step. This is a counterfactual final-update multiplier (including decay and role scaling), not gradient scaling, clipping modification, or an optimizer reset. It does not define an accepted/resumable training algorithm and no candidate is saved as a checkpoint.

Evaluate selected four-question/24-target language CE, raw router penalty, configured total objective and all16-question/96-target aggregate language CE. Record full/scaled parameter displacement norms, actual steps, finite state and loss changes relative to source. Compare QA versus zero at each fixed scale. Require source exactness at scale0 and full-step selected/aggregate/objective reproduction versus prior results within1e-6 at scale1. Scale choices and sources are predefined, no extra rates or favorable endpoint selection.

Descriptive threshold abs(delta)<=1e-6 is numerically unresolved, not statistical significance. Report every scale and source; no best-scale promotion. A smaller positive loss change is reduced harm, not an improvement over source. Nonmonotonic behavior is possible; tiny float32 movements may be unresolved. One next step at historically selected snapshots cannot establish learning, answer accuracy, held-out transfer, English retention, or trajectory stability. Keep prior failed generalization/replication gates unchanged. This task does not implement MTP; record DeepSeek inspirations as hypotheses and revisit after this baseline control.

## Implementation and verification plan

Write focused failing tests first, then a separate update_scale helper and freeze/run CLI; do not alter existing model/training/scoring or prior diagnostic inputs. Test direct QA/zero endpoint identity, exact zero baseline, independently calculated intermediate displacement, retained optimizer state/steps, distinct selected/aggregate losses, no supervision and nonfinite rejection, and source immutability.

Freeze this protocol, source/data/code/tests, numerical environment and every prior QA-control input hash before measuring. Validate source file/state/metadata identity, seed/history-derived next QA and counts, aggregate96targets, all ten branch names per source, loss/control algebra, finite outputs, endpoints and unchanged sources/inputs/settings. Run focused/full regression/static checks and independent review, then measure. Write compact evidence/report, update research/PRD/index/changelog, keep release/main/packageversion0.0.1 untouched. Local research commit only; no push, persistent training, validation/test opening, rankings or raw generation.
