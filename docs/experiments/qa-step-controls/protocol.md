# Copied QA-step optimizer/regularizer controls

## Design and frozen scope

Use all five sources and exact next QA batches from the completed qa-replay-pair protocol. Identical source weights, fresh QA carry and all16-question/96-target aggregate evaluation. No English update, held-out evaluation, rankings, generation, training trajectory, source overwrite, release or push.

Cross loaded versus fresh optimizer state with configured router balance versus zero router balance. Fresh means tx.init(source.params): both moments and optimizer-internal clock reset, while public TrainState.step remains the source counter. Same optimizer transform/hyperparameters, clipping, weight decay and role scales. This isolates combined optimizer history, not moments alone. The coefficient change affects the training objective, not inference. Each branch independently starts at source weights. Add one zero-gradient control per optimizer state: six transitions/source,30 total. At initialization loaded and fresh must agree.

Measure selected-group language CE, raw router penalty, configured total objective and coefficient-specific objective, plus aggregate QA language CE before/after each branch. Record gradient norms, actual update norms, steps and finite state. Compare loaded configured result to prior saved selected/aggregate QA-after values within1e-6. Record controlled differences in aggregate and selected CE for regularizer toggling, history resetting and their interaction. Positive CE change means worse; changes of absolute size at most1e-6 are numerically unresolved, not statistical significance. Zero controls measure history/decay drift, not an additive causal decomposition.

Freeze this method, all source/data/code/tests and every prior pair input hash before measurements. Enforce source hashes/state identity, restored QA selection, numerical settings, no-output-overwrite, all6 branches, objective algebra, counters, finite outputs, prior reproduction and unchanged inputs. All five selected sources required; no budget extensions or favorable-source selection. Separate frozen recipe, raw outputs and report under results/qa-step-controls and docs/evidence/qa-step-controls. Findings apply only to these local steps and selected snapshots; fresh optimizer magnitude and changed clipping are part of the intervention. No optimizer reset recommendation without longer controlled evidence.

## Implementation plan

- Add focused tests for six independent branches versus direct routed trainer/autodiff, fresh optimizer initialization, zero drift controls, coefficient-specific objective algebra, source immutability and supervision rejection. Run red before implementation.
- Implement qa_controls helper only; reuse routed_objective and existing source/QA restoration without changing model/training math.
- Add freeze/run CLI with exact prior-source selection and runtime checks. Test helper; run full regression and static checks. Independently review code/protocol before measurement.
- Freeze inputs then compute30 discarded transitions; audit prior reproduction, source identity, metrics and branch counts.
- Write numerical report and compact evidence; update research/PRD/index/README/Unreleased. Final hashes/static checks/review, local research commit only.
