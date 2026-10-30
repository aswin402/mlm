# Reading comprehension evaluation probes (`v1`, `v2-multihop`, `v3-3hop`, `v4-temporal`)

Original evaluation-only diagnostics:
- **`data/evaluation/reading-v1.json` (1-hop):** 12 counterfactual pairs / 24 questions testing single-hop attribute, location, and ownership binding across forward and backward query orders.
- **`data/evaluation/reading-v2-multihop.json` (2-hop):** 8 counterfactual pairs / 16 questions (`person-room`, `container-color`, `pet-food`, `guard-trainer`, `friend-city`, `key-room`, `team-prize`, `tool-shelf`) testing 2-clause transitive relational composition ($A \to B \to C$).
- **`data/evaluation/reading-v3-3hop.json` (3-hop):** 8 counterfactual pairs / 16 questions (`chain-room`, `chain-color`, `chain-food`, `chain-city`, `chain-owner`, `chain-prize`, `chain-tool`, `chain-drink`) testing 3-clause transitive relational composition ($A \to B \to C \to D$).
- **`data/evaluation/reading-v4-temporal.json` (temporal & multi-attribute):** 8 counterfactual pairs / 16 questions (`temporal-room`, `temporal-color`, `temporal-pet`, `multi-attr-c0`, `multi-attr-c1`, `2hop-temp-fwd`, `2hop-temp-bwd`, `3hop-dist`) testing temporal state-overwrite across sequential clauses, multi-attribute same-subject clause specificity, 2-hop temporal transitive chains, and 3-hop reasoning with unrelated distractor clauses.

Each pair keeps the question and candidate answers fixed while changing a context fact and the correct answer. The expected-answer key only grades frozen-model likelihood rankings or open-ended greedy `generate()` outputs. It is never passed to the network or used for training, checkpoint selection, or inference response rules. No regex or response-keyword classifier is used.

Report individual accuracy, both-members-correct pair accuracy, and prediction-change fraction. A fixed answer preference gets 50% item accuracy and zero pair accuracy on this balanced set. Two-choice guessing has 50% expected item accuracy. This tiny authored set is not a standardized general-understanding benchmark. Mean answer-byte likelihood can still favor particular strings; paired results help expose that bias but do not eliminate all confounds.

Use identical prompt format for comparisons. Chat-formatted results for a dialogue-adapted checkpoint belong to a separate comparison because its training and prompt format differ. Freeze this v1 suite before inspecting results; future revisions must get a new version rather than silently changing questions.


Scorer version 2 conditions on `Answer: ` in plain format or the assistant-marker newline in chat format. It scores only the answer bytes, excluding those prompt separators from the numerator and denominator. Preliminary version-1 scores that included an extra space are retained as superseded artifacts; the question-suite hash is unchanged.
