# Reading probes v1

Original evaluation-only diagnostic: 12 pairs / 24 questions. Each pair keeps the question and candidate answers fixed while changing a context fact and the correct answer. The expected-answer key only grades frozen-model likelihood rankings. It is never passed to the network or used for training, checkpoint selection, or inference response rules. No regex or response-keyword classifier is used.

Report individual accuracy, both-members-correct pair accuracy, and prediction-change fraction. A fixed answer preference gets 50% item accuracy and zero pair accuracy on this balanced set. Two-choice guessing has 50% expected item accuracy. This tiny authored set is not a standardized general-understanding benchmark. Mean answer-byte likelihood can still favor particular strings; paired results help expose that bias but do not eliminate all confounds.

Use identical prompt format for comparisons. Chat-formatted results for a dialogue-adapted checkpoint belong to a separate comparison because its training and prompt format differ. Freeze this v1 suite before inspecting results; future revisions must get a new version rather than silently changing questions.


Scorer version 2 conditions on `Answer: ` in plain format or the assistant-marker newline in chat format. It scores only the answer bytes, excluding those prompt separators from the numerator and denominator. Preliminary version-1 scores that included an extra space are retained as superseded artifacts; the question-suite hash is unchanged.
