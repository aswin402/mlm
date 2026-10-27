# Frozen future-byte answer/position analysis

## Scope

Exploratory read-only follow-up to the failed primary replication gate. Evaluate the common initializer and all six completed control/auxiliary backbones: pilotQA42/E43, replicationQA43/E44 andQA44/E45. Never select favorable checkpoints, update weights, load auxiliary heads for predictions, generate free text, or open development/fresh test files. Existing sixteen training questions only. This analysis does not retroactively replace the primary QA-CE gate or qualify generalization/chat.

Reuse evaluate_questions(format=chat), mean conditional log probability of answer bytes, two fixed choices and existing argmax tie behavior; reuse retrieval aggregate for complete four-question groups/context/query swaps. Expected answer indices only grade scores after neural prediction. Report16-question correctness,4complete groups,8context pairs, score margins (gold−other), exact and near ties(abs<=1e-6), questionwise changes and control-only/auxiliary-only correct transitions. A near tie is a descriptive warning, not a change of prediction.

Teacher-force the exact original training records, including their correct answers/newline/EOS. Offline metadata verifies text==canonical chat prompt+goldanswer+newline, exact contiguous assistant loss span andEOS supervision. Align rows by build_batches natural group order and validate token/mask identity. Partition the96supervised positions into first answer byte16, remaining answer bytes48, newline16 andEOS16; also report answer-only64. Teacher-forced rest-byte loss is conditioned on preceding gold answer bytes, so it is not an independent binding test. First-byte CE is more directly prompt-conditioned, but is not itself proof of binding. Do not infer a spelling/gradient mechanism from these partitions.

Record per-question teacher target NLLs and each partition's count,sum,mean; weighted partition sums must reproduce global main CE and historical QA CE within1e-6. Mean gold answer-byte NLL must match negative existing gold candidate score within1e-5 (padding/batch-layout numerical tolerance). Teacher masks never affect the scorer/model inputs beyond existing ordinary causal teacher forcing. No answer-index parser or rule-based inference.

For each of the three predefined comparisons, report per-role active−control NLL contributions normalized by96; their sum must equal total QA CE change. Report answer-score margins and question-level correctness flips. Historical pilot separate from the two replication seeds; no pooled qualification gate or favorable metric substitution.

## Provenance and implementation plan

Preserve all140 replication inputs and validate all newly completed replication artifacts using explicit published/raw artifact-manifest paths and checkpoint/head hashes. All111 prior continuation inputs and original pilot preservation remain covered transitively. Freeze this method, data, source/state/config hashes for all7models, code/tests, partition masks, scoring recipes, numerical settings, source budgets and original metrics before ranking outputs.

Write failing focused tests for exact record/question/token alignment and corrupt masks, weighted partitions and ranking transitions, plus real-neural scorer/teacher CE agreement and source immutability. New answer_analysis helper/runner only; existing model/trainer/scorer/data/old drivers unchanged. Focused regression, static checks and independent protocol/code review before measurement. Last89-test full regression is prior evidence; no full repeat is planned for new read-only analysis code unless failures or core changes justify it.

Run all7models once, no optimizer steps or saved model files. Audit source/state/file/metadata immutability, exact inputs/versions/settings, all16question identities,4groups,96mask positions, partition counts/disjointness, finite outputs, score/teacher consistency, original CE reproduction and comparison algebra. Publish all outcomes and compact evidence under docs; update research/PRD/index/Unreleased. Keep main/version0.0.1/tag unchanged, local research commit only, no push or promotion.
