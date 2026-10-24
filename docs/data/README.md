# Data used in MLM experiments

`smoke/` is an original sentence-template fixture.

`hf/` contains bounded downloads from public Hugging Face datasets and is excluded from version control. Source rows, URLs, byte/content hashes, observed Hub revisions, and local split details are in `hf/manifest.json`. The dataset viewer serves current rows; the observed Hub SHA is not a claim that the viewer request was revision-pinned. Cached response bytes preserve the exact local snapshot.

- English: [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), dataset-card license CDLA-Sharing-1.0. 1,000 training stories and 32 development / 32 test stories selected from upstream validation.
- Conversation: [HuggingFaceTB/everyday-conversations-llama3.1-2k](https://huggingface.co/datasets/HuggingFaceTB/everyday-conversations-llama3.1-2k), dataset-card license Apache-2.0. 300 training conversations and 32 development / 32 test conversations selected from upstream test_sft.

Upstream held-out rows are partitioned into local development and test subsets after exact-document deduplication against training. Development chooses checkpoints; local test scores are reported after selection. This does not guarantee absence of semantic duplicates or establish performance on the full source datasets. Selection uses bounded first rows rather than a representative random sample.

JSONL preserves original story paragraphs and full conversations. Conversations use textual `<|user|>`, `<|assistant|>`, and optional `<|system|>` delimiters. These are byte sequences, not additional vocabulary tokens. The objective predicts all text, including user turns; it is not assistant-only supervised loss.

The adaptation mix uses all 300 conversations and the first 250 training stories. Both held-out domains are included in the mixed development and test sets. Story rehearsal is an explicit training strategy, not proof that forgetting is solved.

Prepare a fresh snapshot:

```sh
uv run python -m mlm.hfdata --out data/hf --stories 1000 --conversations 300 --development-size 32
```

The command also creates the mixed-stage files. Existing prepared datasets should be reused; choose a different output directory to collect another snapshot. Source licenses and attribution remain applicable; downloading does not relicense the data. No dataset has been uploaded or published by this project.


The separate `assistant-v2/` experiment is prepared by `python -m mlm.supervised_data` from verified cached public rows. It uses 64 conversations and 64 stories plus offline synthetic paired reading examples. Public source attribution/licenses above still apply. Its manifest records exact files, counts, and hashes. Reading labels are dataset-generation logic only, never inference logic.

Optional JSONL `loss_spans` contain sorted, nonoverlapping half-open UTF-8 byte ranges; `score_eos` separately enables the EOS target. Assistant records supervise assistant content/newline and the final EOS. Prompt bytes still update memory. Stories retain full-text supervision. Without spans, the historical all-target objective applies. Names and templates are disjoint across supervised reading splits, not guaranteed unseen in prior English pretraining. See [measured outcomes](../experiments/assistant-learning/report.md).


`retrieval-v1/` is a separate offline controlled-task dataset prepared by `python -m mlm.retrieval_data`. It contains 960 synthetic training records, 480 development records, and 480 test records, plus 32 cached public training stories in full-stage training. Basic subsets contain 64/32/32 single-fact/no-delay documents. Synthetic supervision uses assistant-only UTF-8 byte spans and EOS; stories use full-text targets. Public story attribution and license information above remain applicable.

Task names are disjoint across splits; sentence grammar and four color values are shared intentionally. Filler lengths are exactly 0/64/128 bytes. Context swaps and query swaps are grouped to expose incorrect entity selection. This is not a natural-language chat benchmark. Facts/labels in question JSON are grading metadata only; the model receives the textual prompt, and inference has no symbolic answer lookup. File hashes and bounded public-source provenance are recorded in the manifest. See [experiment details](../experiments/retrieval-curriculum/report.md).


`binding-v1/` is an original focused two-person dataset, prepared by `python -m mlm.binding_data`. It has 960 training documents / 240 four-question groups, 224 development / 56 groups, and 224 test / 56 groups. All unordered entity pairs within each split appear, with both fact orders; color pairs cycle through six combinations in seeded shuffled entity-pair order. Split names are disjoint within this task. Grammar and colors are shared deliberately; this does not test broad wording transfer.

`binding_run` batches all four context/query swaps together and trains on assistant-only spans plus EOS. There is no story rehearsal in this diagnostic. Its labels/factual metadata are offline training/evaluation data, never an inference solver. Dataset/source hashes and pre-test checkpoint locks are in [the focused report](../experiments/focused-binding/report.md). Its post-hoc single-group overfit check is training-only and must not be mixed with held-out results.


`gradual-v1/`, prepared by `python -m mlm.gradual_data`, has a nested 64-group training pool (256 documents) and two separate development/test transfer buckets: new pairs of training-pool names and names absent from the task's training pool. Each evaluation bucket has 48 four-question groups / 192 questions, or 24 cases when requiring both fact orders together. Familiar-name pair identities are disjoint across training/development/test; names intentionally overlap for that bucket. New-name development/test pools are disjoint from each other and training. The first group matches the stored single-group sanity source exactly.

A post-hoc audit found the familiar-name development/test pair roles structurally imbalanced relative to the initial curriculum pairs. This snapshot is preserved as historical evidence; use a new role-balanced confirmation design before claiming robust transfer. [Report and split audit](../experiments/gradual-binding/report.md). All labels/grouping/role analysis remain offline; neural inference receives only the textual prompt.


`confirmation-v1/`, prepared by `python -m mlm.confirmation_data`, is a new role-balanced snapshot: 32 artificial training names, 1,024 training documents / 256 four-question groups, and 1,152 documents in each development/test split. Familiar-name evaluation covers all 32 training names with 16 reserved pairs; new-name pools contain 16 separate labels and eight pairs. All evaluation pairs use six color combinations and both sentence orders. Within/cross-role pairs are balanced separately in every bucket, and every training name has four partners. Role classes are random offline strata, not prompt features.

Familiar/new-name bucket sizes differ (96 versus 48 eight-condition cases), so checkpoint selection gives them equal macro weight. Source, dataset and checkpoint locks are in [the study report](../experiments/balanced-confirmation/report.md). No story is hidden among QA records: rehearsal uses an explicit separate cached English training stream with its own counters/carry. English development constrains one checkpoint policy; the repeatedly used English test remains a retention diagnostic. Names are disjoint within the task, not guaranteed globally novel in pretraining.


`hierarchy-v1/` uses the same balanced confirmation generator with a new seed (920231), preserving old default seed731029 behavior. It contains fresh training/development/test names and pair assignments; counts and within/cross balance follow confirmation-v1. The hierarchy experiment freezes all three variants before opening this new test. See [protocol and results](../experiments/hierarchical-routing/report.md). Role classes are offline evaluation strata and are not the model's router groups or supplied linguistic labels.

`scratch-binding-v1/` is the balanced generator's seed 631204 snapshot, reserved for the from-scratch dense/hierarchical comparison. Counts and role balancing follow confirmation-v1. The English source remains the same bounded TinyStories snapshot; each architecture independently pretrains from random weights before adapting its own checkpoint. Test-only question files add exactly 64 or 128 neutral filler bytes to the zero-delay questions, without changing facts, choices or grading labels. These delayed sets are diagnostic and never used for selection. [Protocol and results](../experiments/from-scratch/report.md).


`transfer-v1/` contains fixed/diverse/mixed subdirectories from the original seed 904217 generator (`python -m mlm.transfer_data`). They share 16 task-training labels, disjoint familiar-name development/test pairings and separate 16-label new-name pools. Fixed has 16 training groups; diverse/mixed have 128 with identical underlying facts/queries and different renderers. Each development/test snapshot has six name/wording buckets of 48 complete eight-condition cases, totaling 2,304 questions. Heldout templates occur in neither training renderer. Model test scoring happens only after all endpoints are locked. Ground truth and role/wording metadata are offline labels, never an inference solver. [Protocol and results](../experiments/transfer/report.md).


`kv-binding-v1/` is transfer_data seed 638101 with the same verified pairing/name/wording split rules as transfer-v1. Only its diverse 128-group pool is trained in the [delta-memory pilot](../experiments/associative/report.md); generated fixed/mixed pools are unused. New checkpoints start from the preserved English-only source, with all model test scoring after both endpoint locks. Labels remain offline training/grading metadata; no entity parser or answer lookup is added to inference.


## Addressing debug subsets

`addressing-v1/groups-1` and `groups-4` copy the first4/16 aligned training records/questions from `kv-binding-v1/diverse`, without relabeling, and the first8 complete development conditions. Training uses the same Bon/Bom pair, one or two color pairs and one or both fact orders. These are previously available debugging subsets, not fresh confirmation datasets. No test files are created/opened. See `results/addressing/protocol.json` for hashes and the report for exposure limits.
