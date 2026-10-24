> Historical guide and experiment log. Run commands from the repository root. See the [release overview](../README.md) for current capabilities. Full checkpoints and run artifacts are local-only.

# User guide and experiment history

An experimental byte-level language model made of small GRU networks connected by a directed graph. JAX performs causal recurrent computation, Flax owns model parameters, and Optax trains them with clipped AdamW.

**Current state:** a tested fixed-graph CPU prototype with English training, matched GRU baselines, and controlled retrieval experiments. An optional learned node readout is implemented but has not solved entity binding. Dynamic growth, rewiring, continual parameter learning, and Transformer comparisons remain future experiments. This is not a general chatbot.

Read [the original idea and research](research.md), [requirements](prd.md), and [measured smoke results](experiments/smoke/report.md), [English/chat results](experiments/english-chat/report.md), and [baseline/reading comparison](experiments/comparison/report.md).

## Setup and checks

The tested environment is Linux x86-64 with Python 3.12 on CPU. uv selects the project Python; the system Python is not modified. GPU support has not been tested.

```sh
uv sync --locked
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run ty check src
```

## Train

```sh
uv run mlm train --train data/smoke/train.txt --validation data/smoke/validation.txt --config configs/smoke.json --out runs/my-first-mlm --steps 300 --eval-every 50
```

`--steps` means additional optimizer updates. Training uses independent document-prefix batches, with a fresh recurrent state per row. Documents are separated by blank lines. Documents longer than `length` are prefix-truncated; the summary reports truncation. This first trainer does not yet stream through all chunks of long documents. Choose appropriately short documents for now. Training/validation identical documents and identical scored prefixes are rejected, but this is not a full corpus decontamination system.

The included data is an original, deliberately simple fixture. It is not TinyStories. Validation is a development set used for checkpoint selection; use a separately prepared untouched test corpus for later research conclusions.

Outputs include:

- `best.mlm`: the best checkpoint among initial and periodic development evaluations.
- `latest.mlm`: latest resumable state, including optimizer state and sampling RNG.
- `metrics.jsonl`: periodically measured training/development metrics.
- `summary.json`: initial/final losses, best checkpoint, timing, memory, and truncation counts.
- `samples.json`: best-checkpoint prompts and greedy continuations.
- `config.json` and `environment.json`: settings, dependencies, backend, and source fingerprint.

Checkpoints use atomic single-file replacement, and state is committed before its metrics. Resume repairs a truncated final metric record while rejecting earlier corruption. Their stored topology is verified when loading. Runs are ignored by version control; small run evidence is preserved under results/.

## Generate and evaluate

```sh
uv run mlm generate runs/my-first-mlm/best.mlm --prompt 'The red ' --max-bytes 64
uv run mlm evaluate runs/my-first-mlm/best.mlm --data data/smoke/validation.txt
uv run mlm evaluate runs/my-first-mlm/best.mlm --data data/smoke/validation.txt --no-messages
```

Generation is greedy, stops at EOS, and changes recurrent state rather than weights. Display replaces invalid UTF-8 sequences. Disabling messages at evaluation tests dependence on graph communication; it is not a fair comparison against a separately trained graph-free baseline.

Loss and perplexity include structural targets where present. Bits/byte counts only payload bytes. Compare perplexities only with the same vocabulary and masking policy.

## Resume

```sh
uv run mlm train --train data/smoke/train.txt --validation data/smoke/validation.txt --out runs/my-first-mlm --resume runs/my-first-mlm/latest.mlm --steps 100
```

Resume restores configuration, parameters, Adam state, step count, and sampler RNG. Corpus identities must match. A nonempty output directory accepts only its own latest checkpoint and rejects newer history. To branch from a different or older checkpoint, use a new empty output directory. Best-checkpoint selection in a new branch starts with that branch's initial state.

## Implementation map

- `config.py`, `data.py`, `graph.py`: validation, byte batches, topology.
- `model.py`: independent GRUs and synchronous graph rounds inside a causal scan.
- `training.py`, `evaluate.py`: optimizer, masked loss, frozen-state evaluation.
- `checkpoint.py`, `generate.py`, `cli.py`: persistence and commands.
- `tests/`: causality, state isolation, NumPy reference, learnability, checkpoint/resume, split overlap, and output protection.

Next research milestone: isolate routing from memory timescales and pretraining mismatch, with retrieval and English retention measured separately. Multi-seed comparisons and a Transformer baseline remain outstanding.


## Train on real English and conversations

Prepare the bounded public datasets using [the data instructions](data/README.md). Then run:

```sh
uv run mlm stream-train --train data/hf/stories/train.jsonl --validation data/hf/stories/validation.jsonl --test data/hf/stories/test.jsonl --config configs/english.json --out runs/english-new --steps 4000 --eval-every 500
uv run mlm stream-train --train data/hf/mixed/train.jsonl --validation data/hf/mixed/validation.jsonl --test data/hf/mixed/test.jsonl --config configs/conversation.json --init runs/english-new/best.mlm --out runs/conversation-new --steps 2000 --eval-every 500
uv run mlm chat runs/conversation-new/best.mlm --message 'Hi!'
```

`stream-train` traverses entire documents. State carries between chunks and is detached for truncated backpropagation; padding freezes state and new documents reset it. Its checkpoints include recurrent carry, document order, per-lane positions, RNG, optimizer, and model state. `--resume` continues the same dataset/configuration; `--init` transfers parameters into a new stage with fresh optimizer and stream state. They are mutually exclusive. Changing datasets belongs to `--init`, not resume.

The conversation stage rehearses English stories. It learns all byte targets, including user turns, rather than assistant-only loss. `chat` is an experimental single-turn role wrapper, not evidence of instruction following or understanding. Inspect raw samples and held-out metrics before judging quality. `--no-messages` trains a graph-message ablation; the plain-GRU and message-ablation comparison is now recorded under results/comparison; a Transformer benchmark remains outstanding.

Evaluate whole documents explicitly:

```sh
uv run mlm evaluate runs/english-new/best.mlm --data data/hf/stories/test.jsonl --full-documents
```

The original `train` command retains its independent-prefix behavior for smoke reproduction. `evaluate` defaults to that historical prefix mode; use `--full-documents` for real-corpus runs. Development-selected test results are not used to update parameters or choose checkpoints. Repeated manual tuning on test results would invalidate that separation.


Current saved dialogue checkpoint: `runs/conversation-v1/best.mlm`. It learned a greeting but failed simple question probes. See the unedited results report; do not treat it as a capable assistant.


## Baselines and paired reading evaluation

Architecture selection is explicit in configuration: `architecture: "graph"` (default, including old checkpoints) or `architecture: "gru"`. The GRU is one recurrent cell with its own embedding/readout and no graph/message weights. For the GRU, `width` sets hidden size; graph-only `nodes`, `degree`, and `rounds` are ignored. `configs/gru-english.json` uses width 208, giving 368,210 parameters against the graph model's 366,722. A separate `--no-messages` graph ablation keeps the graph parameter arrays but disables neighbor information during both training and generation.

```sh
uv run mlm stream-train --train data/hf/stories/train.jsonl --validation data/hf/stories/validation.jsonl --test data/hf/stories/test.jsonl --config configs/english.json --out runs/no-messages-new --steps 4000 --eval-every 500 --no-messages
uv run mlm stream-train --train data/hf/stories/train.jsonl --validation data/hf/stories/validation.jsonl --test data/hf/stories/test.jsonl --config configs/gru-english.json --out runs/gru-new --steps 4000 --eval-every 500
uv run python -m mlm.comprehension runs/english-v1/best.mlm --out results/my-reading.json
```

The [paired reading suite](data/evaluation.md) is evaluation-only. Candidate answers are ranked using the frozen neural model's conditional byte likelihood; the answer key only grades results. It is not a keyword or regex answer generator. The paired contexts expose fixed-answer preferences that can score 50% on individual two-choice questions while getting every complete pair wrong. Chat-format diagnostic results should be reported separately from identically formatted English baselines.

Graph ablation retains at least 102,400 matrix parameters disconnected from data loss (the message projection and message-input half of GRU input matrices). Equal allocated parameter count therefore does not imply equal effective capacity. The plain-GRU comparison matches parameters approximately but has 208 recurrent-state scalars per sequence, versus 512 in the graph. Compare these as explicit architecture/resource trade-offs. One seed and these tiny subsets cannot establish general superiority.


The first matched English comparison is complete: story test losses were 1.122 for MLM, 1.157 without messages, and 1.099 for the parameter-matched plain GRU (lower is better). All tested checkpoints got zero fully correct pairs on the small reading diagnostic. These one-seed findings support further experiments rather than a superiority or comprehension claim. See [the full report](experiments/comparison/report.md).


## Assistant-only learning experiment

Assistant-only supervision and matched graph/GRU training are implemented. Both models completed 2,000 additional optimizer updates. Neither solved any of the 64 held-out counterfactual reading pairs; English story prediction also worsened. These checkpoints are experiments, not validated upgrades. See [results and reproduction commands](experiments/assistant-learning/report.md).

Prepare the offline supervised dataset from the existing public-data cache, then evaluate a saved checkpoint:

```sh
uv run python -m mlm.supervised_data --source data/hf --out data/assistant-v2
uv run python -m mlm.supervised_eval runs/assistant-graph-v1/best.mlm --out results/my-assistant-evaluation.json
```

Reuse existing prepared data instead of overwriting it. JSONL `loss_spans` are half-open UTF-8 byte offsets; `score_eos` controls EOS supervision. Input validity and supervision are separate: prompt bytes update recurrent state without contributing target loss. Full-text story rehearsal remains explicit. Use `stream-train` and `evaluate --full-documents` for supervised files; prefix commands reject them. Prompt-only batches advance memory without changing optimizer state or update count.

Streaming checkpoint selection follows global update multiples of `--eval-every`; off-grid endpoints save latest state but cannot replace best. Resume preserves supervision identity and the selection interval. Resuming into a fresh directory also restores the source best checkpoint; retain it alongside latest.

Synthetic labels are generated offline. Inference uses learned neural weights and recurrent state, with no label generator, regex answer solver, or response lookup. Names/templates are disjoint across the supervised reading splits; this does not guarantee those names were absent from English pretraining. Node growth, rewiring, and learned routing remain unimplemented.


## Retention experiment

An offline gate-bias intervention and reproducible memory probe are available. The graph now changes preferences with context, but individual reading accuracy remains around chance and English loss worsens. This is an experimental initialization change, not a validated architecture upgrade. See [protocol, results, and commands](experiments/memory-retention/report.md). Earlier English checkpoints remain the preferred references.


## Controlled retrieval curriculum

The latest experiment separates single-fact recall from selecting the queried person's fact. After 500 basic and 1,500 mixed binding/rehearsal updates, both graph and GRU answer all 96 single-fact test questions correctly across 0/64/128 filler bytes by candidate likelihood. Neither solves any of 96 complete multi-entity binding groups; multiple-fact accuracy remains around chance. These are narrow synthetic-task findings, not broad language understanding.

The [report](experiments/retrieval-curriculum/report.md) includes fresh-test checkpoint locks, all difficulty cells, raw greedy outputs, supervision exposure, and reproduction commands. English loss remains worse than the original English checkpoints, which are preserved.

```sh
uv run python -m mlm.retrieval_data --out data/retrieval-new --source data/hf
uv run python -m mlm.retrieval_eval runs/retrieval-binding-graph-v1/best.mlm --questions data/retrieval-v1/test-questions.json --out results/my-retrieval-evaluation.json
```

`retrieval_data` is offline dataset generation. `retrieval_eval` uses frozen neural answer likelihoods and evaluates context swaps plus fixed-context query swaps. Neither provides an inference-time answer solver. Tests after repeated tuning require fresh confirmation data.


## Focused binding and optional node readout

The latest [focused experiment](experiments/focused-binding/report.md) trains complete four-question groups with two people and no filler. Checkpoints are selected by complete development-group accuracy, with target loss as a tiebreaker. Graph, GRU, and a learned-readout graph each trained for 2,000 updates. Fresh-test complete groups: 0/56, 1/56, and 0/56. English prediction regressed, so these are not adopted upgrades.

The optional graph setting `node_readout: "attention"` adds 8,256 parameters and computes state-conditioned weights over current node states. Default `"sum"` preserves existing behavior and legacy checkpoints. This is learned neural pooling, not an explicit question parser, growing graph, or online parameter updates. Ordinary stream warm starts require the same readout setting; the experimental grouped runner can initialize new readout parameters while copying shared source tensors.

```sh
uv run python -m mlm.binding_data --out data/binding-new
uv run python -m mlm.binding_run --data data/binding-new --config configs/binding-graph.json --init runs/retrieval-basic-graph-v1/best.mlm --out runs/focused-binding-new --steps 2000 --eval-every 250
```

This dedicated runner starts a fresh optimizer and does not support resume. Use empty output paths. Every batch contains all four context/query swaps, with input validity separate from assistant loss masks. No story rehearsal is included in this diagnostic.

A separate post-hoc single-group sanity run achieves 4/4 correct rankings and raw greedy answers on its memorized training group. That demonstrates tiny-case learnability, not generalization. Next investigate gradual curriculum and optimization before expanding architecture. All 44 tests pass.


## Gradual binding comparison

The unchanged graph completed a 3,000-update curriculum and a matched flat-training control, both starting from the same memorized seed group. Curriculum fits 2/4/8/16 training groups completely, then 29/32 and 50/64. Both checkpoints were selected on development data and locked before testing.

Requiring every assignment/query swap and both sentence orders (eight questions per case), curriculum solves 18/24 familiar-name new-pair cases and 1/24 new-name cases. Flat solves 11/24 and 9/24. The source solves zero in either bucket. This is partial controlled-task transfer, not reliable chat or name-independent retrieval. A discovered early-role imbalance between familiar-name development and test pairs limits interpretation; see [the full report](experiments/gradual-binding/report.md). English prediction worsens severely, so no English/chat checkpoint is replaced.

```sh
uv run python -m mlm.gradual_data --out data/gradual-new
uv run python -m mlm.gradual_run --data data/gradual-new --config configs/gradual-graph.json --init runs/binding-overfit-sanity-v1/final.mlm --out runs/gradual-curriculum-new --strategy curriculum --stage-steps 500 --stories data/hf/stories/test.jsonl
uv run python -m mlm.gradual_eval runs/gradual-curriculum-new/best.mlm --questions data/gradual-new/test-questions.json --out results/gradual-curriculum-new.json
```

Use `--strategy flat` and a fresh output directory for the control. Finish and lock both development-selected models before testing. The standalone runner preserves optimizer state across its stages but does not implement resume. All 47 regression tests pass. Next: balanced confirmation splits, wider name diversity and an explicit English-retention strategy.


## Balanced confirmation and English rehearsal

The fresh [balanced study](experiments/balanced-confirmation/report.md) uses 32 training labels and balanced within/cross-role pairings. Both arms receive the same 2,000 QA updates from the preserved English checkpoint; rehearsal adds 2,000 English updates. All selected models score zero complete eight-condition cases on 96 familiar-name and 48 new-name test cases. This joint acquisition setting does not demonstrate binding. Source, learning rate and task size differ from the earlier toy-transfer experiment, so it is not a balancing-only replication.

Rehearsal preserves and slightly improves English test loss (1.121557 source to 1.087839). The unconstrained task-only candidate worsens it to 3.381755; its English-protected selection correctly falls back to source weights. Rehearsal's protected/unconstrained slots are the same trained model. Retention is a measured success; semantic binding is not. No general-chat model is promoted.

```sh
uv run python -m mlm.confirmation_data --out data/confirmation-new
uv run python -m mlm.confirmation_run --data data/confirmation-new --config configs/confirmation-graph.json --init runs/english-v1/best.mlm --out runs/confirmation-rehearsal-new --steps 2000 --eval-every 500 --rehearsal
```

`best-task.mlm` selects binding development accuracy without an English constraint. `best.mlm` additionally requires English development loss no more than 1.25 times the source baseline; step-zero fallbacks are explicitly recorded. This standalone runner has no resume feature. QA/English updates and bytes are counted separately; rehearsal costs extra compute. All 50 tests pass. Next diagnose source-controlled learning progression before more architecture changes.


## Hierarchical recurrent update prototype

A two-level learned router is implemented: four groups with two GRU nodes each. Config `node_update` selects `"dense"` (legacy default), `"uniform"` (fixed 1/N update rate), or `"hierarchical"` (learned group/child probabilities); `groups: 4` defines the hierarchy. The message graph and sum readout remain fixed. This is an unlabeled routing hierarchy, not trained verb/pronoun experts, a growing network, or a Transformer replacement.

The [three-arm pilot](experiments/hierarchical-routing/report.md) completes 1,000 QA + 1,000 English updates per model on fresh balanced data. All score 0/144 complete test cases. English test loss: dense 1.119035, uniform-slow 1.953038, hierarchical 1.473429. Learned routing improves on fixed slowing here, but does not beat the dense model or demonstrate binding. Initial slowing already changes the dense-trained source's behavior; this is a warm-start pilot, not matched pretraining from scratch.

```sh
uv run python -m mlm.confirmation_data --out data/hierarchy-new --seed 920231
uv run python -m mlm.hierarchy_run --data data/hierarchy-new --config configs/hierarchy-hierarchical.json --init runs/english-v1/best.mlm --out runs/hierarchy-new --steps 1000 --eval-every 250
```

Use the dedicated hierarchy runner for routing-balance auxiliary loss and usage diagnostics. Ordinary trainers support the forward model but do not add that auxiliary loss. `best.mlm` in this runner selects binding development score/QA loss without an English constraint. Standard stream warm starts require matching update modes; the experimental initializer copies shared source weights into variants. Soft routing computes every node, so no sparse speedup is claimed. All 54 tests pass. Source code, tests, config and lockfile snapshots are preserved with the report.


## From-scratch dense versus hierarchical study

The [new comparison](experiments/from-scratch/report.md) independently pretrains both architectures for 4,000 English updates, then adapts each own checkpoint on fresh balanced binding with English rehearsal. Shared initial tensors and data budgets are audited; selections are locked before tests. Separate sanity probes demonstrate small-set memorization, while the full report distinguishes it from held-out binding and English retention. CPU timing and tensor sizes are measured with explicit limits. All 56 tests pass. The model remains an experimental fixed recurrent graph; growing experts and Transformer replacement are unproven.


## Binding acquisition diagnostic

The [nested-set study](experiments/acquisition/report.md) tests independent 1/4/16/64-group task-only trials from each architecture's own English checkpoint. It reports training fit separately from held-out development transfer, actual subset name coverage and English forgetting. Paired source resets, subset IDs, sample traces and byte budgets are audited. These are exploratory fixed-endpoint trials, with no test-based selection or promoted chat model.


## Equal-exposure acquisition control

The [exposure study](experiments/exposure/report.md) gives each of 16 training groups 125 presentations in 2,000 task-only updates, matching the per-group exposure of the successful four-group trials. It preserves the architecture, source and learning rate; paired traces/bytes and source fingerprints are audited. Results distinguish training fit, eight-condition development transfer and English forgetting. This is a development-only diagnostic with no promoted model.


## Shared/node rates and rehearsal

Config supports optional shared_lr_scale, node_lr_scale and router_lr_scale (default 1). They scale completed AdamW updates, including decay; checkpoint loading restores the policy. The [six-arm study](experiments/plasticity/report.md) compares uniform high/low rates and slower shared representations, with and without English rehearsal. It reports binding acquisition, development transfer and English retention separately, and counts the extra rehearsal compute. Existing graph/default behavior is preserved; no chatbot is promoted.


## Pairing and wording transfer study

The [fresh transfer study](experiments/transfer/report.md) compares a fixed pool, varied pairings/colors, and the same episodes with varied wording. Three source-matched rehearsal trials are locked before scoring reserved test pairings, names and wording. Every arm covers all training labels; source, exposure differences, raw neural rankings/generations and complete eight-condition test outcomes are explicit. No architecture or default-checkpoint replacement is claimed.


## Learned associative memory pilot

Optional Config.memory="delta" adds a learned read-before-write key/value matrix to the graph; memory_keys controls its row count and memory_lr_scale its optimizer rate. Default memory="none" remains compatible with existing checkpoints. Matrix state is document-local, reset/frozen/carried with the recurrent state; it is not permanent learning during chat. The [paired pilot](experiments/associative/report.md) records source-preserving zero-gate initialization, fresh locked binding tests, English retention and the added parameter/state/runtime costs. All 68 regression tests pass.


## Memory addressing diagnostics

[Report](experiments/addressing/report.md): actual-forward tracing, exact signed overwrite-aware reads, projected region vectors and frozen readout ablation. Graph and delta memorize one training group; both fail the four-group fixed-budget probe. This is a training diagnostic, with unequal per-group exposure and no new test claims. New helpers live in `src/mlm/addressing.py`; the archived fixed-protocol recipe is `results/addressing/recipe.py` and expects fresh output directories. It refuses to overwrite prior probe runs. Projection postprocessing is `uv run python results/addressing/project_regions.py`.


## Four-group exposure control

[Report](experiments/addressing-exposure/report.md): at2,000QA+2,000English updates, delta fits all4 training groups while the graph control fits0/4. Disabling delta's frozen memory output removes that fitted success. Both still fail complete groups in the tiny reused development case. This is single-seed acquisition evidence, with extra memory parameters and additional total rehearsal compared with earlier probes. The frozen protocol contains exact reproduction commands; existing run directories must be preserved.


## Adaptation-seed replication

[Report](experiments/addressing-replication/report.md): the seed42 four-group delta fit did not repeat at predefined seeds43/44 (8/16 and9/16 training rankings;0/4 groups both). Graph controls score8/16 and0/4. All retain English performance but fail complete tiny development groups. This is fixed-source adaptation replication, not independent English pretraining. The new acquisition gate fails0/2; no model is promoted. Exact commands, frozen inputs, logs, checkpoints and endpoint audits are preserved.


## Initialization and training schedule control

[Report](experiments/addressing-factorial/report.md): in a selected2×2 table, both memory initializations42/43 acquire all4 training groups with combined schedule42; both fail all4 with schedule43. Two cells are new and two are reused historical runs. This is a conditional schedule lead, not generalization or a reversal of the failed replication gate. All203 input/artifact audits pass. Next separate QA ordering from English rehearsal document exposure while holding memory initialization fixed.


## Independent QA and English sampler controls

`mlm.plasticity_run` accepts `--qa-seed` (group ordering) and `--english-seed` (replay stream). Defaults remain config seed and config seed+1; config seed still controls initialization/subset selection. Actual seeds are recorded in protocol, checkpoint metadata and summary.

[QA/replay report](experiments/schedule-components/report.md): with fixed memory initialization42, replay43 acquires all4 training groups at QA42 and43; replay44 acquires0/4 (7/16 and8/16 answers). This selected historical/new grid points to English sampling sensitivity, while all complete development-group scores remain zero.75 regression tests and219 artifact/input checks pass. Future work should test interference rather than fix the replay seed.
