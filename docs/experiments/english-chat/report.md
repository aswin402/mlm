# First real-English and conversation experiments

Date: 2026-10-23. Status: completed bounded CPU experiments, not a capable-chatbot release.

## Data and training

Source data: [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) and [Everyday Conversations](https://huggingface.co/datasets/HuggingFaceTB/everyday-conversations-llama3.1-2k). Source attribution, licenses, byte hashes, and selection limits are in [data instructions](../../data/README.md) and download provenance (`results/datasets.json`; local artifact).

English stage: 1,000 training stories, 32 development stories, and 32 local test stories. Conversation stage: 300 conversations plus 250 rehearsed training stories; development and test each include 32 stories and 32 conversations. All documents are traversed completely, preserving paragraphs and conversation boundaries. These are bounded first-row subsets, not representative random samples of the complete datasets. No model weights or API-generated responses were imported; MLM trained its own byte prediction weights from scratch, followed by parameter-only initialization for stage two.

Architecture: eight independent GRU nodes, width 64, degree two, one synchronous message round per byte, **366,722 parameters**. Batch size four, chunk length 128. State carries across chunks; gradients are detached between chunks. Padding does not update state. Parameters remain fixed during inference; structural growth is disabled.

English training: 4,000 AdamW updates at learning rate 0.001. Dialogue adaptation: 2,000 updates at 0.0003, with a new optimizer and stream initialized from the English best parameters. The dialogue objective predicts all turns; it is not assistant-only supervised loss. Each stage uses seed 42. No hyperparameter search or multi-seed study was performed.

## Measured results

Loss is next-byte cross-entropy in nats, including EOS. Lower is better. Development data selects checkpoints; final local test data does not control training.

| Metric | Before stage | Selected checkpoint |
|---|---:|---:|
| English development loss | 6.007100 | 1.155704 |
| Mixed development loss during adaptation | 2.024821 | 1.222799 |

| Local test domain | English checkpoint | Adapted checkpoint |
|---|---:|---:|
| Stories: loss | 1.121557 | 1.108754 |
| Conversations: loss | 2.784249 | 1.361218 |
| Stories: bits/byte | 1.615753 | 1.596149 |
| Conversations: bits/byte | 4.012230 | 1.962674 |

Story prediction did not degrade in this particular rehearsal experiment. This does not establish that forgetting is solved: there is no no-rehearsal control, new-domain growth experiment, or multi-seed estimate. The same small local story test is used before/after adaptation for a planned retention comparison, rather than as a sequence of tuning signals.

## Actual chat behavior

The trained role wrapper answers `Hi!` with `Hello! How can I help you today?`. Raw generation then attempts to invent a following user turn. The single-turn CLI cuts at a new user marker; raw samples retain everything for inspection.

The probes `What is your name?`, `What color is the sky?`, and `A cat says meow. What animal says meow?` produced repetitive, irrelevant continuations. **The model does not yet demonstrate reliable question answering or comprehension.** It learned useful byte/word patterns and a greeting format. A low next-byte loss is not sufficient evidence of understanding.

See unedited probes (`results/english-chat/chat-probes.json`; local artifact), English generations (`results/english-chat/english/samples.json`; local artifact), and adapted generations (`results/english-chat/conversation/samples.json`; local artifact). These are qualitative observations, not a standardized benchmark. Greedy decoding, no repetition penalties, at most 160 continuation bytes. No favorable samples were selected from repeated stochastic draws.

## Compute and artifacts

English training loop: 290.6 seconds, with 276.4 synchronized update seconds and 1,916,320 payload bytes processed. Dialogue loop: 171.4 seconds, with 157.4 update seconds and 946,437 payload bytes. Compilation, initial/final evaluations, and sample generation are outside loop timing. Timing includes contention from verification processes and is not a controlled performance benchmark.

Local checkpoints: `runs/english-v1/best.mlm` and `runs/conversation-v1/best.mlm`, each with a separate latest checkpoint for exact resume. JSON metrics, source fingerprints, configs, and samples are preserved here. The checkpoint contains sampler RNG/order/offsets, recurrent carry, optimizer moments, topology, and parameters.

## Reproduce and use

See [README](../../usage.md) for data preparation and exact commands. Try the already trained model:

```sh
uv run mlm chat runs/conversation-v1/best.mlm --message 'Hi!'
uv run mlm generate runs/english-v1/best.mlm --prompt 'Once upon a time' --max-bytes 160
uv run mlm evaluate runs/conversation-v1/best.mlm --data data/hf/conversation/test.jsonl --full-documents
```

## Next decisions

Run matched plain-GRU, graph-without-messages, and Transformer baselines before crediting graph topology. Then test larger parameter/data budgets, assistant-only dialogue loss, and a defined comprehension/QA evaluation. A larger or longer run should be justified by those experiments, not by assuming that more data automatically fixes the architecture. Growth and rewiring remain separate research stages.
