# MLM baseline and comprehension comparison

Date: 2026-10-23. Exploratory single-seed measurements.

The graph model outperformed its message-free ablation on this small English test set. A plain GRU with approximately the same allocated parameter count performed slightly better than the graph model. This does not support claiming that the spiderweb architecture beats a standard recurrent model.

| Model | Allocated parameters | Development loss | Story test loss | Test bits/byte |
|---|---:|---:|---:|---:|
| graph | 366,722 | 1.155704 | 1.121557 | 1.615753 |
| no-messages | 366,722 | 1.192971 | 1.157398 | 1.667643 |
| gru | 368,210 | 1.136525 | 1.098890 | 1.583409 |

Loss is next-byte cross-entropy in nats, including document EOS. Lower is better. All selected checkpoints were at update 4,000. Raw metrics and unedited generations are preserved in each model subdirectory.

## Protocol and limits

Same frozen TinyStories subset: 1,000 training / 32 development / 32 local test stories, complete-document processing, byte vocabulary, batch size four, chunk length 128, seed 42, 4,000 updates, learning rate 0.001, clipping, weight decay, and AdamW settings. Selection used development loss every 500 updates. Test loss did not control training. Corpus hashes, processed payload-byte counts, and final sampler snapshots match exactly.

The graph reference is the previously completed english-v1 run. The current loader reproduced its recorded development loss exactly: 1.1557040961972767. Historical graph checkpoints remain compatible. GRU checkpoints explicitly identify their architecture and have no graph topology.

Graph: eight GRUs, width 64, degree two, one communication round. Plain GRU: one cell of width 208 without message or pooling matrices. Parameter counts differ by about 0.4%, but recurrent state differs: 512 scalars per sequence for the graph versus 208 for the GRU.

Equal allocation is not equal effective capacity or compute. Message-free ablation retains at least 102,400 matrix weights disconnected from prediction loss: 4,096 message-projection weights and 98,304 message-input weights. Weight decay can still change those unused values. The ablation tests communication together with its effective parameter paths, rather than isolating topology at identical effective capacity.

Verification jobs overlapped some training runs, and the graph reference was collected earlier. Raw timings are retained, but no controlled latency, energy, or speedup conclusion is drawn. One seed does not estimate training variance; a multi-seed study and Transformer baseline remain outstanding.

## Paired reading results

Original evaluation-only diagnostic: 12 counterfactual pairs / 24 two-choice questions. Each pair changes a fact and correct answer while holding question and choices fixed. Frozen neural likelihoods rank candidates; expected-answer keys only grade results. No answer-key leakage, keyword solver, regex solver, or hardcoded inference response is used.

| Checkpoint | Format | Item accuracy | Both-in-pair accuracy | Changed answer within pair |
|---|---|---:|---:|---:|
| Graph English | plain | 50% | 0% | 0% |
| Message-free English | plain | 50% | 0% | 0% |
| Plain GRU English | plain | 50% | 0% | 0% |
| Dialogue-adapted graph | chat | 50% | 0% | 0% |

Every model kept the same preferred candidate within each pair. A fixed preference gets 50% individual accuracy on this balanced suite. These results do not demonstrate successful counterfactual reading; they do not prove hidden states contain no contextual information. This tiny authored suite is format-dependent, not a standardized general-understanding benchmark. The dialogue checkpoint has extra training and a different format, so its row is a separate diagnostic.

Final scorer version 2 places the appropriate separator inside the prompt and length-normalizes answer bytes only. Plain prompts end in `Answer: `; chat prompts end in the assistant-marker newline. Preliminary version-1 scores that included an extra space are retained under superseded-separator-scorer. Questions and answer keys were unchanged; their SHA-256 is recorded in every final artifact. There was no result-dependent question or prompt tuning.

## Reproduce

Training commands are in [README](../../usage.md). Diagnostic commands:

```sh
uv run python -m mlm.comprehension runs/english-v1/best.mlm --out results/graph-reading-new.json
uv run python -m mlm.comprehension runs/baseline-no-messages/best.mlm --out results/no-messages-reading-new.json
uv run python -m mlm.comprehension runs/baseline-gru/best.mlm --out results/gru-reading-new.json
uv run python -m mlm.comprehension runs/conversation-v1/best.mlm --format chat --out results/chat-reading-new.json
```

## Decision

Keep the graph as a research candidate and the plain GRU as a baseline. Establish multi-seed behavior and an assistant-answer training objective with fresh held-out reading tasks before adding growth. Trainable edge strengths belong in a separate controlled experiment. Rewiring, node growth, and Transformer replacement remain unproven.
