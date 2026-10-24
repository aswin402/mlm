# Retention-gate intervention

This experiment tests a targeted initialization change, not a new architecture. The fixed graph, state dimensions, learned equations, and parameter counts are unchanged. The offline initializer adds a bias toward slower retention, then normal gradient training can change all gates. Inference contains no response rules or answer lookup.

## Rationale and limits

For the implemented GRU convention, h' = z h + (1-z) candidate. At zero other preactivation, adding log(tau-1) gives z = 1-1/tau. We distribute nominal tau geometrically from 2 to 128, across eight graph nodes or 208 plain-GRU channels. We add these offsets to the pretrained biases, so actual input-dependent timescales are not fixed or guaranteed. The graph has repeated channels within each node; the GRU has a finer channel-level distribution. This is not an exactly identical prior distribution across architectures.

The motivation comes from [Tallec and Ollivier, Can recurrent neural networks warp time?](https://arxiv.org/abs/1804.11188). Our deterministic pretrained-bias intervention is chrono-inspired, not a faithful reproduction of that paper's initialization method. Large retention gates can impair writing and gradient learning. Longer-lived state differences need not encode useful facts or support correct answer selection.

## Fixed protocol

Both models start from their respective 4,000-update English checkpoints with fresh optimizers and the bias intervention. They reuse assistant-v2, seed 42, batch size four, unroll 256, learning rate 0.0005, and 2,000 optimizer updates. Development target loss selects among steps 0, 500, 1,000, 1,500, and 2,000. No test split is supplied to training. Existing unmodified assistant-learning runs are the controls. There is no timescale sweep. File hashes are in protocol.json and source provenance accompanies each initialization.

The supervised dataset and old diagnostic have already informed project development. Reused test questions are therefore exploratory diagnostics; a positive result would need fresh held-out confirmation. One seed cannot establish a general advantage. Runtime contention prevents a controlled speed comparison.

The memory probe substitutes equal-byte-length green/white context facts while holding question and neutral filler fixed. It reports final recurrent-state separation and the change in green-versus-white likelihood margin. State separation is not a semantic-memory score. Positive margin effect is directionally appropriate for this single fact, but does not establish broad reading capability.

## Reproduction

Use fresh output paths; source checkpoints and prepared data must already exist.

```sh
uv run python -m mlm.memory_init runs/english-v1/best.mlm runs/memory-init-graph-new/initial.mlm
uv run mlm stream-train --train data/assistant-v2/train.jsonl --validation data/assistant-v2/validation.jsonl --config configs/assistant-graph.json --init runs/memory-init-graph-new/initial.mlm --out runs/memory-graph-new --steps 2000 --eval-every 500
uv run python -m mlm.supervised_eval runs/memory-graph-new/best.mlm --out results/memory-graph-new-evaluation.json
uv run python -m mlm.memory_probe runs/memory-graph-new/best.mlm --out results/memory-graph-new-probe.json
uv run mlm evaluate runs/memory-graph-new/best.mlm --data data/hf/stories/test.jsonl --full-documents
```

For the GRU substitute runs/baseline-gru/best.mlm, configs/assistant-gru.json, and fresh GRU output names. Initialization never modifies its source. Preserve the generated provenance JSON alongside the initial checkpoint.

## Results

Both models completed 2,000 updates and consumed exactly the controls' 915,785 input payload bytes and 82,737 supervised payload bytes. Final stream snapshots, training/development text identities, and supervision masks match the controls. Source/data/config hashes were verified. Graph selection chose step 2,000; GRU chose step 500. Later GRU development loss worsened; final weights were not substituted for best.

| Measure | Graph control | Graph retention | GRU control | GRU retention |
|---|---:|---:|---:|---:|
| Best development target loss | 0.268223 | 0.278613 | 0.386889 | 0.401177 |
| Test target loss | 0.234342 | 0.230059 | 0.244349 | 0.281644 |
| Individual reading correct /128 | 64 | 66 | 64 | 62 |
| Complete reading pairs correct /64 | 0 | 19 | 0 | 2 |
| Preferred-answer changes /64 | 0 | 36 | 0 | 6 |
| Story test loss | 1.267662 | 1.849269 | 1.169831 | 1.855628 |

Lower losses are better. Target loss scores assistant spans and EOS, not all input bytes. The original English checkpoints' story losses were 1.121557 (graph) and 1.098890 (GRU). Neither new memory checkpoint should replace those English checkpoints.

The graph is now context-sensitive, but **reliable reading has not been demonstrated**. Of 64 pairs, 19 are both correct, 17 are both wrong, and 28 have one correct answer. The GRU has two both correct, four both wrong, and 58 with one correct. Individual accuracy remains around 50%. Independent fair random guesses would average 25% complete-pair accuracy; a fixed answer averages 50% individual and zero complete pairs on these balanced swaps. Thus 29.7% graph pair accuracy is not evidence of above-chance semantic understanding. Changes from fixed preferences must not be advertised as broad comprehension improvement.

On the original reused 24-question diagnostic, graph scores 15/24 individual and 3/12 complete pairs; GRU 14/24 and 2/12. These tiny previously inspected sets cannot establish generalization. On the fixed first 16 raw greedy probes, graph scores 6/16 and GRU 5/16. Unedited outputs remain in the evaluation JSON files; there is no answer extraction or repair.

## Memory observation

The retention initialization alone increases state separation for equal-length fact substitutions. After adaptation, graph relative state separation is 0.7602 with zero filler bytes and 0.05631 with 288 filler bytes, versus 0.002621 and about 2e-7 in the graph control. GRU shows a similar direction of change. This is numerical sensitivity, not a guarantee that the relevant entity/fact can be retrieved.

At zero and 36 filler bytes, the trained graph and GRU prefer the correct candidate in both green/white contexts in this one familiar-template probe. At 144 and 288 bytes, both lose at least one of the two correct preferences, despite nonzero state separation. The graph answer-margin context effect falls from 1.2822 to 0.02062 nats per answer byte by 288 filler bytes. This supports further work on retention and retrieval, without identifying a unique cause of broader reading failures.

## Decision and verification

Keep the intervention as an optional research tool. Do not promote these checkpoints to the default English/chat model. A next experiment should separately test learning to bind a fact to the queried entity using simple controlled retrieval tasks, then increase distractors and delay, while tracking English retention. Freeze new validation/test splits before further tuning; fresh confirmation is required. Growth is premature until useful retrieval has a measurable baseline.

All 36 tests passed in the full regression run. After strengthening metadata overwrite protection, the three focused initialization tests passed again; lint, formatting, and type checks also passed. Neural inference equations and parameter counts are unchanged. `memory_init.py` is offline only; `memory_probe.py` and answer labels are evaluation only.
