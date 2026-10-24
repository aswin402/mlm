# Hierarchical recurrent update routing

This is a two-level learned update-routing prototype inside the existing eight-node recurrent graph, not a completed growing or grammar-labelled architecture. Four router groups contain two independent GRU subnetworks each. There are no assigned verb/pronoun experts, symbolic question parser, inference answer rules, learned edges or node growth. A source snapshot is preserved under code/ with hashes for reproducibility as the workspace evolves.

## Mechanism

At each byte, concatenate its learned embedding with the mean previous node state. A root softmax predicts four group probabilities; group-specific child softmaxes predict probabilities for the two nodes inside each group. Their product p_i is a normalized distribution over eight recurrent nodes. Within each synchronous GRU round:

h_i(new) = h_i(old) + p_i * (GRU_i(input, messages, h_i(old)) - h_i(old)).

Gates use only current/past input and respect document resets. Padding preserves state and contributes no routing mass. Soft probabilities normally give every node a partial update; unselected states are not literally frozen as they would be under a hard sparse gate. This prototype therefore tests learned multirate memory updates, not actual sparse execution.

The bounded directed message graph and sum readout remain unchanged. Grouping is an organizational routing hierarchy; internal/cross-group rewiring or separate parent memories are not added. Each child is itself a small neural network with its own GRU parameters. Category specialization would require subsequent probing/data and is not demonstrated by module names or routing probabilities.

Hierarchical routing is established prior art in [sparsely gated/hierarchical MoE research](https://arxiv.org/html/1701.06538v1). This particular recurrent state-update adaptation is a hypothesis to test, not a claim that nesting experts is a newly discovered neural principle.

## Controls and training

Three arms share the same eight width-64 GRUs, 512-float recurrent state, source English weights, data, seed and budget:

- Dense: every node uses its ordinary GRU update (nominal gate mass eight).
- Uniform slow: every node blends at p_i=1/8 (nominal gate mass one).
- Hierarchical: learned group/child probabilities, also total gate mass one.

The fixed-slow arm isolates slowing memory from learning a router. Both router layers start at zero, so hierarchical and uniform models initially match exactly after shared-weight transfer. They do not initially reproduce the dense English model's behavior: changing the update timescale changes the recurrent computation. Initial English/QA losses must therefore be reported, not treated as equivalent initial predictions.

Hierarchical adds 1,548 parameters (368,270 total); controls each have 366,722. The parameter/state distinction is explicit; this is not an exactly parameter-matched test. Existing defaults remain dense and legacy checkpoints load. Ordinary stream warm starts reject update-mode mismatch, while the experimental initializer copies shared tensors and initializes the new router parameters.

The hierarchical training objective adds 0.01*(N*sum(mean_route_probability_i^2)-1), with valid input positions only and zero penalty for empty batches. This encourages balanced usage; it does not guarantee specialization or correct routing. Reported language loss and checkpoint tiebreaks exclude the auxiliary penalty. Usage/entropy diagnostics are accumulated over training positions; dense/uniform reference probabilities are uniform, while their actual nominal update masses differ.

Fresh role-balanced data use generator seed 920231, separate from the previous test snapshot. There are 32 training labels, 16 new development and 16 new test labels; within/cross-role pair counts are balanced. Each arm starts from runs/english-v1/best.mlm with fresh optimizer, seed42, LR0.001, batch4, unroll128. Each receives 1,000 QA-group updates and 1,000 explicit English streaming updates, with independent recurrent carry for English. Model selection at initial and every250 QA cycles maximizes macro eight-condition development correctness across familiar/new-name pools, then minimizes pure QA loss; ties keep earlier weights. English development is monitored, not a feasibility constraint in this experiment. best.mlm here is therefore not the English-protected policy of the previous runner.

Data/config/source/model/runner hashes are frozen before training. All three selected hashes must be locked before the fresh task and English tests are scored. There is no sweep or Transformer baseline. These models still compute every node; no sparse speedup follows. Runtime under shared CPU contention is not a controlled benchmark.

## Reproduction

Use fresh paths. Preserve the source checkpoint and cached English data.

```sh
uv run python -m mlm.confirmation_data --out data/hierarchy-new --seed 920231
uv run python -m mlm.hierarchy_run --data data/hierarchy-new --config configs/hierarchy-hierarchical.json --init runs/english-v1/best.mlm --out runs/hierarchy-new --steps 1000 --eval-every 250
uv run python -m mlm.gradual_eval runs/hierarchy-new/best.mlm --questions data/hierarchy-new/test-questions.json --out results/hierarchy-new-test.json
```

Repeat training with hierarchy-dense.json and hierarchy-uniform.json under fresh output names, then lock all selected checkpoints before testing. The runner has no resume feature. The preserved source snapshot can be placed on PYTHONPATH via code/src when reproducing this exact implementation. Current experiment files and hashes identify which code/config/data were used.

## Transformer replacement standard

The [Transformer paper](https://arxiv.org/abs/1706.03762) establishes an attention-based architecture with empirical sequence-task results. This prototype has not been compared directly with a Transformer under matched data and resource budgets. A replacement claim would need competitive held-out language quality and retrieval, controlled training/inference cost, memory/device measurements, longer-context tests and scaling across seeds/model sizes. Small synthetic successes, parameter counts, or a visually brain-like hierarchy do not meet that standard.

## Measured results

All three arms completed 1,000 QA plus 1,000 English updates. Each consumed exactly 388,000 QA input bytes / 20,000 supervised QA payload bytes, and 479,511 English input/supervised payload bytes. QA/English sampler traces match; only state-update mechanism and router parameters/objective differ. Source/config/data/runner hashes were verified and all three selected checkpoint hashes locked before test.

All development primary binding scores remain zero, so the pure QA-loss tiebreak selects dense QA step1,000, uniform step750, and hierarchical step1,000. The uniform test result is therefore for its earlier selected checkpoint, not its final state.

| Variant | Parameters | Selected QA step | Familiar-name cases /96 | New-name cases /48 | English test loss | QA test loss |
|---|---:|---:|---:|---:|---:|---:|
| Dense updates | 366,722 | 1,000 | 0 | 0 | 1.119035 | 0.235698 |
| Fixed slow updates | 366,722 | 750 | 0 | 0 | 1.953038 | 0.174190 |
| Hierarchical learned updates | 368,270 | 1,000 | 0 | 0 | 1.473429 | 0.159689 |

Each case requires all eight assignment/query/sentence-order predictions correct. All within/cross-role test strata also have zero complete cases. Individual two-choice accuracy is 49.74%, 50.52%, and 50.87%. Lower QA byte loss does not demonstrate binding: formatting and teacher-forced word completion can improve without reliable query-dependent retrieval.

Learned routing improves English prediction relative to the fixed-slow control in this pilot, but remains worse than dense updates. There is no demonstrated binding benefit. These are one-seed selected-checkpoint results, not statistical or scaling conclusions. The unchanged source English reference has story-test loss 1.121557; the repeated story set is a retention monitor, not a fresh broad language benchmark.

Raw greedy samples are stored separately for the fixed first full group from each of two pools (eight samples per model), with whitespace/case normalization only. They are not full-test generation scores. All item likelihoods and errors remain in the JSON evidence.

## Router activity and limits

Both router layers changed from zero: root-kernel L2 norm is about 5.83, child-kernel 5.28, with nonzero biases. The actual soft router trained, rather than staying a fixed uniform mask. Cumulative QA mean usage is approximately [0.088, 0.096, 0.187, 0.206, 0.100, 0.094, 0.122, 0.107]. English usage is approximately [0.043, 0.038, 0.104, 0.542, 0.037, 0.031, 0.143, 0.061].

This shows input/domain-dependent utilization, not verified linguistic expertise. English traffic concentrates in node3 despite the auxiliary balance penalty. Balanced usage is encouraged, not guaranteed; entropy and mean batch penalty are saved to make this visible. No verbs/pronouns/nouns categories were assigned or demonstrated.

Algebraically, if the original GRU retention gate is z_i, the outer update gate gives an effective retention coefficient 1-p_i*(1-z_i). This parameterizes memory timescales and couples node updates through a global learned router. It is not evidence of an entirely new learning principle. The router uses a global pooled previous state, so disabling explicit neighbor messages would still leave this global routing path; a future message ablation must account for it.

The controls isolate dense versus uniformly slowed versus learned slowed updates, but do not isolate hierarchical factorization from an ordinary flat learned gate. A flat learned-router control remains outstanding before claiming that hierarchy specifically adds value. The hierarchical arm also has extra parameters and a balancing auxiliary; their contributions are not independently separated.

A major initialization limitation is that all shared weights came from a dense-trained English model. Uniform/hierarchical initial English development loss is 3.770138, versus dense 1.155704. Hierarchical and uniform match each other initially exactly, but neither is function-equivalent to the dense source. This is a warm-start adaptation pilot, not a fair from-scratch assessment of each architecture's best language-model potential. Adequate matched pretraining would be needed for that stronger comparison.

## Decision

The first two-level routing implementation is functional and testable, but it has not solved the current learning failure. It is not adopted as the default English/chat model. Default dense behavior and earlier checkpoints remain preserved. There is no node growth, arbitrary-depth category tree, sparse kernel dispatch, online weight learning, or Transformer-replacement evidence.

All 54 tests passed, plus Ruff and type checks. Tests include legacy behavior, normalized/factored routes, fixed-slow equivalence, exact blend math, nonzero router gradients, context dependence, causality, resets, padding, chunk state, checkpoint replay, empty-batch optimizer safety, and an end-to-end routed training run. Source snapshots and hashes preserve this implementation for future review.

Further work should distinguish learned flat gating from hierarchical gating, address pretraining mismatch, and demonstrate actual retrieval before increasing depth or expert count. Hard routing/dispatch and Transformer baselines require separate quality/cost experiments. A model hierarchy that resembles a brain or web is an engineering hypothesis, not proof of intelligence or an automatic successor to Transformers.
