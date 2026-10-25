# Mini Learning Model (MLM): Research and Feasibility

Date: 2026-10-23
Status: Initial research synthesis and proposed experimental direction.
Evidence: Public papers and official project documentation consulted during the project discussion. No architecture has been implemented or benchmarked in this workspace.
Companion document: [Product requirements](prd.md).

## 1. Preserve the original idea

The founder's idea is a new language-model architecture named **Mini Learning Model**: small neural networks connected by nodes and edges like a spiderweb. A dot can itself contain a smaller web of computational units. Nodes and connections can grow, change, and reorganize rather than remaining a permanently fixed structure. The model should eventually adapt during use and aim for smaller size, lower computation, faster execution, and broad device accessibility.

The ambition is to investigate an alternative to Transformers, not merely rename a small Transformer. The spiderweb metaphor is the starting intuition; it is not yet a mathematical architecture or evidence of better performance.

Preserve these essential ideas:

- Small neural modules as the computational building blocks.
- Sparse communication between modules.
- Potential hierarchy: modules may contain internal subnetworks.
- Learned selection of useful computation.
- Adaptive connectivity and bounded capacity growth.
- Investigation of learning or memory updates during operation.
- Practical efficiency measured on actual hardware.

The initial prototype will simplify this vision deliberately. A fixed graph comes first; growth, hierarchy, and continual adaptation remain explicit later research stages rather than forgotten requirements.

Terminology: MLM here means Mini Learning Model, not the established abbreviation for masked language modeling. The initial objective is causal next-token prediction.

## 2. Assessment

A research prototype is technically feasible. Existing work supports individual mechanisms such as sparse rewiring, capacity expansion, recurrence, and neural memory updates during inference. The reviewed evidence does not establish that their combination will outperform Transformers or run at useful quality and speed on every device.

Confidence assessment:

| Claim | Assessment |
|---|---|
| Small neural modules can communicate through a graph | Established computational approach; our specific implementation is untested |
| Connectivity can change during training | Supported by dynamic sparse training research |
| Capacity can expand across tasks | Supported in continual-learning settings |
| A language model can update neural memory during inference | Supported by TTT and Titans research |
| Our proposed graph will improve language quality per unit of compute | Open hypothesis |
| Arbitrary sparsity will produce lower latency | Not established; depends on kernels and hardware |
| The exact proposed combination is novel | Undetermined; requires a more systematic novelty review |
| One implementation can deliver useful operation on every device | Not a credible initial requirement |

## 3. Separate the forms of adaptation

1. **State adaptation:** recurrent activations change with each token; model parameters need not change.
2. **Routing adaptation:** different tokens activate different modules or edges.
3. **Topology adaptation:** connections are added, removed, or redirected.
4. **Parameter adaptation:** trainable weights change through learning.
5. **Capacity adaptation:** additional trainable modules or parameters are allocated.
6. **Memory adaptation:** a dedicated state or learned memory is updated during a sequence or session.

These must have separate switches and measurements. An animated graph or changing hidden state is not proof of structural growth. Ordinary LLMs update weights during training and generally keep them fixed during inference; they are not inherently incapable of adaptation.

## 4. Related research and projects

### 4.1 Dynamic sparse connectivity: RigL

[Rigging the Lottery: Making All Tickets Winners](https://arxiv.org/abs/1911.11134), with [official code](https://github.com/google-research/rigl), learns connectivity by removing and adding connections during training using weight and gradient information.

Relevance: supports changing a sparse network's wiring rather than assuming fixed wiring is optimal. Its principal setup maintains a fixed parameter budget; it does not establish the benefits of unlimited growth or our graph-of-modules design. Gradient-based growth scoring also has a computational cost that must be counted.

### 4.2 Capacity growth: Dynamically Expandable Networks

[Lifelong Learning with Dynamically Expandable Networks](https://arxiv.org/abs/1708.01547) selectively retrains and expands a network across tasks, including mechanisms intended to limit semantic drift.

Relevance: growth is a legitimate continual-learning mechanism. Limitation: task-sequence results cannot be assumed to transfer to open-ended language modeling or autonomous growth on arbitrary user text.

### 4.3 Local growth: neural cellular automata

[Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/) demonstrates learned local update rules that grow and regenerate patterns from seeds.

Relevance: a useful conceptual connection to local communication, shared rules, and self-organization. Limitation: growth of active cell states is different from creating new trainable capacity. Pattern regeneration is not evidence of language understanding.

### 4.4 Neural memory: Test-Time Training

[Learning to (Learn at Test Time): RNNs with Expressive Hidden States](https://arxiv.org/abs/2407.04620), with [official PyTorch implementation](https://github.com/test-time-training/ttt-lm-pytorch), treats recurrent hidden state as a small learning model updated on test sequences.

Relevance: particularly close to small networks acting as adaptive memory. Limitation: neural memory introduces computation, state, and optimization complexity; its benefits must be demonstrated separately from topology growth.

### 4.5 Neural memory: Titans

[Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663) investigates neural long-term memory that learns during inference.

Relevance: informs a later memory subsystem. Limitation: adaptive memory does not by itself imply a growing graph, permanent factual learning, or low inference cost.

### 4.6 Nonstandard sequence backbones: Mamba and RWKV

[Mamba](https://arxiv.org/abs/2312.00752), with [official implementation](https://github.com/state-spaces/mamba), investigates selective state-space sequence modeling. [RWKV](https://arxiv.org/abs/2305.13048) investigates recurrent language modeling with parallelizable training formulations.

Relevance: language modeling need not rely on standard full self-attention. Their results are architecture- and setting-specific; reported speedups are not predictions for MLM. Recurrent formulations compress past context into state, creating capacity and retrieval trade-offs.

### 4.7 Small language experiments: TinyStories

[TinyStories](https://arxiv.org/abs/2305.07759) demonstrates coherent story generation in a restricted distribution with models below 10 million parameters.

Relevance: an accessible first language-learning experiment. Limitation: coherent simple stories do not establish broad reasoning, factual knowledge, multilingual competence, or general assistant quality.

## 5. Candidate approaches

| Approach | Strength | Main risk | Position |
|---|---|---|---|
| Sparse recurrent graph of small modules | Direct test of the founder's concept | Routing, optimization, and irregular execution overhead | Recommended research direction |
| Established recurrent backbone plus adaptive memory | Stronger reference point and fewer simultaneous unknowns | Less architectural departure | Baseline or fallback |
| Cellular-automata language model | Strong local self-organization emphasis | Most speculative language-learning and execution behavior | Later exploratory branch |

A graph is not automatically attention-free. If every module scores every other module each token, routing may reintroduce expensive all-pairs computation. Candidate-neighbor sets and bounded degree should be the initial assumption.

## 6. Proposed architecture, not a validated design

### 6.1 Data flow

Token IDs → embeddings → input injection → recurrent graph updates → graph readout → vocabulary logits → next-token loss.

The graph persists within a sequence. Its state resets at independent sequence boundaries by default. Reading position t must never depend on token t+1 or later. Prediction and state-update ordering must be documented explicitly.

### 6.2 Modules and graph

Each module contains a small gated update network and a state vector. Nodes initially share a width so their execution can be batched. A bounded number of neighbor edges carries messages; an explicit readout produces token predictions. Begin with a flat graph. Nested subnetworks and multiple graph levels are later ablations.

An illustrative update is:

m_i = aggregate over selected neighbors j of gate(i,j,input) * message(h_j)

h_i_next = gated_update(h_i, m_i, input)

This is a conceptual family, not the final equation. Choose normalization, parameter sharing, update order, and readout through small controlled experiments. Synchronous updates must read previous-step states so execution order does not silently change the model.

### 6.3 Routing

First run all modules in the small fixed graph. Add input-dependent routing later. A hard top-k operation does not provide ordinary gradients through the discrete selection itself; a later design must specify a training method and compare its cost. Track module usage and collapse onto a few modules.

### 6.4 Growth and rewiring

Run the structural controller between training intervals, not in every token computation initially. Consider removing persistently unused edges, adding candidate connections, or activating modules after sustained evidence of a capacity bottleneck. Loss spikes alone are insufficient evidence: they may indicate bad data, optimization problems, or domain shift.

Every structural event needs a reason, resource-budget check, initialization rule, optimizer-state update, graph-invariant check, and checkpoint. Growth must be compared against a model with the same final capacity from the beginning.

### 6.5 Memory and online learning

Separate ordinary recurrent state from optional fast-weight memory and durable parameter updates. Default evaluation keeps learned model parameters frozen. Later test-time learning must specify which variables change, available supervision, learning rate, reset boundaries, retention tests, and rollback behavior. Never use hidden evaluation labels for adaptation.

## 7. Stack decision record

The user accepted the stack discussion and requested project documents. The last recommended default was JAX + Flax; no versions or target hardware have been selected. Record this as the **working proposal**, not an explicit user choice between every framework.

| Component | Proposed choice | Reason / boundary |
|---|---|---|
| Training and numerical research | JAX + Flax | Differentiation, compiled tensor computation, neural modules |
| Optimizer | Optax | Proposed complement; verify compatibility when pinning versions |
| Environment and dependencies | uv | Reproducible Python project management |
| Lint / format | Ruff | Python code quality |
| Static types | ty | Working preference; Pyrefly remains an alternative |
| Configuration validation | Pydantic | Validate settings and budgets outside computation loops |
| Data analysis | Polars when needed | Corpus metadata and experiment tables |
| API | FastAPI later | Only after local inference is useful |
| Serialization | Default sufficient implementation first | Add msgspec or orjson only for a measured need |
| Rust integration | PyO3 + maturin if needed | Optional native extension packaging |
| Custom kernels | JAX-native path first; Mojo optional | Profile and validate before adding another language |

### 7.1 JAX and growth

[JAX JIT documentation](https://docs.jax.dev/en/latest/201/jit.html) explains specialization and recompilation when shapes or static values change. Use fixed-shaped edge arrays, parameter pools, states, and active masks within a capacity stage. Updating edge-index values can express rewiring without changing array shapes. Increase physical capacity at explicit checkpoints, accepting recompilation and optimizer-state migration.

Logical activation of reserved slots is not physical allocation growth. Allocated parameters, trainable parameters, active parameters, serialized size, and executed work must be reported separately. A dense calculation followed by masking does not constitute sparse execution.

[Flax NNX documentation](https://flax.readthedocs.io/en/latest/guides/pytree.html) describes module and state integration. A Flax object graph is a software abstraction, not automatically the learned spiderweb topology.

### 7.2 Framework alternatives

- [Burn](https://github.com/tracel-ai/burn): strongest considered Rust-first alternative for training and inference; validate required operations on the intended backend.
- [Candle](https://github.com/huggingface/candle): supports training and custom operations, with useful Rust deployment examples; proposed primarily for a later runtime evaluation.
- [tinygrad](https://docs.tinygrad.org/): attractive for framework-level experimentation; adds the risk of dividing attention between model and framework engineering.
- PyTorch: remains a practical eager-execution fallback if structural experiments prove cumbersome in JAX.

Choose one primary training implementation. Porting a novel graph requires behavior and state semantics, not just loading its weights.

### 7.3 Mojo

Mojo is allowed and remains an option. [Its documentation](https://docs.modular.com/mojo/manual/) describes Python interoperability; [binding limitations](https://docs.modular.com/mojo/manual/python/mojo-from-python/) and [hardware requirements](https://docs.modular.com/mojo/requirements/) must be checked against the pinned release and target device.

A Mojo operation used for training needs correct gradients and integration with the host framework. Python importability is not sufficient for compilation, autodiff, or zero-copy device access. [JAX FFI](https://docs.jax.dev/en/latest/ffi.html) is a possible integration mechanism, not proof of a ready-made Mojo backend. Compare any kernel with a reference implementation and include call, synchronization, and transfer costs.

## 8. Experimental program

### Stage A: Establish a trustworthy evaluation pipeline

Select one target machine and record CPU, GPU, RAM, operating system, framework versions, dtype, and power settings where available. Establish deterministic data splits and a common tokenizer. Train a small Transformer and a gated recurrent baseline with equivalent preprocessing.

### Stage B: Fixed graph

Use an exploratory total parameter range of about 1–10 million, adjustable to hardware. Verify causal correctness, finite gradients, checkpoint resume, state reset, and ability to overfit a tiny training subset. Then evaluate held-out data. This range is a proposal, not a proven optimum.

### Stage C: Routing and rewiring

Introduce routing separately from rewiring. Compare each against the fixed graph and account for controller and scoring costs. Keep a random-rewiring control where feasible to distinguish learned selection from generic perturbation.

### Stage D: Bounded growth

Compare against fixed models with the same initial capacity, same final capacity, and approximately matched total training compute. Include a reserved-but-inactive pool control to distinguish new physical capacity from activating preallocated parameters.

### Stage E: Continual adaptation

Use a documented domain sequence. Evaluate new-domain learning and old-domain retention after each stage. Compare memory-only adaptation, parameter updates, growth-only, and their combinations. Distinguish session-specific adaptation from durable learning.

### Stage F: Runtime optimization

Only after correctness and a repeatable learning result, optimize the measured bottleneck. Evaluate CPU inference first on the selected device, then other devices individually. Custom Mojo or Rust kernels are justified by measured end-to-end benefit.

## 9. Measurements and comparison rules

Required: held-out negative log-likelihood, perplexity with a common tokenizer, peak host/device memory, checkpoint bytes, parameter counts, training wall time, first-token latency, steady generation throughput, and structural-controller overhead.

For a recurrent graph, report node count, edge count, degree, update rounds per token, selected modules, and executed operations where measurable. Track allocation growth, compile time, and recompilation count. Include adaptation memory and optimizer state in relevant measurements.

Benchmark timings must synchronize asynchronous device work and separate compilation/warm-up from steady execution. Use identical prompts, sequence lengths, batch sizes, and generation settings. Report at least three training seeds for headline research claims, with mean and dispersion; initial smoke runs may use one seed and must be labeled accordingly.

Matched parameter count and matched compute are different comparisons. Report both when claiming efficiency. A lower loss at higher cost is not an efficiency win. Energy claims require actual measurements or an explicitly identified estimate with methodology.

## 10. Main risks and mitigations

| Risk | Mitigation / experiment |
|---|---|
| Sparse work is slower than dense work | Benchmark actual gather, aggregation, and routing kernels |
| Local graph cannot retrieve distant context | Retrieval probes and controlled message-depth or shortcut experiments |
| Growth increases memory without quality gains | Hard budgets and fixed-final-capacity control |
| Routing collapses | Usage metrics and controlled routing alternatives |
| Continual learning forgets old domains | Retention evaluation, replay or selective updates as later ablations |
| Training leaks future tokens | Causality tests covering graph state and memory updates |
| Structural changes corrupt resume | Serialize topology, optimizer state, RNG state, and controller state |
| JAX compilation dominates | Fixed capacity stages, bounded shapes, compile-time reporting |
| Too many languages delay research | Single reference implementation before native kernels |
| Apparent novelty duplicates existing work | Focused related-work review once equations and controller are specified |

## 11. Research success and stopping rules

A working graph language model is an engineering milestone, not evidence of superiority. A research success requires a reproducible advantage on a declared quality/resource or adaptation/retention comparison. Exact improvement thresholds must be set after baseline variance and hardware costs are known, before the main comparison run.

If a feature repeatedly fails to improve the relevant comparison under a documented tuning budget, record the negative result and simplify or revise it. Do not add hierarchy, routing, growth, and test-time learning together to hide an unexplained failure.

## 12. Open decisions

- Available training hardware and resource budget.
- First inference device, RAM ceiling, and useful latency target.
- Confirmation of JAX + Flax versus a Rust-first Burn implementation.
- Dataset subset, tokenizer, sequence length, and training-token budget.
- Initial module equations, degree, state width, and readout.
- Definition and timing of the first structural-growth experiment.
- Whether eventual online learning is session-local or durable.

## 13. Evidence boundaries

This is a targeted initial literature and tooling review, not an exhaustive systematic survey or novelty certification. Paper demonstrations apply to their own experimental conditions. Framework capabilities and backend support can change; pin versions and recheck official documentation before implementation. All architecture choices above are proposals until tested.


## 14. Follow-up: the two mini-AGI repositories

Reviewed: 2026-10-23. Method: read repository documentation, configuration, and selected source files through public web access. Neither project was installed, executed, trained, or independently benchmarked. Links below target moving main branches, not pinned snapshots; cached GitHub and raw views showed different run counters. Reproduction must first pin a commit and matching weights/configuration. Do not combine counters from different snapshots.

### 14.1 muellerberndt/mini-agi: an agent layer

[Repository](https://github.com/muellerberndt/mini-agi); [inspected miniagi.py](https://github.com/muellerberndt/mini-agi/blob/main/miniagi.py).

The inspected implementation constructs external language-model clients for action selection and summarization. It builds prompts containing objectives, prior observations, and optional criticism, then interprets commands. Its memory is application context and summarized history. This is useful agent orchestration, but it does not implement MLM's growing trainable graph or demonstrate a new neural backbone.

Possible later use: a model-independent task loop for evaluating an already capable MLM. It should not be the initial architecture foundation. An agent's apparent improvement from summaries or tool feedback must not be counted as neural parameter learning.

### 14.2 volotat/mini-AGI: relevant model-level engineering

[Repository](https://github.com/volotat/mini-AGI); [README](https://github.com/volotat/mini-AGI/blob/main/README.md).

The author describes a toy continual-learning byte model trained on an 8 GB GPU, with a growing expert pool backed by disk. This is much closer to MLM's capacity-adaptation goal. The README also explicitly says generation does not train the weights. Shared forward code is not equivalent to learning on every generated token.

The reported continual-learning experiments favor a slower shared trunk and interleaving. They do not establish elimination of forgetting in arbitrary streams. Published examples are undertrained and include clear errors; project naming is not evidence of AGI. These are author-reported findings, not results reproduced here.

Source-level observations:

| Inspected source | Observation | Implication for MLM |
|---|---|---|
| [model.py](https://github.com/volotat/mini-AGI/blob/main/minagi/model.py) | PyTorch decoder components include attention, rotary positions, RMSNorm, and SwiGLU | This is not evidence of an attention-free replacement |
| [recur.py](https://github.com/volotat/mini-AGI/blob/main/minagi/recur.py) | Reuses blocks with learned halting and a PonderNet-style objective | Adaptive depth is different from adding modules |
| [paged.py](https://github.com/volotat/mini-AGI/blob/main/minagi/paged.py) | Implements disk/RAM/device expert residency, per-expert state handling, and stable identities | A useful engineering reference for later offloading |
| [pool.py](https://github.com/volotat/mini-AGI/blob/main/minagi/pool.py) | Contains expert routing, pool expansion, pruning, and distinct resident-pool behavior | Do not assume all pool variants follow one growth/pruning rule |
| [plasticity.py](https://github.com/volotat/mini-AGI/blob/main/minagi/plasticity.py) | Adjusts learning rate using held-out observations | Such data is a development/control set, not an untouched final test |
| [forgetting_probe.py](https://github.com/volotat/mini-AGI/blob/main/replication/forgetting_probe.py) | Provides before/after domain scoring, sequential/interleaved modes, and optimizer-state controls | Useful experiment structure; not independent validation |

The [inspected configuration](https://github.com/volotat/mini-AGI/blob/main/config.yaml) specifies a 512-wide trunk, eight attention heads, an initial 64 experts, top-8 selection, and 32 resident experts. These are reference-project settings, not MLM defaults. At those dimensions, a three-matrix SwiGLU expert has approximately 3 × 512 × 2048 = 3,145,728 matrix parameters. An initially modest active footprint can therefore coexist with hundreds of millions of stored expert parameters.

Inference: paging expands accessible capacity but does not make total storage small. Router work, transfer bandwidth, cache misses, host memory, optimizer state, and recurrent applications remain real costs. Top-k per application is not necessarily top-k distinct experts per generated token when a block repeats.

Another inference requiring testing: selecting a resident expert set using requests aggregated across a training window could create a future-information path unless the admission protocol is prefix-causal. This review did not execute a leakage test or establish a defect. MLM must test prefix invariance for routing and paging as well as attention.

### 14.3 What to borrow and what to verify

Borrow principles: stable module IDs, optimizer state attached to module identity, explicit growth events, newborn protection, separation of total capacity from resident capacity, and before/after retention probes. These are design lessons, not a proposal to copy the full implementation.

Before reproducing the second project: pin source and checkpoint revisions; inspect evaluation boundaries; compare sequential and interleaved domain schedules; retain optimizer state across restarts; separate raw decoding from repetition controls; measure storage and transfer costs; and evaluate an untouched test set. Repeated validation-driven learning-rate or growth decisions require a separate final test.

Interleaving is a meaningful baseline, but it does not logically guarantee zero forgetting. Repeatedly training all domains may still reduce some capabilities. Likewise, an unused module may hold a rare useful skill: usage alone is not sufficient justification for pruning in MLM.

## 15. Additional research that sharpens the design

| Source | Finding relevant to MLM | Boundary |
|---|---|---|
| [Gated Graph Sequence Neural Networks](https://arxiv.org/abs/1511.05493) | Gated recurrent message passing can learn graph computations and sequence outputs | Graph-input task results do not establish an internal graph language-model advantage |
| [Growing Neural Gas](https://papers.neurips.cc/paper_files/paper/1994/hash/d56b9fc4b0f1be8871f5e1c40c0067e7-Abstract.html) | Nodes and connections can be added incrementally to learn input topology | Vector quantization and topology learning are not next-token prediction |
| [Net2Net](https://arxiv.org/abs/1511.05641) | Some widening/deepening transformations preserve an existing network's function | Arbitrary graph surgery is not automatically function-preserving |
| [Switch Transformers](https://arxiv.org/abs/2101.03961) | Conditional expert activation separates total parameters from active computation | Routing and communication remain costs; the backbone remains Transformer-based |
| [ModuleFormer](https://arxiv.org/abs/2306.04640) | Sparse modules support experiments in extension and specialization | A particularly relevant novelty comparison, not proof of our graph design |
| [PonderNet](https://arxiv.org/abs/2107.05407) | Learned halting can trade computation against prediction performance | Extra recurrent depth is not extra capacity and does not guarantee reasoning |
| [Loss of plasticity in deep continual learning](https://doi.org/10.1038/s41586-024-07711-7) | Continued learning can lose adaptability; replacing low-utility units can help in studied settings | Plasticity and retention are separate properties; language-model transfer needs testing |
| [Graph communication bottlenecks](https://arxiv.org/abs/2006.05205) | Message passing can compress distant information through narrow paths | More edges or rounds might help but can undermine efficiency |

Together these support feasibility of the components. They do not validate a novel combination or guarantee useful scaling.

## 16. A concrete, falsifiable first MLM design

This section specifies a proposed experiment, not an approved final architecture or measured result.

### 16.1 Minimal graph model

Start with 16 modules, state width 128, four incoming neighbors per module, and two synchronous message rounds per token. These numbers are starting hypotheses. Each node is a small gated neural network, preserving the idea of a dot containing computation. Test shared update weights versus module-specific weights; the former saves parameters but does not add trainable capacity merely by adding states.

Use a fixed ring with a few seeded long-range connections as one topology. Compare a seeded random degree-bounded graph and a chain. Define edges as computational paths between persistent module states, not tokens or external knowledge entities.

Let x_t be the current token embedding and h_(i,t-1) module i's previous state. Initialize a round with h_(i,t,0) = h_(i,t-1). For round r:

    m_(i,t,r) = sum over j in N(i) of a_(ij,t,r) * W_m h_(j,t,r)
    z_(i,t,r) = concat(x_t, m_(i,t,r))
    h_(i,t,r+1) = GRU_i(z_(i,t,r), h_(i,t,r))

Initially a is a normalized fixed edge weight. Later learn soft gates over the same bounded neighbor set. Use a readout of the final module states, such as mean pooling followed by a projection, to predict token t+1. Keep self-state through the gated update. All modules read previous-round states; none reads a partly updated neighbor.

The model consumes x_t, updates state, then predicts x_(t+1). Reset state at document boundaries. Train with causal cross-entropy and documented truncated backpropagation through time. Detaching between chunks limits gradient reach; report it and keep baseline comparisons explicit about context differences.

### 16.2 Cost and memory expectations

For N modules, width d, degree k, and R rounds, independent gated updates typically include O(R N d²) work. With a shared message projection computed once per node per round, aggregation adds roughly O(R N k d). Per-edge dense transforms would be more expensive. Readout and vocabulary projection add further work. These are rough algorithmic counts, not runtime predictions.

State memory is O(Nd) per sequence; training additionally stores activations over the unroll and optimizer state. Serial token recurrence can reduce training throughput despite compact state. This model may lose to a Transformer at short context; that outcome must remain visible.

### 16.3 Introduce growth without destroying existing behavior

First test a residual module branch with a zero-initialized output projection, initialized internal features, and an unchanged existing route. This can preserve the old output at insertion for an additive construction; the output projection can receive gradients initially, while internal gradients may begin only after it becomes nonzero. Verify this numerically.

Do not zero every parameter or both a gate and an output projection without checking gradients: an apparently harmless insertion can make the new branch unable to learn. Changing a normalized router's candidate set may change existing outputs even when a new expert's output is small. Function preservation must be tested for the actual equations.

Compare random initialization, warm-started modules, and a validated function-preserving transformation where applicable. Protect a newborn for a declared training interval, expose it to learning, then assess improvement and retention. Usage is diagnostic rather than a complete utility measure.

### 16.4 Growth controller as a hypothesis

At fixed training intervals, check resource headroom, optimization health, persistent development-loss plateau, and module load. Trial at most one bounded structural change per event. Save the parent state, train the trial under a fixed budget, and compare with a no-growth continuation from the same parent using equivalent data and compute accounting.

Keep or reject the change using a development criterion fixed before the trial. Report rejected trials and their cost. Reserve final test data for final evaluation; repeated architecture decisions on the same development set can overfit it.

In JAX, start with fixed capacity stages. Logical activation of slots and physical reallocation are separate experiments. At reallocation, copy existing parameters and optimizer moments explicitly, initialize new moments, migrate graph IDs, and measure recompilation cost.

## 17. Execution order and decision gates

1. **Synthetic learnability:** delayed copy, associative recall with unseen key/value combinations, and simple pattern prediction. Vary delays and graph distances. A tiny-data overfit is only a debugging gate.
2. **Fixed-graph language:** common-tokenizer TinyStories comparison against GRU and Transformer baselines. Publish loss, latency, memory, and training cost even if the graph loses.
3. **Connectivity study:** compare fixed, randomly rewired, and learned-rewired graphs at the same edge budget.
4. **Capacity study:** compare growth to fixed initial/final capacity, reserved-slot activation, and fixed-budget replacement.
5. **Retention study:** train domain A, adapt to B, then evaluate both without training on their test sets. Add A/B interleaving as a separate rehearsal baseline. Compare slower shared updates, replay, frozen old modules, and new-module learning independently.
6. **Optimization:** only after a repeatable benefit, profile actual sparse execution and consider Mojo kernels or a Rust runtime.

The first publishable research question is: **Can a bounded recurrent graph improve new-domain learning versus old-domain retention at comparable total training cost and deployed memory?** Broad Transformer replacement and universal device support remain longer-term questions.

## 18. Updated feasibility verdict

We can implement this design with current frameworks. The second mini-AGI project is a relevant engineering reference for growing expert capacity; the first is an agent example. Neither validates MLM's internal spiderweb architecture. The strongest next step is a small causal graph model with a controlled growth experiment, not a combined agent, paged billion-parameter pool, hierarchy, and online learner.

No code, training results, or independent repository reproduction has been produced by this follow-up. The research recommendation is conditional on subsequent measurements.


## 19. MLM v0.1: selected algorithm for the first implementation

Decision date: 2026-10-23. Status: concrete experimental specification; unimplemented and unbenchmarked. This section narrows the alternatives in sections 6 and 16 into a reproducible first candidate. Earlier options remain research alternatives, not simultaneous implementation requirements.

Working name: **MLM recurrent module graph**. This is a descriptive project name, not a claim of an established or novel algorithm.

### 19.1 Core logic

A node is a small trainable gated recurrent network with its own parameters and state. Edges transport state vectors between nodes. Reusing the graph over a sequence provides temporal memory; running several synchronous rounds provides communication within one token step.

Use byte-level causal prediction initially: IDs 0–255 are bytes, 256 is beginning-of-document, and 257 is end-of-document. UTF-8 text is encoded into bytes before modeling and decoded with a documented invalid-sequence policy for display. Byte sequences are longer than subword sequences, so report bytes/second and bits/byte; do not compare their perplexity directly against a different vocabulary.

At token t, consume x_t and predict x_(t+1). Never feed the target into routing, state, or growth statistics before the prediction is scored.

Notation: N nodes, state width d, fixed incoming degree k, R rounds, learned embedding e_t, and predecessor indices E[i,q].

    h_i^(0) = previous token's final state_i
    v_j^(r) = W_msg h_j^(r)
    m_i^(r) = (1 / sqrt(k)) * sum_q v_(E[i,q])^(r)
    h_i^(r+1) = GRU_i(concat(e_t, m_i^(r)), h_i^(r))
    z_t = (1 / sqrt(N_initial)) * sum_i W_read_i h_i^(R)
    logits_t = W_vocab LayerNorm(z_t) + b_vocab
    loss_t = cross_entropy(logits_t, x_(t+1))

Embedding, message, state, and readout widths are d in the initial implementation. GRU input width is 2d. Every round reads the complete previous-round state, not partially updated neighbors. The initial graph uses fixed edges and fixed aggregation coefficients; gates and discrete routing are disabled.

The sum readout and fixed N_initial denominator replace the earlier illustrative mean-pooling option. This permits a later zero-output branch to be inserted without changing the contributions of existing nodes. Node-specific GRU and readout parameters make nodes distinct. Message projection is shared to avoid one dense matrix per edge.

Input reaches every node in v0.1. That is intentionally simple, but it may let the model ignore communication. An edge-disabled ablation is therefore required to determine whether the spiderweb actually helps beyond a collection of recurrent modules.

### 19.2 Initial topology and profiles

Use deterministic directed ring-plus-shortcut predecessors. For the language profile, node i receives from (i-offset) modulo N for offsets 1, 2, 4, and 8. These give valid distinct neighbors for N=16. For the tiny profile use offsets 1 and 3 with N=8. The GRU already preserves self-state, so no explicit self-edge is required.

| Setting | CPU correctness profile | Initial language profile |
|---|---|---|
| Nodes | 8 | 16 |
| State / embedding width | 32 | 128 |
| Incoming edges per node | 2 | 4 |
| Message rounds per byte | 1 | 2 |
| Unroll length | 32 | 128 |
| Batch size | 2 | 4, lower if profiling requires |
| Compute dtype | float32 | float32 initially |
| Growth, rewiring, routing | Off | Off |

These are starting configurations, not tuned optima or throughput commitments. Approximate GRU matrix parameters alone scale as N × 3 × (2d+d) × d; count actual parameters in the implementation rather than describing the small profile as a million-parameter model. Compilation and execution must be profiled before raising scale.

## 20. Training algorithm and data rules

### 20.1 Supervised self-prediction

Use ordinary backpropagation through time with teacher forcing. The training targets are the next observed bytes; no human labels or external language-model API are required. Optimize with AdamW through Optax. Initial exploratory settings: learning rate 3e-4, beta1 0.9, beta2 0.999, epsilon 1e-8, weight decay 0.01 on weight matrices, and global gradient-norm clipping at 1.0. Do not decay biases or normalization parameters. These hyperparameters require validation, not claims of optimality.

For the first smoke experiment, use a constant learning rate and no structural controller. This isolates the architecture from a validation-driven scheduler. If a later scheduler is introduced, record its settings and compare it consistently across models.

Use JAX scan operations to express recurrence over tokens and a bounded loop or scan over graph rounds. Flax owns parameters; recurrent state is explicit input/output. Batch independent documents with separate states. For contiguous chunks of one document, carry state and stop its gradient at chunk boundaries; reset when a new document begins. Never reuse a training state during validation.

### 20.2 Data progression

1. Generate deterministic synthetic sequence tasks for correctness and memory probes. Separate train and evaluation examples, including unseen associative-recall combinations.
2. Overfit a tiny training-only text fixture to confirm optimization works. This is a debugging result, not a language benchmark.
3. Train on a bounded TinyStories subset. Select and record the exact source revision and subset size during setup; split at document level before producing windows. Maintain separate training, development, and final-test sets and record content hashes.
4. Expand tokens, context, and scale only after the smoke profile passes and its runtime is measured.

Use the same byte vocabulary, document boundaries, prediction masks, and data budget for direct GRU/Transformer comparisons. Exclude padding from loss. Report structural-marker loss separately or explicitly identify whether it is included; bits/byte on payload bytes uses natural-log loss divided by ln(2). Record generated output without repetition controls as the primary sample; label any decoding controls separately.

### 20.3 Training step

    read a batch and its document-reset mask
    reset only states belonging to newly started documents
    compute next-byte logits and masked mean loss
    differentiate with respect to model parameters
    check finite loss and gradients; clip gradient norm
    apply AdamW; return explicit recurrent states
    detach carried states before the next training chunk
    log byte count, loss, gradient norm, elapsed time, and resource use
    checkpoint parameters, optimizer, RNG, topology, and data position

Evaluation uses frozen parameters, fresh states, and a documented prefix warm-up policy. Timing synchronizes JAX device work. Model selection uses development data; the final test does not choose checkpoints or controller settings.

## 21. Structural-learning algorithms after the fixed graph

### 21.1 Learned soft edges before hard routing

First replace fixed coefficients with learned scalar sigmoid gates on the same bounded edges, keeping a fixed aggregation denominator. Compare against the fixed graph. This is differentiable edge weighting, not topology growth and not a compute saving: every listed edge is still evaluated.

Only after that experiment consider candidate-edge selection. Hard top-k membership has no ordinary gradient; a specific estimator or external search rule must be specified before claiming end-to-end learned routing. No such estimator is required for v0.1.

### 21.2 Rewiring as a controlled search

Proposed first rewiring experiment: at a declared checkpoint, estimate edge importance using a fixed development probe, then trial replacement of one low-importance edge with a seeded candidate. Compare against no change and random replacement using fixed continuation budgets. Importance must measure contribution, such as loss change on edge ablation, rather than usage alone. Candidate scoring and all trials count toward training cost.

Do not mix this experiment with node addition. Preserve valid indices and degree, and retain the pre-change checkpoint. Search may overfit development data; final-test evaluation remains separate.

### 21.3 Growth through an additive branch

Add one new node with initialized GRU parameters and incoming edges from existing nodes. Initialize its readout projection to zero and leave its outgoing edges absent. Existing graph updates, logits, and readout denominator remain unchanged at insertion, subject to numerical verification.

The zero readout can receive gradients from the loss. Initially the new GRU may receive no gradient through that projection; it begins learning after the projection becomes nonzero. Do not zero all internal weights. Train the trial for a fixed budget before judging it. Add outgoing communication in a separate later intervention, since changing existing neighborhoods can perturb old behavior.

Physical growth requires expanding parameter arrays, preserving old optimizer moments, creating zero moments for new parameters, and recompiling. If a reserved pool is used instead, call it logical activation and report its allocated memory from the beginning.

A first growth trigger can be a sustained development plateau after the warm-up period, with finite optimization, resource headroom, and no untrained newborn still on trial. Before running it, fix the evaluation interval, plateau window, minimum gain, trial duration, and capacity ceiling from baseline variability and measured runtime. This is a future experiment gate, not an autonomous controller already specified for production.

Accept a trial only under its predeclared development quality/resource criterion. Compare with fixed-initial-capacity, fixed-final-capacity, reserved-slot, and no-growth continuation controls. Count rejected trials. Growth alone does not guarantee better retention.

## 22. Testing and implementation order

### 22.1 Required correctness gates

- **Prefix invariance:** two inputs with identical prefixes and different suffixes yield matching prefix logits within a stated numerical tolerance.
- **State isolation:** a new document's outputs do not depend on the preceding independent document after reset.
- **Synchronous messages:** compare vectorized graph updates against a small explicit reference using prior-round states.
- **Gradients:** initial trainable paths have finite gradients; a tiny-data training loss decreases and the fixture can be overfit under a fixed debugging budget.
- **Checkpoint resume:** restored parameters, topology, optimizer, RNG, and data position reproduce the next step within the documented tolerance.
- **Graph contribution:** compare full edges against zero-message and topology controls.
- **Growth insertion:** adding the isolated zero-readout branch initially preserves logits; its readout learns and its internal parameters subsequently receive gradients.
- **Migration:** old parameters and optimizer moments survive expansion; new moments are initialized correctly.

### 22.2 Work packages

1. Environment and validated configs: pyproject.toml, lockfile, and CPU backend check.
2. Data and state semantics: byte encoding, document splits, batches, reset masks, and deterministic synthetic tasks.
3. Fixed graph: graph generation, node updates, readout, and correctness tests.
4. Training and artifacts: optimizer, metrics, checkpoint/resume, and local generation.
5. Baselines and benchmarks: matched-protocol GRU and Transformer, held-out loss, runtime, memory, and multiple seeds.
6. Separate research branches: edge gates, rewiring, additive growth, and retention probes, each with its own controls.

Suggested implementation boundaries: src/mlm/config.py, data.py, graph.py, model.py, train.py, checkpoint.py, evaluate.py, and generate.py, with numerical and state tests under tests/. Exact interfaces belong to the implementation plan, not this research record.

The first deliverable is a reproducible CPU smoke run with actual loss and test output. The next deliverable is a fixed-graph versus baseline language comparison. Neither deliverable should be described as a trained general assistant.

## 23. Local readiness and immediate feasibility

Read-only inspection on 2026-10-23 found:

- Workspace contains research.md and prd.md, with no model implementation yet.
- Linux x86-64, Python 3.14.4, uv available.
- 16 logical CPUs and approximately 14.5 GiB reported system RAM; this is total RAM, not measured free memory.
- PCI inspection identifies AMD Barcelo integrated graphics. No NVIDIA utility was found. GPU compute compatibility and available device memory were not established.
- JAX, Flax, Optax, and PyTorch are absent from the inspected Python interpreter. This does not establish whether unrelated environments have them installed.

We can build and test a tiny CPU model on this machine, subject to dependency installation and actual runtime checks. Use a project-managed Python version supported by the pinned dependencies rather than assume the system Python is suitable. CPU feasibility is not a promise of practical full-corpus training speed. The initial run must measure step time and memory before estimating a larger training budget.

Current status: algorithm specified; implementation, dependency setup, tests, and training have not started. This update authorizes no claim of passing tests or learned language ability. Model quality, useful speed, structural-growth benefits, and Transformer competitiveness remain experimental outcomes.


## 24. First implementation and measured smoke run

Implementation date: 2026-10-23. The earlier readiness statements describe the pre-implementation state. A fixed-graph reference model now exists in src/mlm with JAX, Flax Linen parameter ownership, Optax AdamW, Pydantic settings, byte-level document batches, CLI training/evaluation/generation, and atomic checkpoints. Dependencies are locked in uv.lock.

The first scope deliberately trains independent document prefixes; it does not yet carry states across corpus chunks in the training CLI. The model API supports explicit state and passes chunk-equivalence tests, but a document-streaming sampler/trainer is a subsequent task. Prefix truncation is reported and exact scored-prefix overlap across training/development is rejected.

Eleven tests cover configuration/data behavior, aggregation, full NumPy-reference GRU rounds and readout, prefix causality, reset isolation, chunk equivalence, message dependence, finite gradients and loss improvement, exact next-step and end-to-end resume, generation, checkpoint selection, scored-prefix overlap, and output-directory protection. Ruff and ty checks passed during implementation; see the final execution record for current check outcomes.

The reviewed CPU smoke run used 100,546 parameters, eight nodes, width 32, degree two, and one round per byte. After 300 updates, training loss fell from 6.129411 to 0.205798 nats; development loss fell from 6.091121 to 2.046138. The best evaluated development result was 1.653910 at step 50, after which overfitting was evident. A best-checkpoint greedy sample continued `The ` as `red cat rests.`.

These are one-seed engineering results on 36 training and 18 development template sentences, not evidence of general language competence, growing capacity, or superiority over baselines. No independent final test or real-corpus training was performed. Detailed settings, timings, artifacts, limits, and reproduction commands are in [the smoke report](experiments/smoke/report.md). Next: document streaming and fair baselines on a real corpus, then isolated rewiring/growth studies.


## 25. Full-document English and dialogue training

Completed 2026-10-23. A resumable document-streaming trainer now covers all bytes, carries state between chunks, freezes padded state, and resets at new documents. Checkpoints preserve sampler order/offsets/RNG and recurrent carry. Warm-starting a new corpus transfers parameters with a fresh optimizer; exact resume preserves the original stream and optimizer. JSONL retains paragraphs and multi-turn conversation boundaries.

Public Hugging Face sources: TinyStories and Everyday Conversations. Bounded subsets contain 1,000 training stories and 300 training conversations, with distinct local development and test documents. Cached row responses, source links, dataset-card license identifiers, selection ranges, and hashes are recorded. The viewer API was not revision-pinned; exact local snapshot hashes must be used for reproduction. See [data details](data/README.md).

A 366,722-parameter MLM trained for 4,000 English updates, then 2,000 conversation/English-rehearsal updates. English development loss improved from 6.0071 to 1.1557 nats. Story test loss was 1.1216 before adaptation and 1.1088 after; conversation test loss improved from 2.7842 to 1.3612. These one-seed results measure byte prediction, not semantic understanding. No matched baseline or growth experiment has been run.

Qualitative probes found a correct greeting, repetitive story continuations, and irrelevant answers to simple factual and reading questions. The model is not yet a reliable chatbot. Rehearsal coincided with maintained story prediction in this experiment, but it is not proof of general continual-learning retention. The dialogue objective currently includes all turns, not assistant-only targets.

Twenty-one regression tests passed during this milestone, covering streaming coverage, padded-state freezing, exact carry/optimizer/sampler restoration, independent test scoring, conversation formatting, split deduplication, message-mode generation, and interrupted best-checkpoint recovery in addition to the original tests. The complete measured record, limitations, samples, and commands are in [the experiment report](experiments/english-chat/report.md).


## 26. Matched recurrent baselines and paired reading diagnostic

Completed 2026-10-23. Added a plain single-cell GRU architecture, architecture-aware checkpoint initialization, and an evaluation-only neural-likelihood reading scorer. Historical graph checkpoints still load; the current implementation reproduced the graph reference development loss exactly.

The message-free ablation and plain GRU each trained from scratch for 4,000 updates on the same English subset, seed, optimizer settings, batch order, and document-streaming protocol as the graph reference. Corpus hashes, processed-byte counts, and final sampler snapshots match. Plain GRU width 208 gives 368,210 parameters; graph allocation is 366,722, a difference of about 0.4%.

Story test loss in nats: graph 1.121557, message-free ablation 1.157398, plain GRU 1.098890. Thus the graph beats its ablation in this run, while the simpler parameter-matched GRU is slightly better than the graph. This is one seed and a small test subset, not a statistical or general architecture ranking. The message-free model retains at least 102,400 unused matrix weights, and graph/GRU state sizes differ; parameter matching is not compute or effective-capacity matching. Runtime measurements were not collected under controlled contention.

An original frozen diagnostic contains 12 counterfactual pairs / 24 two-choice questions. Model likelihoods rank candidate answers; labels only grade the results and never enter neural scoring. Every checkpoint scored 50% individual accuracy and 0% complete-pair accuracy, with no preferred-answer changes within pairs. The adapted chat checkpoint was scored separately using its role format. These findings expose fixed-answer preferences on this diagnostic and provide no successful reading-comprehension evidence. They do not establish that the models encode no contextual information.

A review caught a shared-separator normalization issue in the preliminary scorer. Final scorer version 2 treats the appropriate separator as part of the prompt and scores only answer bytes. Regression tests verify alignment, unequal-length normalization, and plain/chat separators. Preliminary artifacts are preserved as superseded; the questions and answer key were unchanged. There is no keyword or regex answer solver.

Twenty-six tests pass, covering the original pipeline, GRU gradients/causality/padding/chunk state/checkpoint recovery, parameter budgets, and comprehension scoring. The [comparison report](experiments/comparison/report.md) contains the protocol, raw evidence, limitations, and reproduction commands.

Decision: keep the graph as a research candidate and the plain GRU as a baseline. Prioritize multi-seed comparisons and a supervised assistant-answer objective with fresh held-out reading tasks. Learned edge strengths, structural rewiring, node growth, and a Transformer baseline remain separate outstanding experiments. No baseline result justifies a Transformer-replacement claim.


## 27. Assistant-only supervision and failed context reading

Completed 2026-10-23. Implemented explicit assistant byte-span supervision, separate input/target masks, objective-aware exact resume, zero-target optimizer skipping, and global development-selection intervals. Added offline synthetic reading practice with balanced counterfactual pairs and disjoint names/templates within this stage. Training contains 800 unique pairs repeated eight times, 64 cached public conversations, and 64 full-text stories. Each architecture received the same 2,000 updates, 915,785 consumed payload bytes, and 82,737 supervised payload bytes from its corresponding English checkpoint.

Graph best step 1,000: development target loss 0.268223, test target loss 0.234342. GRU best step 1,500: development 0.386889, test 0.244349. These restricted answer losses include formatting/EOS and cannot be compared with earlier full-text losses. Both models remained at 50% individual accuracy, zero of 64 complete held-out pairs, and no preferred-answer flips. The original frozen diagnostic also remained at zero complete pairs. Raw greedy answers became short category defaults; this is not comprehension evidence.

Story test loss worsened from 1.121557 to 1.267662 for graph and from 1.098890 to 1.169831 for GRU. Preserve the English checkpoints. In a controlled color-substitution probe, state differences decayed rapidly with added neutral text and answer-score changes were tiny. This suggests testing retention/readout mechanisms next; it does not prove a single cause of failure. Optimization, capacity, and task design remain possible contributors. Familiar training-style probes also failed paired reading.

All 33 regression tests passed. See [full protocol, evidence, and commands](experiments/assistant-learning/report.md). Offline label rules never participate in inference. Growth, learned edges, routing, Transformer benchmarking, and multi-seed validation remain outstanding.


## 28. Retention-gate intervention: context sensitivity without reliable reading

Completed 2026-10-23. Following the memory diagnostic, tested deterministic chrono-inspired offsets to pretrained retention-gate biases. Nominal timescales span 2–128 bytes across graph nodes or GRU channels; all gates remain trainable. No state, parameters, topology, runtime answer rules, or update equations were added. Motivation: [Tallec and Ollivier](https://arxiv.org/abs/1804.11188); this is not an exact reproduction of their initialization method.

Both architectures completed 2,000 updates with identical training exposure to the assistant-learning controls. Graph development-selected step 2,000 reaches 66/128 individual answers and 19/64 complete pairs; GRU-selected step 500 reaches 62/128 and 2/64. Graph changes answers in 36/64 pairs, but 17 pairs are both wrong. Independent random guesses average 25% complete pairs; 29.7% paired accuracy and 51.6% individual accuracy do not establish reliable comprehension or above-chance semantic performance.

The familiar color-substitution probe shows longer-lived state and answer-score differences, and correct preferences at short delays, but longer delays still fail. Story test loss worsens to 1.849269 graph and 1.855628 GRU, versus 1.267662/1.169831 for the unmodified assistant controls. Preserve original English checkpoints. Record this as a useful mechanism diagnostic with a large quality tradeoff, not an accepted model upgrade.

All 36 tests pass; initializer integrity and overwrite protection, frozen scoring, source hashes, and matched stream exposure were checked. [Report and raw evidence](experiments/memory-retention/report.md). Next: controlled entity–fact retrieval with increasing distractors/delay, fresh confirmation splits, and retention measurement before structural growth.


## 29. Controlled recall versus entity–fact binding

Completed 2026-10-23. Implemented a fresh synthetic English retrieval curriculum with disjoint task names, four equal-byte-length answer colors, shared grammar, 0/1/3 distractors, and 0/64/128 filler bytes. Context swaps and fixed-context query swaps form four-question binding groups. This removes unfamiliar wording as a confound while testing whether the question selects the correct person's fact. Labels exist only in offline preparation/evaluation.

Each architecture starts from its English-derived retention initialization, trains 500 basic updates and 1,500 binding/rehearsal updates, with fresh optimizers per stage. Unroll 384 contains every synthetic example (maximum 268 input bytes), preserving the gradient path to facts. Data/config/source hashes were frozen; both selected checkpoint hashes were locked before retrieval test evaluation. Input bytes, scored bytes, masks, and sampler snapshots match between architectures.

Both basic models score 32/32 fresh short single-fact questions and 16/16 complete pairs. After the second stage, both score 96/96 single-fact questions across all three delays. Multiple-fact individual accuracy is only 48.18% graph and 50.78% GRU, and both solve zero of 96 complete four-question binding groups. Overall accuracy (58.54%/60.63%) masks this split. Fixed 30-example raw greedy samples score 50% each; likelihood ranking and unrestricted generation must remain separate measures.

Story loss is 1.577292 graph and 1.264509 GRU after the second stage, better than after the basic stage but worse than the original English checkpoints. Only 32/992 training documents are stories, yet they contribute 124,221/152,091 supervised payload bytes (81.7%); this is not an exact gradient-share measure. No rehearsal ablation isolates its effect.

The result establishes narrow single-fact recall under one shared grammar and seed, not entity binding or general comprehension. Data, curriculum, unroll, and rehearsal changed together; no unshifted-gate control on the new task isolates retention-prior benefit. The paired-entity coverage is bounded, and English retention uses a repeatedly inspected diagnostic. Next isolate one distractor/no delay with varied entity combinations and an explicitly accounted objective; query-conditioned memory/readout remains an architecture hypothesis.

All 39 tests pass, with static checks passing. See [protocol, per-cell evidence, raw outputs, and commands](experiments/retrieval-curriculum/report.md). Node growth, rewiring, learned routing, Transformer superiority, and device-efficiency claims remain unproven.


## 30. Focused binding, learned node readout, and a learnability check

Completed 2026-10-23. Added fresh two-person/no-delay data with all 120 unordered pairs of 16 training names, both fact orders, and complete four-question context/query groups. Development/test each have eight disjoint names and 56 groups. Paired groups, rather than independent documents, form batches. Selection maximizes complete development-group correctness, then minimizes target loss. Each example fits the 128-byte unroll. No rehearsal obscures the task objective in this diagnostic.

Graph and GRU each trained 2,000 updates from their single-fact basic sources. Both remained at zero complete development groups, triggering the predeclared learned node-readout experiment before test inspection. The variant uses additive attention conditioned on node zero's recurrent state, with zero initial score vector for uniform equivalence. Motivation: [Bahdanau et al.](https://arxiv.org/abs/1409.0473). It is state-conditioned, not a parsed-question encoder or a novel attention principle. It adds 8,256 parameters without increasing the 512-scalar graph state; fixed message aggregation and topology remain unchanged. Default sum readout and legacy loading are preserved.

All three runs saw identical group order, masks, and exposure: 776,000 input and 40,000 supervised payload bytes each. Hashes and development selection were verified, and all selected checkpoints locked before the fresh test. Graph scores 115/224 individual answers and 0/56 complete groups; GRU 121/224 and 1/56; learned readout 111/224 and 0/56. Selected-model training-group correctness is also poor: 0/240, 7/240, and 1/240. These outcomes do not demonstrate reliable binding or a readout benefit.

English story losses worsen to 7.439665, 8.119605, and 7.668399. Preserve earlier English checkpoints and do not promote these task experiments. Learned readout parameters changed and pass gradient/state tests, but architectural functionality is not evidence of useful learned behavior. Low target loss near 0.117 does not establish semantic success.

A separate post-hoc training-only sanity check uses the original sum-readout graph and one four-question group for 500 updates at learning rate 0.003. It achieves 4/4 correct likelihood rankings and 4/4 raw greedy answers on that memorized group. This establishes tiny-case fitting, not held-out generalization. Dataset size and learning rate changed together, so the cause of the larger experiment's failure remains unresolved. Next scale a controlled curriculum gradually and separate known-name/novel-pair generalization from held-out-name transfer before increasing graph capacity.

All 44 tests pass, with Ruff and type checks passing. [Full protocol, scores, source checks, optional readout equations, and sanity evidence](experiments/focused-binding/report.md). Growth, rewiring, online weight adaptation, device-efficiency claims and Transformer superiority remain unimplemented or unproven.


## 31. Gradual binding and matched flat training

Completed 2026-10-23. Kept the original sum-readout graph unchanged. Both arms start from the same single-group memorization checkpoint with fresh optimizer, LR 0.003, seed 42, batch four and unroll 128. Curriculum adds 2/4/8/16/32/64 groups at 500 updates per stage with continuous optimizer state; flat sees all 64 groups for the same 3,000-update budget. Both consume 1,164,000 input and 60,000 supervised payload bytes. No rehearsal or architecture changes were introduced.

The curriculum fits active training groups 2/2, 4/4, 8/8, 16/16, then 29/32 and 50/64, preserving the seed group throughout recorded endpoints. Flat fits all 64 by 2,000 updates and again at 3,000. Checkpoint selection uses fixed equal-sized familiar-name/new-pair and new-name development pools, complete four-question-group accuracy first and target loss second, with eligible endpoints starting at 1,000 updates after actual name coverage. Selected curriculum step 2,500 has 32 active groups; flat selects 2,000 with 64. Both hashes were locked before test scoring.

A secondary order-robust metric, added after initial development observations but before test, requires all eight assignment/query/fact-order conditions. The source scores 0/24 cases in each pool. Curriculum scores 18/24 familiar-name cases and 1/24 new-name cases; flat scores 11/24 and 9/24. Individual test accuracy is 96.88%/63.54% curriculum and 75.52%/78.13% flat across those pools. Primary four-question test groups are 52/96 versus 44/96, while secondary eight-condition cases are 19/48 versus 20/48. This is limited transfer with different error tradeoffs, not a clean overall winner or general binding mastery.

An avoidable construction imbalance was found after scoring: familiar-name development pairs always join the same early pair-role class, while familiar-name test pairs join opposite classes. Exact pair/text disjointness holds, but these pools differ structurally, which could reward a coarse shortcut. The audit does not prove an internal model representation; it limits the strong familiar-name test result and requires balanced replication. Existing test evidence is preserved, not silently replaced.

Selected English story loss worsens from the seed source's 4.743716 to 12.834514 curriculum and 10.170839 flat. Original English checkpoints remain preserved; neither task model is promoted. Fixed raw greedy samples score 7/8 and 6/8 but are not full-test generation benchmarks. One seed, eight training names and one shared grammar cannot establish broad language comprehension.

All 47 tests pass with lint, formatting and type checks. [Full protocol, stage curves, primary/secondary metrics, source controls and split audit](experiments/gradual-binding/report.md). Next: new role-balanced confirmation splits with more diverse names and explicit English retention before more architecture changes. Growth, rewiring, online weight learning and efficiency/superiority claims remain outstanding.


## 32. Balanced acquisition and explicit English rehearsal

Completed 2026-10-24. Created a fresh balanced task with 32 training artificial names, four partners each, 64 training entity pairs and 256 four-question groups. Within-role/cross-role pair counts are equal in every evaluation pool. Familiar development/test matchings cover all 32 training names with disjoint pair identities; 16-name development/test novel pools are disjoint. All labels and role strata remain offline, never inference inputs.

Both arms start from the preserved 4,000-update English graph with a fresh optimizer, LR 0.001, batch four and unroll 128. Both receive identical order/exposure for 2,000 QA updates: 776,000 input and 40,000 supervised payload bytes. Rehearsal adds 2,000 full-document English updates and 958,088 English payload bytes, for twice the optimizer calls. This matches QA exposure, not total compute. Rehearsal checkpoints are evaluated after complete QA+English cycles.

Selection uses macro complete eight-condition development correctness across familiar/new-name buckets, then QA loss. An unconstrained best-task slot is reported separately from best constrained by English development loss <=1.25 times the initial baseline. Both policies allow initial state; four selected hashes were locked before task and English test scoring.

All selected models score 0/96 familiar-name and 0/48 new-name complete test cases, 50% individual ranking accuracy, and no context/query preference flips. Within/cross strata also score zero. Task-only unconstrained selection is QA step 1,500; its English test loss is 3.381755. Its protected selection is step zero, verified parameter-identical to the source, not a new learned success. Rehearsal selects QA step 2,000 / optimizer step 4,000 under both policies (identical weights), with English test loss 1.087839 versus source 1.121557. This is retention of byte-prediction quality without demonstrated binding acquisition. English test is a reused monitor, not new independent generalization evidence.

This study changes source (clean English rather than memorized binding), learning rate, task size and name diversity relative to the earlier partial-transfer experiment. It cannot isolate balancing alone or establish that earlier narrow transfer was nonexistent. Larger balanced acquisition still fails within this budget; optimization, progression and memory initialization remain unresolved. Next hold source and learning rate fixed while increasing a balanced learning ladder, retaining explicit language evaluation. No architecture change or growth was introduced.

All 50 tests and static checks pass. [Protocol, policy/source audit, objective counters, role-stratified scores and limits](experiments/balanced-confirmation/report.md). Retention-compatible research weights are saved; no reliable chatbot or general binding upgrade is claimed.


## 33. Hierarchical update routing: implemented, no Transformer-replacement evidence

Completed 2026-10-24. Implemented four groups of two recurrent nodes with a learned root/child softmax router. Current byte embedding and mean prior node state determine p_i; each node blends its GRU update by p_i. A fixed-uniform 1/8 control isolates slowed memory from learning gates; the legacy dense mode remains unchanged by default. Gates are causal and ignore padded positions. Source weights, fixed message graph, width64/eight nodes and sum readout are shared. Category labels, learned topology, node growth and hard sparse dispatch are not implemented.

Hierarchy builds on [hierarchical MoE](https://arxiv.org/html/1701.06538v1), not a newly established attention/learning principle. Hierarchical adds 1,548 parameters (368,270 total versus 366,722 controls), with the same 512-scalar recurrent state. Balance regularization encourages uniform mean routing but does not guarantee it. Soft routing still computes all candidates; ordinary forward calls avoid storing diagnostic routes unless requested. Standard stream warm starts validate update mode/group compatibility; legacy checkpoints retain dense defaults.

Fresh balanced seed920231 data support three matched-budget arms, each receiving 1,000 QA and 1,000 English updates. Each sees 388,000 QA input /20,000 supervised bytes and 479,511 English payload bytes; QA/English sampler traces match. Source/config/data/code hashes and selected checkpoints were locked before testing. Hierarchical and uniform controls initially match exactly, but both alter dense-trained English predictions: initial English development losses are 3.770138 versus dense1.155704. This limits architecture-potential comparisons; matched pretraining remains outstanding.

All selected models score zero of 96 familiar-name and 48 new-name complete eight-condition cases. Individual ranking accuracy is 49.74% dense, 50.52% uniform and 50.87% hierarchical. English test loss is 1.119035, 1.953038 and 1.473429 respectively. Hierarchical routing learns and improves language adaptation relative to fixed slowing, but not dense updates, and provides no demonstrated binding benefit. No flat learned-router control isolates hierarchy itself, and there is no direct Transformer comparison.

Cumulative hierarchical English traffic concentrates about54% on one node, despite the balance penalty; QA usage is more diffuse. This is input/domain-sensitive utilization, not verified linguistic expertise. All router parameters change from zero, so the router is functional; module activity does not imply understanding. No sparse compute or runtime advantage is established.

All 54 tests pass with lint/format/type checks. [Full mechanism, code snapshot, controls, frozen evidence and limitations](experiments/hierarchical-routing/report.md). The [Transformer architecture](https://arxiv.org/abs/1706.03762) is not replaced by these findings. Competitive language/retrieval, controlled training/inference costs, devices, longer context and scaling would need direct evaluation before such a claim.


## 34. From-scratch architecture comparison

Both dense and hierarchical graphs now have independently trained English checkpoints:4,000 updates each, identical shared initialization and 1,916,320 training payload bytes. Independent memorization probes fit one training group in both modes; four-group ranking is 62.5% dense and 100% hierarchical. Probe weights never seed English training. Each architecture then adapts its own English checkpoint with 1,000 QA plus 1,000 English rehearsal updates on fresh balanced data.

dense: English test loss 1.121557, adapted 1.126128; complete binding cases 0/144. hierarchical: English test loss 1.323535, adapted 1.321262; complete binding cases 0/144.

[Full protocol, locked tests, delayed probes and CPU measurements](experiments/from-scratch/report.md). This removes the prior inherited-weight mismatch without proving specialization, growth, arbitrary-device portability or Transformer replacement. Soft routing still computes all nodes. Narrow task acquisition remains the next gate; preserve these negative outcomes and use fresh confirmation data after changes. All 56 tests pass.


## 35. Independent nested-set acquisition diagnostic

Completed independent 1/4/16/64-group, 500-update task-only trials for dense and hierarchical models, resetting each to its own English checkpoint with fresh optimizer. Same paired subsets, traces and bytes; no test data used. Fixed endpoints, LR 0.003, seed 42.

dense: 1 groups: 100.0% complete training groups; 4 groups: 100.0% complete training groups; 16 groups: 0.0% complete training groups; 64 groups: 0.0% complete training groups.

hierarchical: 1 groups: 100.0% complete training groups; 4 groups: 100.0% complete training groups; 16 groups: 6.2% complete training groups; 64 groups: 0.0% complete training groups.

[Full training/development results, name coverage and English forgetting](experiments/acquisition/report.md). Memorization and held-out transfer are distinct requirements. Fixed budgets reduce presentations per example at larger sizes, so outcomes do not prove a capacity limit. These task-only checkpoints are diagnostic and do not replace English models. Next control exposure per example and compare any curriculum with matched flat training; use fresh confirmation after changes.


## 36. mini-AGI and Ornith inspirations: 2026-10-24

Method: inspected public repository pages, selected source, configuration, model card and author technical articles. Neither project was installed, trained or independently benchmarked. Branch URLs move; cached pages have different retrieval dates, and a consistent commit-pinned snapshot was not obtained. Conclusions describe the inspected material, not a reproduced release. Earlier section14 remains a historical review.

mini-AGI supplies model-level references: its [decoder code](https://github.com/volotat/mini-AGI/blob/main/minagi/model.py) uses attention, so its growth mechanism does not demonstrate an attention-free replacement. Its [recurrent core](https://github.com/volotat/mini-AGI/blob/main/minagi/recur.py) reuses blocks with learned halting, related to [PonderNet](https://arxiv.org/abs/2107.05407). Extra processing depth is distinct from extra stored capacity.

Inspect actual growth behavior: [AutoGrow and parameter grouping](https://github.com/volotat/mini-AGI/blob/main/minagi/pool.py) implement guarded speculative additions, small birth contributions and survival checks. Older top-of-file saturation/entropy wording differs from this implementation. Experts have no assigned linguistic labels. The parameter split places router weights with the pool in the inspected implementation; avoid assuming every description of the trunk uses identical parameter groups.

[Paging](https://github.com/volotat/mini-AGI/blob/main/minagi/paged.py) separates stored, RAM-cached and device-resident experts and associates optimizer moments with expert identity. Reduced residency does not eliminate storage or transfers. Any later MLM offloading needs prefix-causal admission tests, preserved optimizer identities and actual end-to-end measurements.

The [author's retention experiments](https://github.com/volotat/mini-AGI#how-continual-learning-works) report benefits from slower shared updates and interleaved domains. These are bounded author results. Interleaving does not mathematically prevent forgetting. The [probe implementation](https://github.com/volotat/mini-AGI/blob/main/replication/forgetting_probe.py) compares optimizer restoration/cold starts and can hold growth fixed, useful controls for MLM's deliberate optimizer resets.

Ornith is a different layer. The [1.0 article](https://ornith.ai/ornith_1_0.html) describes post-training pretrained Qwen/Gemma models while jointly improving task scaffolds and solutions. A scaffold is a task-solving procedure, not a neural connection graph. The [1.5 article](https://ornith.ai/ornith_1_5.html) adds proposed tasks and rewards validity, learnable difficulty and novelty. Its self-improvement still requires training updates and verification. The [repository](https://github.com/ornith-ai/Ornith-1) inspected primarily exposes documentation/assets/model links; a runnable end-to-end training implementation was not established. Published scores were not independently validated here.

MLM proposals, not implemented outcomes:

1. Complete equal-exposure acquisition controls before attributing larger-set failure to capacity.
2. Compare slower shared embedding/message/output updates with faster node-specific updates; treat router rates separately. Include uniform-rate and rehearsal controls at explicit matched budgets. Measure binding gain versus English loss, not retention alone.
3. Use externally verifiable training tasks near measured competence, retain easy replay, and reserve disjoint confirmation. Our current MLM should not author or certify its own answers. Scheduler rules are training controls, not an inference answer solver.
4. Compare fixed message-round counts before learned halting. Train changed recurrent dynamics fairly; report processing cost as well as accuracy.
5. Investigate guarded growth and paging only after acquisition/retention gates pass. Newborn protection, stable IDs and counterfactual utility tests matter more than assigning nodes names such as verbs or pronouns. Usage alone cannot establish semantic specialization or safe pruning.

These repositories offer hypotheses for learning rules, capacity management and training feedback. Neither establishes that MLM's fixed GRU graph can replace Transformers, that unlimited growth solves binding, or that arbitrary devices can host the resulting model.


## 37. Equal-exposure acquisition control

The unchanged acquisition runner completed paired 16-group/2,000-update task-only trials from each own English checkpoint, matching the four-group study's 125 presentations per group. Same config, initial parameters and fresh optimizer as the previous 16-group trials; update traces share the first 500 group IDs and paired byte counters match. No test set or checkpoint selection was used. Four times the original update budget is explicitly counted.

dense: 51.56% training answer ranking, 0/16 complete training groups, 0/144 complete development cases; English development loss 9.424430.

hierarchical: 95.31% training answer ranking, 14/16 complete training groups, 0/144 complete development cases; English development loss 9.336111.

[Protocol, exposure controls, name coverage and measured results](experiments/exposure/report.md). Fixed-endpoint exploratory results are not a capacity proof or independent final confirmation. Task-only English forgetting prevents promotion. Next isolate shared/node learning rates and rehearsal before any growth or deeper hierarchy.


## 38. Shared/node rate and rehearsal study

Implemented opt-in shared/node/router learning-rate multipliers on completed AdamW updates, preserving default behavior, optimizer-state structure and legacy checkpoint loading. Six fixed-endpoint hierarchical trials compare high uniform, low uniform and split representation rates, each with/without English rehearsal. Same own English source, 16 groups and 2,000 QA updates; rehearsal adds 2,000 English updates. No test data or endpoint selection.

high-task: 14/16 complete training groups; 0/144 complete development cases; English development loss 9.336111.

high-rehearsal: 16/16 complete training groups; 0/144 complete development cases; English development loss 1.319806.

low-task: 12/16 complete training groups; 0/144 complete development cases; English development loss 3.537662.

low-rehearsal: 5/16 complete training groups; 0/144 complete development cases; English development loss 1.307681.

split-task: 16/16 complete training groups; 0/144 complete development cases; English development loss 6.268202.

split-rehearsal: 16/16 complete training groups; 0/144 complete development cases; English development loss 1.305334.

[Frozen protocol, costs, optimizer/default reproduction and full results](experiments/plasticity/report.md). This is an exploratory fixed-rate/data-schedule study, not an adaptive growth mechanism. Acquisition/retention and held-out transfer remain separate gates. No model is promoted without fresh confirmation.


## 39. Diversity and wording transfer confirmation

Completed three matched update-budget rehearsal trials from the same English-only source: fixed 16, diverse 128 and mixed-wording 128 training groups. New seed 904217 task pools reserve familiar-name pairings and new-name labels for development/test; every arm covers the same 16 training labels. Canonical/mixed underlying episodes and sample traces match; wording differs. No architecture/optimizer/trainer changes or old QA initializers.

fixed: 0/16 complete training groups; 0/288 complete fresh test cases; English development loss 1.303425.

diverse: 0/128 complete training groups; 0/288 complete fresh test cases; English development loss 1.296879.

mixed: 0/128 complete training groups; 0/288 complete fresh test cases; English development loss 1.301324.

[Full test buckets, source baseline, locked checkpoints, raw samples and exposure limits](experiments/transfer/report.md). Fixed versus diverse changes partner/color diversity and repetitions per example at a fixed budget; it is not a capacity proof. Transfer requires at least 90% complete cases in every fresh name/wording bucket, separately from acquisition/retention. No model is promoted without that gate and further confirmation.


## 40. Learned delta-memory architecture pilot

Implemented optional memory=delta on the fixed sum-readout graph, using learned key/value/query/write projections and a document-local delta-rule matrix. Read-before-write, reset/padding/chunk contracts and checkpoint continuation are tested. Zero output gain preserves source predictions mathematically; packed CPU scan roundoff is bounded and real initial English loss differs by 7.01e-9. Default memory=none preserves legacy graph behavior. Extra 16,514 parameters and state 2,048→6,144 bytes are explicit.

none: 0/128 complete training groups, 0/288 complete new test cases; English development loss 1.296104.

delta: 0/128 complete training groups, 0/288 complete new test cases; English development loss 1.291094.

[Full mechanism, primary-paper links, source/trace/lock audits and measured state/CPU costs](experiments/associative/report.md). Both arms use the same English source and 2,000 QA + 2,000 English updates on new seed 638101 diverse data. This is an adaptation pilot with extra parameters/state/compute, not a new delta-rule invention, a parameter-matched proof or an optimized parallel implementation. Learning useful language addresses remains distinct from toy memory algebra. No model is promoted; 68 tests and review passed.


## 41. Learned memory addressing diagnostic

A frozen-controller tracer now reconstructs the delta read as an exact signed sum of earlier writes, accounting for later overwrites and excluding current/future writes. Actual model steps, full-prompt execution and output projection are checked. Canonical fact metadata only annotates the results offline. No model or training behavior changed.

On the failed pilot's first eight training prompts, final entity-swap query cosine was0.9999717–0.9999768. Gates increased to about0.80–0.89 across facts/question and0.83 on the assistant header. Header writes had the largest mean projected conditional region norm; graph state can carry earlier facts into these writes, so this is not proof of erasure. Frozen readout removal changed9/512 training rankings, reduced accuracy51.76%→50%, and left complete groups0/128. It worsened assistant-target loss0.123376→0.221659. Memory contributes without reliable binding.

Four independent source-initialized500QA+500English probes: graph and delta each fit1/1 training group and all four fixed raw answers; both complete0/4 groups on the larger subset, at50% question accuracy. All complete0/2 tiny development groups. Each four-group item gets one quarter the exposure of the single-group item; this does not isolate diversity/order effects or prove capacity limits. Disabling memory readout preserves all fitted single-group rankings. No new test set was opened or model promoted.

See [full report](experiments/addressing/report.md), frozen protocol, traces, signed projected vectors and integrity audits.71 tests passed; static checks and independent review passed. Next separate exposure from diversity, then consider one learned query-retention/write-selectivity change with a matched control. Growth and Transformer-replacement claims remain unsupported.


## 42. Four-group exposure control: positive narrow acquisition

With unchanged model/training algorithms, independent original-source initialization and2,000QA+2,000English updates, delta acquires16/16 training answers and4/4 complete groups; graph remains8/16 and0/4. Both receive exactly500 presentations per group, matching earlier single-group QA exposure. Delta fixed first-group raw generation is4/4, graph2/4. Frozen post-hoc memory-output removal drops delta to8/16 and0/4 (8 flips), showing coadapted readout dependence.

Both score4/8 and0/2 complete groups on the tiny reused development case; no new test files opened. English dev loss is1.291641 graph/1.286896 delta versus source1.352696. Additional total optimization/rehearsal confounds comparison with shorter runs; delta also has16,514 extra parameters. This revises the earlier finite-budget acquisition failure, not the generalization verdict. One seed/two names cannot establish general binding or Transformer replacement. See [report](experiments/addressing-exposure/report.md), frozen protocol and audits. Next replicate the exact acquisition comparison across predefined seeds before fresh transfer confirmation.


## 43. Adaptation-seed replication: acquisition not stable

Predefined seeds43 and44 independently repeat graph/delta four-group adaptation from the same original English checkpoint, each2,000QA+2,000English,500 presentations/group. Delta training rankings8/16 and9/16, complete groups0/4 at both; graph8/16 and0/4 at both. The predefined both-new-seeds acquisition gate fails0/2. All trials pass retention (Englishdev≈1.255–1.266 versus source1.352696), but all complete0/2 tiny reused development groups. Raw fixed first-group outputs2/4 each. No test opened, budget extended, replacement seed or promotion.

This qualifies section42: seed42 narrow acquisition remains real but is not reliably repeated in these new seeds. Seed42 is a selected observed reference, excluded from the new2-seed denominator. Shared graph initialization is fixed; adaptation seed changes QA order, English stream and delta-extra initialization together, so no individual cause is identified. Neural/training code unchanged; all74 input hashes, exact checkpoint budgets/source tensors and matched within-seed sampling audited. See [report](experiments/addressing-replication/report.md). Next decouple memory initialization and sampling/rehearsal order before proposing an addressing fix or fresh transfer confirmation.


## 44. Initialization versus combined schedule: conditional schedule pattern

Completed the missing cells of a post-hoc2×2 control using saved step-zero memory initializations42/43 and combined schedule seeds42/43. New init42/schedule43 scores8/16 training rankings,0/4 groups; new init43/schedule42 scores16/16,4/4. Reused diagonals42/42 and43/43 are16/16,4/4 and8/16,0/4 respectively. Both tested initializations acquire with schedule42; neither acquires with43. Primary complete-group schedule contrast−100pp, average initialization contrast0 and interaction0 over this selected table. Initialization can still affect other behavior: schedule43 query flips differ across initializations.

All new runs use fresh optimizers and2,000QA+2,000English updates,500/group, with every non-memory tensor exactly original-source matched. English losses1.270391/1.285907 pass retention; tiny development4/8 and3/8,0/2 groups both. No fresh test/ablation/promotion.203 artifact/input checks passed; recorded historical software/backend/source fields match, but old numerical flags were not recorded. Factor levels were outcome-selected and diagonals reused; no population or independent replication claim. QA order and English document exposure/order remain bundled. Prior replication gate remains failed. See [report](experiments/addressing-factorial/report.md). Next separate QA and English RNGs with fixed memory initialization before changing architecture.


## 45. QA versus English replay sampling: 2026-10-25

Added optional independent QA/English sampler seeds to the plasticity runner, preserving legacy defaults, config initialization/subset seed and all neural/optimizer/scoring math. Full75-test suite passed; actual preserved old/new-default paths match parameters and Adam moments bitwise after8QA/replay pairs. Old driver/tests preserved and only those two files intentionally changed.

With fixed memory initializer42, English43 yields16/16 training rankings and4/4 groups under both QA42 and43. English44 yields7/16 and8/16,0/4 groups. New crosses are QA42/E44 andQA43/E43; other two cells are historical/outcome-selected. Complete-group English contrast−100pp, QA contrast0, interaction0; question-accuracy contrasts−53.125pp English,+3.125pp QA,+6.25pp interaction over this table. All cells retain English loss<1.29 but complete0/2 reused development groups. No new test, ablation or promotion; prior independent-seed gate remains failed.

All219 input/endpoint checks passed, including115 frozen hashes, actual seed policies and separate QA/English schedule/counter matches. English streams differ in document exposure/order and200 total bytes, so replay-order versus content effects and gradient interference are not identified. Historical driver-version and missing QA42/E43 numerical flags limit causal interpretation. See [report](experiments/schedule-components/report.md). Next measure QA/replay gradient and optimizer-step interference on frozen training probes before choosing a training intervention; selecting a favorable replay seed is not a general fix.


## 46. Frozen replay/optimizer diagnostic: local harm, cause unconfirmed

Completed80 predefined training-only probes over the common initializer and all4 fixed-init42 endpoints. Raw QA-loss change from a copied English Adam step is snapshot-dependent; replay44 is not consistently worse than43. At fittedQ42/E43 all16 replaysteps harm QA CE (15/16 relative to zero), while failedQ43/E44 has16/16 raw helpful steps. A zero-gradient optimizer control moves QA loss substantially, so raw harm alone cannot be assigned to the new English gradient. No post-step answer ranking or generalization was measured.

Each source retains actual Adam moments, clipping, decay and role scaling; independent candidates are discarded. QA CE covers16questions/96assistant-newline-EOS targets. Streams43/44 use positions0,1,2,3,499,999,1499,1999 with frozen-document-prefix carry, not historical changing-weight carry. Endpoints follow the final English update, so these are extra-English steps without the normal preceding QA phase.72 frozen hashes,10 unchanged core checks,78 regression tests and504 runtime audits pass; checkpoint files remain unchanged. See [report](experiments/replay-interference/report.md). Next test the copied actual QA→English next-loop pair with saved carry/sampler state before selecting an intervention. Earlier failed acquisition replication and generalization verdicts remain unchanged.


## 47. Actual-next QA→English pairs: mixed local effects

Five restored next-loop pairs use complete QA history, actual English stream/carry and loaded optimizer states, with matched English-only and zero-gradient controls:25 copied transitions. QA raises aggregate CE at3/5 snapshots; English after QA raises it at2/5, while control-adjusted effects have different membership. Selected-group and aggregate changes can disagree. These results do not isolate Adam or show universal replay erasure.

All80 regression tests and191 runtime audits pass;78 frozen hashes including72 prior inputs remain unchanged. No checkpoint, neural/training math, ranking, held-out evaluation or release changed. Next isolate optimizer-history and router-balance contributions to the QA step before a separately frozen intervention. Earlier failed replication/generalization gates remain unchanged. See [report](experiments/qa-replay-pair/report.md).
