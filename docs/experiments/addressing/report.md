# Learned-memory addressing diagnostic

The memory is active, but this experiment finds no reliable entity–color binding advantage. Both graph control and delta model memorize one group; both fail the four-group probe at this fixed budget. No architecture or training behavior changed, and no new test scores were computed.

## Protocol

The frozen [plan](../../archive/README.md#historical-plans) and [protocol](../../evidence/addressing/protocol.json) specify the analysis. The reproducible [recipe](../../evidence/addressing/recipe.py) uses the original English-only checkpoint independently for each trial, fresh AdamW, seed42, identical split role rates, batch4, 500 QA plus500 English updates. Four trials total4,000 optimizer updates. None starts from the failed pilot or another probe.

Training subsets are the first4 and16 records of the old diverse pool: one group (two assignments ×two queried entities), or four groups (same Bon/Bom pair, gold/lime and blue/pink, both fact orders). Tiny development is the first complete eight-condition case from the old development pool. It is exploratory and previously available, not a fresh test. Each one-group trial sees its group500 times; each four-group trial sees each group125 times. Therefore these probes do not isolate diversity, order reversal or exposure as causal factors. No checkpoint selection or promotion.

## Small-set results

Candidate ranking is the unchanged mean neural answer-byte likelihood scorer. Raw answers below use the fixed first training group and only strip/casefold normalization. They are training samples.

| Trial | Training questions correct | Complete training groups | Raw samples correct | English dev loss |
|---|---:|---:|---:|---:|
| addressing-none-g1-v1 | 100.00% | 1/1 | 4/4 | 1.361780 |
| addressing-delta-g1-v1 | 100.00% | 1/1 | 4/4 | 1.359972 |
| addressing-none-g4-v1 | 50.00% | 0/4 | 2/4 | 1.367070 |
| addressing-delta-g4-v1 | 50.00% | 0/4 | 2/4 | 1.363914 |

All probes complete zero of two tiny development groups. One-group raw success is memorization, not evidence of general binding or English chat readiness. Four-group failures at500 QA updates do not prove a capacity limit. English development losses are around1.36 versus1.352696 for the source; this is retention evidence on a reused development set.

## What the failed pilot memory learned

We traced the first eight original training prompts (one Bon/Bom gold/lime case, both fact orders), both before and after the earlier2,000QA+2,000English pilot. Prompt bytes alone enter the tracer. Facts/labels only annotate regions afterward; no symbolic lookup or answer repair enters inference.

The tracer reconstructs keys, queries, values, write gates, old-matrix reads, delta updates, pooled residuals and output logits from actual graph states. It checks token stepping against an uninstrumented full-prompt forward. For a read at t, the exact signed contribution of an earlier write s is beta_s × q_tᵀ A_(t−1)…A_(s+1) k_s × v_s, with A_j=I−beta_j k_j k_jᵀ. The backward algorithm excludes the current and all future writes. Coefficients can be negative; they are not attention probabilities.

Across the eight trained prompts, mean write gates were0.802 for fact1,0.885 for fact2,0.887 for the question and0.831 for the assistant header. Their initial values were roughly0.11–0.12. Mean absolute coefficient shares were14.71%,9.91%,43.29%,28.44% respectively, with the remainder BOS/user header. This statistic is conditional on the realized controller trajectory, not a probability or causal attribution.

The [projection analysis](../../evidence/addressing/project_regions.py) sums signed vectors within each region before applying the learned output kernel and gain. Mean projected region norms were0.1773 fact1,0.0747 fact2,0.0950 question and0.5708 assistant header. These norms do not add because region vectors can cancel. All initial projected contributions are exactly zero because the initial gain is zero. Region sums reconstruct the actual read and projected residual. Header/question writes can contain earlier factual information carried by the graph, so these measurements do not prove facts were erased.

Final query cosines for fixed-context entity swaps were0.9999717–0.9999768. Recalled-vector cosines were0.9794–0.9868. Mean projected residual/base-pooling norm ratio was0.08048 before layernorm. This is consistent with weak entity discrimination at the final read; similarity alone cannot prove why binding fails.

Maximum conditional-read reconstruction error on these eight trained pilot prompts was4.64e-07. Full traces, raw coefficients, values, reconstruction errors and query swaps are archived in the per-checkpoint JSON files. The residual decomposition is linear before layernorm; it is not a decomposition of final answer scores.

## Frozen readout ablation

Only memory_gain is set to zero in a copied parameter tree. No weights are retrained or written back. The graph/controller/matrix trajectory remains the same for fixed teacher-forced inputs; memory writes and their compute still execute. This tests dependence on a coadapted output channel, not the benefit of training without memory.

| Failed pilot metric | Readout on | Readout off |
|---|---:|---:|
| Training accuracy,512 questions | 51.7578% | 50% |
| Complete training groups | 0/128 | 0/128 |
| Assistant-target training loss, including newline/EOS | 0.123376 | 0.221659 |
| Mean correct-minus-wrong candidate score | 0.005023 | 0.000185 |
| English dev loss | 1.291094 | 1.289253 |

Nine training predictions flip. The memory improves training likelihood but does not solve binding. The loss and candidate-ranking metrics use different masks: training loss includes assistant answer/newline/EOS, while candidate scores include answer bytes only.

For the fitted one-group delta probe, all four training rankings stay correct with the readout disabled (zero flips), although target loss worsens0.001597→0.011597. The graph can support that fitted ranking without the memory output. For the four-group delta probe, two rankings flip, accuracy remains50%, and complete groups remain0/4; target loss worsens0.118962→0.307034. These results show active contribution without reliable binding.

## Verification and artifacts

The complete suite passed:71 tests in185.74s. New tests verify signed overwrite reconstruction, exclusion of current/future writes, actual/full-prompt equality, byte-region coverage and readout-ablation immutability. Ruff lint/format and ty checks passed for the new diagnostic helper/tests. Independent review found no helper issue and requested projected region contributions; the separate postprocessor implements those and passed follow-up review. No model, training, inference or dataset-generation algorithm changed.

Frozen inputs are checked in integrity.json (`results/addressing/integrity.json`; local artifact); all83 hashes from the earlier associative pilot also match prior-input-integrity.json (`results/addressing/prior-input-integrity.json`; local artifact). Source/config/test snapshots, protocol, logs, per-run source hashes, fresh optimizer fingerprints, sampler traces and counters are saved. Checkpoints and raw generations are under runs/addressing-*-v1. No original model files were overwritten.

## Next experiment

First separate exposure from diversity with a predeclared equal-exposure control, and retain the original source initialization. If exploring an addressing change, test one learned mechanism at a time: a controller that preserves the queried entity representation through the assistant header, or a more selective learned write gate. Keep keys/queries/writes learned from bytes, with no category dictionary, entity parser or label-driven routing. Compare against the unchanged graph at matched exposure and report training acquisition before fresh transfer testing.

This diagnostic does not validate growing nodes, category-specialized webs, sparse execution, broad language understanding or a Transformer replacement. It narrows the problem to learned addressing and acquisition under the present setup.
