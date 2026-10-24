# Fixed-source adaptation-seed replication

The four-group acquisition result did not replicate at the two additional predefined seeds. Delta scored8/16 (seed43) and9/16 (seed44) training rankings, with0/4 complete groups at each seed. Both graph controls scored8/16 and0/4 groups. The predefined delta acquisition gate fails:0/2 new seeds succeeded. English retention passes at both seeds for both models.

The earlier seed42 delta result remains16/16 and4/4, but it is a separately labeled, previously observed reference. It does not turn the new replication result into a success. These data show possible narrow acquisition with unstable outcomes under this adaptation setup, not reliable general binding.

## Predefined protocol

The [plan](../../archive/README.md#historical-plans) and [protocol](../../evidence/addressing-replication/protocol.json) fixed seeds43/44, both model arms, and all budgets before training. Every trial starts from the same original English checkpoint with a fresh AdamW optimizer and the existing role rates. Each receives2,000QA+2,000English updates (4,000 optimizer updates per trial;16,000 total). Every training group is presented exactly500 times. No extension, replacement seed, checkpoint selection, ablation, new test scoring or model promotion occurred.

This is adaptation-seed replication, not independent pretraining. Shared graph tensors are copied exactly from the same English source at every seed. The seed changes QA group ordering and English document ordering; delta's additional memory parameters also change initialization. These factors vary together, so a failure cannot be assigned specifically to memory initialization, QA order or English rehearsal order. Within a seed, graph and delta receive identical QA order, byte budgets and English stream. Delta retains its extra16,514 parameters and matrix state, so this is not a parameter-count or compute-matched comparison.

The dataset remains16 aligned training records for Bon/Bom, two color pairs (gold/lime, blue/pink), both fact orders, complementary assignments and both queried entities. Metadata only grades outputs; neural inference is unchanged. The eight-condition tiny development case is reused exploratory data and no test file is opened.

The acquisition endpoint requires all16 candidate rankings and all4 complete training groups correct. The aggregate gate requires both new delta seeds to pass (2/2). A separate retention gate requires English development loss≤1.6908700424, which is1.25×source1.3526960339. Raw first-group samples and tiny development are secondary. All predefined trials stay in the denominator.

## New-seed results

| Adaptation seed | Memory | Training rankings | Complete groups | Fixed raw training samples | English dev loss |
|---|---|---:|---:|---:|---:|
| 43 | none | 8/16 | 0/4 | 2/4 | 1.265716 |
| 43 | delta | 8/16 | 0/4 | 2/4 | 1.264488 |
| 44 | none | 8/16 | 0/4 | 2/4 | 1.255028 |
| 44 | delta | 9/16 | 0/4 | 2/4 | 1.256495 |

Both models pass retention at both seeds, with English losses lower than the original source on this reused development set. Acquisition successes are0/2 for graph and0/2 for delta; joint acquisition+retention successes are0/2 each. Delta's one additional correct ranking at seed44 does not complete a group and cannot establish a stable advantage.

Every new trial scores4/8 on the tiny development case,0/2 complete groups and0/1 complete eight-condition cases. All four fixed first-group raw generations score2/4 after strip/casefold, with no answer extraction or repair. The raw sample set is only four training prompts; it is not all16 generations or a chat benchmark.

For context only, the selected seed42 reference had graph8/16,0/4 groups,2/4 raw samples,English1.291641; delta16/16,4/4 groups,4/4 raw samples,English1.286896. Do not pool that observed success with the new seeds to claim the predefined replication gate passed. Two new seeds and a tiny task cannot estimate a general success rate or prove a capacity limit.

## Verification and reproducibility

All four frozen commands finished with exit0. The existing driver rejects nonfinite updates. The [audit](../../evidence/addressing-replication/audit.json) verifies all74 frozen input hashes, runner identity, all four dataset groups and exactly500 presentations each, fresh initial optimizers,4000 steps in actual saved final states, config/counter agreement with actual checkpoints, exact shared source tensors and zero initial delta gate. Within each seed, QA order, byte counts and final English stream agree across models. Checkpoints, full scores, fixed raw samples, optimizer/source fingerprints, sampler histories, config and environment are stored under `runs/addressing-replication-{none,delta}-s{43,44}-v1`. Logs are alongside this report. The launcher refuses existing output directories.

Neural/training/test code did not change. The prior71-test passing result is the regression basis; the full suite was not rerun for this experiment. Fresh Ruff lint and ty checks passed for launcher/active analyzer; analyzer formatting passed. The frozen launcher retains a cosmetic blank-line formatting deviation noted in [verification](../../evidence/addressing-replication/verification.json). Independent review found a completeness gap in the first analyzer draft; it was fixed to require16 questions and4 groups before freezing.

An optional type check later found ambiguous heterogeneous dictionary inference in the analyzer. The original is preserved as `analyze-frozen-v1.py`; the active version adds only an `Any` import and dictionary annotation. Normalized executable ASTs are identical and independent review confirmed no scoring/gate change. Amendment (`results/addressing-replication/analysis-type-amendment.json`; local artifact) records hashes and timing. The experiment protocol and model code were not changed.

## Decision and next step

Do not promote the successful seed42 checkpoint as a reliable general model, or spend a fresh transfer test to imply acquisition is already stable. Preserve all failed and successful trials. The strongest supported conclusion is: document-local delta memory can participate in fitting this task, but that fit is not reproducible across these two new adaptation seeds.

Next, separate memory initialization from QA/rehearsal ordering in a predeclared factorial diagnostic while keeping source weights, data, learning rates and budgets fixed. This can distinguish an initialization problem from a schedule/order problem before selecting a learned addressing intervention. No new inference parser, category dictionary, hardcoded answers, node growth or Transformer-replacement claim follows from the current evidence.
