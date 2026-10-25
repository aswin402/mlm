# Frozen replay-interference diagnostic protocol

## Purpose and scope

Test local optimizer-step interference on predefined training probes, not generalization, acquisition improvement or a historical causal explanation. Model/optimizer/training math and saved checkpoints remain unchanged. Work is isolated on research/replay-interference; v0.0.1 stays on main. No new training run, disk checkpoint update, test-set reading, favorable-seed selection, architecture change or new release/push is part of this diagnostic.

## Predefined samples

Use the common step-zero initializer42 plus all four fixed-init42 endpoints from the QA/replay component table: Q42/E43 and Q43/E44 historical endpoints, and Q42/E44 and Q43/E43 crossed endpoints. Load full Adam state/config, never reset moments for the primary probe. Use all16 training questions (four complete groups) for one aggregate, position-weighted assistant-target CE including newline/EOS; it is not candidate ranking or all-input loss.

From each English training DocumentStream seed43/44, choose zero-based batch positions0,1,2,3,499,999,1499,1999, B4,L128. Selection is fixed before diagnostics:16 probes/checkpoint,80 total. Reconstruct the document-local incoming recurrent carry by reading only each active document's already-consumed prefix with the frozen checkpoint weights; padding freezes rows, empty/new-document prefixes start at zero and normal resets apply. Stop gradients through this reconstructed carry. This is a frozen-context counterfactual, not the carry from historical training with changing weights. English document exposure/order differs, and the16 samples are not IID repetitions or the whole corpus.

## Measurements

For each checkpoint/probe, start independently from the same loaded state. Compute g_Q as the gradient of aggregate masked QA language CE only. Compute g_E as the exact existing routed English training objective gradient, including router balance. Apply the existing TrainState optimizer to g_E, retaining moments, global clipping, weight decay and completed per-role update scaling. Also apply a zero gradient from that identical original state as a momentum-and-decay control. Neither candidate becomes the starting state for another probe.

Also record the copied step's effect on its own English objective and language CE, with incoming carry held fixed. Record baseline QA CE, QA CE after the copied English step, QA CE after the copied zero-gradient step, actual changes, English-minus-zero QA-loss difference, raw g_Q/g_E cosine, first-order g_Q dot actual parameter delta, and nonlinear remainder. Report raw gradient norms, actual update norms and signed dot contributions by shared/node/router/memory role; these contributions sum to the full first-order estimate. Undefined cosine for zero norm is null. Verify finite results and supervised positions. Tiny absolute QA-loss changes within1e-6 are numerically unresolved for count summaries; report raw numbers too.

Primary descriptive comparison: per frozen checkpoint, paired English44-versus43 differences in actual QA-loss change across the same eight positions, plus harmful/helpful/unresolved counts at threshold1e-6. Include zero-control-adjusted values. A positive copied-step effect establishes local harm on these probes, not a trajectory mechanism or broad superiority. Gradient cosine alone cannot establish actual Adam interference; the actual QA CE is recomputed after each copied update.

## Implementation and verification

Add src/mlm/interference.py and focused tests only; add a bounded diagnostic CLI/runner with compact evidence under docs/evidence/replay-interference and raw output under ignored results/replay-interference. Preserve source/data/checkpoint hashes before analysis and afterward. Tests first: closed-form masked CE on a tiny known model with populated Adam moments/decay/role scaling; optimizer candidate against make_routed_step; gradient/actual-step accounting and source immutability; frozen-prefix reconstruction against literal sequential frozen reading including resets/padding; reject empty supervision/nonfinite results. Run meaningful targeted/full regression and static checks, plus independent protocol/code/report review before final claims.

Freeze protocol, exact selections, core/data/driver/checkpoint hashes, configs/environment and batch/prefix fingerprints before the real diagnostic. Archive per-probe and per-role metrics, compact summary, numerical controls and audits. Update documentation under docs and mark any new functionality Unreleased without changing v0.0.1. Do not hardcode a favorable replay seed or change training based on exploratory results without a separately frozen intervention.
