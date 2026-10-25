# Frozen replay/optimizer interference diagnostic

The diagnostic finds local QA-loss interference, but does not establish why replay 44 failed acquisition. English replay steps can hurt masked QA CE at fitted snapshots, while momentum and decay alone also move QA loss. Replay44 is not consistently more harmful than 43 across the frozen states. No training intervention or checkpoint change was made.

A phase limitation matters: endpoint checkpoints were saved after their final English update. These probes add another English step without the normally intervening QA step. Their incoming context is reconstructed using frozen weights, not the historical carry produced while weights changed. This is a local counterfactual, not a replay of the alternating training trajectory.

## Fixed measurements

The [predefined protocol](protocol.md) selected the common initializer plus all four fixed-init42 QA/replay endpoints. Each checkpoint is tested independently against English 43/44 at zero-based stream batch positions 0,1,2,3,499,999,1499,1999:80 probes total. Selection was frozen before computation. No validation or test dataset was read.

QA loss is global token-weighted cross-entropy over all 16 training questions, four complete groups and 96 supervised assistant/newline/EOS positions, evaluated from fresh document states. This is not answer-candidate ranking, raw generation accuracy or a measure of generalization. English gradients use the unchanged training objective, including router balance. QA gradients use language CE only.

Every probe starts from its loaded checkpoint's parameters, Adam moments and counter. The copied English update retains actual global clipping, Adam history, weight decay and per-role completed-update scaling. A separate zero-gradient update starts from the same original state. It advances moments/decay; it is not a no-op. Candidate states are discarded rather than chained across probes. No model checkpoint was written.

## Actual QA-loss changes

Positive change means worse QA CE. Means below are over eight selected positions per stream. Absolute changes≤1e−6 are treated as numerically unresolved for count summaries; this is an operational tolerance, not a statistical significance test. The initializer has a much larger loss scale and should not be pooled with fitted endpoints.

| Frozen checkpoint | QA CE before | Zero-gradient ΔQA | Mean replay43 ΔQA | Mean replay 44 ΔQA | Mean 44−43 difference |
|---|---:|---:|---:|---:|---:|
| initial | 6.86677 | +8.10623e-06 | +0.0321449 | -0.0375937 | -0.0697386 |
| q42_e43 | 0.000819306 | +2.38816e-05 | +3.55194e-05 | +2.97067e-05 | -5.81269e-06 |
| q42_e44 | 0.116436 | +4.93526e-05 | +4.21712e-05 | +4.20073e-05 | -1.63913e-07 |
| q43_e43 | 0.00121691 | +1.43349e-05 | +9.34051e-06 | +1.24077e-05 | +3.06717e-06 |
| q43_e44 | 0.115562 | -0.00015232 | -0.000182431 | -0.000175895 | +6.53602e-06 |

Replay44's mean effect is lower at the initializer and Q42/E43 snapshot, nearly equal at Q42/E44 (difference−1.64e−7), and higher at both Q43 snapshots. Position-matched differences are mixed. The positions match stream indices, not identical documents. These dependent, selected samples do not estimate a population-level replay effect.

All 80 copied steps reduce their own sampled English training objective by more than1e−6. That does not make their QA effect uniformly harmful or helpful.

## Momentum/decay control

“Relative harm” below means QA CE after the English-gradient step exceeds QA CE after the zero-gradient step. It is a controlled optimizer counterfactual, not an additive decomposition of historical training causes.

| Frozen checkpoint | Probe stream | Raw harmful | Harmful relative to zero | Mean English-minus-zero QA CE |
|---|---:|---:|---:|---:|
| initial | 43 | 5/8 | 5/8 | +0.0321368 |
| initial | 44 | 2/8 | 2/8 | -0.0376018 |
| q42_e43 | 43 | 8/8 | 8/8 | +1.16378e-05 |
| q42_e43 | 44 | 8/8 | 7/8 | +5.82514e-06 |
| q42_e44 | 43 | 8/8 | 2/8 | -7.18143e-06 |
| q42_e44 | 44 | 8/8 | 2/8 | -7.34534e-06 |
| q43_e43 | 43 | 5/8 | 3/8 | -4.99441e-06 |
| q43_e43 | 44 | 7/8 | 2/8 | -1.92724e-06 |
| q43_e44 | 43 | 0/8 | 1/8 | -3.01115e-05 |
| q43_e44 | 44 | 0/8 | 1/8 | -2.35755e-05 |

At fitted Q42/E43, both streams hurt QA in all 16 raw probes and 15/16 relative-to-zero probes. This is direct local conflict on these inputs, with its own English objective improving. The absolute mean raw QA changes are only about3e−5; no post-step candidate rankings or generated answers were evaluated, so this is not evidence that a correct answer actually flipped.

At failed Q42/E44, all 16 raw probes look harmful, but the control-adjusted counts are 4 harmful, 11 helpful and 1 unresolved. The positive zero-gradient drift is larger than either stream's mean raw increase. At failed Q43/E44, all 16 raw probes are helpful, while two are harmful relative to zero. Raw English-step change and the incremental effect of supplying its gradient answer different questions.

At fitted Q43/E43, 12 raw probes are harmful, 3 helpful and 1 unresolved; relative to zero, 5 are harmful, 9 helpful and 2 unresolved. An English gradient can soften pre-existing optimizer drift while the completed step still raises QA loss.

## Gradient and role measurements

The raw cosine is between the aggregate QA-language gradient and the English training-total gradient. Positive first-order g_Q·Δθ predicts a local QA-loss increase along the actual completed optimizer update. Both are secondary to recomputed CE. Nonlinear remainders are material, especially at initialization and failed Q42/E44, so cosine or a linear approximation alone cannot establish interference.

Shared/node/router/memory gradient norms, actual update norms and signed first-order contributions are archived for every probe. Contributions sum to the complete first-order projection. At the two fitted endpoints the memory role has the largest positive mean projection, but this includes optimizer history and decay and is not a causal ablation of memory or proof of fact erasure. No layer-freezing intervention was tested.

## Verification and artifacts

The complete regression suite passed 78 tests in 240.93s. The strengthened three diagnostic tests passed in 27.90s before replacing a deprecated test-only norm alias. Tests cover closed-form nonuniform router-balance gradients, QA language-only gradients, unequal QA token lengths with newline/EOS, exact trainer-equivalent copied updates with populated moments/clipping/decay/role scaling, zero controls, state immutability, document-prefix carry versus sequential reading, and explicit prefix gradient detachment. Empty supervision and nonfinite inputs are rejected. Ruff lint/format and ty checks pass for the diagnostic code.

Independent review found no material helper/runner errors and verified all 72 frozen hashes, five source mappings, optimizer steps 0/4000, CPU environment, exact selections and all ten unchanged model/optimizer core hashes before analysis. The executed process also records actual JAX numerical and whitelisted thread/environment settings. The exact executed metadata wrapper is preserved as text with its hash; an unused-import cleanup of the reusable local recorder occurred after the run without modifying frozen measurement code.

All 504 runtime audits passed. They verify regenerated QA/probe fingerprints, complete 80-probe coverage, source-state byte hashes before/after each checkpoint, unchanged input/checkpoint files, independent counters, supervised positions, role contribution/count accounting, and constant baseline/zero controls within each snapshot. Independent result review also recomputed 80-row means, counts and paired differences. Model parameters and saved Adam states on disk remain unchanged.

Compact numerical evidence is under [docs/evidence/replay-interference](../../evidence/replay-interference/); per-checkpoint JSON contains all 16 rows and role metrics. Full source files, original checkpoints and datasets remain local. The diagnostic lives on research/replay-interference; main and tag v0.0.1 were not changed or pushed by this work.

To repeat with the required local training caches/checkpoints and a fresh output directory:

```sh
uv run python -m mlm.interference_run freeze --out results/my-replay-probe
uv run python -m mlm.interference_run run --out results/my-replay-probe
```

The freeze phase records selections/source/data/state hashes before loss/gradient probes. Existing diagnostic outputs are protected against overwrite. Frozen checkpoints and English data are not distributed with the package. No architecture, optimizer, training-loop, scorer, Config schema, held-out evaluation or release version changed.

## Decision and next experiment

Local interference exists, but this check does not confirm that replay 44's gradients caused its acquisition failures or justify a favorable-seed workaround. Much of the raw change can reflect the optimizer state already present in a snapshot, and these are extra-English-step probes rather than the normal alternating phase.

Next, predefine a copied QA→English pair using the checkpoint's saved stream/carry and the reconstructed next QA batch. Measure QA loss after each phase, with the zero-gradient control starting after QA. This tests the actual next-loop phase while keeping weights on disk fixed. Only then select a bounded, separately frozen training intervention if the evidence supports it. The earlier failed replication gate and unestablished generalization remain unchanged.
