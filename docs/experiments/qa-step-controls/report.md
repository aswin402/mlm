# QA optimizer-history and router-regularizer controls

Completed 2026-10-26 on `research/qa-step-controls`. **Resetting Adam at the unchanged learning rate worsens aggregate QA CE at all four fitted snapshots, under both training objectives. Disabling router regularization alone gives mixed local results. Neither is supported as a general fix.** This is a copied-state diagnostic; source model files and training algorithms remain unchanged.

## Frozen design

The [predefined protocol](protocol.md) crosses loaded optimizer state versus fresh `tx.init(source.params)` with configured router coefficient0.01 versus coefficient0. Each history also has an independent zero-gradient optimizer control: six transitions at each of five selected prior sources,30 total. Fresh resets both moments and optimizer-internal clock, keeps source weights/optimizer settings and public TrainState.step, and is a joint history intervention, not a moment-only test. Public steps0/4000 become1/4001.

Use the exact next QA groups and complete-history restoration from the prior pair diagnostic, fresh QA carry, four-question/24-target selected groups and all16-question/96-target aggregate language CE. No English updates, answer ranking, generation, validation, fresh test, trajectory or persistent training. Source labels identify historical sampling policies; they are not a controlled seed-population comparison. The source initializer is not pooled with fitted endpoints.

Configured total objective is selected-group language CE plus0.01 times raw router penalty. The language-only branch optimizes selected-group CE. Both objective definitions are evaluated after every branch, and the same optimizer transform, clipping, decay and role scales are retained. Zero controls still apply optimizer history/decay; differences against them are conditional controls, not additive causal decompositions.

## Aggregate QA CE changes

Positive means worse. Full precision and all contrasts are in [summary evidence](../../evidence/qa-step-controls/summary.json). Absolute changes at most1e-6 are numerically unresolved under the descriptive protocol threshold; this is not statistical significance.

| Source | Loaded + configured | Loaded + language only | Loaded zero | Fresh + configured | Fresh + language only | Fresh zero |
|---|---:|---:|---:|---:|---:|---:|
| initial | -2.690497 | -2.690263 | +9.059906e-06 | -2.690497 | -2.690263 | +9.059906e-06 |
| q42_e43 | +1.370907e-05 | +1.518923e-05 | +2.387166e-05 | +0.7962638 | +0.6099135 | +1.192093e-07 |
| q42_e44 | +3.510714e-05 | +3.227592e-05 | +4.929304e-05 | +0.04934384 | +0.04507263 | +1.713634e-07 |
| q43_e43 | +8.225441e-06 | +8.702278e-06 | +1.432491e-05 | +0.04002049 | +9.813905e-05 | +1.788139e-07 |
| q43_e44 | -0.0003755689 | -0.0004002601 | -0.000152275 | +0.02431334 | +0.02599893 | -3.427267e-07 |

The original loaded/configured branch reproduces the prior QA-after selected/aggregate losses within1e-6 at every source. At initialization, loaded and fresh candidate states agree exactly. At all four fitted sources, both fresh-gradient branches increase selected-group CE, aggregate CE and their active training objective. Fresh zero drift is below1e-6 at these fitted sources.

Disabling regularization with loaded history slightly improves aggregate CE at q42_e44 and q43_e44, worsens it at q42_e43, and changes q43_e43 by less than1e-6. It does not resolve the positive raw QA CE changes at q42_e43, q42_e44 or q43_e43. With fresh history, the coefficient effect is much larger and also mixed, indicating a local interaction between history and objective rather than a universal regularizer verdict.

## Selected language versus active objective

The following table concerns loaded/configured QA updates only; selected language and configured total objective are evaluated on the same selected group, not the aggregate.

| Source | Selected language change | Configured total-objective change | Aggregate language change | Aggregate change minus loaded zero |
|---|---:|---:|---:|---:|
| initial | -3.192992 | -3.191816 | -2.690497 | -2.690506 |
| q42_e43 | +1.446402e-05 | -9.818934e-06 | +1.370907e-05 | -1.016259e-05 |
| q42_e44 | +3.254414e-05 | +1.947582e-05 | +3.510714e-05 | -1.418591e-05 |
| q43_e43 | -3.33786e-06 | -1.053687e-05 | +8.225441e-06 | -6.099464e-06 |
| q43_e44 | -0.001164958 | -0.001301162 | -0.0003755689 | -0.0002232939 |

At q42_e43, selected language CE rises while configured total objective falls: the step improves its actual training objective despite the language tradeoff. At q42_e44, both selected language and configured objective rise, so regularization alone does not explain that increase. Removing the coefficient still leaves a language-only loss increase there. At q43_e43, selected language and total objective improve while aggregate language worsens, showing a selected-group versus full-task tradeoff.

Every loaded-gradient QA branch improves aggregate CE relative to its own zero-gradient control, even when raw aggregate CE increases. Thus the new QA gradient moderates optimizer-history/decay drift at the three raw-harm fitted sources. This does not prove a unique cause in the nonlinear optimizer or a beneficial long-run trajectory.

## Update sizes

| Source | Loaded configured norm | Fresh configured norm | Fresh / loaded | Loaded language norm | Fresh language norm |
|---|---:|---:|---:|---:|---:|
| initial | 1.702982 | 1.702982 | 1 | 1.702995 | 1.702995 |
| q42_e43 | 0.2417128 | 1.683363 | 6.964308 | 0.2410795 | 1.659252 |
| q42_e44 | 0.2355368 | 1.665826 | 7.072466 | 0.2351069 | 1.663787 |
| q43_e43 | 0.2356164 | 1.651125 | 7.007681 | 0.235478 | 1.61962 |
| q43_e44 | 0.2602256 | 1.747531 | 6.715444 | 0.2576218 | 1.746614 |

Resetting history produces configured update norms about6.7–7.1 times the loaded norms at fitted sources. Raw gradient norms match across history branches for a given coefficient, as expected from identical parameters/data; optimizer state changes the update. Different update magnitudes and coordinate directions are part of this reset intervention. Their individual causal contributions were not isolated, and a smaller-rate fresh optimizer was not tested. These results support rejecting the tested unchanged-rate reset, not rejecting all resets or proving that scalar update size is the sole cause.

## Verification

The test was written first and failed because the helper did not exist. Direct routed-trainer comparisons cover all branches at startup and after a nonzero optimizer history, including exact fresh-state equivalence at initialization, distinct selected/aggregate targets, objective and contrast algebra, zero controls, source immutability, no-supervision rejection and NaN rejection. Review identified finite-output and contrast-coverage gaps; both were fixed before freeze. The initial full regression was intentionally interrupted after22 passing tests to apply those fixes and is not counted as a complete run.

Final full regression:81 passed in288.43s. Strengthened focused test:1 passed in33.22s. Ruff, formatting and ty pass. Independent code review cleared the design before freeze. The frozen run passes391 audits, preserves88 input hashes including all78 prior pair inputs, checks all five source file/state identities, exact next QA selection, source counters, baseline/prior result reproduction, six branches, objective/contrast algebra, finite values, immutability and numerical settings. Compact JSON copies preserve the original raw bytes; see [verification](../../evidence/qa-step-controls/verification.json). The required Graphify update remains unavailable because its installed Python module is missing.

## Decision and limits

Keep loaded optimizer state for now. Do not apply an unchanged-rate optimizer reset or globally remove the router penalty based on these local results. A useful next predefined check is a bounded QA update-size control with loaded history and objective/aggregate-loss measurements, before any longer intervention. This remains a hypothesis, not a demonstrated improvement.

These are single steps at five historically selected snapshots on the reused tiny training task. There is no evidence here of better answer accuracy, held-out binding, English retention after these branches, chat, dynamic graph growth, sparse execution or superiority over Transformers. Earlier failed independent-seed acquisition and generalization gates remain unchanged. Source weights, main, v0.0.1 tag and package version are unchanged; this research is local and unpushed.
