# Context-contrastive sampling replication: acquisition did not repeat

The fixed weight-100 recipe failed its acquisition gate on the new QA44/English45 schedule. Both freshly trained arms finished at 8/16 training answers, zero complete binding groups and zero context/query prediction flips. The selected QA43/English44 pilot's 16/16 and 4/4 result remains historical evidence; it is not a reliable replicated recipe.

## Fixed comparison

Both arms restarted from the same original step-zero graph/delta-memory checkpoint, including the full optimizer. Each received exactly 128 QA and 128 English updates, 32 visits per group, with seeds 44/45. The active arm used the unchanged same-answer context-contrastive objective at weight 100. The control used the exact legacy update. No new weight, budget extension or conditional arm was tried. No architecture, inference parameters or optimizer settings changed; 384,784 parameters remain. This is independent sampling replication, not independent model-initialization replication.

| Schedule and arm | Training answers | Complete groups | Main QA CE | Fixed English training-probe CE |
|---|---:|---:|---:|---:|
| Historical43/44 control | 8/16 | 0/4 | 0.127620 | 1.191786 |
| Historical43/44 weight 100 | 16/16 | 4/4 | 0.022787 | 1.233599 |
| New44/45 control | 8/16 | 0/4 | 0.126892 | 1.206518 |
| New44/45 weight 100 | 8/16 | 0/4 | 0.139620 | 1.201739 |

New active minus control English CE is −0.004778385, so the relative English direction gate passes. The primary acquisition gate fails and the joint gate fails. The new comparison denominator is one; the selected pilot is not included. Both new English endpoints exceed initial 1.157155, so relative improvement does not establish retention. Active context loss 0.692999840 remains close to context-independent ln(2), unlike the pilot's 0.001522773. The new active arm does not acquire binding even on the repeated training fixture. No held-out comprehension, generalization, chat or device-efficiency claim follows.

Both QA ordering and English replay sampling changed together. This experiment establishes sensitivity to the combined sampling schedule; it cannot identify which component caused the difference or establish English replay as the cause.

## Implementation, review and verification

Implemented the separate driver first, then added audit tests, as requested. The driver reuses the original context objective, legacy optimizer and unchanged canonical evaluator. It freezes source state/configuration, input hashes, candidate and QA batch fingerprints, every English batch hash, complete QA order, independently reconstructed expected exposure and numerical settings before optimization. It checks its own protocol digest and settings and all input hashes before/after each arm.

Runtime accepted-update, main supervised-target/byte, candidate-position and context-term counts are accumulated and compared with the independent frozen expectations. Per arm: 128 QA updates, 128 English updates, 3,072 QA targets, 2,560 QA supervised bytes, 61,440 English targets and 61,376 English supervised bytes. The active arm additionally evaluates 4,096 candidate answer-byte positions and 512 context comparisons. Main exposure matches; compute does not, because the active arm evaluates both candidates. Finite guards accepted every update; final optimizer step 256 and 32 visits/group were required.

Review caught missing validation/freezing of the previous experiment's raw JSONs. This was fixed before training: 11 mandatory raw/public JSON pairs must match, including protocols, diagnostics, histories and summaries. Regression coverage rejects altered raw history, missing raw summary, count/visit mismatches, changed protocol/input/settings, and incorrect success-gate comparisons. The previous executed runner and all old artifacts remain unchanged.

Eight focused tests passed in 13.18 seconds after the fix; Ruff and ty passed. This is not a new full-suite run. A final independent audit reconstructs schedule/fingerprints/exposure, reloads and rescores both saved checkpoints, verifies source immutability and all 225 frozen inputs, and compares the fresh control's complete state with the previous same-schedule control. All 260 final audit checks passed, as recorded in [verification](../../evidence/context-replication/verification.json). The requested graphify update failed because the installed CLI cannot import its package.

## Decision and next step

Do not promote weight 100, combine more architectural ideas with it, or treat the favorable pilot as general language learning. A useful next diagnostic is to separate QA ordering from English replay sampling, or inspect gradient interference on existing frozen snapshots, before changing the recipe. Any new training comparison should have its own fixed protocol and budget. Keep fresh evaluation closed until training acquisition becomes repeatable.

[Protocol](protocol.md) · [Evidence](../../evidence/context-replication/summary.json) · [Historical pilot](../context-contrastive/report.md). New checkpoints remain local-only in results/context-replication; no push, release or version change.
