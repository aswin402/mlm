# Actual-next QA → English pair diagnostic

Completed 2026-10-26 on `research/qa-replay-pair`. This training-only diagnostic finds mixed local effects. English replay does not uniformly undo QA learning, and the QA step can itself increase full-task QA loss. No architecture or training intervention was made.

## Scope and method

The [predefined protocol](protocol.md) uses the common initializer and four fixed-memory-initialization endpoints from the previous diagnostic. Initial step is0; fitted sources are step4000 with2,000 QA and2,000 English updates. Each source retains its loaded Adam moments, clipping, decay and role scaling.

Reconstruct the actual next QA group from the complete saved permutation history. Restore the English stream snapshot, independently validate its entire history-derived state, and use the exact saved float32 English recurrent carry. QA uses fresh carry; its outgoing carry never replaces English carry. All16 training questions contribute96 assistant/newline/EOS target positions to token-weighted aggregate QA CE.

Per source, copy one QA update; then independently copy English and zero-gradient optimizer updates from that QA candidate. Also copy English-only and zero-gradient updates directly from the original source with identical English batch/carry. Five sources produce25 copied transitions. Model files are never overwritten. Final protocol was frozen before measurements; an earlier selection-only draft remains ignored locally and contained no measurements.

## Aggregate QA loss

Positive change means worse CE. Values are rounded here; [compact evidence](../../evidence/qa-replay-pair/summary.json) preserves full precision. The initializer has a very different loss scale and should not be pooled with fitted endpoints.

| Source | Next QA group | Before | After QA | After QA→English | Net change |
|---|---|---:|---:|---:|---:|
| initial | train-0003 | 6.86676788 | 4.17627096 | 3.49049497 | -3.37627 |
| q42_e43 | train-0001 | 0.000819305598 | 0.000833014667 | 0.00085686642 | +3.75608e-05 |
| q42_e44 | train-0001 | 0.116436392 | 0.116471499 | 0.116527729 | +9.13367e-05 |
| q43_e43 | train-0002 | 0.00121690833 | 0.00122513378 | 0.00121999776 | +3.08943e-06 |
| q43_e44 | train-0002 | 0.115561672 | 0.115186103 | 0.114974089 | -0.000587583 |

## Phase and optimizer controls

Zero change below means QA loss after a zero-gradient optimizer step minus QA loss after QA. Such a step still applies loaded optimizer history and decay. English-minus-zero compares two candidate states from the same QA candidate; it is a conditional control, not an additive causal decomposition.

| Source | QA change | English after QA | Zero after QA | English minus zero | Matched English-only change |
|---|---:|---:|---:|---:|---:|
| initial | -2.6905 | -0.685776 | -0.870174 | +0.184398 | +0.243089 |
| q42_e43 | +1.37091e-05 | +2.38518e-05 | +1.67489e-05 | +7.10285e-06 | +3.26335e-05 |
| q42_e44 | +3.51071e-05 | +5.62295e-05 | +6.34417e-05 | -7.21216e-06 | +4.13954e-05 |
| q43_e43 | +8.22544e-06 | -5.13601e-06 | +1.01725e-05 | -1.53085e-05 | +1.97685e-06 |
| q43_e44 | -0.000375569 | -0.000212014 | -0.000199094 | -1.29193e-05 | -0.000178829 |

English after QA raises aggregate QA CE at2/5 snapshots and lowers it at3/5. Relative to the zero-gradient control it is harmful at2/5 and helpful at3/5, but the membership differs. At q42_e44, raw English harm is smaller than zero-gradient drift. At initialization, English helps in raw terms but helps less than zero.

Matched English-only and English-after-QA effects change sign at initial and q43_e43. Batch and incoming English carry are fixed within each comparison; preceding QA changes parameters, Adam moments and step count together. This supports a conditional preceding-phase effect without isolating Adam as its cause. At q43_e44, preceding QA makes raw English change more helpful while reducing its advantage relative to zero. All ten English branches lower their own training objective and language CE.

## Selected group versus full task

| Source | Selected-group CE before QA | After QA | Change |
|---|---:|---:|---:|
| initial | 7.0979228 | 3.90493083 | -3.19299 |
| q42_e43 | 0.00103211403 | 0.00104657805 | +1.4464e-05 |
| q42_e44 | 0.11653769 | 0.116570234 | +3.25441e-05 |
| q43_e43 | 0.000905315101 | 0.000901977241 | -3.33786e-06 |
| q43_e44 | 0.11454197 | 0.113377012 | -0.00116496 |

At q43_e43, QA improves its selected group while worsening aggregate CE; subsequent English partially offsets that increase. At q42_e43 and q42_e44, even selected-group language CE increases. QA optimizes language loss plus router regularization, and uses loaded optimizer history; these measurements do not separately identify regularization, momentum, clipping or role scaling. Post-QA total QA objective was not measured, so a selected-language-loss increase does not establish an objective-increasing update.

## Verification and limits

Full regression:80 tests passed in254.51s; strengthened focused tests:2 passed in31.80s. Audit-only CLI guards were subsequently tightened and checked by the actual frozen run. All191 runtime audit flags pass. All78 frozen input hashes, including all72 prior diagnostic inputs, remain unchanged. Source checkpoint file/state identity, metadata/carry immutability, next-data fingerprints, phase algebra, counters and gradient/update role accounting were checked. Independent review reproduced numerical tables and phase algebra without finding material errors. Final static and publication-integrity checks are recorded in [verification](../../evidence/qa-replay-pair/verification.json).

One next pair at five historically selected snapshots cannot establish a training trajectory, seed-population effect or a fix. Absolute changes at most1e-6 are numerically unresolved by the predefined descriptive threshold, not a statistical significance rule. No post-step answer ranking, generation, validation or fresh test was evaluated. Existing acquisition-replication and generalization failures remain unchanged. Different source weights, histories and next batches prevent treating this table as a controlled English43-versus44 comparison.

Next: separately specify a copied QA-step control that isolates optimizer-history and router-balance effects before choosing any longer training intervention. Dynamic growth, sparse execution, reliable chat and Transformer superiority remain unproven. Version0.0.1, main and release tag remain unchanged; this research is local and unpushed.
