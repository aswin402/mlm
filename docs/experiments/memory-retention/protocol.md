# Retention-gate warm-start experiment

Goal: test whether longer initial recurrent retention improves context-sensitive reading without adding state or parameters. Authorized as the next memory experiment. No runtime solver or topology change.

Design: copy each architecture's English checkpoint into a fresh optimizer state. Add log(tau-1) only to its learned retention-gate bias, with tau geometrically spaced from 2 to 128 bytes. Graph distributes tau across nodes, GRU across hidden channels. Since h'=z*h+(1-z)*candidate and z=sigmoid(log(tau-1))=1-1/tau at zero other input, this sets a nominal retention prior, not a guaranteed actual timescale. Existing trained gate contributions remain, so this is chrono-inspired bias intervention, not the original chrono initialization algorithm.

Source: Tallec and Ollivier, https://arxiv.org/abs/1804.11188 . Saturation/slow writing may harm learning; failure remains reportable.

1. Add offline `mlm.memory_init` with finite range validation, source hash/provenance, fresh optimizer, output protection, no mutation of source. Tests verify only retention slice changes for both architectures, source preservation, unchanged budget, and checkpoint roundtrip.
2. Prepare two initial checkpoints from the same English sources used by assistant-learning. Freeze protocol before training. Reuse assistant-v2 data, seed/configuration, 2,000 updates and validation every 500; no hyperparameter sweep. No test data enters training/selection. Original unmodified assistant runs remain controls.
3. Run both stages, score development-selected checkpoints using frozen reading evaluation, then measure story retention. Existing tests are reused diagnostic sets: results are exploratory and need fresh confirmation before a positive generalization claim.
4. Save raw artifacts and report including negative outcomes. Update research, PRD and usage. Verify tests, lint and types. No git commit: workspace has no repository.
