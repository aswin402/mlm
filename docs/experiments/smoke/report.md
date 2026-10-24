# MLM v0.1 CPU smoke result

Run date: 2026-10-23. This is an engineering smoke experiment, not a language capability benchmark.

- Architecture: 8 independent GRU modules, width 32, two incoming edges, one synchronous round per byte.
- Parameters: 100,546.
- Data: 36 original template sentences for training; 18 different sentences for development. Grammar overlaps deliberately; some development words are unseen. No document was truncated. No independent final test was used.
- Optimizer: clipped AdamW, learning rate 0.003, batch size 4, unroll 64, seed 42.
- Updates: 300; development evaluation every 50 updates.
- Backend: CPU; exact versions and source digest are in environment.json and uv.lock.

| Metric | Before training | Final step 300 |
|---|---:|---:|
| Training next-token loss, nats, including EOS | 6.129411 | 0.205798 |
| Development next-token loss, nats, including EOS | 6.091121 | 2.046138 |
| Development payload bits/byte | 8.734793 | 3.099455 |

Best evaluated development checkpoint: step 50, loss 1.653910. Its greedy continuation of `The ` was `red cat rests.`. A continuation of `The red bird` was ` rests.`. Full decoding metadata is in samples.json.

Development loss worsened after step 50 despite continued low training loss. The run exhibits overfitting; best.mlm preserves the best evaluated checkpoint, while latest.mlm preserves the resumable latest state.

Measured training-loop time: 7.323 seconds, including periodic evaluation and checkpoint saves but excluding initial evaluation, training-step compilation, final evaluation, and sample generation. Training-step compilation took 1.135 seconds. Synchronized updates took 4.359 seconds, processing 23,101 payload bytes at about 5299 bytes/second. This is training throughput, not generation speed. Peak process RSS was 418.1 MiB; it includes the Python/JAX process and is not just model memory. The latest resumable checkpoint was 1,208,492 bytes.

## Interpretation

The code learns a tiny text distribution, runs causal graph computation, generates continuations, and can resume training exactly in the tested environment. It does not establish better quality or efficiency than a Transformer or a GRU. There is one seed, a tiny synthetic corpus, no matched baseline, no independent final test, and no capacity growth.

## Reproduce

```sh
uv sync --locked
uv run mlm train --train data/smoke/train.txt --validation data/smoke/validation.txt --config configs/smoke.json --out runs/reproduction --steps 300 --eval-every 50
uv run mlm generate runs/reproduction/best.mlm --prompt 'The ' --max-bytes 64
uv run mlm evaluate runs/reproduction/best.mlm --data data/smoke/validation.txt
```

Use an unused output directory. The source fingerprint in environment.json describes this run's source. Numerical and runtime variation across machines and package/backend versions is expected. Checkpoints are local under runs/smoke-reviewed; text/JSON run evidence is copied here. The earlier runs/smoke experiment preceded a bias-decay correction and is not the result reported in this table.
