"""Postprocess frozen traces; account for cancellation and the learned output projection."""

import hashlib
import json
from pathlib import Path

import numpy as np

from mlm.checkpoint import load_checkpoint

OUT = Path("results/addressing")


def main():
    cases = {
        "pilot-initial": "results/associative/initial-delta.mlm",
        "pilot-trained": "runs/associative-delta-v1/final.mlm",
        **{f"addressing-delta-g{g}-v1": f"runs/addressing-delta-g{g}-v1/final.mlm" for g in [1, 4]},
    }
    results = {}
    for label, checkpoint in cases.items():
        state, _, _ = load_checkpoint(Path(checkpoint))
        kernel = np.asarray(state.params["memory_output_kernel"], dtype=np.float64)
        gain = float(state.params["memory_gain"])
        rows = json.loads((OUT / f"{label}-traces.json").read_text())
        output = []
        for row in rows:
            trace = row["trace"]
            writes = np.array(trace["effective_read_coefficients"])[:, None] * np.array(
                trace["values"]
            )
            vectors = {
                name: writes[start:end].sum(0) for name, (start, end) in row["regions"].items()
            }
            recalled = sum(vectors.values())
            np.testing.assert_allclose(recalled, trace["last_recalled"], atol=1e-5, rtol=1e-5)
            projected = {name: gain * (v @ kernel) for name, v in vectors.items()}
            total = sum(projected.values())
            np.testing.assert_allclose(
                np.linalg.norm(total), trace["memory_residual_norms"][-1], atol=1e-5, rtol=1e-5
            )
            output.append(
                {
                    "id": row["question"]["id"],
                    "regions": {
                        name: {
                            "signed_recalled_vector": vectors[name].tolist(),
                            "projected_residual_vector": value.tolist(),
                            "projected_residual_l2": float(np.linalg.norm(value)),
                        }
                        for name, value in projected.items()
                    },
                    "total_projected_residual_l2": float(np.linalg.norm(total)),
                }
            )
        results[label] = {
            "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
            "prompts": output,
        }
    results["scope"] = (
        "Conditional linear residual decomposition before layernorm; "
        "region norms are not additive or causal effects. "
        "Question/header writes can contain earlier facts through graph state."
    )
    (OUT / "projected-regions.json").write_text(json.dumps(results, indent=2) + "\n")
    (OUT / "projection-code-sha256.json").write_text(
        json.dumps({__file__: hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2)
        + "\n"
    )
    print("Projected region vectors reconstruct the actual residuals")


if __name__ == "__main__":
    main()
