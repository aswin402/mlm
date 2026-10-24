"""Check recorded historical execution identity and record current numerical settings."""

import json
import os
from pathlib import Path

import jax

from mlm.cli import environment

OUT = Path("results/addressing-factorial")
current = environment()
paths = [
    "runs/addressing-exposure-delta-g4-v1/environment.json",
    "runs/addressing-replication-delta-s43-v1/environment.json",
]
checks = {path: json.loads(Path(path).read_text()) == current for path in paths}
result = {
    "current": current,
    "historical_recorded_fields_match": checks,
    "current_numerical_settings": {
        name: str(getattr(jax.config, name))
        for name in ["jax_enable_x64", "jax_default_matmul_precision", "jax_disable_jit"]
    },
    "current_execution_environment": {
        name: os.environ.get(name)
        for name in [
            "JAX_ENABLE_X64",
            "JAX_DEFAULT_MATMUL_PRECISION",
            "JAX_PLATFORMS",
            "JAX_PLATFORM_NAME",
            "XLA_FLAGS",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
        ]
    },
    "limit": (
        "Historical artifacts did not record these numerical/thread flags; "
        "only recorded source/software/backend/platform fields can be checked. "
        "Historical-versus-new execution equivalence is not fully proven."
    ),
}
(OUT / "environment-preflight.json").write_text(json.dumps(result, indent=2) + "\n")
if not all(checks.values()):
    raise AssertionError("Recorded historical environment differs")
print("Both historical recorded environments match current source/software/backend/platform")
