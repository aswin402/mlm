"""Audit the intentional runner-only source change and record execution settings."""

import hashlib
import json
import os
from pathlib import Path

import jax

from mlm.cli import environment

OUT = Path("results/schedule-components")
current = environment()
legacy_driver = OUT / "legacy/src/mlm/plasticity_run.py"
legacy_digest = hashlib.sha256(
    b"".join(
        legacy_driver.read_bytes() if p.name == "plasticity_run.py" else p.read_bytes()
        for p in sorted(Path("src/mlm").glob("*.py"))
    )
).hexdigest()
checks = {}
historical = [
    "runs/addressing-exposure-delta-g4-v1/environment.json",
    "runs/addressing-factorial-i42-s43-v1/environment.json",
]
for path in historical:
    old = json.loads(Path(path).read_text())
    checks[path + ":software_backend"] = {k: v for k, v in old.items() if k != "source_sha256"} == {
        k: v for k, v in current.items() if k != "source_sha256"
    }
    checks[path + ":only_driver_source_change"] = old["source_sha256"] == legacy_digest
prior = json.loads(Path("results/addressing-factorial/protocol.json").read_text())["sha256"]
changed = {}
for name, expected in prior.items():
    if name.startswith(("src/", "tests/")):
        actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
        if actual != expected:
            changed[name] = {"old": expected, "new": actual}
checks["only_authorized_driver_and_test_changed"] = set(changed) == {
    "src/mlm/plasticity_run.py",
    "tests/test_plasticity.py",
}
for name in changed:
    old = OUT / "legacy" / name
    checks[name + ":legacy_preserved"] = hashlib.sha256(old.read_bytes()).hexdigest() == prior[name]
checks["source_module_set_unchanged"] = {str(p) for p in Path("src/mlm").glob("*.py")} == {
    p for p in prior if p.startswith("src/mlm/") and p.endswith(".py")
}
settings = {
    name: str(getattr(jax.config, name))
    for name in ["jax_enable_x64", "jax_default_matmul_precision", "jax_disable_jit"]
}
variables = {
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
}
known = json.loads(Path("results/addressing-factorial/i42-s43-execution.json").read_text())
checks["known_historical_flags_match"] = (
    known["numerical_settings"] == settings and known["execution_environment"] == variables
)
result = {
    "current": current,
    "current_numerical_settings": settings,
    "current_execution_environment": variables,
    "legacy_source_digest": legacy_digest,
    "checks": checks,
    "intentional_source_changes": changed,
    "limit": (
        "QA42/English43 historical numerical flags were not recorded; "
        "source version differs only in reviewed sampler controls and metadata."
    ),
}
(OUT / "environment-preflight.json").write_text(json.dumps(result, indent=2) + "\n")
if not all(checks.values()):
    raise AssertionError([k for k, v in checks.items() if not v])
print(f"All {len(checks)} historical execution/source compatibility checks passed")
