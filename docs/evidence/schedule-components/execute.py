"""Validate frozen child settings, record execution evidence, invoke unchanged driver."""

import json
import os
import sys
from pathlib import Path

import jax

from mlm.cli import environment
from mlm.plasticity_run import main

OUT = Path("results/schedule-components")
protocol = json.loads((OUT / "protocol.json").read_text())
label = sys.argv[1]
trial = next(t for t in protocol["commands"] if f"q{t['qa_seed']}-e{t['english_seed']}" == label)
frozen = json.loads((OUT / "environment-preflight.json").read_text())
settings = {name: str(getattr(jax.config, name)) for name in frozen["current_numerical_settings"]}
variables = {name: os.environ.get(name) for name in frozen["current_execution_environment"]}
actual = {
    "environment": environment(),
    "numerical_settings": settings,
    "execution_environment": variables,
    "driver_command": trial["command"],
}
assert actual["environment"] == frozen["current"]
assert settings == frozen["current_numerical_settings"]
assert variables == frozen["current_execution_environment"]
with (OUT / f"{label}-execution.json").open("x") as handle:
    json.dump(actual, handle, indent=2)
    handle.write("\n")
sys.argv = ["mlm.plasticity_run", *trial["command"][5:]]
main()
