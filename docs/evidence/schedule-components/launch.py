"""Run only the two predefined crossed cells, preserving existing diagonals."""

import hashlib
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

OUT = Path("results/schedule-components")
protocol = json.loads((OUT / "protocol.json").read_text())
for name, expected in protocol["sha256"].items():
    if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
        raise RuntimeError(f"Frozen input changed: {name}")
for trial in protocol["commands"]:
    if Path(trial["run"]).exists():
        raise RuntimeError(f"Refusing to overwrite {trial['run']}")


def run(trial):
    label = f"q{trial['qa_seed']}-e{trial['english_seed']}"
    print(f"Starting {label}", flush=True)
    with (OUT / f"{label}.log").open("x") as log:
        child_env = dict(os.environ)
        frozen = json.loads((OUT / "environment-preflight.json").read_text())
        for name, value in frozen["current_execution_environment"].items():
            if value is None:
                child_env.pop(name, None)
            else:
                child_env[name] = value
        command = ["uv", "run", "python", str(OUT / "execute.py"), label]
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, env=child_env)
    print(f"Finished {label}", flush=True)


with ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(run, protocol["commands"]))
print("Both crossed trials completed", flush=True)
