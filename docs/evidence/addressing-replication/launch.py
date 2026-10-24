"""Execute the frozen commands with at most two concurrent trials."""
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

OUT = Path("results/addressing-replication")
protocol = json.loads((OUT / "protocol.json").read_text())
for name, expected in protocol["sha256"].items():
    if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
        raise RuntimeError(f"Frozen input changed: {name}")
for trial in protocol["commands"]:
    if Path(trial["run"]).exists():
        raise RuntimeError(f"Refusing to overwrite {trial['run']}")


def run(trial):
    label = f"{trial['memory']}-s{trial['seed']}"
    print(f"Starting {label}", flush=True)
    with (OUT / f"{label}.log").open("x") as log:
        subprocess.run(trial["command"], check=True, stdout=log, stderr=subprocess.STDOUT)
    print(f"Finished {label}", flush=True)
    return label


with ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(run, protocol["commands"]))
print("All four predefined trials completed", flush=True)
