"""Post-hoc frozen training-only readout dependence check; no retraining."""

import json
from pathlib import Path

from mlm.addressing import read_disabled
from mlm.checkpoint import load_checkpoint
from mlm.comprehension import evaluate_questions
from mlm.retrieval_eval import aggregate

out = Path("results/addressing-exposure")
state, cfg, _ = load_checkpoint(Path("runs/addressing-exposure-delta-g4-v1/final.mlm"))
questions = json.loads(Path("data/addressing-v1/groups-4/train-questions.json").read_text())[
    "questions"
]
results = {}
for mode, model in [("on", state), ("off", read_disabled(state))]:
    scored = evaluate_questions(model, cfg, questions, format="chat")
    results[mode] = {"overall": aggregate(scored["items"]), "items": scored["items"]}
results["prediction_flips"] = sum(
    a["prediction"] != b["prediction"]
    for a, b in zip(results["on"]["items"], results["off"]["items"], strict=True)
)
results["scope"] = (
    "Post-hoc training-only check selected after observing fitted delta endpoint; "
    "only copied memory_gain zeroed; no retraining, no checkpoint mutation or test scoring. "
    "Coadapted output dependence does not isolate the training benefit of memory."
)
(out / "ablation.json").write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps({k: v for k, v in results.items() if k not in ["on", "off"]}))
print(json.dumps({k: results[k]["overall"] for k in ["on", "off"]}, indent=2))
