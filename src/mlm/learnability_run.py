"""Training-only random-initialization sanity, never transferred to English pretraining."""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from mlm.binding_run import build_batches
from mlm.checkpoint import save_checkpoint
from mlm.comprehension import evaluate_questions
from mlm.config import Config
from mlm.hierarchy_run import make_routed_step
from mlm.model import initial_state
from mlm.retrieval_eval import aggregate
from mlm.training import create_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Choose a fresh probe path")
    args.out.mkdir(parents=True)
    cfg = Config.model_validate_json(args.config.read_text())
    records = [
        json.loads(line) for line in Path("data/gradual-v1/train.jsonl").read_text().splitlines()
    ][:16]
    questions = json.loads(Path("data/gradual-v1/train-questions.json").read_text())["questions"][
        :16
    ]
    batches, _ = build_batches(records, questions, cfg)
    state = create_state(cfg)
    step = make_routed_step(cfg)
    rng = np.random.default_rng(cfg.seed)
    rows = []
    for groups in [1, 4]:
        for _ in range(500):
            batch = jax.tree.map(jnp.asarray, batches[int(rng.integers(groups))])
            state, _, metrics = step(state, initial_state(cfg, 4), batch)
            if not bool(metrics["finite"]):
                raise FloatingPointError("Nonfinite learnability update")
        ranked = evaluate_questions(state, cfg, questions[: groups * 4], format="chat")
        row = {
            "steps": int(state.step),
            "training_groups": groups,
            "metrics": aggregate(ranked["items"]),
            "scope": "training-only sanity; no generalization and no pretraining transfer",
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        save_checkpoint(
            args.out / f"groups-{groups}.mlm",
            state,
            cfg,
            {"mode": "training-only-scratch-probe", "messages": True},
        )
        (args.out / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
        (args.out / f"groups-{groups}-scores.json").write_text(json.dumps(ranked, indent=2) + "\n")


if __name__ == "__main__":
    main()
