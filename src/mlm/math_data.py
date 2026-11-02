"""100% Hugging Face Parquet-backed 4-Stage Mathematics Curriculum Builder for MLM v2.

Streams real mathematics datasets via Rust `polars` from Hugging Face Parquet shards:
  - `openai/gsm8k` (Grade School Math 8K word problems & step-by-step derivations)
  - `microsoft/orca-math-word-problems-200k` (200K math word problems & solutions)
  - `meta-math/MetaMathQA` (395K GSM8K + Hendrycks MATH algebra/geometry/number theory QA)

Contains ZERO synthetic loops or hardcoded question/answer lists.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from mlm.curated_data import (
    _first_complete_sentences,
    _make_supervised_chat_record,
    _make_supervised_think_record,
)
from mlm.parquet_stream import read_hf_parquet_rows


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _parse_gsm8k_answer(raw_answer: str) -> tuple[str, str]:
    """Parse a GSM8K answer (`reasoning #### final_val`) into `(steps_text, final_val)`."""
    cleaned = re.sub(r"<<[^>]+>>", "", str(raw_answer))
    if "####" in cleaned:
        steps_part, final_part = cleaned.split("####", 1)
        final_val = final_part.strip()
    else:
        steps_part = cleaned
        final_val = ""
    lines = [" ".join(ln.strip().split()) for ln in steps_part.splitlines() if ln.strip()]
    steps_text = " ".join(lines)
    if steps_text and steps_text[-1] not in ".!?":
        steps_text += "."
    return steps_text, final_val


def _parse_metamath_response(raw_response: str) -> tuple[str, str]:
    """Parse a MetaMathQA response (`reasoning The answer is: X`) into `(steps_text, final_val)`."""
    text = " ".join(str(raw_response).strip().split())
    marker = "The answer is:"
    if marker in text:
        steps_part, final_part = text.rsplit(marker, 1)
        steps_clean = steps_part.strip()
        final_val = final_part.strip().rstrip(".")
        return steps_clean, final_val
    return text, ""


def prepare_curated_math(
    root: Path = Path("data/curated-math-v1"),
    hf_cache: Path = Path("data/hf"),
    orca_rows: int = 3500,
    metamath_rows: int = 4500,
) -> dict:
    """Build the 4-stage Mathematics Node curriculum 100% from Hugging Face datasets."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    # 1. Load real Hugging Face datasets via Rust Polars Parquet streaming
    gsm_train = read_hf_parquet_rows(hf_cache, "openai/gsm8k", "main", "train", max_rows=7473)
    gsm_test_path = hf_cache / "gsm8k" / "test.jsonl"
    if gsm_test_path.exists():
        gsm_test = [
            json.loads(line)
            for line in gsm_test_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        gsm_test = read_hf_parquet_rows(hf_cache, "openai/gsm8k", "main", "test", max_rows=500)

    orca_data = read_hf_parquet_rows(
        hf_cache, "microsoft/orca-math-word-problems-200k", "default", "train", max_rows=orca_rows
    )
    metamath_data = read_hf_parquet_rows(
        hf_cache, "meta-math/MetaMathQA", "default", "train", max_rows=metamath_rows
    )

    # =========================================================================
    # STAGE 1: Math Foundations (4 Core Graph Nodes)
    # Real mathematical problem + solution documents from Orca-Math & GSM8K
    # =========================================================================
    s1_train: list[dict] = []
    s1_val: list[dict] = []

    for idx, r in enumerate(orca_data[:1200]):
        q = " ".join(str(r.get("question", "")).strip().split())
        a = _first_complete_sentences(str(r.get("answer", "")), max_chars=280)
        if q and a and len(q) <= 220:
            rec = {"text": f"Problem: {q}\nSolution: {a}"}
            if idx % 16 == 0 and len(s1_val) < 64:
                s1_val.append(rec)
            else:
                s1_train.append(rec)

    for r in gsm_train[:600]:
        q = " ".join(str(r.get("question", "")).strip().split())
        steps_clean, num_clean = _parse_gsm8k_answer(r.get("answer", ""))
        if q and steps_clean and num_clean and len(q) <= 220 and len(steps_clean) <= 260:
            s1_train.append(
                {"text": f"Problem: {q}\nSolution: {steps_clean} Final answer: {num_clean}."}
            )

    # =========================================================================
    # STAGE 2: Arithmetic & Word Problems (4 -> 6 Graph Nodes)
    # Supervised QA from GSM8K + Orca-Math-200K
    # =========================================================================
    s2_train: list[dict] = []
    s2_val: list[dict] = []

    for r in gsm_train:
        q = " ".join(str(r.get("question", "")).strip().split())
        steps_clean, num_clean = _parse_gsm8k_answer(r.get("answer", ""))
        if q and steps_clean and num_clean and len(q) <= 220 and len(steps_clean) <= 240:
            ans_str = f"The answer is {num_clean}. {steps_clean}"
            rec = _make_supervised_chat_record(q, ans_str)
            if len(rec["text"].encode("utf-8")) <= 480:
                s2_train.append(rec)

    for r in gsm_test:
        q = " ".join(str(r.get("question", "")).strip().split())
        steps_clean, num_clean = _parse_gsm8k_answer(r.get("answer", ""))
        if q and steps_clean and num_clean and len(q) <= 220 and len(steps_clean) <= 240:
            ans_str = f"The answer is {num_clean}. {steps_clean}"
            rec = _make_supervised_chat_record(q, ans_str)
            if len(rec["text"].encode("utf-8")) <= 480 and len(s2_val) < 64:
                s2_val.append(rec)

    for idx, r in enumerate(orca_data[1200:]):
        q = " ".join(str(r.get("question", "")).strip().split())
        a = _first_complete_sentences(str(r.get("answer", "")), max_chars=260)
        if q and a and len(q) <= 200:
            rec = _make_supervised_chat_record(q, a)
            if len(rec["text"].encode("utf-8")) <= 480:
                if idx % 25 == 0 and len(s2_val) < 64:
                    s2_val.append(rec)
                else:
                    s2_train.append(rec)

    # =========================================================================
    # STAGE 3: Algebra, Geometry & MetaMathQA (6 -> 8 Graph Nodes)
    # Supervised QA from `meta-math/MetaMathQA` (MATH + GSM8K augmentations)
    # =========================================================================
    s3_train: list[dict] = []
    s3_val: list[dict] = []

    for idx, r in enumerate(metamath_data):
        q = " ".join(str(r.get("query", "")).strip().split())
        steps_clean, final_val = _parse_metamath_response(r.get("response", ""))
        if not q or not steps_clean or not final_val:
            continue
        if len(q) > 220 or len(steps_clean) > 260 or len(final_val) > 32:
            continue
        ans_str = f"The answer is {final_val}. {steps_clean}"
        rec = _make_supervised_chat_record(q, ans_str)
        if len(rec["text"].encode("utf-8")) <= 480:
            if idx % 28 == 0 and len(s3_val) < 64:
                s3_val.append(rec)
            else:
                s3_train.append(rec)

    # Replay a slice of Stage 2 GSM8K inside Stage 3 to prevent catastrophic forgetting
    s3_train.extend(s2_train[:400])

    # =========================================================================
    # STAGE 4: Ornith-1.5 `<|think|>` Math Reasoning (8 -> 10 Graph Nodes)
    # Real step-by-step reasoning traces from GSM8K & MetaMathQA inside `<|think|>`
    # =========================================================================
    s4_train: list[dict] = []
    s4_val: list[dict] = []

    for r in gsm_train[:1800]:
        q = " ".join(str(r.get("question", "")).strip().split())
        steps_clean, num_clean = _parse_gsm8k_answer(r.get("answer", ""))
        if q and steps_clean and num_clean and len(q) <= 180 and len(steps_clean) <= 200:
            final_reply = f"The answer is {num_clean}."
            rec = _make_supervised_think_record(q, steps_clean, final_reply)
            if len(rec["text"].encode("utf-8")) <= 480:
                s4_train.append(rec)

    for r in gsm_test[:200]:
        q = " ".join(str(r.get("question", "")).strip().split())
        steps_clean, num_clean = _parse_gsm8k_answer(r.get("answer", ""))
        if q and steps_clean and num_clean and len(q) <= 180 and len(steps_clean) <= 200:
            final_reply = f"The answer is {num_clean}."
            rec = _make_supervised_think_record(q, steps_clean, final_reply)
            if len(rec["text"].encode("utf-8")) <= 480 and len(s4_val) < 48:
                s4_val.append(rec)

    for idx, r in enumerate(metamath_data[:2000]):
        q = " ".join(str(r.get("query", "")).strip().split())
        steps_clean, final_val = _parse_metamath_response(r.get("response", ""))
        if q and steps_clean and final_val and len(q) <= 180 and len(steps_clean) <= 200:
            if len(final_val) <= 24:
                final_reply = f"The answer is {final_val}."
                rec = _make_supervised_think_record(q, steps_clean, final_reply)
                if len(rec["text"].encode("utf-8")) <= 480:
                    if idx % 30 == 0 and len(s4_val) < 48:
                        s4_val.append(rec)
                    else:
                        s4_train.append(rec)

    s1_dir = root / "stage1_math_foundations"
    s2_dir = root / "stage2_math_arithmetic_word"
    s3_dir = root / "stage3_math_algebra_geometry"
    s4_dir = root / "stage4_math_reasoning_think"

    _write_jsonl(s1_dir / "train.jsonl", s1_train)
    _write_jsonl(s1_dir / "validation.jsonl", s1_val[:64] or s1_train[:32])
    _write_jsonl(s2_dir / "train.jsonl", s2_train)
    _write_jsonl(s2_dir / "validation.jsonl", s2_val[:64] or s2_train[:32])
    _write_jsonl(s3_dir / "train.jsonl", s3_train)
    _write_jsonl(s3_dir / "validation.jsonl", s3_val[:64] or s3_train[:32])
    _write_jsonl(s4_dir / "train.jsonl", s4_train)
    _write_jsonl(s4_dir / "validation.jsonl", s4_val[:48] or s4_train[:24])

    manifest = {
        "sources": [
            "openai/gsm8k",
            "microsoft/orca-math-word-problems-200k",
            "meta-math/MetaMathQA",
        ],
        "counts": {
            "stage1_train": len(s1_train),
            "stage1_val": len(s1_val[:64]),
            "stage2_train": len(s2_train),
            "stage2_val": len(s2_val[:64]),
            "stage3_train": len(s3_train),
            "stage3_val": len(s3_val[:64]),
            "stage4_train": len(s4_train),
            "stage4_val": len(s4_val[:48]),
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    info = prepare_curated_math()
    print(json.dumps(info, indent=2))
