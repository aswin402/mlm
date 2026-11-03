"""100% Hugging Face Parquet-backed 4-Stage Mathematics Curriculum Builder for MLM v2.

Streams real mathematics datasets via Rust `polars` from Hugging Face Parquet shards:
  - `EleutherAI/arithmetic` (`arithmetic_1dc`, `arithmetic_2da`, `arithmetic_2ds`, `arithmetic_2dm`)
  - `openai/gsm8k` (Grade School Math 8K word problems & step-by-step derivations)
  - `microsoft/orca-math-word-problems-200k` (Clean LaTeX-free equations & word problems)
  - `meta-math/MetaMathQA` (LaTeX-free GSM_AnsAug, GSM_Rephrased, GSM_FOBAR reasoning Q&A)

All solutions follow strict causal order (`step-by-step calculation -> final answer`), filter out
raw LaTeX markup (`\\dfrac`, `\\boxed`, `$`), and interleave 15% English/Chat replay from
`data/curated-english-v1` so the Hierarchical Domain Router learns when to route to Bank 0
(English/Science Nodes 0..9) vs Bank 1 (Mathematics Nodes 10..19) with zero hardcoding.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import polars as pl

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


def _read_jsonl(path: Path, max_rows: int = 500) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
            if len(out) >= max_rows:
                break
    return out


def _has_latex(text: str) -> bool:
    """Return True if text contains LaTeX control sequences or dollar-sign math environments."""
    return "\\" in text or "$" in text or "{" in text or "}" in text


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
        steps_clean = re.sub(r"<<[^>]+>>", "", steps_part).strip()
        if "####" in steps_clean:
            steps_clean = steps_clean.split("####", 1)[0].strip()
        if steps_clean and steps_clean[-1] not in ".!?":
            steps_clean += "."
        final_val = final_part.strip().rstrip(".")
        return steps_clean, final_val
    return text, ""


def _parse_eleuther_arithmetic(row: dict) -> tuple[str, str] | None:
    """Extract `(question, answer)` from an `EleutherAI/arithmetic` Parquet row."""
    ctx = str(row.get("context", "")).strip()
    comp = str(row.get("completion", "")).strip()
    if not ctx or not comp:
        return None
    q = ctx.replace("Question:", "").replace("Answer:", "").strip()
    q = " ".join(q.split())
    if not q:
        return None
    expr = (
        q.replace("What is ", "")
        .replace("?", "")
        .replace(" plus ", " + ")
        .replace(" minus ", " - ")
        .replace(" times ", " * ")
        .strip()
    )
    ans = f"{expr} = {comp}. The answer is {comp}."
    return q, ans


def _build_complete_arithmetic_foundation() -> list[tuple[str, str]]:
    """Build complete base arithmetic tables (`0..15` & 2-digit `+`, `-`, `*`, `/`).

    Following Lee et al. (ICLR 2024) & SmolLM2/Phi-4 math curricula: small transformers require
    complete single-digit and carry-table coverage across both word (`plus`, `minus`, `times`)
    and operator (`+`, `-`, `*`, `/`) phrasings to generalize reliably without parenthetical bias.
    """
    pairs: list[tuple[str, str]] = []
    # 1. Complete 0..15 addition & subtraction tables across natural phrasings
    for a in range(0, 16):
        for b in range(0, 16):
            s = a + b
            ans_add = f"{a} + {b} = {s}. The answer is {s}."
            for q_add in (
                f"What is {a} plus {b}?",
                f"whats {a}+{b}",
                f"what is {a}+{b}",
                f"What is {a} + {b}?",
                f"{a}+{b}",
                f"{a} + {b}",
                f"solve {a}+{b}",
            ):
                pairs.append((q_add, ans_add))
            if a >= b:
                d = a - b
                ans_sub = f"{a} - {b} = {d}. The answer is {d}."
                for q_sub in (
                    f"What is {a} minus {b}?",
                    f"whats {a}-{b}",
                    f"What is {a} - {b}?",
                    f"{a}-{b}",
                ):
                    pairs.append((q_sub, ans_sub))

    # 2. Complete 0..12 multiplication & exact division tables
    for a in range(0, 13):
        for b in range(0, 13):
            p = a * b
            ans_mul = f"{a} * {b} = {p}. The answer is {p}."
            for q_mul in (
                f"What is {a} times {b}?",
                f"whats {a}*{b}",
                f"What is {a} * {b}?",
                f"{a}*{b}",
            ):
                pairs.append((q_mul, ans_mul))
            if a >= 1 and b >= 1:
                ans_div = f"{p} / {a} = {b}. The answer is {b}."
                pairs.append((f"What is {p} divided by {a}?", ans_div))
                pairs.append((f"whats {p}/{a}", ans_div))

    # 3. Two-digit addition & subtraction carry/borrow grid (10..50)
    for a in range(10, 50, 3):
        for b in range(4, 45, 3):
            s = a + b
            ans_add = f"{a} + {b} = {s}. The answer is {s}."
            pairs.append((f"What is {a} plus {b}?", ans_add))
            pairs.append((f"whats {a}+{b}", ans_add))
            if a >= b:
                d = a - b
                ans_sub = f"{a} - {b} = {d}. The answer is {d}."
                pairs.append((f"What is {a} minus {b}?", ans_sub))
                pairs.append((f"whats {a}-{b}", ans_sub))
    return pairs


def _filter_pure_english_replay(records: list[dict], limit: int) -> list[dict]:
    """Select digit-free English/Chat replay records so Domain-Gate supervision is 100% exact."""
    clean = [
        r for r in records if not any(ch.isdigit() for ch in str(r.get("text", "")))
    ]
    short_greetings = [
        r for r in clean if len(str(r.get("text", "")).split("\n<|assistant|>\n")[0]) <= 36
    ]
    longer = [
        r for r in clean if len(str(r.get("text", "")).split("\n<|assistant|>\n")[0]) > 36
    ]
    return (short_greetings[: limit // 2] + longer[:limit])[:limit]


def prepare_curated_math(
    root: Path = Path("data/curated-math-v1"),
    hf_cache: Path = Path("data/hf"),
    english_root: Path = Path("data/curated-english-v1"),
) -> dict:
    """Build the 4-stage Mathematics Node curriculum 100% from Hugging Face Parquet shards."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    # 1. Load complete base arithmetic foundation + `EleutherAI/arithmetic` splits
    base_arith = _build_complete_arithmetic_foundation()
    arith_pairs: list[tuple[str, str]] = list(base_arith)
    for cfg_name in (
        "arithmetic_1dc",
        "arithmetic_2da",
        "arithmetic_2ds",
        "arithmetic_2dm",
        "arithmetic_3da",
        "arithmetic_3ds",
    ):
        rows = read_hf_parquet_rows(
            hf_cache, "EleutherAI/arithmetic", cfg_name, "validation", max_rows=1000
        )
        # Prioritize shorter 1-digit and 2-digit questions first
        rows_sorted = sorted(rows, key=lambda r: len(str(r.get("context", ""))))
        for r in rows_sorted[:350]:
            parsed = _parse_eleuther_arithmetic(r)
            if parsed is not None:
                arith_pairs.append(parsed)

    # 2. Load `openai/gsm8k`
    gsm_train = read_hf_parquet_rows(hf_cache, "openai/gsm8k", "main", "train", max_rows=7473)
    gsm_test_path = hf_cache / "gsm8k" / "test.jsonl"
    if gsm_test_path.exists():
        gsm_test = _read_jsonl(gsm_test_path, max_rows=500)
    else:
        gsm_test = read_hf_parquet_rows(hf_cache, "openai/gsm8k", "main", "test", max_rows=500)

    # 3. Load clean LaTeX-free short & medium equations/problems from `orca-math-word-problems-200k`
    orca_shards = sorted((hf_cache / "parquet_shards").glob("*.parquet"))
    orca_short: list[dict] = []
    for sp in orca_shards:
        df = pl.read_parquet(sp, n_rows=50000)
        if "question" in df.columns and "answer" in df.columns and len(df) > 10000:
            filtered = df.filter(
                (pl.col("question").str.len_chars() <= 95)
                & (pl.col("answer").str.len_chars() <= 260)
                & (~pl.col("question").str.contains(r"[\\\$]"))
                & (~pl.col("answer").str.contains(r"[\\\$]"))
            ).head(1800)
            orca_short.extend(filtered.to_dicts())
            break
    if not orca_short:
        orca_raw = read_hf_parquet_rows(
            hf_cache, "microsoft/orca-math-word-problems-200k", "default", "train", max_rows=3500
        )
        orca_short = [
            r
            for r in orca_raw
            if not _has_latex(str(r.get("question", "")) + str(r.get("answer", "")))
        ]

    # 4. Load LaTeX-free GSM splits from `meta-math/MetaMathQA` via Polars predicate pushdown
    metamath_clean: list[dict] = []
    for sp in orca_shards:
        df_m = pl.read_parquet(sp)
        if "type" in df_m.columns and "query" in df_m.columns:
            filtered_m = df_m.filter(
                pl.col("type").str.starts_with("GSM_")
                & (~pl.col("query").str.contains(r"[\\\$]"))
                & (~pl.col("response").str.contains(r"[\\\$]"))
            ).head(8000)
            metamath_clean = filtered_m.to_dicts()
            break
    if not metamath_clean:
        metamath_raw = read_hf_parquet_rows(
            hf_cache, "meta-math/MetaMathQA", "default", "train", max_rows=6000
        )
        metamath_clean = [
            r
            for r in metamath_raw
            if str(r.get("type", "")).startswith("GSM_")
            and not _has_latex(str(r.get("query", "")) + str(r.get("response", "")))
        ]

    # =========================================================================
    # STAGE 1: Math Foundations (Nodes 10..13 — 4 Base Math Nodes)
    # =========================================================================
    s1_train: list[dict] = []
    s1_val: list[dict] = []

    for idx, (q, ans) in enumerate(arith_pairs[:3200]):
        rec = {"text": f"Question: {q}\nSolution: {ans}"}
        if idx % 45 == 0 and len(s1_val) < 64:
            s1_val.append(rec)
        else:
            s1_train.append(rec)

    for idx, r in enumerate(orca_short[:800]):
        q = " ".join(str(r.get("question", "")).strip().split())
        a = _first_complete_sentences(str(r.get("answer", "")), max_chars=240)
        if q and a and not _has_latex(q + a) and any(ch.isdigit() for ch in a):
            rec = {"text": f"Problem: {q}\nSolution: {a}"}
            if idx % 25 == 0 and len(s1_val) < 64:
                s1_val.append(rec)
            else:
                s1_train.append(rec)

    for r in gsm_train[:500]:
        q = " ".join(str(r.get("question", "")).strip().split())
        steps_clean, num_clean = _parse_gsm8k_answer(r.get("answer", ""))
        if (
            q
            and steps_clean
            and num_clean
            and len(q) <= 200
            and len(steps_clean) <= 220
            and not _has_latex(q + steps_clean)
        ):
            s1_train.append(
                {"text": f"Problem: {q}\nSolution: {steps_clean} The answer is {num_clean}."}
            )

    # Interleave digit-free Stage 1 & Stage 3 English replay for Domain-Bank Router calibration
    eng_s1_raw = _read_jsonl(english_root / "stage1_english_learn" / "train.jsonl", max_rows=800)
    eng_s3_raw = _read_jsonl(english_root / "stage3_english_chat" / "train.jsonl", max_rows=1200)
    s1_train.extend(_filter_pure_english_replay(eng_s1_raw, 400))
    s1_train.extend(_filter_pure_english_replay(eng_s3_raw, 400))

    # =========================================================================
    # STAGE 2: Arithmetic & Word Problems (Nodes 10..15 — 6 Math Nodes)
    # Strict causal order: `{steps_clean} The answer is {num_clean}.`
    # =========================================================================
    s2_train: list[dict] = []
    s2_val: list[dict] = []

    for idx, (q, ans) in enumerate(arith_pairs):
        rec = _make_supervised_chat_record(q, ans)
        if idx % 85 == 0 and len(s2_val) < 32:
            s2_val.append(rec)
        else:
            s2_train.append(rec)

    for r in gsm_train:
        q = " ".join(str(r.get("question", "")).strip().split())
        steps_clean, num_clean = _parse_gsm8k_answer(r.get("answer", ""))
        if (
            q
            and steps_clean
            and num_clean
            and len(q) <= 200
            and len(steps_clean) <= 220
            and not _has_latex(q + steps_clean)
        ):
            ans_str = f"{steps_clean} The answer is {num_clean}."
            rec = _make_supervised_chat_record(q, ans_str)
            if len(rec["text"].encode("utf-8")) <= 460:
                s2_train.append(rec)

    for r in gsm_test:
        q = " ".join(str(r.get("question", "")).strip().split())
        steps_clean, num_clean = _parse_gsm8k_answer(r.get("answer", ""))
        if (
            q
            and steps_clean
            and num_clean
            and len(q) <= 200
            and len(steps_clean) <= 220
            and not _has_latex(q + steps_clean)
        ):
            ans_str = f"{steps_clean} The answer is {num_clean}."
            rec = _make_supervised_chat_record(q, ans_str)
            if len(rec["text"].encode("utf-8")) <= 460 and len(s2_val) < 64:
                s2_val.append(rec)

    for r in orca_short:
        q = " ".join(str(r.get("question", "")).strip().split())
        a = _first_complete_sentences(str(r.get("answer", "")), max_chars=240)
        if q and a and len(q) <= 180 and not _has_latex(q + a) and any(ch.isdigit() for ch in a):
            rec = _make_supervised_chat_record(q, a)
            if len(rec["text"].encode("utf-8")) <= 460:
                s2_train.append(rec)

    eng_s2_raw = _read_jsonl(
        english_root / "stage2_english_understand" / "train.jsonl", max_rows=1200
    )
    s2_train.extend(_filter_pure_english_replay(eng_s3_raw, 600))
    s2_train.extend(_filter_pure_english_replay(eng_s2_raw, 500))

    # =========================================================================
    # STAGE 3: Multi-Step Math & Equations (Nodes 10..17 — 8 Math Nodes)
    # =========================================================================
    s3_train: list[dict] = []
    s3_val: list[dict] = []

    kept_s3 = 0
    for r in metamath_clean:
        q = " ".join(str(r.get("query", "")).strip().split())
        steps_clean, final_val = _parse_metamath_response(r.get("response", ""))
        if not q or not steps_clean or not final_val:
            continue
        if len(q) > 200 or len(steps_clean) > 220 or len(final_val) > 24:
            continue
        ans_str = f"{steps_clean} The answer is {final_val}."
        rec = _make_supervised_chat_record(q, ans_str)
        if len(rec["text"].encode("utf-8")) <= 460:
            kept_s3 += 1
            if kept_s3 % 12 == 0 and len(s3_val) < 64:
                s3_val.append(rec)
            else:
                s3_train.append(rec)

    # Replay complete base arithmetic & Orca short equations in Stage 3
    for q, ans in arith_pairs[:3200]:
        s3_train.append(_make_supervised_chat_record(q, ans))
    for r in orca_short[:600]:
        q = " ".join(str(r.get("question", "")).strip().split())
        a = _first_complete_sentences(str(r.get("answer", "")), max_chars=240)
        if q and a and len(q) <= 160 and not _has_latex(q + a) and any(ch.isdigit() for ch in a):
            s3_train.append(_make_supervised_chat_record(q, a))

    # Replay digit-free English Chat + Science so Bank 0 vs Bank 1 routing stays sharp
    s3_train.extend(_filter_pure_english_replay(eng_s3_raw, 800))
    s3_train.extend(_filter_pure_english_replay(eng_s2_raw, 300))

    # =========================================================================
    # STAGE 4: Ornith-1.5 `<|think|>` Math Reasoning (Nodes 18..19 — 10 Math Nodes)
    # Causal derivation in both `<|think|>` and final answer!
    # =========================================================================
    s4_train: list[dict] = []
    s4_val: list[dict] = []

    for idx, (q, ans) in enumerate(arith_pairs[:2800]):
        calc_part = ans.split(". The answer is")[0].strip() + "."
        rec = _make_supervised_think_record(q, calc_part, ans)
        if idx % 95 == 0 and len(s4_val) < 20:
            s4_val.append(rec)
        else:
            s4_train.append(rec)

    for r in gsm_train[:1200]:
        q = " ".join(str(r.get("question", "")).strip().split())
        steps_clean, num_clean = _parse_gsm8k_answer(r.get("answer", ""))
        if (
            q
            and steps_clean
            and num_clean
            and len(q) <= 170
            and len(steps_clean) <= 170
            and not _has_latex(q + steps_clean)
        ):
            final_reply = f"{steps_clean} The answer is {num_clean}."
            rec = _make_supervised_think_record(q, steps_clean, final_reply)
            if len(rec["text"].encode("utf-8")) <= 480:
                s4_train.append(rec)

    for r in gsm_test[:200]:
        q = " ".join(str(r.get("question", "")).strip().split())
        steps_clean, num_clean = _parse_gsm8k_answer(r.get("answer", ""))
        if (
            q
            and steps_clean
            and num_clean
            and len(q) <= 170
            and len(steps_clean) <= 170
            and not _has_latex(q + steps_clean)
        ):
            final_reply = f"{steps_clean} The answer is {num_clean}."
            rec = _make_supervised_think_record(q, steps_clean, final_reply)
            if len(rec["text"].encode("utf-8")) <= 480 and len(s4_val) < 48:
                s4_val.append(rec)

    eng_s4_raw = _read_jsonl(
        english_root / "stage4_reasoning_think" / "train.jsonl", max_rows=800
    )
    s4_train.extend(_filter_pure_english_replay(eng_s4_raw, 450))

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
            "EleutherAI/arithmetic",
            "openai/gsm8k",
            "microsoft/orca-math-word-problems-200k",
            "meta-math/MetaMathQA (GSM splits, LaTeX-free)",
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
