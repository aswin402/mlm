"""Download and curate real-world English, Chat, and Reading QA datasets from Hugging Face.

Sources curated (excluding synthetic toy templates or toddler fairy tales):
1. Encyclopedic & Textbook English (`english`):
   - `wikimedia/wikipedia` (`20231101.simple`): Simple English Wikipedia articles (CC-BY-SA-3.0)
   - `HuggingFaceTB/cosmopedia-100k`: Textbook-grade educational explanations (Apache-2.0)
2. Multi-Turn English Conversation & Grounded Instruction QA (`conversation`):
   - `HuggingFaceTB/smoltalk` (`everyday-conversations`): Multi-turn dialogues (Apache-2.0)
   - `databricks/databricks-dolly-15k`: Human-written QA & instructions (CC-BY-SA-3.0)
3. Real-World Reading Comprehension & Science Reasoning (`qa` + benchmark JSONs):
   - `allenai/sciq`: Science reading comprehension + 4 choices (CC-BY-NC-3.0)
   - `rajpurkar/squad` (`plain_text`): Wikipedia reading comprehension QA (CC-BY-SA-4.0)
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from mlm.data import corpus_hash
from mlm.hfdata import fetch_rows, get_json


def _clean_paragraph(text: str, max_chars: int = 640) -> str:
    """Extract a clean, complete multi-sentence English paragraph bounded to max_chars."""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    joined = " ".join(lines)
    if len(joined) <= max_chars:
        return joined
    cut = joined[:max_chars]
    last_dot = max(cut.rfind(". "), cut.rfind(".\n"))
    if last_dot >= 120:
        return cut[: last_dot + 1].strip()
    return cut.strip()


def _make_supervised_chat_record(user_text: str, assistant_text: str) -> dict:
    """Format a `<|user|>...<|assistant|>...` record with assistant-only `loss_spans`."""
    prefix = f"<|user|>\n{user_text.strip()}\n<|assistant|>\n"
    reply = f"{assistant_text.strip()}\n"
    full_text = prefix + reply
    start_b = len(prefix.encode("utf-8"))
    end_b = len(full_text.encode("utf-8"))
    return {
        "text": full_text,
        "loss_spans": [[start_b, end_b]],
        "score_eos": True,
    }


def fetch_raw_items(root: Path, dataset: str, config: str, split: str, count: int) -> list[dict]:
    """Fetch raw row dicts from Hugging Face datasets-server with local SHA-256 caching."""
    from urllib.parse import urlencode

    rows_out: list[dict] = []
    cache = root / "raw"
    cache.mkdir(parents=True, exist_ok=True)
    for offset in range(0, count, 100):
        length = min(100, count - offset)
        url = "https://datasets-server.huggingface.co/rows?" + urlencode(
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": offset,
                "length": length,
            }
        )
        key = hashlib.sha256(url.encode()).hexdigest()
        path = cache / f"{key}.json"
        if path.exists():
            raw = path.read_bytes()
        else:
            raw, _ = get_json(url)
            path.write_bytes(raw)
        result = json.loads(raw)
        for item in result.get("rows", []):
            if not item.get("truncated_cells"):
                rows_out.append(item["row"])
    return rows_out


def prepare_curated_english(
    root: Path,
    wiki_count: int = 300,
    cosmo_count: int = 200,
    smoltalk_count: int = 300,
    dolly_count: int = 200,
    sciq_count: int = 300,
    squad_count: int = 300,
    eval_size: int = 32,
) -> dict:
    """Download and build `data/curated-english-v1` splits and QA benchmarks."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    # 1. Encyclopedic & Textbook English (Simple Wikipedia + Cosmopedia)
    wiki_rows = fetch_raw_items(root, "wikimedia/wikipedia", "20231101.simple", "train", wiki_count)
    cosmo_rows = fetch_raw_items(
        root, "HuggingFaceTB/cosmopedia-100k", "default", "train", cosmo_count
    )

    english_docs: list[str] = []
    for r in wiki_rows:
        title = r.get("title", "").strip()
        body = _clean_paragraph(r.get("text", ""), max_chars=560)
        if len(body) >= 80:
            doc = f"{title}: {body}" if title and not body.startswith(title) else body
            english_docs.append(doc)
    for r in cosmo_rows:
        body = _clean_paragraph(r.get("text", ""), max_chars=560)
        if len(body) >= 80:
            english_docs.append(body)
    english_docs = list(dict.fromkeys(english_docs))

    # 2. Multi-Turn Chat & Grounded Instruction QA (SmolTalk + Dolly-15k)
    smol_train, _ = fetch_rows(
        root, "HuggingFaceTB/smoltalk", "everyday-conversations", "train", smoltalk_count
    )
    smol_val, _ = fetch_rows(
        root, "HuggingFaceTB/smoltalk", "everyday-conversations", "test", eval_size * 2
    )
    dolly_rows = fetch_raw_items(
        root, "databricks/databricks-dolly-15k", "default", "train", dolly_count
    )

    chat_records: list[dict] = []
    for txt in smol_train:
        if len(txt.encode("utf-8")) <= 900:
            chat_records.append({"text": txt})
    for r in dolly_rows:
        inst = r.get("instruction", "").strip()
        ctx = _clean_paragraph(r.get("context", ""), max_chars=380)
        resp = _clean_paragraph(r.get("response", ""), max_chars=320)
        if inst and resp and len(inst) + len(ctx) + len(resp) <= 760:
            user_q = f"{ctx}\n{inst}" if ctx else inst
            chat_records.append(_make_supervised_chat_record(user_q, resp))

    # 3. Real-World Reading Comprehension & Science QA (SciQ + SQuAD)
    sciq_train_rows = fetch_raw_items(root, "allenai/sciq", "default", "train", sciq_count)
    sciq_val_rows = fetch_raw_items(root, "allenai/sciq", "default", "validation", 200)
    squad_train_rows = fetch_raw_items(root, "rajpurkar/squad", "plain_text", "train", squad_count)
    squad_val_rows = fetch_raw_items(root, "rajpurkar/squad", "plain_text", "validation", 200)

    qa_train_records: list[dict] = []
    for r in sciq_train_rows:
        sup = _clean_paragraph(r.get("support", ""), max_chars=360)
        q = r.get("question", "").strip()
        ans = r.get("correct_answer", "").strip()
        if sup and q and ans and len(sup) >= 40:
            qa_train_records.append(_make_supervised_chat_record(f"{sup}\n{q}", ans))

    for r in squad_train_rows:
        ctx = _clean_paragraph(r.get("context", ""), max_chars=400)
        q = r.get("question", "").strip()
        ans_list = r.get("answers", {}).get("text", [])
        ans = ans_list[0].strip() if ans_list else ""
        if ctx and q and ans and len(ans) <= 48 and ans.lower() in ctx.lower():
            qa_train_records.append(_make_supervised_chat_record(f"{ctx}\n{q}", ans))

    # Build held-out SciQ & SQuAD multiple-choice + extractive evaluation benchmarks
    sciq_eval_questions: list[dict] = []
    qa_val_records: list[dict] = []
    for idx, r in enumerate(sciq_val_rows):
        sup = _clean_paragraph(r.get("support", ""), max_chars=360)
        q = r.get("question", "").strip()
        ans = r.get("correct_answer", "").strip()
        d1 = r.get("distractor1", "").strip()
        d2 = r.get("distractor2", "").strip()
        d3 = r.get("distractor3", "").strip()
        if sup and q and ans and d1 and d2 and d3 and len(sup) >= 40:
            choices = [ans, d1, d2, d3]
            sciq_eval_questions.append(
                {
                    "id": f"sciq-{idx}",
                    "category": "sciq-science",
                    "context": sup,
                    "question": q,
                    "choices": choices,
                    "answer_index": 0,
                }
            )
            qa_val_records.append(_make_supervised_chat_record(f"{sup}\n{q}", ans))
            if len(sciq_eval_questions) >= eval_size:
                break

    squad_eval_questions: list[dict] = []
    for idx, r in enumerate(squad_val_rows):
        ctx = _clean_paragraph(r.get("context", ""), max_chars=380)
        q = r.get("question", "").strip()
        ans_list = r.get("answers", {}).get("text", [])
        ans = ans_list[0].strip() if ans_list else ""
        if (
            ctx
            and q
            and ans
            and 3 <= len(ans) <= 32
            and " " not in ans
            and ans.isalpha()
            and ans in ctx
        ):
            # Pick a plausible distractor word from the same passage so MC & open-ended both work
            words = [
                w.strip(".,;:!?()\"'")
                for w in ctx.split()
                if len(w.strip(".,;:!?()\"'")) >= 4
                and w.strip(".,;:!?()\"'").isalpha()
                and w.strip(".,;:!?()\"'").lower() != ans.lower()
            ]
            if words:
                distractor = words[0]
                squad_eval_questions.append(
                    {
                        "id": f"squad-{idx}",
                        "category": "squad-extractive",
                        "context": ctx,
                        "question": q,
                        "choices": [ans, distractor],
                        "answer_index": 0,
                    }
                )
                if len(squad_eval_questions) >= eval_size:
                    break

    # Partition English stream into train / validation / test
    eng_val = english_docs[-2 * eval_size : -eval_size]
    eng_test = english_docs[-eval_size:]
    eng_train = english_docs[: -2 * eval_size]

    chat_val = [{"text": t} for t in smol_val[:eval_size]]
    chat_test = [{"text": t} for t in smol_val[eval_size : 2 * eval_size]]

    # Write files
    eng_dir = root / "english"
    chat_dir = root / "conversation"
    qa_dir = root / "qa"
    mix_dir = root / "mixed"
    for d in (eng_dir, chat_dir, qa_dir, mix_dir):
        d.mkdir(parents=True, exist_ok=True)

    def _write_jsonl(path: Path, rows: list[dict]):
        path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
            encoding="utf-8",
        )

    _write_jsonl(eng_dir / "train.jsonl", [{"text": t} for t in eng_train])
    _write_jsonl(eng_dir / "validation.jsonl", [{"text": t} for t in eng_val])
    _write_jsonl(eng_dir / "test.jsonl", [{"text": t} for t in eng_test])

    _write_jsonl(chat_dir / "train.jsonl", chat_records)
    _write_jsonl(chat_dir / "validation.jsonl", chat_val)
    _write_jsonl(chat_dir / "test.jsonl", chat_test)

    _write_jsonl(qa_dir / "train.jsonl", qa_train_records)
    _write_jsonl(qa_dir / "validation.jsonl", qa_val_records)

    # Combined curriculum mix: Encyclopedic English + Multi-Turn Chat + Grounded Reading QA
    mixed_train = [{"text": t} for t in eng_train] + chat_records + qa_train_records
    mixed_val = [{"text": t} for t in eng_val[:16]] + chat_val[:16] + qa_val_records[:16]
    _write_jsonl(mix_dir / "train.jsonl", mixed_train)
    _write_jsonl(mix_dir / "validation.jsonl", mixed_val)

    (qa_dir / "sciq-validation-questions.json").write_text(
        json.dumps({"questions": sciq_eval_questions}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (qa_dir / "squad-validation-questions.json").write_text(
        json.dumps({"questions": squad_eval_questions}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sources": [
            "wikimedia/wikipedia (20231101.simple)",
            "HuggingFaceTB/cosmopedia-100k",
            "HuggingFaceTB/smoltalk (everyday-conversations)",
            "databricks/databricks-dolly-15k",
            "allenai/sciq",
            "rajpurkar/squad",
        ],
        "counts": {
            "english_train": len(eng_train),
            "english_validation": len(eng_val),
            "chat_train": len(chat_records),
            "chat_validation": len(chat_val),
            "qa_train": len(qa_train_records),
            "sciq_eval_questions": len(sciq_eval_questions),
            "squad_eval_questions": len(squad_eval_questions),
            "mixed_train": len(mixed_train),
        },
        "english_train_hash": corpus_hash(eng_train),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare curated Hugging Face datasets for MLM.")
    parser.add_argument("--out", type=Path, default=Path("data/curated-english-v1"))
    args = parser.parse_args(argv)
    manifest = prepare_curated_english(args.out)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
