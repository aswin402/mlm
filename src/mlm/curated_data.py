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
from mlm.hfdata import get_json


def _clean_paragraph(text: str, max_chars: int = 640) -> str:
    """Extract a clean, complete multi-sentence English paragraph bounded to max_chars."""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    joined = " ".join(lines)
    if len(joined) <= max_chars:
        return joined
    cut = joined[:max_chars]
    last_dot = max(cut.rfind(". "), cut.rfind(".\n"))
    if last_dot >= min(40, max_chars // 3):
        return cut[: last_dot + 1].strip()
    return cut.strip()


def _first_complete_sentences(text: str, max_chars: int = 110) -> str:
    """Extract 1-2 complete sentences ending in '.', '!', or '?' within max_chars."""
    joined = " ".join(ln.strip() for ln in text.strip().splitlines() if ln.strip())
    if not joined:
        return ""
    if len(joined) <= max_chars and joined[-1] in ".!?":
        return joined
    cut = joined[:max_chars]
    best = -1
    for sep in (". ", "! ", "? ", ".", "!", "?"):
        idx = cut.rfind(sep)
        if idx > best:
            best = idx
    if best >= 6:
        return cut[: best + 1].strip()
    return ""


def _make_supervised_chat_record(user_text: str, assistant_text: str) -> dict:
    """Format a `<|user|>...<|assistant|>...` record with assistant-only `loss_spans`."""
    prefix = f"<|user|>\n{user_text.strip()}\n<|assistant|>\n"
    reply = assistant_text.strip()
    full_text = prefix + reply
    start_b = len(prefix.encode("utf-8"))
    end_b = len(full_text.encode("utf-8"))
    return {
        "text": full_text,
        "loss_spans": [[start_b, end_b]],
        "score_eos": True,
    }


def _make_supervised_think_record(
    user_text: str, thought_text: str, answer_text: str = ""
) -> dict:
    """Format a DeepSeek-R1 `<|think|>...<|/think|>` record supervised after `<|think|>`."""
    prefix = f"<|user|>\n{user_text.strip()}\n<|assistant|>\n<|think|>"
    reply = f"{thought_text.strip()}<|/think|>"
    full_text = prefix + reply
    start_b = len(prefix.encode("utf-8"))
    end_b = len(full_text.encode("utf-8"))
    return {
        "text": full_text,
        "loss_spans": [[start_b, end_b]],
        "score_eos": True,
        "answer": answer_text.strip(),
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


def _extract_turn_records(rows: list[dict], max_total_bytes: int = 126) -> list[dict]:
    """Extract deduplicated supervised (user, assistant) chat records fitting max_total_bytes."""
    records: list[dict] = []
    seen: set[str] = set()
    for r in rows:
        msgs = r.get("messages", [])
        for i in range(0, len(msgs) - 1, 2):
            if msgs[i].get("role") == "user" and msgs[i + 1].get("role") == "assistant":
                u = " ".join(msgs[i].get("content", "").strip().split())
                a = _first_complete_sentences(msgs[i + 1].get("content", ""), max_chars=92)
                if not u or not a or len(u) > 72 or "\n" in u:
                    continue
                key = u.lower()
                if key in seen:
                    continue
                rec = _make_supervised_chat_record(u, a)
                if len(rec["text"].encode("utf-8")) <= max_total_bytes:
                    seen.add(key)
                    records.append(rec)
    return records


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

    # 2. Multi-Turn Chat & Grounded Instruction QA (SmolTalk + Magpie-Ultra + Alpaca + Dolly-15k)
    smol_train_rows = fetch_raw_items(
        root, "HuggingFaceTB/smoltalk", "everyday-conversations", "train", smoltalk_count
    )
    smol_val_rows = fetch_raw_items(
        root, "HuggingFaceTB/smoltalk", "everyday-conversations", "test", eval_size * 2
    )
    magpie_rows = fetch_raw_items(root, "HuggingFaceTB/smoltalk", "smol-magpie-ultra", "train", 100)
    alpaca_rows = fetch_raw_items(root, "tatsu-lab/alpaca", "default", "train", 200)
    dolly_rows = fetch_raw_items(
        root, "databricks/databricks-dolly-15k", "default", "train", dolly_count
    )

    chat_records: list[dict] = _extract_turn_records(smol_train_rows + magpie_rows)
    seen_users = {
        r["text"].split("\n<|assistant|>\n")[0].replace("<|user|>\n", "").strip().lower()
        for r in chat_records
    }
    for r in alpaca_rows:
        if not r.get("input", "").strip():
            inst = " ".join(r.get("instruction", "").strip().split())
            resp = _first_complete_sentences(r.get("output", ""), max_chars=92)
            if inst and resp and len(inst) <= 72 and inst.lower() not in seen_users:
                rec = _make_supervised_chat_record(inst, resp)
                if len(rec["text"].encode("utf-8")) <= 126:
                    seen_users.add(inst.lower())
                    chat_records.append(rec)

    for r in dolly_rows:
        inst = " ".join(r.get("instruction", "").strip().split())
        ctx = _clean_paragraph(r.get("context", ""), max_chars=240)
        resp = _first_complete_sentences(r.get("response", ""), max_chars=92)
        if not inst or not resp:
            continue
        if not ctx and len(inst) <= 72 and inst.lower() not in seen_users:
            rec = _make_supervised_chat_record(inst, resp)
            if len(rec["text"].encode("utf-8")) <= 126:
                seen_users.add(inst.lower())
                chat_records.append(rec)

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

    val_turn_records = _extract_turn_records(smol_val_rows)
    chat_val = val_turn_records[:eval_size]
    chat_test = val_turn_records[eval_size : 2 * eval_size] or val_turn_records[:eval_size]

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
            "HuggingFaceTB/smoltalk (smol-magpie-ultra)",
            "tatsu-lab/alpaca",
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
