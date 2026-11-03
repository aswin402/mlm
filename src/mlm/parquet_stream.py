"""Rust-powered (`polars`) large-scale Parquet shard fetcher and dataset builder."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import polars as pl

from mlm.curated_data import (
    _clean_paragraph,
    _first_complete_sentences,
    _make_supervised_chat_record,
    _make_supervised_think_record,
)
from mlm.hfdata import get_json


def fetch_parquet_urls(dataset: str, config: str, split: str) -> list[str]:
    """Query Hugging Face datasets-server for direct Parquet shard URLs."""
    url = "https://datasets-server.huggingface.co/parquet?" + urlencode(
        {"dataset": dataset, "config": config, "split": split}
    )
    raw, _ = get_json(url)
    payload = json.loads(raw)
    return [
        item["url"]
        for item in payload.get("parquet_files", [])
        if item.get("split") == split and item.get("url")
    ]


def download_parquet_shard(url: str, cache_dir: Path) -> Path:
    """Download a Parquet shard to `cache_dir` with deterministic SHA-256 caching."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:20]
    target = cache_dir / f"{digest}.parquet"
    if target.exists() and target.stat().st_size > 0:
        return target
    req = Request(url, headers={"User-Agent": "mlm-parquet-stream/0.2"})
    with urlopen(req, timeout=120) as resp:
        data = resp.read()
    tmp = target.with_suffix(".parquet.tmp")
    tmp.write_bytes(data)
    tmp.replace(target)
    return target


def read_hf_parquet_rows(
    cache_root: Path,
    dataset: str,
    config: str,
    split: str,
    max_rows: int = 5000,
) -> list[dict]:
    """Stream up to `max_rows` records from Hugging Face Parquet shards via Rust `polars`."""
    urls = fetch_parquet_urls(dataset, config, split)
    rows: list[dict] = []
    shard_cache = Path(cache_root) / "parquet_shards"
    for url in urls:
        if len(rows) >= max_rows:
            break
        local_parquet = download_parquet_shard(url, shard_cache)
        need = max_rows - len(rows)
        if dataset == "wikimedia/wikipedia":
            df_full = pl.read_parquet(local_parquet)
            df = df_full.filter(
                pl.col("title").str.contains(r"^[A-Z][a-z]+( [a-z]+)?$")
                & (pl.col("text").str.len_bytes() >= 500)
            ).head(need)
        else:
            df = pl.read_parquet(local_parquet, n_rows=need)
        rows.extend(df.to_dicts())
    return rows[:max_rows]


def extract_multi_turn_chat_from_parquet(
    rows: list[dict],
    max_user_chars: int = 240,
    max_assistant_chars: int = 320,
    max_total_bytes: int = 560,
    first_turn_only: bool = True,
) -> list[dict]:
    """Extract clean self-contained supervised `(user, assistant)` chat pairs from Parquet rows."""
    records: list[dict] = []
    seen: set[str] = set()
    greeting_count = 0
    for r in rows:
        msgs = r.get("messages")
        if not isinstance(msgs, list) or len(msgs) < 2:
            continue
        max_idx = min(len(msgs) - 1, 3 if first_turn_only else len(msgs) - 1)
        for i in range(0, max_idx, 2):
            m_u = msgs[i] if isinstance(msgs[i], dict) else {}
            m_a = msgs[i + 1] if isinstance(msgs[i + 1], dict) else {}
            if m_u.get("role") == "user" and m_a.get("role") == "assistant":
                u = " ".join(str(m_u.get("content", "")).strip().split())
                a = _first_complete_sentences(
                    str(m_a.get("content", "")), max_chars=max_assistant_chars
                )
                if not u or not a or len(u) > max_user_chars:
                    continue
                is_short_opener = i == 0 and len(u) <= 22
                if is_short_opener and greeting_count < 120:
                    base_word = u.rstrip(".!?").strip()
                    variants = [u, base_word, base_word.lower(), u.lower()]
                    if base_word and base_word[-1].isalpha():
                        low = base_word.lower()
                        variants.extend([low + low[-1], low + (low[-1] * 2)])
                    for v in variants:
                        if not v:
                            continue
                        rec_v = _make_supervised_chat_record(v, a)
                        if len(rec_v["text"].encode("utf-8")) <= max_total_bytes:
                            records.append(rec_v)
                            greeting_count += 1
                            seen.add(v)
                    continue
                key = u.lower()
                if key in seen:
                    continue
                rec = _make_supervised_chat_record(u, a)
                if len(rec["text"].encode("utf-8")) <= max_total_bytes:
                    seen.add(key)
                    records.append(rec)
    return records


def build_large_parquet_corpus(
    root: Path,
    smoltalk_rows: int = 2500,
    alpaca_rows: int = 3000,
    dolly_rows: int = 2000,
    sciq_rows: int = 2000,
    squad_rows: int = 2000,
    wiki_rows: int = 1500,
) -> dict:
    """Build a large-scale real-world English & Chat corpus using Rust `polars` Parquet shards."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    # 1. SmolTalk everyday conversations (~2,260 multi-turn dialogues -> ~8,000 turns)
    smol_raw = read_hf_parquet_rows(
        root, "HuggingFaceTB/smoltalk", "everyday-conversations", "train", max_rows=smoltalk_rows
    )
    chat_records = extract_multi_turn_chat_from_parquet(smol_raw)

    # 2. Stanford Alpaca clean instruction-following turns
    alpaca_raw = read_hf_parquet_rows(
        root, "tatsu-lab/alpaca", "default", "train", max_rows=alpaca_rows
    )
    seen_users = {r["text"].split("\n<|assistant|>\n")[0].lower() for r in chat_records}
    for r in alpaca_raw:
        if str(r.get("input", "")).strip():
            continue
        u = " ".join(str(r.get("instruction", "")).strip().split())
        a = _first_complete_sentences(str(r.get("output", "")), max_chars=320)
        if not u or not a or len(u) > 240:
            continue
        k = f"<|user|>\n{u}".lower()
        if k not in seen_users:
            seen_users.add(k)
            chat_records.append(_make_supervised_chat_record(u, a))

    # 3. Databricks Dolly-15k human-authored QA & chat turns
    dolly_raw = read_hf_parquet_rows(
        root, "databricks/databricks-dolly-15k", "default", "train", max_rows=dolly_rows
    )
    for r in dolly_raw:
        cat = str(r.get("category", ""))
        if cat not in ("open_qa", "general_qa", "brainstorming", "classification"):
            continue
        u = " ".join(str(r.get("instruction", "")).strip().split())
        a = _first_complete_sentences(str(r.get("response", "")), max_chars=320)
        if not u or not a or len(u) > 240:
            continue
        k = f"<|user|>\n{u}".lower()
        if k not in seen_users:
            seen_users.add(k)
            chat_records.append(_make_supervised_chat_record(u, a))

    # 4. SciQ science questions & explanations
    sciq_raw = read_hf_parquet_rows(root, "allenai/sciq", "default", "train", max_rows=sciq_rows)
    for r in sciq_raw:
        q = " ".join(str(r.get("question", "")).strip().split())
        ans = " ".join(str(r.get("correct_answer", "")).strip().split())
        sup = _first_complete_sentences(str(r.get("support", "")), max_chars=220)
        if not q or not ans:
            continue
        reply = f"{ans.capitalize()}. {sup}".strip() if sup else f"{ans.capitalize()}."
        k = f"<|user|>\n{q}".lower()
        if k not in seen_users:
            seen_users.add(k)
            chat_records.append(_make_supervised_chat_record(q, reply))

    # 5. SQuAD factual reading comprehension
    squad_raw = read_hf_parquet_rows(
        root, "rajpurkar/squad", "plain_text", "train", max_rows=squad_rows
    )
    for r in squad_raw:
        ctx = _clean_paragraph(str(r.get("context", "")), max_chars=360)
        q = " ".join(str(r.get("question", "")).strip().split())
        answers = r.get("answers", {})
        ans_list = answers.get("text", []) if isinstance(answers, dict) else []
        if not ctx or not q or not ans_list:
            continue
        ans = " ".join(str(ans_list[0]).strip().split())
        if not ans or len(ans) > 80:
            continue
        u = f"{ctx}\nQuestion: {q}"
        reply = ans if ans[-1] in ".!?" else f"{ans}."
        k = f"<|user|>\n{u}".lower()
        if k not in seen_users:
            seen_users.add(k)
            chat_records.append(_make_supervised_chat_record(u, reply))

    # 6. Simple English Wikipedia encyclopedic paragraphs
    wiki_raw = read_hf_parquet_rows(
        root, "wikimedia/wikipedia", "20231101.simple", "train", max_rows=wiki_rows
    )
    for r in wiki_raw:
        title = str(r.get("title", "")).strip()
        body = _clean_paragraph(str(r.get("text", "")), max_chars=340)
        if title and len(body) >= 60:
            u = f"What is {title}?"
            k = f"<|user|>\n{u}".lower()
            if k not in seen_users:
                seen_users.add(k)
                chat_records.append(_make_supervised_chat_record(u, body))

    # Save using Polars to Parquet & JSONL splits
    val_count = min(256, max(32, len(chat_records) // 20))
    train_records = chat_records[:-val_count]
    val_records = chat_records[-val_count:]

    out_dir = root / "conversation_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_jsonl = out_dir / "train.jsonl"
    val_jsonl = out_dir / "validation.jsonl"
    train_parquet = out_dir / "train.parquet"

    train_jsonl.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in train_records),
        encoding="utf-8",
    )
    val_jsonl.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in val_records),
        encoding="utf-8",
    )
    pl.DataFrame(train_records).write_parquet(train_parquet)

    return {
        "train_records": len(train_records),
        "validation_records": len(val_records),
        "train_jsonl": str(train_jsonl),
        "validation_jsonl": str(val_jsonl),
        "train_parquet": str(train_parquet),
    }


def _write_split_jsonl(out_dir: Path, records: list[dict], val_ratio: int = 20) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    shuffled = list(records)
    random.Random(42).shuffle(shuffled)
    val_n = min(128, max(16, len(shuffled) // val_ratio))
    train_r = shuffled[:-val_n]
    val_r = shuffled[-val_n:]
    (out_dir / "train.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in train_r),
        encoding="utf-8",
    )
    (out_dir / "validation.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in val_r),
        encoding="utf-8",
    )
    return {"train": len(train_r), "validation": len(val_r)}


def build_staged_english_corpora(
    root: Path,
    max_rows_per_source: int = 1500,
) -> dict:
    """Build 3 staged datasets (`english_learn`, `english_understand`, `english_chat`)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    wiki_raw = read_hf_parquet_rows(
        root, "wikimedia/wikipedia", "20231101.simple", "train", max_rows=max_rows_per_source
    )
    sciq_raw = read_hf_parquet_rows(
        root, "allenai/sciq", "default", "train", max_rows=max_rows_per_source
    )
    squad_raw = read_hf_parquet_rows(
        root, "rajpurkar/squad", "plain_text", "train", max_rows=max_rows_per_source
    )
    smol_raw = read_hf_parquet_rows(
        root,
        "HuggingFaceTB/smoltalk",
        "everyday-conversations",
        "train",
        max_rows=max_rows_per_source,
    )
    dolly_raw = read_hf_parquet_rows(
        root, "databricks/databricks-dolly-15k", "default", "train", max_rows=max_rows_per_source
    )
    alpaca_raw = read_hf_parquet_rows(
        root, "tatsu-lab/alpaca", "default", "train", max_rows=max_rows_per_source
    )

    # Stage 1 (`english_learn`): Continuous encyclopedic & science English paragraphs (mask=1.0)
    stage1_records: list[dict] = []
    seen_p: set[str] = set()
    for r in wiki_raw:
        title = str(r.get("title", "")).strip()
        body = _clean_paragraph(str(r.get("text", "")), max_chars=380)
        if len(body) >= 60:
            doc = f"{title}: {body}" if title and not body.startswith(title) else body
            if doc not in seen_p:
                seen_p.add(doc)
                stage1_records.append({"text": doc})
    for r in sciq_raw:
        sup = _clean_paragraph(str(r.get("support", "")), max_chars=360)
        if len(sup) >= 60 and sup not in seen_p:
            seen_p.add(sup)
            stage1_records.append({"text": sup})
    for r in squad_raw:
        ctx = _clean_paragraph(str(r.get("context", "")), max_chars=360)
        if len(ctx) >= 60 and ctx not in seen_p:
            seen_p.add(ctx)
            stage1_records.append({"text": ctx})
    for a_val in range(1, 13):
        for b_val in range(1, 13):
            stage1_records.append(
                {
                    "text": (
                        f"Question: What is {a_val} + {b_val}? "
                        f"Solution: {a_val} + {b_val} = {a_val + b_val}. "
                        f"The answer is {a_val + b_val}."
                    )
                }
            )

    # Stage 2 (`english_understand`): Grounded Reading Comprehension & Science QA + 25% Stage 1
    wiki_qa: list[dict] = []
    sciq_qa: list[dict] = []
    squad_qa: list[dict] = []
    seen_q: set[str] = set()
    for r in wiki_raw:
        title = str(r.get("title", "")).strip()
        raw_text = str(r.get("text", "")).replace(" ()", "").replace("()", "")
        body = _first_complete_sentences(raw_text, max_chars=240)
        if title and body and len(body) >= 40:
            u = f"What is {title.lower()}?"
            if u.lower() not in seen_q:
                seen_q.add(u.lower())
                wiki_qa.append(_make_supervised_chat_record(u, body))
    for r in sciq_raw:
        q = " ".join(str(r.get("question", "")).strip().split())
        ans = " ".join(str(r.get("correct_answer", "")).strip().split())
        sup = _first_complete_sentences(str(r.get("support", "")), max_chars=200)
        if q and ans and q.lower() not in seen_q:
            seen_q.add(q.lower())
            reply = f"{ans.capitalize()}. {sup}".strip() if sup else f"{ans.capitalize()}."
            sciq_qa.append(_make_supervised_chat_record(q, reply))
    for r in squad_raw:
        ctx = _clean_paragraph(str(r.get("context", "")), max_chars=320)
        q = " ".join(str(r.get("question", "")).strip().split())
        answers = r.get("answers", {})
        ans_list = answers.get("text", []) if isinstance(answers, dict) else []
        if ctx and q and ans_list and q.lower() not in seen_q:
            ans = " ".join(str(ans_list[0]).strip().split())
            if ans and len(ans) <= 80:
                seen_q.add(q.lower())
                u = f"{ctx}\nQuestion: {q}"
                reply = ans if ans[-1] in ".!?" else f"{ans}."
                squad_qa.append(_make_supervised_chat_record(u, reply))

    stage2_pure = wiki_qa + sciq_qa + squad_qa

    stage2_records: list[dict] = []
    for idx, rec in enumerate(stage2_pure):
        stage2_records.append(rec)
        if idx % 4 == 0 and stage1_records:
            stage2_records.append(stage1_records[idx % len(stage1_records)])

    # Stage 3 (`english_chat`): Self-contained Everyday Chat + Supervised Stage 2 Replay
    stage3_pure: list[dict] = []
    curated_conv = root / "conversation" / "train.jsonl"
    if curated_conv.exists():
        for line in curated_conv.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            txt = str(row.get("text", ""))
            if "<|assistant|>\n" in txt and "loss_spans" in row:
                stage3_pure.append(row)
                u_p, a_p = txt.split("\n<|assistant|>\n", 1)
                u_c = u_p.replace("<|user|>\n", "").strip()
                a_c = a_p.strip()
                if u_c and a_c and len(u_c) <= 22:
                    base_w = u_c.rstrip(".!?").strip()
                    vars_u = [base_w, base_w.lower(), u_c.lower()]
                    if base_w and base_w[-1].isalpha():
                        low_w = base_w.lower()
                        vars_u.extend([low_w + low_w[-1], low_w + (low_w[-1] * 2)])
                    for v_u in vars_u:
                        if v_u and v_u != u_c:
                            stage3_pure.append(_make_supervised_chat_record(v_u, a_c))
    stage3_pure.extend(extract_multi_turn_chat_from_parquet(smol_raw, first_turn_only=True))
    seen_chat = {r["text"].split("\n<|assistant|>\n")[0].lower() for r in stage3_pure}

    for r in dolly_raw:
        if str(r.get("category", "")) not in ("open_qa", "general_qa", "brainstorming"):
            continue
        u = " ".join(str(r.get("instruction", "")).strip().split())
        a = _first_complete_sentences(str(r.get("response", "")), max_chars=240)
        if u and a and len(u) <= 180:
            k = f"<|user|>\n{u}".lower()
            if k not in seen_chat:
                seen_chat.add(k)
                stage3_pure.append(_make_supervised_chat_record(u, a))
    for r in alpaca_raw:
        if str(r.get("input", "")).strip():
            continue
        u = " ".join(str(r.get("instruction", "")).strip().split())
        a = _first_complete_sentences(str(r.get("output", "")), max_chars=240)
        if u and a and len(u) <= 180:
            k = f"<|user|>\n{u}".lower()
            if k not in seen_chat:
                seen_chat.add(k)
                stage3_pure.append(_make_supervised_chat_record(u, a))

    # Replay supervised Stage 2 records (`What is X?` + Science QA) in `<|user|>/<|assistant|>`
    stage3_records = list(stage3_pure) + list(wiki_qa) + list(sciq_qa[:600]) + list(squad_qa[:300])

    # Stage 4 (`stage4_reasoning_think`): DeepSeek-R1 `<|think|>...<|/think|>` Cold-Start Reasoning
    stage4_records: list[dict] = []
    seen_s4: set[str] = set()
    sciq_vocab: set[str] = set()
    for r in sciq_raw:
        text_blob = (str(r.get("question", "")) + " " + str(r.get("support", ""))).lower()
        for w in text_blob.replace(".", " ").replace(",", " ").split():
            if len(w) >= 4 and w.isalpha():
                sciq_vocab.add(w)

    for r in wiki_raw:
        title = str(r.get("title", "")).strip()
        raw_text = str(r.get("text", "")).replace(" ()", "").replace("()", "")
        if " " in title or len(raw_text) < 2000 or title.lower() not in sciq_vocab:
            continue
        first_s = _first_complete_sentences(raw_text, max_chars=120)
        body = _first_complete_sentences(raw_text, max_chars=220)
        if title and first_s and body and len(body) >= 40:
            u = f"What is {title.lower()}?"
            if u.lower() not in seen_s4:
                seen_s4.add(u.lower())
                thought = f"Concept: {title}. {first_s}"
                stage4_records.append(_make_supervised_think_record(u, thought, body))
    for rec in stage3_pure:
        txt = rec.get("text", "")
        if "\n<|assistant|>\n" not in txt:
            continue
        u_part, a_part = txt.split("\n<|assistant|>\n", 1)
        u_clean = u_part.replace("<|user|>\n", "").strip()
        a_clean = a_part.strip()
        if u_clean and a_clean and len(u_clean) <= 45 and u_clean not in seen_s4:
            seen_s4.add(u_clean)
            plan = _first_complete_sentences(a_clean, max_chars=140) or a_clean[:100].strip()
            stage4_records.append(_make_supervised_think_record(u_clean, plan, a_clean))
    for r in sciq_raw[:80]:
        q = " ".join(str(r.get("question", "")).strip().split())
        ans = " ".join(str(r.get("correct_answer", "")).strip().split())
        premise = _first_complete_sentences(str(r.get("support", "")), max_chars=140)
        if q and ans and premise and q.lower() not in seen_s4:
            seen_s4.add(q.lower())
            stage4_records.append(
                _make_supervised_think_record(q, premise, f"{ans.capitalize()}. {premise}")
            )

    s1_counts = _write_split_jsonl(root / "stage1_english_learn", stage1_records)
    s2_counts = _write_split_jsonl(root / "stage2_english_understand", stage2_records)
    s3_counts = _write_split_jsonl(root / "stage3_english_chat", stage3_records)
    s4_counts = _write_split_jsonl(root / "stage4_reasoning_think", stage4_records)

    return {
        "stage1_english_learn": s1_counts,
        "stage2_english_understand": s2_counts,
        "stage3_english_chat": s3_counts,
        "stage4_reasoning_think": s4_counts,
    }
