"""Bounded public Hugging Face row downloads with cached-response provenance."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from mlm.data import corpus_hash


def render_row(row):
    if "text" in row:
        return row["text"].strip()
    pieces = []
    for message in row["messages"]:
        role = message["role"]
        if role not in ("system", "user", "assistant"):
            raise ValueError(f"Unknown conversation role: {role}")
        pieces.append(f"<|{role}|>\n{message['content'].strip()}\n")
    return "".join(pieces)


def disjoint_splits(train, heldout, development_size):
    train = list(dict.fromkeys(train))
    seen = set(train)
    remaining = []
    for text in heldout:
        if text and text not in seen:
            seen.add(text)
            remaining.append(text)
    if len(remaining) < 2 * development_size:
        raise ValueError("Not enough unique held-out documents")
    return {
        "train": train,
        "validation": remaining[:development_size],
        "test": remaining[development_size : 2 * development_size],
    }


def get_json(url):
    request = Request(url, headers={"User-Agent": "mini-learning-model-research/0.1"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=20) as response:
                return response.read(), dict(response.headers)
        except (HTTPError, URLError, TimeoutError) as error:
            if isinstance(error, HTTPError) and error.code not in (429, 500, 502, 503, 504):
                raise
            if attempt == 2:
                raise
            print(f"Retrying transient dataset error: {error}", flush=True)
    raise RuntimeError("Dataset request failed")


def fetch_rows(root, dataset, config, split, count):
    documents, pages = [], []
    cache = root / "raw"
    cache.mkdir(parents=True, exist_ok=True)
    for offset in range(0, count, 100):
        url = "https://datasets-server.huggingface.co/rows?" + urlencode(
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": offset,
                "length": min(100, count - offset),
            }
        )
        key = hashlib.sha256(url.encode()).hexdigest()
        path = cache / f"{key}.json"
        if path.exists():
            raw = path.read_bytes()
            headers = {"cache": "local"}
        else:
            raw, headers = get_json(url)
            path.write_bytes(raw)
        result = json.loads(raw)
        rows = result["rows"]
        if len(rows) != min(100, count - offset):
            raise ValueError("Dataset response has fewer rows than requested")
        for item in rows:
            if item.get("truncated_cells"):
                raise ValueError("Dataset viewer truncated a cell; refusing incomplete documents")
            documents.append(render_row(item["row"]))
        pages.append(
            {
                "url": url,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "cache": str(path.relative_to(root)),
                "headers": headers,
            }
        )
        print(f"Downloaded {dataset}/{split}: {offset + len(rows)}/{count}", flush=True)
    return documents, pages


def prepare_mix(root):
    """Deterministic conversation adaptation with a bounded story rehearsal subset."""
    root = Path(root)
    directory = root / "mixed"
    directory.mkdir(exist_ok=True)
    manifest = {
        "parent_manifest": "../manifest.json",
        "purpose": "conversation adaptation with English rehearsal",
        "splits": {},
    }
    for split in ("train", "validation", "test"):
        stories = [
            json.loads(line)
            for line in (root / "stories" / f"{split}.jsonl").read_text().splitlines()
        ]
        chats = [
            json.loads(line)
            for line in (root / "conversation" / f"{split}.jsonl").read_text().splitlines()
        ]
        stories = stories[:250] if split == "train" else stories
        path = directory / f"{split}.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in chats + stories),
            encoding="utf-8",
        )
        manifest["splits"][split] = {
            "stories": len(stories),
            "conversations": len(chats),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def prepare(root, stories=1000, conversations=300, development_size=32):
    root = Path(root)
    if (root / "manifest.json").exists():
        raise ValueError("Dataset manifest already exists; reuse it or choose a new output path")
    manifest = {
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {},
        "revision_policy": (
            "Viewer API serves current rows; cached responses and hashes are the "
            "reproducibility record. Hub SHA is observed metadata, "
            "not a pinned viewer revision."
        ),
        "selection": "First bounded rows, not a representative random sample",
    }
    sources = [
        (
            "stories",
            "roneneldan/TinyStories",
            "default",
            "train",
            "validation",
            stories,
            "cdla-sharing-1.0",
            "https://huggingface.co/datasets/roneneldan/TinyStories",
        ),
        (
            "conversation",
            "HuggingFaceTB/everyday-conversations-llama3.1-2k",
            "default",
            "train_sft",
            "test_sft",
            conversations,
            "apache-2.0",
            "https://huggingface.co/datasets/HuggingFaceTB/everyday-conversations-llama3.1-2k",
        ),
    ]
    for name, dataset, config, train_split, heldout_split, count, license_id, card in sources:
        raw, _ = get_json("https://huggingface.co/api/datasets/" + dataset)
        hub_sha = json.loads(raw)["sha"]
        train, pages = fetch_rows(root, dataset, config, train_split, count)
        heldout, heldout_pages = fetch_rows(
            root, dataset, config, heldout_split, min(100, 2 * development_size + 16)
        )
        splits = disjoint_splits(train, heldout, development_size)
        directory = root / name
        directory.mkdir(parents=True, exist_ok=True)
        records = {}
        for split, documents in splits.items():
            path = directory / f"{split}.jsonl"
            path.write_text(
                "".join(
                    json.dumps({"text": text}, ensure_ascii=False) + "\n" for text in documents
                ),
                encoding="utf-8",
            )
            records[split] = {
                "documents": len(documents),
                "payload_bytes": sum(len(text.encode("utf-8")) for text in documents),
                "content_hash": corpus_hash(documents),
                "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        manifest["sources"][name] = {
            "dataset": dataset,
            "config": config,
            "train_source_split": train_split,
            "heldout_source_split": heldout_split,
            "observed_hub_sha": hub_sha,
            "license": license_id,
            "dataset_card": card,
            "pages": pages + heldout_pages,
            "splits": records,
        }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    prepare_mix(root)
    print(f"Prepared data and manifest under {root}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/hf"))
    parser.add_argument("--stories", type=int, default=1000)
    parser.add_argument("--conversations", type=int, default=300)
    parser.add_argument("--development-size", type=int, default=32)
    args = parser.parse_args()
    if not (
        1 <= args.stories <= 10000
        and 1 <= args.conversations <= 2000
        and 1 <= args.development_size <= 42
    ):
        parser.error("Counts exceed the bounded downloader limits")
    prepare(args.out, args.stories, args.conversations, args.development_size)


if __name__ == "__main__":
    main()
