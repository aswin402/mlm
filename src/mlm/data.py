"""Byte vocabulary and independent document-prefix batches."""

import hashlib
import json
from pathlib import Path

import numpy as np

BOS = 256
EOS = 257
VOCAB_SIZE = 258


def encode(text: str) -> np.ndarray:
    return np.array([BOS, *text.encode("utf-8"), EOS], dtype=np.int32)


def read_documents(path: Path) -> list[str]:
    if path.suffix == ".jsonl":
        documents = [
            json.loads(line)["text"]
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not all(isinstance(doc, str) for doc in documents):
            raise ValueError("Every JSONL document must contain a text string")
    else:
        documents = [part.strip() for part in path.read_text(encoding="utf-8").split("\n\n")]
    documents = [part for part in documents if part]
    if not documents:
        raise ValueError(f"No nonempty documents in {path}")
    return documents


def corpus_hash(documents: list[str]) -> str:
    digest = hashlib.sha256()
    for document in documents:
        payload = document.encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def make_batch(documents: list[str], indices, length: int) -> dict[str, np.ndarray]:
    """Each row starts fresh; long documents are explicitly prefix-truncated."""
    if length < 1 or len(indices) == 0:
        raise ValueError("Batch and length must be nonempty")
    shape = (len(indices), length)
    inputs = np.full(shape, EOS, dtype=np.int32)
    targets = inputs.copy()
    mask = np.zeros(shape, dtype=np.float32)
    resets = np.zeros(shape, dtype=bool)
    resets[:, 0] = True
    for row, index in enumerate(indices):
        tokens = encode(documents[int(index)])
        count = min(length, len(tokens) - 1)
        inputs[row, :count] = tokens[:count]
        targets[row, :count] = tokens[1 : count + 1]
        mask[row, :count] = 1
    return {"inputs": inputs, "targets": targets, "mask": mask, "resets": resets}


def supervision_hash(masks):
    if masks is None:
        return None
    digest = hashlib.sha256()
    for mask in masks:
        array = np.asarray(mask, dtype=np.uint8)
        digest.update(len(array).to_bytes(8, "big"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def read_training_data(path):
    """JSONL may specify half-open UTF-8 byte loss_spans and score_eos."""
    path = Path(path)
    if path.suffix != ".jsonl":
        return read_documents(path), None, None
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    documents, masks = [], []
    explicit = any("loss_spans" in row or "score_eos" in row for row in records)
    for row in records:
        text = row.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError("Each training record needs nonempty text")
        size = len(text.encode("utf-8"))
        mask = np.zeros(size + 1, np.float32)
        spans = row.get("loss_spans", [[0, size]])
        previous_end = 0
        for start, end in spans:
            if (
                type(start) is not int
                or type(end) is not int
                or not previous_end <= start < end <= size
            ):
                raise ValueError("Invalid, unsorted, or overlapping loss spans")
            mask[start:end] = 1
            previous_end = end
        score_eos = row.get("score_eos", "loss_spans" not in row)
        if type(score_eos) is not bool:
            raise ValueError("score_eos must be boolean")
        mask[-1] = float(score_eos)
        if not mask.any():
            raise ValueError("Each document needs at least one supervised target")
        documents.append(text)
        masks.append(mask)
    if not documents:
        raise ValueError("No training documents")
    return documents, masks if explicit else None, supervision_hash(masks) if explicit else None
