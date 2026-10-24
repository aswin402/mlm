from mlm.hfdata import disjoint_splits, render_row


def test_chat_format_preserves_roles_and_paragraphs():
    row = {
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello\n\nHow are you?"},
        ]
    }
    text = render_row(row)
    assert text == "<|user|>\nHi\n<|assistant|>\nHello\n\nHow are you?\n"


def test_documents_are_deduplicated_before_splitting():
    splits = disjoint_splits(["a", "b", "b"], ["a", "c", "d", "e", "f"], 2)
    assert splits == {"train": ["a", "b"], "validation": ["c", "d"], "test": ["e", "f"]}


def test_rehearsal_mix_is_reproducible_and_preserves_documents(tmp_path):
    import json

    from mlm.data import read_documents
    from mlm.hfdata import prepare_mix

    for family, count in (("stories", 260), ("conversation", 3)):
        directory = tmp_path / family
        directory.mkdir()
        for split in ("train", "validation", "test"):
            documents = [
                {"text": f"{family}-{split}-{i}\n\nsecond paragraph"}
                for i in range(count if split == "train" else 2)
            ]
            (directory / f"{split}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in documents)
            )
    prepare_mix(tmp_path)
    before = (tmp_path / "mixed/manifest.json").read_text()
    prepare_mix(tmp_path)
    assert before == (tmp_path / "mixed/manifest.json").read_text()
    documents = read_documents(tmp_path / "mixed/train.jsonl")
    assert len(documents) == 253
    assert all("\n\nsecond paragraph" in doc for doc in documents)
    assert len(read_documents(tmp_path / "mixed/test.jsonl")) == 4
