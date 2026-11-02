"""Rust-backed Byte-Level BPE subword tokenizer using Hugging Face `tokenizers`."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL_TOKENS = [
    "<|bos|>",
    "<|eos|>",
    "<|pad|>",
    "<|user|>",
    "<|assistant|>",
    "<|system|>",
]

BOS_TOKEN = "<|bos|>"
EOS_TOKEN = "<|eos|>"
PAD_TOKEN = "<|pad|>"

BOS_ID = 0
EOS_ID = 1
PAD_ID = 2
USER_ID = 3
ASSISTANT_ID = 4
SYSTEM_ID = 5


class BPETokenizer:
    """Fast Rust-backed Byte-Level BPE tokenizer with byte-span supervision mask mapping."""

    def __init__(self, backend: Tokenizer) -> None:
        self._tokenizer = backend
        self.bos_id: int = self._tokenizer.token_to_id(BOS_TOKEN) or BOS_ID
        self.eos_id: int = self._tokenizer.token_to_id(EOS_TOKEN) or EOS_ID
        self.pad_id: int = self._tokenizer.token_to_id(PAD_TOKEN) or PAD_ID

    @property
    def vocab_size(self) -> int:
        return int(self._tokenizer.get_vocab_size())

    @classmethod
    def train_from_iterator(
        cls,
        texts: Iterable[str],
        vocab_size: int = 8192,
        min_frequency: int = 2,
    ) -> BPETokenizer:
        """Train a Byte-Level BPE tokenizer in Rust on an iterator of UTF-8 strings."""
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Digits(individual_digits=True),
                pre_tokenizers.ByteLevel(add_prefix_space=False),
            ]
        )
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=int(vocab_size),
            min_frequency=int(min_frequency),
            special_tokens=list(SPECIAL_TOKENS),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
        tok.train_from_iterator(texts, trainer=trainer)
        return cls(tok)

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._tokenizer.save(str(p))

    @classmethod
    def load(cls, path: Path | str) -> BPETokenizer:
        return cls(Tokenizer.from_file(str(Path(path))))

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> np.ndarray:
        """Encode UTF-8 text into `int32` BPE token IDs."""
        enc = self._tokenizer.encode(text, add_special_tokens=False)
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        ids.extend(enc.ids)
        if add_eos:
            ids.append(self.eos_id)
        return np.asarray(ids, dtype=np.int32)

    def decode(
        self,
        token_ids: Sequence[int] | np.ndarray,
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode `int32` BPE token IDs back into a valid UTF-8 string."""
        ids = [int(t) for t in token_ids]
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def encode_with_mask(
        self,
        text: str,
        loss_spans: list[list[int]] | None = None,
        score_eos: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode `text` to `[BOS, *ids, EOS]` and map UTF-8 byte `loss_spans` to target mask.

        Returns:
            tokens: `int32` array of shape `(T + 2,)` (`[BOS, t_0, ..., t_{T-1}, EOS]`)
            mask: `float32` array of shape `(T + 1,)` aligned with targets `tokens[1:]`
                  (`[t_0, ..., t_{T-1}, EOS]`).
        """
        enc = self._tokenizer.encode(text, add_special_tokens=False)
        ids = enc.ids
        n_subwords = len(ids)
        tokens = np.empty(n_subwords + 2, dtype=np.int32)
        tokens[0] = self.bos_id
        if n_subwords > 0:
            tokens[1 : n_subwords + 1] = ids
        tokens[n_subwords + 1] = self.eos_id

        if loss_spans is None:
            mask = np.ones(n_subwords + 1, dtype=np.float32)
            mask[-1] = 1.0 if score_eos else 0.0
            return tokens, mask

        if text.isascii():
            char_to_byte = None
        else:
            # Build character-index -> UTF-8 byte-offset lookup table for non-ASCII text
            char_to_byte = np.zeros(len(text) + 1, dtype=np.int32)
            running_bytes = 0
            for idx, ch in enumerate(text):
                running_bytes += len(ch.encode("utf-8"))
                char_to_byte[idx + 1] = running_bytes

        mask = np.zeros(n_subwords + 1, dtype=np.float32)
        offsets = enc.offsets
        text_len = len(text)
        for tok_idx, (c_start, c_end) in enumerate(offsets):
            if char_to_byte is None:
                b_start = min(c_start, text_len)
                b_end = min(c_end, text_len)
            else:
                b_start = int(char_to_byte[min(c_start, text_len)])
                b_end = int(char_to_byte[min(c_end, text_len)])
            for s_start, s_end in loss_spans:
                # A subword token is supervised if its end byte extends into the span
                # (handles leading `Ġ` space tokens attached to the first assistant word).
                if b_end > s_start and b_start < s_end:
                    mask[tok_idx] = 1.0
                    break
        mask[-1] = 1.0 if score_eos else 0.0
        if not mask.any() and n_subwords > 0:
            mask[-2] = 1.0
        return tokens, mask

    def get_vocab_tokens_and_scores(self) -> tuple[list[str], list[float], list[int]]:
        """Extract ordered vocabulary strings, scores, and token types for GGUF export."""
        vocab = self._tokenizer.get_vocab(with_added_tokens=True)
        size = self.vocab_size
        tokens = [""] * size
        scores = [0.0] * size
        # GGUF token types: 1 = NORMAL, 3 = CONTROL
        token_types = [1] * size
        special_set = set(SPECIAL_TOKENS)
        for tok_str, tok_id in vocab.items():
            if 0 <= tok_id < size:
                tokens[tok_id] = tok_str
                scores[tok_id] = -float(tok_id)
                if tok_str in special_set:
                    token_types[tok_id] = 3
        return tokens, scores, token_types
