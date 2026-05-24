"""Word and subword vocabulary construction for FastText training."""

from __future__ import annotations

import json
import pickle
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from src.tokenizer import generate_ngrams

UNK_TOKEN = "<UNK>"
PAD_TOKEN = "<PAD>"


@dataclass
class Vocabulary:
    """Bidirectional mappings for words and subword n-grams."""

    min_freq: int = 5
    min_n: int = 3
    max_n: int = 6

    word2idx: Dict[str, int] = field(default_factory=dict)
    idx2word: Dict[int, str] = field(default_factory=dict)
    subword2idx: Dict[str, int] = field(default_factory=dict)
    idx2subword: Dict[int, str] = field(default_factory=dict)
    word_freq: Counter = field(default_factory=Counter)
    word_ngrams: Dict[int, List[int]] = field(default_factory=dict)
    negative_sampling_probs: Optional[np.ndarray] = None
    subword_cache: Optional[np.ndarray] = None
    subword_lengths: Optional[np.ndarray] = None
    max_subword_len: int = 1

    @property
    def vocab_size(self) -> int:
        return len(self.word2idx)

    @property
    def subword_vocab_size(self) -> int:
        return len(self.subword2idx)

    @property
    def unk_idx(self) -> int:
        return self.word2idx[UNK_TOKEN]

    @property
    def pad_idx(self) -> int:
        return self.word2idx[PAD_TOKEN]

    def build_from_tokens(self, tokens: Iterable[str]) -> None:
        """Build word and subword vocabularies from an iterable of tokens."""
        self.word_freq = Counter(tokens)
        self._build_word_vocab()
        self._build_subword_vocab()
        self._build_word_ngram_indices()
        self._build_negative_sampling_distribution()
        self._build_subword_cache()

    def _build_word_vocab(self) -> None:
        self.word2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        idx = 2
        for word, freq in self.word_freq.most_common():
            if freq < self.min_freq:
                continue
            self.word2idx[word] = idx
            idx += 1
        self.idx2word = {index: word for word, index in self.word2idx.items()}

    def _build_subword_vocab(self) -> None:
        self.subword2idx = {PAD_TOKEN: 0}
        idx = 1
        for word in self.word2idx:
            if word in {PAD_TOKEN, UNK_TOKEN}:
                continue
            for ngram in generate_ngrams(word, self.min_n, self.max_n):
                if ngram not in self.subword2idx:
                    self.subword2idx[ngram] = idx
                    idx += 1
        self.idx2subword = {index: token for token, index in self.subword2idx.items()}

    def _build_word_ngram_indices(self) -> None:
        self.word_ngrams = {}
        for word, word_idx in self.word2idx.items():
            if word in {PAD_TOKEN, UNK_TOKEN}:
                self.word_ngrams[word_idx] = []
                continue
            ngram_ids = [
                self.subword2idx[ngram]
                for ngram in generate_ngrams(word, self.min_n, self.max_n)
            ]
            self.word_ngrams[word_idx] = ngram_ids

    def _build_negative_sampling_distribution(self) -> None:
        """Unigram distribution raised to the 3/4 power (word2vec standard)."""
        freqs = np.zeros(self.vocab_size, dtype=np.float64)
        for word, idx in self.word2idx.items():
            if word in {PAD_TOKEN, UNK_TOKEN}:
                continue
            freqs[idx] = float(self.word_freq.get(word, 0))
        powered = np.power(freqs, 0.75)
        total = powered.sum()
        if total <= 0:
            self.negative_sampling_probs = np.full(self.vocab_size, 1.0 / self.vocab_size)
        else:
            self.negative_sampling_probs = powered / total

    def word_to_idx(self, word: str) -> int:
        return self.word2idx.get(word, self.unk_idx)

    def encode_corpus(self, tokens: Sequence[str]) -> List[int]:
        return [self.word_to_idx(token) for token in tokens]

    def _build_subword_cache(self) -> None:
        """Dense [vocab_size, max_ngrams] table for fast batch collation."""
        max_len = max((len(ngrams) for ngrams in self.word_ngrams.values()), default=1)
        max_len = max(max_len, 1)
        cache = np.zeros((self.vocab_size, max_len), dtype=np.int32)
        lengths = np.zeros(self.vocab_size, dtype=np.int32)
        for word_idx, ngrams in self.word_ngrams.items():
            if not ngrams:
                continue
            length = len(ngrams)
            lengths[word_idx] = length
            cache[word_idx, :length] = np.asarray(ngrams, dtype=np.int32)
        self.subword_cache = cache
        self.subword_lengths = lengths
        self.max_subword_len = max_len

    def get_subword_indices(self, word_idx: int) -> List[int]:
        return self.word_ngrams.get(word_idx, [])

    def get_subword_indices_for_word(self, word: str) -> List[int]:
        return self.get_subword_indices(self.word_to_idx(word))

    def oov_rate(self, tokens: Sequence[str]) -> float:
        if not tokens:
            return 0.0
        oov = sum(1 for token in tokens if token not in self.word2idx)
        return oov / len(tokens)

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "min_freq": self.min_freq,
            "min_n": self.min_n,
            "max_n": self.max_n,
            "word2idx": self.word2idx,
            "subword2idx": self.subword2idx,
            "word_freq": dict(self.word_freq),
        }
        with open(directory / "vocab.json", "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        with open(directory / "vocab.pkl", "wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, directory: str | Path) -> "Vocabulary":
        directory = Path(directory)
        with open(directory / "vocab.pkl", "rb") as handle:
            vocab = pickle.load(handle)
        if not isinstance(vocab, cls):
            raise TypeError("Loaded object is not a Vocabulary instance.")
        return vocab


def build_vocabulary(
    tokens: Iterable[str],
    min_freq: int = 5,
    min_n: int = 3,
    max_n: int = 6,
) -> Vocabulary:
    """Convenience helper to build a vocabulary from tokens."""
    vocab = Vocabulary(min_freq=min_freq, min_n=min_n, max_n=max_n)
    vocab.build_from_tokens(tokens)
    return vocab
