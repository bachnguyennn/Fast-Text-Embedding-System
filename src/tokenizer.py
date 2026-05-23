"""Text cleaning and FastText-style character n-gram generation."""

from __future__ import annotations

import re
import string
from typing import Iterable, Iterator, List, Sequence

# FastText boundary markers for subword n-grams.
WORD_BOUNDARY_LEFT = "<"
WORD_BOUNDARY_RIGHT = ">"


def clean_text(text: str) -> str:
    """Lowercase text and remove punctuation."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """Clean text and split into whitespace-delimited tokens."""
    cleaned = clean_text(text)
    if not cleaned:
        return []
    return cleaned.split()


def iter_tokens_from_file(path: str, chunk_size: int = 1_000_000) -> Iterator[str]:
    """Stream tokens from a large text file without loading it entirely."""
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        buffer = ""
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            buffer += chunk.lower()
            while True:
                match = re.search(r"[a-z]+", buffer)
                if not match:
                    buffer = buffer[-50:] if len(buffer) > 50 else buffer
                    break
                yield match.group(0)
                buffer = buffer[match.end() :]
        while True:
            match = re.search(r"[a-z]+", buffer)
            if not match:
                break
            yield match.group(0)
            buffer = buffer[match.end() :]


def generate_ngrams(
    word: str,
    min_n: int = 3,
    max_n: int = 6,
) -> List[str]:
    """
    Generate character n-grams for a word using FastText conventions.

    The word is wrapped with boundary symbols, e.g. ``where`` -> ``<where>``,
    then all contiguous character n-grams from ``min_n`` to ``max_n`` are extracted.
    """
    if not word:
        return []

    bounded = f"{WORD_BOUNDARY_LEFT}{word}{WORD_BOUNDARY_RIGHT}"
    ngrams: List[str] = []
    for n in range(min_n, max_n + 1):
        if len(bounded) < n:
            continue
        for start in range(len(bounded) - n + 1):
            ngrams.append(bounded[start : start + n])
    return ngrams


def generate_ngrams_batch(
    words: Sequence[str],
    min_n: int = 3,
    max_n: int = 6,
) -> List[List[str]]:
    """Generate n-grams for a batch of words."""
    return [generate_ngrams(word, min_n=min_n, max_n=max_n) for word in words]


def count_tokens(tokens: Iterable[str]) -> int:
    """Count tokens in an iterable."""
    return sum(1 for _ in tokens)
