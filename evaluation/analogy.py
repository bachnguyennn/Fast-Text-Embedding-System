"""Google semantic/syntactic analogy evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from tqdm import tqdm

from src.tokenizer import generate_ngrams
from src.utils import load_embeddings_vec


@dataclass
class AnalogyLookup:
    word2idx: Dict[str, int]
    vectors: np.ndarray
    use_subwords: bool = False
    _words: List[str] = field(default_factory=list, repr=False)
    _norm_vectors: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._words = [""] * len(self.vectors)
        for word, idx in self.word2idx.items():
            if idx < len(self._words):
                self._words[idx] = word
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        self._norm_vectors = (self.vectors / norms).astype(np.float32)

    @classmethod
    def from_vec(cls, path: str | Path, use_subwords: bool = False) -> "AnalogyLookup":
        word2idx, vectors = load_embeddings_vec(path)
        return cls(word2idx=word2idx, vectors=vectors, use_subwords=use_subwords)

    @classmethod
    def from_gensim(cls, model, use_subwords: bool = False) -> "AnalogyLookup":
        words = list(model.key_to_index.keys())
        word2idx = {word: idx for idx, word in enumerate(words)}
        vectors = np.vstack([model.get_vector(word) for word in words]).astype(np.float32)
        return cls(word2idx=word2idx, vectors=vectors, use_subwords=use_subwords)

    def _vector(self, word: str) -> np.ndarray | None:
        word = word.lower()
        if word in self.word2idx:
            return self.vectors[self.word2idx[word]]
        if not self.use_subwords:
            return None
        ngrams = generate_ngrams(word)
        ngram_vecs = [self.vectors[self.word2idx[ngram]] for ngram in ngrams if ngram in self.word2idx]
        if not ngram_vecs:
            return None
        return np.mean(ngram_vecs, axis=0)

    def analogy(self, a: str, b: str, c: str, exclude: Iterable[str] = ()) -> str | None:
        vec_a = self._vector(a)
        vec_b = self._vector(b)
        vec_c = self._vector(c)
        if vec_a is None or vec_b is None or vec_c is None:
            return None

        target = vec_b - vec_a + vec_c
        target_norm = np.linalg.norm(target)
        if target_norm == 0:
            return None
        target = (target / target_norm).astype(np.float32)

        assert self._norm_vectors is not None
        scores = self._norm_vectors @ target

        excluded = {token.lower() for token in exclude}
        best_word = None
        best_score = -1e9
        for idx, word in enumerate(self._words):
            if not word or word.startswith("<"):
                continue
            if word.lower() in excluded:
                continue
            score = float(scores[idx])
            if score > best_score:
                best_score = score
                best_word = word
        return best_word


def load_google_analogy_questions(path: str | Path | None = None) -> List[Tuple[str, str, str, str, str]]:
    """
    Load Google analogy questions.

    Returns tuples of (section, a, b, c, expected_d).
    """
    if path is not None and Path(path).exists():
        raw_path = Path(path)
    else:
        raw_path = Path("data/raw/questions-words.txt")
        if not raw_path.exists():
            url = "https://raw.githubusercontent.com/tmikolov/word2vec/master/questions-words.txt"
            import urllib.request

            raw_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, raw_path)

    questions: List[Tuple[str, str, str, str, str]] = []
    section = "unknown"
    with open(raw_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith(":"):
                if line.startswith(":"):
                    section = line[1:].strip()
                continue
            parts = line.split()
            if len(parts) != 4:
                continue
            a, b, c, d = parts
            questions.append((section, a.lower(), b.lower(), c.lower(), d.lower()))
    return questions


def evaluate_analogies(
    lookup: AnalogyLookup,
    questions: List[Tuple[str, str, str, str, str]] | None = None,
) -> Dict[str, float | int]:
    if questions is None:
        questions = load_google_analogy_questions()

    assert lookup._norm_vectors is not None
    norm_matrix = lookup._norm_vectors
    words = lookup._words

    invalid_mask = np.array(
        [not word or word.startswith("<") for word in words],
        dtype=bool,
    )

    total = 0
    correct = 0
    section_totals: Dict[str, int] = {}
    section_correct: Dict[str, int] = {}

    for section, a, b, c, expected in tqdm(questions, desc="Google analogies"):
        vec_a = lookup._vector(a)
        vec_b = lookup._vector(b)
        vec_c = lookup._vector(c)
        if vec_a is None or vec_b is None or vec_c is None:
            continue

        target = vec_b - vec_a + vec_c
        target_norm = np.linalg.norm(target)
        if target_norm == 0:
            continue
        target = (target / target_norm).astype(np.float32)
        scores = norm_matrix @ target
        scores[invalid_mask] = -np.inf
        for word in (a, b, c):
            idx = lookup.word2idx.get(word)
            if idx is not None:
                scores[idx] = -np.inf

        best_idx = int(np.argmax(scores))
        if not np.isfinite(scores[best_idx]):
            continue
        best_word = words[best_idx]

        total += 1
        section_totals[section] = section_totals.get(section, 0) + 1
        if best_word == expected:
            correct += 1
            section_correct[section] = section_correct.get(section, 0) + 1

    accuracy = correct / max(total, 1)
    section_accuracy = {
        section: section_correct.get(section, 0) / max(section_totals[section], 1)
        for section in section_totals
    }
    return {
        "total_questions": total,
        "correct": correct,
        "accuracy": accuracy,
        "section_accuracy": section_accuracy,
    }
