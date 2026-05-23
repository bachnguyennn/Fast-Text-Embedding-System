"""Utility helpers for data download, reproducibility, and embedding I/O."""

from __future__ import annotations

import random
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch

from src.tokenizer import iter_tokens_from_file, tokenize


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def download_text8(raw_dir: str | Path = "data/raw") -> Path:
    """Download and extract the text8 corpus if missing."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    text_path = raw_dir / "text8"
    if text_path.exists():
        return text_path

    zip_path = raw_dir / "text8.zip"
    url = "http://mattmahoney.net/dc/text8.zip"
    if not zip_path.exists():
        print(f"Downloading text8 from {url} ...")
        urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(raw_dir)
    return text_path


def load_corpus(path: str | Path, max_tokens: int | None = None) -> List[str]:
    """Load and tokenize a corpus file."""
    tokens = list(iter_tokens_from_file(str(path)))
    if max_tokens is not None:
        tokens = tokens[:max_tokens]
    return tokens


def preprocess_and_save_corpus(
    raw_path: str | Path,
    processed_dir: str | Path = "data/processed",
    max_tokens: int | None = None,
) -> Tuple[List[str], Path]:
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    tokens = load_corpus(raw_path, max_tokens=max_tokens)
    output_path = processed_dir / "corpus_tokens.txt"
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(" ".join(tokens))
    return tokens, output_path


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def most_similar(
    word: str,
    word2idx: dict,
    idx2word: dict,
    vectors: np.ndarray,
    top_k: int = 10,
) -> List[Tuple[str, float]]:
    if word not in word2idx:
        return []
    query = vectors[word2idx[word]]
    scores = vectors @ query / (
        np.linalg.norm(vectors, axis=1) * np.linalg.norm(query) + 1e-10
    )
    ranked = np.argsort(-scores)
    results: List[Tuple[str, float]] = []
    for idx in ranked:
        token = idx2word[int(idx)]
        if token.startswith("<"):
            continue
        if token == word:
            continue
        results.append((token, float(scores[idx])))
        if len(results) >= top_k:
            break
    return results


def save_embeddings_vec(vocab, vectors: torch.Tensor | np.ndarray, path: str | Path) -> None:
    """Save embeddings in word2vec/FastText .vec text format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(vectors, torch.Tensor):
        vectors = vectors.cpu().numpy()

    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"{vectors.shape[0]} {vectors.shape[1]}\n")
        for idx in range(vectors.shape[0]):
            word = vocab.idx2word.get(idx)
            if word is None or word.startswith("<"):
                continue
            vector_str = " ".join(f"{value:.6f}" for value in vectors[idx])
            handle.write(f"{word} {vector_str}\n")


def load_embeddings_vec(path: str | Path) -> Tuple[dict, np.ndarray]:
    path = Path(path)
    word2idx: dict = {}
    vectors: List[np.ndarray] = []
    with open(path, "r", encoding="utf-8") as handle:
        header = handle.readline().strip().split()
        if len(header) != 2:
            raise ValueError(f"Invalid .vec header in {path}")
        for line in handle:
            parts = line.rstrip().split(" ")
            word = parts[0]
            vec = np.asarray(parts[1:], dtype=np.float32)
            word2idx[word] = len(vectors)
            vectors.append(vec)
    return word2idx, np.vstack(vectors)


def get_vector_for_word(
    word: str,
    word2idx: dict,
    vectors: np.ndarray,
    subword_vectors: dict | None = None,
    ngrams: Iterable[str] | None = None,
) -> np.ndarray | None:
    """Lookup a word vector, optionally composing FastText subword vectors for OOV."""
    if word in word2idx:
        return vectors[word2idx[word]]
    if subword_vectors is None or ngrams is None:
        return None
    ngram_vecs = [subword_vectors[ngram] for ngram in ngrams if ngram in subword_vectors]
    if not ngram_vecs:
        return None
    return np.mean(ngram_vecs, axis=0)
