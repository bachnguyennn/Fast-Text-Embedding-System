"""Word similarity evaluation on WordSim-353 and SimLex-999."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.tokenizer import generate_ngrams
from src.utils import get_vector_for_word, load_embeddings_vec


@dataclass
class EmbeddingLookup:
    word2idx: Dict[str, int]
    vectors: np.ndarray
    subword_vectors: Dict[str, np.ndarray] | None = None

    @classmethod
    def from_vec(cls, path: str | Path) -> "EmbeddingLookup":
        word2idx, vectors = load_embeddings_vec(path)
        return cls(word2idx=word2idx, vectors=vectors)

    @classmethod
    def from_gensim(cls, model) -> "EmbeddingLookup":
        words = list(model.key_to_index.keys())
        word2idx = {word: idx for idx, word in enumerate(words)}
        vectors = np.vstack([model.get_vector(word) for word in words]).astype(np.float32)
        return cls(word2idx=word2idx, vectors=vectors)

    def vector(self, word: str, use_subwords: bool = False) -> np.ndarray | None:
        word = word.lower()
        if word in self.word2idx:
            return self.vectors[self.word2idx[word]]
        if not use_subwords:
            return None
        ngrams = generate_ngrams(word)
        ngram_vecs = [self.vectors[self.word2idx[ngram]] for ngram in ngrams if ngram in self.word2idx]
        if ngram_vecs:
            return np.mean(ngram_vecs, axis=0)
        if self.subword_vectors:
            return get_vector_for_word(
                word,
                self.word2idx,
                self.vectors,
                subword_vectors=self.subword_vectors,
                ngrams=ngrams,
            )
        return None

    def similarity(self, word_a: str, word_b: str, use_subwords: bool = False) -> float | None:
        vec_a = self.vector(word_a, use_subwords=use_subwords)
        vec_b = self.vector(word_b, use_subwords=use_subwords)
        if vec_a is None or vec_b is None:
            return None
        denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
        if denom == 0:
            return None
        return float(np.dot(vec_a, vec_b) / denom)


def load_wordsim353(path: str | Path | None = None) -> pd.DataFrame:
    """Load WordSim-353 dataset from a public mirror if local file is absent."""
    if path is not None and Path(path).exists():
        df = pd.read_csv(path)
    else:
        url = (
            "https://raw.githubusercontent.com/AdrienGuille/DistributionalSemantics/"
            "master/evaluation_data/wordsim353.tsv"
        )
        df = pd.read_csv(url, sep="\t", header=None, names=["word1", "word2", "score"])

    df.columns = [col.strip().lower() for col in df.columns]
    if "word1" not in df.columns:
        df = df.rename(columns={df.columns[0]: "word1", df.columns[1]: "word2", df.columns[2]: "score"})
    return df


def load_simlex999(path: str | Path | None = None) -> pd.DataFrame:
    if path is not None and Path(path).exists():
        df = pd.read_csv(path)
    else:
        url = "https://www.dropbox.com/s/0jpa1x8vpmk3ych/EN-SIM999.txt?dl=1"
        df = pd.read_csv(url, sep="\t")
        if "SimLex999" in df.columns:
            df = df.rename(columns={"SimLex999": "score"})

    df.columns = [col.strip().lower() for col in df.columns]
    if "word1" not in df.columns:
        df = df.rename(columns={df.columns[0]: "word1", df.columns[1]: "word2", df.columns[2]: "score"})
    return df


def evaluate_word_pairs(
    lookup: EmbeddingLookup,
    pairs: pd.DataFrame,
    use_subwords: bool = False,
) -> Tuple[float, int, float]:
    """
    Compute Spearman correlation between human scores and cosine similarities.

    Returns:
        spearman_r, num_evaluated_pairs, coverage
    """
    human_scores: List[float] = []
    model_scores: List[float] = []
    total = len(pairs)

    for _, row in pairs.iterrows():
        sim = lookup.similarity(str(row["word1"]), str(row["word2"]), use_subwords=use_subwords)
        if sim is None:
            continue
        human_scores.append(float(row["score"]))
        model_scores.append(sim)

    if len(model_scores) < 2:
        return float("nan"), len(model_scores), len(model_scores) / max(total, 1)

    rho, _ = spearmanr(human_scores, model_scores)
    coverage = len(model_scores) / max(total, 1)
    return float(rho), len(model_scores), coverage


def run_wordsim_evaluation(
    lookup: EmbeddingLookup,
    use_subwords: bool = False,
) -> Dict[str, float | int]:
    wordsim = load_wordsim353()
    simlex = load_simlex999()

    ws_rho, ws_n, ws_cov = evaluate_word_pairs(lookup, wordsim, use_subwords=use_subwords)
    sl_rho, sl_n, sl_cov = evaluate_word_pairs(lookup, simlex, use_subwords=use_subwords)

    return {
        "wordsim353_spearman": ws_rho,
        "wordsim353_pairs": ws_n,
        "wordsim353_coverage": ws_cov,
        "simlex999_spearman": sl_rho,
        "simlex999_pairs": sl_n,
        "simlex999_coverage": sl_cov,
    }
