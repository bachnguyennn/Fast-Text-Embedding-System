"""Skip-Gram dataset with negative sampling for word embedding training."""

from __future__ import annotations

import random
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.vocabulary import UNK_TOKEN, Vocabulary


class SkipGramDataset(Dataset):
    """
    Generate (center, context, negatives) training examples.

    Each positive pair comes from a sliding context window. Negative samples are
    drawn from the unigram distribution raised to the 3/4 power.
    """

    def __init__(
        self,
        token_ids: Sequence[int],
        vocab: Vocabulary,
        window_size: int = 5,
        num_negatives: int = 10,
        max_pairs: int | None = None,
        seed: int = 42,
    ) -> None:
        self.token_ids = list(token_ids)
        self.vocab = vocab
        self.window_size = window_size
        self.num_negatives = num_negatives
        self.rng = random.Random(seed)

        self.pairs: List[Tuple[int, int]] = []
        for center_pos, center_id in enumerate(self.token_ids):
            if center_id in {vocab.pad_idx, vocab.unk_idx}:
                continue
            start = max(0, center_pos - window_size)
            end = min(len(self.token_ids), center_pos + window_size + 1)
            for context_pos in range(start, end):
                if context_pos == center_pos:
                    continue
                context_id = self.token_ids[context_pos]
                if context_id in {vocab.pad_idx, vocab.unk_idx}:
                    continue
                self.pairs.append((center_id, context_id))

        if max_pairs is not None and len(self.pairs) > max_pairs:
            self.rng.shuffle(self.pairs)
            self.pairs = self.pairs[:max_pairs]

        if vocab.negative_sampling_probs is None:
            raise ValueError("Vocabulary must define negative_sampling_probs.")

        self._neg_population = np.arange(vocab.vocab_size)
        self._neg_probs = vocab.negative_sampling_probs

    def __len__(self) -> int:
        return len(self.pairs)

    def _sample_negatives(self, center_id: int, context_id: int) -> List[int]:
        negatives: List[int] = []
        while len(negatives) < self.num_negatives:
            samples = self.rng.choices(self._neg_population, weights=self._neg_probs, k=self.num_negatives)
            for sample in samples:
                if sample not in {center_id, context_id} and sample not in negatives:
                    negatives.append(int(sample))
                if len(negatives) >= self.num_negatives:
                    break
        return negatives

    def __getitem__(self, index: int) -> dict:
        center_id, context_id = self.pairs[index]
        negatives = self._sample_negatives(center_id, context_id)
        center_subwords = self.vocab.get_subword_indices(center_id)
        return {
            "center_id": center_id,
            "context_id": context_id,
            "negatives": negatives,
            "center_subwords": center_subwords,
        }


def collate_skipgram_batch(batch: List[dict]) -> dict:
    """Pad subword indices and stack tensors for DataLoader."""
    max_subwords = max(len(item["center_subwords"]) for item in batch)
    max_subwords = max(max_subwords, 1)

    center_ids = torch.tensor([item["center_id"] for item in batch], dtype=torch.long)
    context_ids = torch.tensor([item["context_id"] for item in batch], dtype=torch.long)
    negatives = torch.tensor([item["negatives"] for item in batch], dtype=torch.long)

    subword_ids = torch.zeros((len(batch), max_subwords), dtype=torch.long)
    subword_mask = torch.zeros((len(batch), max_subwords), dtype=torch.bool)
    for row, item in enumerate(batch):
        ngrams = item["center_subwords"]
        length = len(ngrams)
        if length == 0:
            subword_mask[row, 0] = True
            continue
        subword_ids[row, :length] = torch.tensor(ngrams, dtype=torch.long)
        subword_mask[row, :length] = True

    return {
        "center_ids": center_ids,
        "context_ids": context_ids,
        "negatives": negatives,
        "subword_ids": subword_ids,
        "subword_mask": subword_mask,
    }


def build_skipgram_dataset(
    tokens: Sequence[str],
    vocab: Vocabulary,
    window_size: int = 5,
    num_negatives: int = 10,
    max_pairs: int | None = None,
    seed: int = 42,
) -> SkipGramDataset:
    token_ids = vocab.encode_corpus(tokens)
    return SkipGramDataset(
        token_ids=token_ids,
        vocab=vocab,
        window_size=window_size,
        num_negatives=num_negatives,
        max_pairs=max_pairs,
        seed=seed,
    )
