"""Skip-Gram dataset with negative sampling for word embedding training."""

from __future__ import annotations

import random
from typing import List, Sequence, Tuple, Union

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


def _sample_negatives_batch(
    center_ids: np.ndarray,
    context_ids: np.ndarray,
    vocab: Vocabulary,
    num_negatives: int,
    rng: np.random.Generator,
) -> np.ndarray:
    batch_size = center_ids.shape[0]
    negatives = rng.choice(
        vocab.vocab_size,
        size=(batch_size, num_negatives),
        replace=True,
        p=vocab.negative_sampling_probs,
    ).astype(np.int64)
    for col in range(num_negatives):
        bad = (negatives[:, col] == center_ids) | (negatives[:, col] == context_ids)
        while bad.any():
            negatives[bad, col] = rng.choice(
                vocab.vocab_size,
                size=int(bad.sum()),
                replace=True,
                p=vocab.negative_sampling_probs,
            )
            bad = (negatives[:, col] == center_ids) | (negatives[:, col] == context_ids)
    return negatives


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


def collate_skipgram_batch_fast(batch: List[dict], vocab: Vocabulary) -> dict:
    """Vectorized collation using precomputed subword cache."""
    center_ids_np = np.asarray([item["center_id"] for item in batch], dtype=np.int64)
    context_ids_np = np.asarray([item["context_id"] for item in batch], dtype=np.int64)
    rng = np.random.default_rng()
    negatives_np = _sample_negatives_batch(
        center_ids_np, context_ids_np, vocab, batch[0]["num_negatives"], rng
    )

    center_ids = torch.from_numpy(center_ids_np)
    context_ids = torch.from_numpy(context_ids_np)
    negatives = torch.from_numpy(negatives_np)

    if vocab.subword_cache is not None and vocab.subword_lengths is not None:
        sw = vocab.subword_cache[center_ids_np]
        lengths = vocab.subword_lengths[center_ids_np]
        max_len = int(lengths.max()) if len(lengths) else 1
        max_len = max(max_len, 1)
        subword_ids = torch.from_numpy(sw[:, :max_len].copy())
        subword_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
        for row, length in enumerate(lengths):
            if length > 0:
                subword_mask[row, :length] = True
            else:
                subword_mask[row, 0] = True
    else:
        return collate_skipgram_batch(batch)

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
    streaming: bool = False,
    samples_per_epoch: int | None = None,
    fast: bool = True,
) -> Union[SkipGramDataset, "StreamingSkipGramDataset", "FastCorpusSkipGramDataset"]:
    token_ids = vocab.encode_corpus(tokens)
    if fast:
        return FastCorpusSkipGramDataset(
            token_ids=token_ids,
            vocab=vocab,
            window_size=window_size,
            num_negatives=num_negatives,
            samples_per_epoch=samples_per_epoch,
            seed=seed,
        )
    if streaming:
        return StreamingSkipGramDataset(
            token_ids=token_ids,
            vocab=vocab,
            window_size=window_size,
            num_negatives=num_negatives,
            samples_per_epoch=samples_per_epoch,
            seed=seed,
        )
    return SkipGramDataset(
        token_ids=token_ids,
        vocab=vocab,
        window_size=window_size,
        num_negatives=num_negatives,
        max_pairs=max_pairs,
        seed=seed,
    )


class StreamingSkipGramDataset(Dataset):
    """
    Memory-efficient skip-gram dataset for large corpora.

    Samples (center, context) pairs on the fly instead of materializing all pairs.
    """

    def __init__(
        self,
        token_ids: Sequence[int],
        vocab: Vocabulary,
        window_size: int = 5,
        num_negatives: int = 10,
        samples_per_epoch: int | None = None,
        seed: int = 42,
    ) -> None:
        self.token_ids = np.asarray(token_ids, dtype=np.int32)
        self.vocab = vocab
        self.window_size = window_size
        self.num_negatives = num_negatives
        self.samples_per_epoch = int(samples_per_epoch or len(self.token_ids))
        self.rng = np.random.default_rng(seed)

        if len(self.token_ids) == 0:
            raise ValueError("token_ids is empty.")
        if vocab.negative_sampling_probs is None:
            raise ValueError("Vocabulary must define negative_sampling_probs.")

        self._neg_population = np.arange(vocab.vocab_size)
        self._neg_probs = vocab.negative_sampling_probs

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _sample_negatives(self, center_id: int, context_id: int) -> List[int]:
        negatives: List[int] = []
        while len(negatives) < self.num_negatives:
            samples = self.rng.choice(
                self._neg_population,
                size=self.num_negatives,
                replace=True,
                p=self._neg_probs,
            )
            for sample in samples:
                sample = int(sample)
                if sample not in {center_id, context_id} and sample not in negatives:
                    negatives.append(sample)
                if len(negatives) >= self.num_negatives:
                    break
        return negatives

    def __getitem__(self, index: int) -> dict:
        del index
        for _ in range(20):
            pos = int(self.rng.integers(0, len(self.token_ids)))
            center_id = int(self.token_ids[pos])
            if center_id in {self.vocab.pad_idx, self.vocab.unk_idx}:
                continue

            offset = int(self.rng.integers(1, self.window_size + 1))
            if self.rng.random() < 0.5:
                offset = -offset
            context_pos = pos + offset
            if context_pos < 0 or context_pos >= len(self.token_ids):
                continue
            context_id = int(self.token_ids[context_pos])
            if context_id in {self.vocab.pad_idx, self.vocab.unk_idx}:
                continue

            negatives = self._sample_negatives(center_id, context_id)
            center_subwords = self.vocab.get_subword_indices(center_id)
            return {
                "center_id": center_id,
                "context_id": context_id,
                "negatives": negatives,
                "center_subwords": center_subwords,
            }

        return self.__getitem__(0)


class FastCorpusSkipGramDataset(Dataset):
    """
    One sample per corpus position per epoch (word2vec-style), with O(1) __getitem__.

    Avoids Python retry loops in StreamingSkipGramDataset; negatives sampled in collate.
    """

    def __init__(
        self,
        token_ids: Sequence[int],
        vocab: Vocabulary,
        window_size: int = 5,
        num_negatives: int = 10,
        samples_per_epoch: int | None = None,
        seed: int = 42,
    ) -> None:
        self.token_ids = np.asarray(token_ids, dtype=np.int32)
        self.vocab = vocab
        self.window_size = window_size
        self.num_negatives = num_negatives
        self.samples_per_epoch = int(samples_per_epoch or len(self.token_ids))
        self.rng = np.random.default_rng(seed)
        self._epoch_seed = seed
        self._positions = np.arange(self.samples_per_epoch, dtype=np.int64)
        self._offsets = self._draw_offsets(self.samples_per_epoch)
        self._reshuffle_epoch()

        if len(self.token_ids) == 0:
            raise ValueError("token_ids is empty.")
        self._pad = vocab.pad_idx
        self._unk = vocab.unk_idx

    def _draw_offsets(self, n: int) -> np.ndarray:
        offsets = self.rng.integers(1, self.window_size + 1, size=n, dtype=np.int32)
        signs = self.rng.integers(0, 2, size=n, dtype=np.int32) * 2 - 1
        return offsets * signs

    def _reshuffle_epoch(self) -> None:
        corpus_len = len(self.token_ids)
        self._positions = self.rng.integers(0, corpus_len, size=self.samples_per_epoch, dtype=np.int64)
        self._offsets = self._draw_offsets(self.samples_per_epoch)

    def set_epoch(self, epoch: int) -> None:
        self.rng = np.random.default_rng(self._epoch_seed + epoch)
        self._reshuffle_epoch()

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int) -> dict:
        pos = int(self._positions[index])
        offset = int(self._offsets[index])
        center_id = int(self.token_ids[pos])
        context_pos = pos + offset
        if context_pos < 0 or context_pos >= len(self.token_ids):
            context_pos = max(0, min(len(self.token_ids) - 1, context_pos))
        context_id = int(self.token_ids[context_pos])

        if center_id in {self._pad, self._unk}:
            center_id = self._unk
        if context_id in {self._pad, self._unk}:
            context_id = self._unk

        return {
            "center_id": center_id,
            "context_id": context_id,
            "num_negatives": self.num_negatives,
        }
