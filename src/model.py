"""Skip-Gram and FastText models with negative sampling."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGramModel(nn.Module):
    """Plain Skip-Gram without subword information."""

    def __init__(self, vocab_size: int, embedding_dim: int = 100) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.input_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.uniform_(self.input_embeddings.weight, -0.5 / self.embedding_dim, 0.5 / self.embedding_dim)
        nn.init.zeros_(self.output_embeddings.weight)

    def get_input_vector(
        self,
        center_ids: torch.Tensor,
        subword_ids: torch.Tensor | None = None,
        subword_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del subword_ids, subword_mask
        return self.input_embeddings(center_ids)

    def get_output_vector(self, word_ids: torch.Tensor) -> torch.Tensor:
        return self.output_embeddings(word_ids)

    def forward(
        self,
        center_ids: torch.Tensor,
        context_ids: torch.Tensor,
        negatives: torch.Tensor,
        subword_ids: torch.Tensor | None = None,
        subword_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        center_vec = self.get_input_vector(center_ids, subword_ids, subword_mask)
        context_vec = self.get_output_vector(context_ids)
        negative_vec = self.get_output_vector(negatives)

        pos_score = torch.sum(center_vec * context_vec, dim=-1)
        pos_loss = F.logsigmoid(pos_score)

        neg_score = torch.bmm(negative_vec, center_vec.unsqueeze(-1)).squeeze(-1)
        neg_loss = F.logsigmoid(-neg_score).sum(dim=-1)

        loss = -(pos_loss + neg_loss).mean()
        return loss

    def get_word_vectors(self) -> torch.Tensor:
        """Return input-side word vectors for evaluation."""
        return self.input_embeddings.weight.detach()


class FastTextModel(SkipGramModel):
    """
    FastText Skip-Gram model.

    Input representation = word embedding + sum(subword embeddings).
    Output side remains standard word embeddings for negative sampling.
    """

    def __init__(
        self,
        vocab_size: int,
        subword_vocab_size: int,
        embedding_dim: int = 100,
    ) -> None:
        super().__init__(vocab_size=vocab_size, embedding_dim=embedding_dim)
        self.subword_vocab_size = subword_vocab_size
        self.subword_embeddings = nn.Embedding(subword_vocab_size, embedding_dim, padding_idx=0)
        nn.init.uniform_(
            self.subword_embeddings.weight,
            -0.5 / self.embedding_dim,
            0.5 / self.embedding_dim,
        )
        with torch.no_grad():
            self.subword_embeddings.weight[0].zero_()

    def get_input_vector(
        self,
        center_ids: torch.Tensor,
        subword_ids: torch.Tensor | None = None,
        subword_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        word_vec = self.input_embeddings(center_ids)
        if subword_ids is None or subword_mask is None:
            return word_vec

        subword_vec = self.subword_embeddings(subword_ids)
        subword_vec = subword_vec * subword_mask.unsqueeze(-1).float()
        subword_sum = subword_vec.sum(dim=1)
        counts = subword_mask.sum(dim=1, keepdim=True).clamp(min=1).float()
        return word_vec + subword_sum / counts

    def get_word_vectors(self, vocab) -> torch.Tensor:
        """
        Compose full FastText vectors for all vocabulary words.

        For in-vocabulary words this matches training-time input vectors.
        """
        device = self.input_embeddings.weight.device
        word_ids = torch.arange(self.vocab_size, device=device)
        subword_lists = [vocab.get_subword_indices(int(idx)) for idx in word_ids.tolist()]
        max_len = max((len(items) for items in subword_lists), default=1)
        max_len = max(max_len, 1)

        subword_ids = torch.zeros((self.vocab_size, max_len), dtype=torch.long, device=device)
        subword_mask = torch.zeros((self.vocab_size, max_len), dtype=torch.bool, device=device)
        for row, ngrams in enumerate(subword_lists):
            if not ngrams:
                subword_mask[row, 0] = True
                continue
            length = len(ngrams)
            subword_ids[row, :length] = torch.tensor(ngrams, dtype=torch.long, device=device)
            subword_mask[row, :length] = True

        return self.get_input_vector(word_ids, subword_ids, subword_mask).detach()


def build_model(
    model_type: Literal["skipgram", "fasttext"],
    vocab_size: int,
    subword_vocab_size: int = 0,
    embedding_dim: int = 100,
) -> SkipGramModel:
    if model_type == "skipgram":
        return SkipGramModel(vocab_size=vocab_size, embedding_dim=embedding_dim)
    if model_type == "fasttext":
        if subword_vocab_size <= 0:
            raise ValueError("FastText requires a positive subword_vocab_size.")
        return FastTextModel(
            vocab_size=vocab_size,
            subword_vocab_size=subword_vocab_size,
            embedding_dim=embedding_dim,
        )
    raise ValueError(f"Unsupported model_type: {model_type}")
