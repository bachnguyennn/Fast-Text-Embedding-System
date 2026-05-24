#!/usr/bin/env python3
"""Quick MPS benchmark: batch size 1024 vs 2048 (50 steps each)."""

from __future__ import annotations

import sys
import time
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import build_skipgram_dataset, collate_skipgram_batch_fast
from src.device import get_device, supports_autocast
from src.trainer import TrainConfig, Trainer
from src.utils import download_text8, load_corpus, set_seed, subsample_corpus
from src.vocabulary import Vocabulary


def benchmark(batch_size: int, steps: int = 80) -> float:
    device = get_device()
    processed = ROOT / "data" / "processed"
    vocab = Vocabulary.load(processed)
    if vocab.subword_cache is None:
        vocab._build_subword_cache()

    tokens = load_corpus(download_text8(ROOT / "data" / "raw"), max_tokens=200_000)
    tokens = subsample_corpus(tokens, word_freq=vocab.word_freq, threshold=1e-5, seed=42)
    dataset = build_skipgram_dataset(
        tokens=tokens,
        vocab=vocab,
        num_negatives=15,
        fast=True,
        samples_per_epoch=batch_size * steps,
    )
    collate_fn = partial(collate_skipgram_batch_fast, vocab=vocab)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=False,
    )

    config = TrainConfig(model_type="fasttext", embedding_dim=300, batch_size=batch_size, use_amp=True)
    trainer = Trainer.from_vocab(vocab, config, device=device)
    trainer.model.train()
    use_amp = supports_autocast(device)
    autocast_device = "mps" if device.type == "mps" else "cpu"

    start = time.perf_counter()
    for step, batch in enumerate(loader):
        if step >= steps:
            break
        center_ids = batch["center_ids"].to(device)
        context_ids = batch["context_ids"].to(device)
        negatives = batch["negatives"].to(device)
        subword_ids = batch["subword_ids"].to(device)
        subword_mask = batch["subword_mask"].to(device)
        trainer.optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.autocast(device_type=autocast_device, dtype=torch.float16):
                loss = trainer.model(center_ids, context_ids, negatives, subword_ids, subword_mask)
        else:
            loss = trainer.model(center_ids, context_ids, negatives, subword_ids, subword_mask)
        loss.backward()
        trainer.optimizer.step()

    elapsed = time.perf_counter() - start
    return steps / elapsed


def main() -> None:
    set_seed(42)
    results = {}
    for bs in (1024, 2048):
        print(f"\n--- batch_size={bs} ---")
        it_s = benchmark(bs)
        results[bs] = it_s
        print(f"{it_s:.2f} batches/sec")

    best = max(results, key=results.get)
    print(f"\nRecommended batch_size: {best} ({results[best]:.2f} it/s)")


if __name__ == "__main__":
    main()
