#!/usr/bin/env python3
"""End-to-end pipeline for training Skip-Gram and FastText embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.dataset import build_skipgram_dataset
from src.trainer import TrainConfig, train_embeddings
from src.utils import download_text8, preprocess_and_save_corpus, set_seed
from src.vocabulary import build_vocabulary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Skip-Gram / FastText embeddings from scratch.")
    parser.add_argument("--model-type", choices=["skipgram", "fasttext"], default="fasttext")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--embedding-dim", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--num-negatives", type=int, default=10)
    parser.add_argument("--min-freq", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=None, help="Limit corpus size for quick runs.")
    parser.add_argument("--max-pairs", type=int, default=None, help="Cap number of training pairs.")
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--models-dir", type=str, default="models")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    raw_dir = Path(args.data_dir) / "raw"
    processed_dir = Path(args.data_dir) / "processed"
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    text_path = download_text8(raw_dir)
    tokens, _ = preprocess_and_save_corpus(text_path, processed_dir, max_tokens=args.max_tokens)
    print(f"Loaded {len(tokens):,} tokens from {text_path}")

    vocab = build_vocabulary(tokens, min_freq=args.min_freq)
    vocab.save(processed_dir)
    print(
        f"Vocabulary: {vocab.vocab_size:,} words | "
        f"{vocab.subword_vocab_size:,} subwords | "
        f"OOV rate={vocab.oov_rate(tokens):.2%}"
    )

    dataset = build_skipgram_dataset(
        tokens=tokens,
        vocab=vocab,
        window_size=args.window_size,
        num_negatives=args.num_negatives,
        max_pairs=args.max_pairs,
        seed=args.seed,
    )
    print(f"Training pairs: {len(dataset):,}")

    config = TrainConfig(
        model_type=args.model_type,
        embedding_dim=args.embedding_dim,
        window_size=args.window_size,
        num_negatives=args.num_negatives,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_pairs=args.max_pairs,
        seed=args.seed,
        checkpoint_dir=str(models_dir),
    )

    trainer = train_embeddings(vocab, dataset, config)
    summary = {
        "model_type": args.model_type,
        "tokens": len(tokens),
        "vocab_size": vocab.vocab_size,
        "subword_vocab_size": vocab.subword_vocab_size,
        "pairs": len(dataset),
        "oov_rate": vocab.oov_rate(tokens),
        "history": trainer.history,
    }
    with open(models_dir / f"{args.model_type}_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Training complete. Embeddings saved to {models_dir / f'{args.model_type}_final.vec'}")


if __name__ == "__main__":
    main()
