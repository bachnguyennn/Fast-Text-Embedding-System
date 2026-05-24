#!/usr/bin/env python3
"""End-to-end pipeline for training Skip-Gram and FastText embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.dataset import build_skipgram_dataset
from src.device import get_device
from src.trainer import TrainConfig, train_embeddings
from src.utils import download_text8, load_corpus, set_seed, subsample_corpus
from src.vocabulary import Vocabulary, build_vocabulary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Skip-Gram / FastText embeddings from scratch.")
    parser.add_argument("--model-type", choices=["skipgram", "fasttext"], default="fasttext")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs when training from scratch.")
    parser.add_argument("--target-epochs", type=int, default=None, help="Stop after this epoch (for resume).")
    parser.add_argument("--resume-from", type=str, default=None, help="Checkpoint .pt to continue training.")
    parser.add_argument("--embedding-dim", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--num-negatives", type=int, default=15)
    parser.add_argument("--min-freq", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=None, help="Limit corpus (None = full text8).")
    parser.add_argument("--max-pairs", type=int, default=None, help="Only for non-streaming mode.")
    parser.add_argument(
        "--streaming",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Legacy streaming dataset (slow). Default uses fast corpus iterator.",
    )
    parser.add_argument(
        "--fast-dataset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use FastCorpusSkipGramDataset (recommended).",
    )
    parser.add_argument(
        "--subsample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Word2vec subsampling of frequent tokens.",
    )
    parser.add_argument("--subsample-threshold", type=float, default=1e-5)
    parser.add_argument("--samples-per-epoch", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=0.0025)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def _ensure_vocab_cache(vocab: Vocabulary) -> None:
    if vocab.subword_cache is None:
        vocab._build_subword_cache()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    raw_dir = Path(args.data_dir) / "raw"
    processed_dir = Path(args.data_dir) / "processed"
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    if (processed_dir / "vocab.pkl").exists() and args.resume_from:
        vocab = Vocabulary.load(processed_dir)
        _ensure_vocab_cache(vocab)
        text_path = download_text8(raw_dir)
        tokens = load_corpus(text_path, max_tokens=args.max_tokens)
        if args.subsample:
            tokens = subsample_corpus(
                tokens,
                word_freq=vocab.word_freq,
                threshold=args.subsample_threshold,
                seed=args.seed,
            )
        raw_token_count = len(tokens)
        print(f"Resumed vocab from {processed_dir} | train tokens: {len(tokens):,}")
    else:
        text_path = download_text8(raw_dir)
        tokens = load_corpus(text_path, max_tokens=args.max_tokens)
        raw_token_count = len(tokens)
        print(f"Loaded {raw_token_count:,} raw tokens from {text_path}")
        vocab = build_vocabulary(tokens, min_freq=args.min_freq)
        if args.subsample:
            tokens = subsample_corpus(
                tokens,
                word_freq=vocab.word_freq,
                threshold=args.subsample_threshold,
                seed=args.seed,
            )
            print(f"After subsampling: {len(tokens):,} tokens ({len(tokens) / raw_token_count:.1%} of raw)")
        vocab.save(processed_dir)

    print(
        f"Vocabulary: {vocab.vocab_size:,} words | "
        f"{vocab.subword_vocab_size:,} subwords | "
        f"OOV rate={vocab.oov_rate(tokens):.2%}"
    )

    samples_per_epoch = args.samples_per_epoch or len(tokens)
    use_fast = args.fast_dataset and not args.streaming
    dataset = build_skipgram_dataset(
        tokens=tokens,
        vocab=vocab,
        window_size=args.window_size,
        num_negatives=args.num_negatives,
        max_pairs=args.max_pairs,
        seed=args.seed,
        streaming=args.streaming and not use_fast,
        samples_per_epoch=samples_per_epoch,
        fast=use_fast,
    )
    print(
        f"Training samples per epoch: {len(dataset):,} | "
        f"fast={use_fast} | batch_size={args.batch_size} | device={device}"
    )

    target_epochs = args.target_epochs or args.epochs
    config = TrainConfig(
        model_type=args.model_type,
        embedding_dim=args.embedding_dim,
        window_size=args.window_size,
        num_negatives=args.num_negatives,
        batch_size=args.batch_size,
        epochs=args.epochs,
        target_epochs=target_epochs,
        learning_rate=args.learning_rate,
        max_pairs=args.max_pairs,
        seed=args.seed,
        checkpoint_dir=str(models_dir),
        save_epoch_checkpoints=True,
        use_amp=args.use_amp,
        fast_dataset=use_fast,
        num_workers=args.num_workers,
    )

    resume_path = args.resume_from
    if resume_path is None and target_epochs > args.epochs:
        candidate = models_dir / f"{args.model_type}_best.pt"
        if candidate.exists():
            resume_path = str(candidate)

    trainer = train_embeddings(
        vocab,
        dataset,
        config,
        resume_from=resume_path,
        device=device,
    )
    summary = {
        "model_type": args.model_type,
        "embedding_dim": args.embedding_dim,
        "target_epochs": target_epochs,
        "tokens_raw": raw_token_count,
        "tokens_train": len(tokens),
        "vocab_size": vocab.vocab_size,
        "subword_vocab_size": vocab.subword_vocab_size,
        "samples_per_epoch": len(dataset),
        "batch_size": args.batch_size,
        "device": str(device),
        "fast_dataset": use_fast,
        "use_amp": args.use_amp,
        "oov_rate": vocab.oov_rate(tokens),
        "history": trainer.history,
    }
    with open(models_dir / f"{args.model_type}_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Training complete. Embeddings saved to {models_dir / f'{args.model_type}_final.vec'}")


if __name__ == "__main__":
    main()
