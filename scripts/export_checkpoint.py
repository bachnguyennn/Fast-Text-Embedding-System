#!/usr/bin/env python3
"""Export .vec file from a saved checkpoint (no retraining)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model import FastTextModel, SkipGramModel, build_model
from src.trainer import TrainConfig, Trainer
from src.utils import set_seed
from src.vocabulary import Vocabulary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export embeddings from checkpoint.")
    parser.add_argument("--checkpoint", default="models/fasttext_best.pt")
    parser.add_argument("--output", default=None, help="Default: models/{model_type}_final.vec")
    parser.add_argument("--vocab-dir", default="data/processed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(42)
    ckpt_path = ROOT / args.checkpoint
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = TrainConfig(**payload["config"])
    vocab = Vocabulary.load(ROOT / args.vocab_dir)

    trainer = Trainer.from_vocab(vocab, config, device=torch.device("cpu"))
    trainer.model.load_state_dict(payload["model_state_dict"])
    if payload.get("history"):
        trainer.history = payload["history"]

    out = args.output or str(ROOT / "models" / f"{config.model_type}_final.vec")
    trainer.export_embeddings(vocab, out)
    history_path = ROOT / "models" / f"{config.model_type}_history.json"
    trainer.save_history(history_path)
    print(f"Exported {out}")
    print(f"History -> {history_path} ({len(trainer.history)} epochs)")


if __name__ == "__main__":
    main()
