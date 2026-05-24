"""Training loop, optimization, and checkpoint management."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import (
    FastCorpusSkipGramDataset,
    SkipGramDataset,
    StreamingSkipGramDataset,
    collate_skipgram_batch,
    collate_skipgram_batch_fast,
)
from src.device import get_device, supports_autocast
from src.model import FastTextModel, SkipGramModel, build_model
from src.utils import save_embeddings_vec, set_seed


@dataclass
class TrainConfig:
    model_type: str = "fasttext"
    embedding_dim: int = 100
    window_size: int = 5
    num_negatives: int = 10
    batch_size: int = 512
    epochs: int = 5
    target_epochs: int | None = None
    learning_rate: float = 0.003
    weight_decay: float = 0.0
    max_pairs: int | None = None
    num_workers: int = 0
    seed: int = 42
    checkpoint_dir: str = "models"
    log_every: int = 100
    save_epoch_checkpoints: bool = True
    use_amp: bool = True
    fast_dataset: bool = True


@dataclass
class Trainer:
    model: SkipGramModel
    config: TrainConfig
    device: torch.device
    optimizer: torch.optim.Optimizer
    scheduler: ReduceLROnPlateau
    history: List[Dict[str, float]] = field(default_factory=list)
    start_epoch: int = 0

    @classmethod
    def from_vocab(
        cls,
        vocab,
        config: TrainConfig,
        device: torch.device | None = None,
    ) -> "Trainer":
        set_seed(config.seed)
        if device is None:
            device = get_device()
        else:
            print(f"Using device: {device}")
        model = build_model(
            model_type=config.model_type,  # type: ignore[arg-type]
            vocab_size=vocab.vocab_size,
            subword_vocab_size=vocab.subword_vocab_size,
            embedding_dim=config.embedding_dim,
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
        return cls(model=model, config=config, device=device, optimizer=optimizer, scheduler=scheduler)

    def load_checkpoint(self, path: str | Path, vocab) -> None:
        path = Path(path)
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model_state_dict"])
        if "optimizer_state_dict" in payload:
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        if "scheduler_state_dict" in payload:
            self.scheduler.load_state_dict(payload["scheduler_state_dict"])
        if payload.get("history"):
            self.history = list(payload["history"])
        self.start_epoch = int(payload.get("epoch", len(self.history)))
        print(f"Resumed from {path.name} at epoch {self.start_epoch}")

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        vocab,
        use_amp: bool,
    ) -> float:
        if isinstance(dataloader.dataset, FastCorpusSkipGramDataset):
            dataloader.dataset.set_epoch(epoch)

        self.model.train()
        total_loss = 0.0
        num_batches = 0
        total_epochs = self.config.target_epochs or self.config.epochs
        progress = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{total_epochs}",
            leave=False,
        )
        autocast_device = "mps" if self.device.type == "mps" else "cuda" if self.device.type == "cuda" else "cpu"

        for step, batch in enumerate(progress):
            center_ids = batch["center_ids"].to(self.device, non_blocking=False)
            context_ids = batch["context_ids"].to(self.device, non_blocking=False)
            negatives = batch["negatives"].to(self.device, non_blocking=False)
            subword_ids = batch["subword_ids"].to(self.device, non_blocking=False)
            subword_mask = batch["subword_mask"].to(self.device, non_blocking=False)

            self.optimizer.zero_grad(set_to_none=True)
            if use_amp and supports_autocast(self.device):
                with torch.autocast(device_type=autocast_device, dtype=torch.float16):
                    loss = self.model(
                        center_ids=center_ids,
                        context_ids=context_ids,
                        negatives=negatives,
                        subword_ids=subword_ids,
                        subword_mask=subword_mask,
                    )
            else:
                loss = self.model(
                    center_ids=center_ids,
                    context_ids=context_ids,
                    negatives=negatives,
                    subword_ids=subword_ids,
                    subword_mask=subword_mask,
                )
            loss.backward()
            self.optimizer.step()

            loss_value = float(loss.detach().item())
            total_loss += loss_value
            num_batches += 1
            if step % self.config.log_every == 0:
                progress.set_postfix(loss=f"{loss_value:.4f}", device=str(self.device))

        return total_loss / max(num_batches, 1)

    def fit(
        self,
        dataset: SkipGramDataset | StreamingSkipGramDataset | FastCorpusSkipGramDataset,
        vocab,
    ) -> List[Dict[str, float]]:
        collate_fn = collate_skipgram_batch
        if self.config.fast_dataset and isinstance(dataset, FastCorpusSkipGramDataset):
            collate_fn = partial(collate_skipgram_batch_fast, vocab=vocab)

        pin_memory = False
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=not isinstance(dataset, FastCorpusSkipGramDataset),
            num_workers=self.config.num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            persistent_workers=False,
        )

        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        total_epochs = self.config.target_epochs or self.config.epochs
        use_amp = self.config.use_amp and supports_autocast(self.device)
        if use_amp:
            print("AMP enabled (float16 autocast)")

        best_loss = min((row["loss"] for row in self.history), default=float("inf"))

        for epoch in range(self.start_epoch, total_epochs):
            start = time.time()
            avg_loss = self.train_epoch(dataloader, epoch, vocab, use_amp=use_amp)
            self.scheduler.step(avg_loss)
            elapsed = time.time() - start
            record = {
                "epoch": epoch + 1,
                "loss": avg_loss,
                "lr": float(self.optimizer.param_groups[0]["lr"]),
                "seconds": elapsed,
            }
            self.history.append(record)
            print(
                f"Epoch {epoch + 1}/{total_epochs} | "
                f"loss={avg_loss:.4f} | lr={record['lr']:.6f} | "
                f"time={elapsed:.1f}s ({elapsed / 60:.1f} min) | device={self.device}"
            )

            if self.config.save_epoch_checkpoints:
                self.save_checkpoint(
                    checkpoint_dir / f"{self.config.model_type}_epoch_{epoch + 1}.pt",
                    vocab,
                    epoch=epoch + 1,
                )

            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_checkpoint(
                    checkpoint_dir / f"{self.config.model_type}_best.pt",
                    vocab,
                    epoch=epoch + 1,
                )

        final_path = checkpoint_dir / f"{self.config.model_type}_final.vec"
        self.export_embeddings(vocab, final_path)
        self.save_history(checkpoint_dir / f"{self.config.model_type}_history.json")
        return self.history

    def export_embeddings(self, vocab, output_path: str | Path, export_batch_size: int = 512) -> None:
        self.model.eval()
        export_device = torch.device("cpu")
        self.model.to(export_device)
        if self.device.type == "mps":
            torch.mps.empty_cache()
        with torch.no_grad():
            if isinstance(self.model, FastTextModel):
                vectors = self.model.get_word_vectors(vocab, batch_size=export_batch_size)
            else:
                vectors = self.model.get_word_vectors()
        save_embeddings_vec(vocab, vectors, output_path)
        self.model.to(self.device)
        print(f"Exported embeddings -> {output_path}")

    def save_checkpoint(self, path: str | Path, vocab, epoch: int) -> None:
        path = Path(path)
        payload = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config.__dict__,
            "history": self.history,
            "epoch": epoch,
            "vocab_size": vocab.vocab_size,
            "subword_vocab_size": vocab.subword_vocab_size,
        }
        torch.save(payload, path)

    def save_history(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.history, handle, indent=2)


def train_embeddings(
    vocab,
    dataset: SkipGramDataset,
    config: TrainConfig,
    resume_from: str | Path | None = None,
    device: torch.device | None = None,
) -> Trainer:
    trainer = Trainer.from_vocab(vocab, config, device=device)
    if resume_from is not None:
        trainer.load_checkpoint(resume_from, vocab)
    trainer.fit(dataset, vocab)
    return trainer
