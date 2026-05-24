"""Training loop, optimization, and checkpoint management."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import SkipGramDataset, StreamingSkipGramDataset, collate_skipgram_batch
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
    learning_rate: float = 0.003
    weight_decay: float = 0.0
    max_pairs: int | None = None
    num_workers: int = 0
    seed: int = 42
    checkpoint_dir: str = "models"
    log_every: int = 100
    save_epoch_checkpoints: bool = False


@dataclass
class Trainer:
    model: SkipGramModel
    config: TrainConfig
    device: torch.device
    optimizer: torch.optim.Optimizer
    scheduler: ReduceLROnPlateau
    history: List[Dict[str, float]] = field(default_factory=list)

    @classmethod
    def from_vocab(
        cls,
        vocab,
        config: TrainConfig,
        device: torch.device | None = None,
    ) -> "Trainer":
        set_seed(config.seed)
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
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

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{self.config.epochs}", leave=False)

        for step, batch in enumerate(progress):
            batch = {key: value.to(self.device) for key, value in batch.items()}
            self.optimizer.zero_grad(set_to_none=True)
            loss = self.model(
                center_ids=batch["center_ids"],
                context_ids=batch["context_ids"],
                negatives=batch["negatives"],
                subword_ids=batch["subword_ids"],
                subword_mask=batch["subword_mask"],
            )
            loss.backward()
            self.optimizer.step()

            loss_value = float(loss.item())
            total_loss += loss_value
            num_batches += 1
            if step % self.config.log_every == 0:
                progress.set_postfix(loss=f"{loss_value:.4f}")

        return total_loss / max(num_batches, 1)

    def fit(self, dataset: SkipGramDataset | StreamingSkipGramDataset, vocab) -> List[Dict[str, float]]:
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=collate_skipgram_batch,
            pin_memory=self.device.type == "cuda",
        )

        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_loss = float("inf")

        for epoch in range(self.config.epochs):
            start = time.time()
            avg_loss = self.train_epoch(dataloader, epoch)
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
                f"Epoch {epoch + 1}/{self.config.epochs} | "
                f"loss={avg_loss:.4f} | lr={record['lr']:.6f} | time={elapsed:.1f}s"
            )

            if self.config.save_epoch_checkpoints:
                checkpoint_path = checkpoint_dir / f"{self.config.model_type}_epoch_{epoch + 1}.pt"
                self.save_checkpoint(checkpoint_path, vocab)

            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_checkpoint(checkpoint_dir / f"{self.config.model_type}_best.pt", vocab)

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

    def save_checkpoint(self, path: str | Path, vocab) -> None:
        path = Path(path)
        payload = {
            "model_state_dict": self.model.state_dict(),
            "config": self.config.__dict__,
            "history": self.history,
            "vocab_size": vocab.vocab_size,
            "subword_vocab_size": vocab.subword_vocab_size,
        }
        torch.save(payload, path)

    def save_history(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.history, handle, indent=2)


def train_embeddings(vocab, dataset: SkipGramDataset, config: TrainConfig) -> Trainer:
    trainer = Trainer.from_vocab(vocab, config)
    trainer.fit(dataset, vocab)
    return trainer
