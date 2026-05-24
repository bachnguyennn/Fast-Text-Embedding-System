#!/usr/bin/env python3
"""Generate all portfolio figures for README and reports."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "reports" / "figures"


def plot_training_curves() -> None:
    histories = []
    for name, label in [("skipgram", "Skip-Gram (ours)"), ("fasttext", "FastText (ours)")]:
        path = ROOT / "models" / f"{name}_history.json"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as handle:
            history = json.load(handle)
        for row in history:
            row["model"] = label
        histories.extend(history)

    if not histories:
        print("No training history found; skip loss curves.")
        return

    import pandas as pd

    df = pd.DataFrame(histories)
    plt.figure(figsize=(9, 5))
    sns.lineplot(data=df, x="epoch", y="loss", hue="model", marker="o")
    plt.title("Training Loss (Skip-Gram vs FastText)")
    plt.xlabel("Epoch")
    plt.ylabel("Negative Sampling Loss")
    plt.tight_layout()
    out = FIGURES / "training_loss_curves.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


def plot_tsne(vec_name: str, title: str, output: str, max_words: int = 400) -> None:
    from src.utils import load_embeddings_vec

    path = ROOT / "models" / f"{vec_name}_final.vec"
    if not path.exists():
        print(f"Missing {path}; skip t-SNE.")
        return

    word2idx, vectors = load_embeddings_vec(path)
    words = [w for w in word2idx if not w.startswith("<")]
    if len(words) > max_words:
        rng = np.random.default_rng(42)
        words = list(rng.choice(words, size=max_words, replace=False))
    sample = vectors[[word2idx[w] for w in words]]

    proj = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto").fit_transform(sample)
    plt.figure(figsize=(11, 9))
    plt.scatter(proj[:, 0], proj[:, 1], s=14, alpha=0.65, c="#2E86AB")
    for i, word in enumerate(words[:70]):
        plt.annotate(word, (proj[i, 0], proj[i, 1]), fontsize=7, alpha=0.85)
    plt.title(title)
    plt.tight_layout()
    out_path = FIGURES / output
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def plot_nearest_neighbors() -> None:
    from src.utils import load_embeddings_vec, most_similar

    queries = ["king", "queen", "man", "woman", "computer", "beautiful", "science", "running"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, model_name in zip(axes, ["skipgram", "fasttext"]):
        path = ROOT / "models" / f"{model_name}_final.vec"
        if not path.exists():
            ax.set_visible(False)
            continue
        word2idx, vectors = load_embeddings_vec(path)
        idx2word = {i: w for w, i in word2idx.items()}
        lines = []
        for q in queries:
            neighbors = most_similar(q, word2idx, idx2word, vectors, top_k=5)
            if neighbors:
                nn = ", ".join(f"{w}({s:.2f})" for w, s in neighbors)
            else:
                nn = "OOV"
            lines.append(f"{q:12} → {nn}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        label = "Skip-Gram (ours)" if model_name == "skipgram" else "FastText (ours)"
        ax.set_title(label, fontweight="bold", fontsize=12, pad=12)
        ax.text(
            0.04,
            0.96,
            "\n".join(lines),
            va="top",
            ha="left",
            fontsize=9.5,
            family="monospace",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8f9fa", edgecolor="#dee2e6"),
        )

    fig.suptitle("Qualitative Nearest-Neighbor Comparison (cosine similarity)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = FIGURES / "nearest_neighbors.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    import sys

    sys.path.insert(0, str(ROOT))
    FIGURES.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    plot_training_curves()
    plot_tsne("skipgram", "t-SNE: Skip-Gram Embeddings (300d)", "tsne_skipgram.png")
    plot_tsne("fasttext", "t-SNE: FastText Embeddings (300d)", "tsne_fasttext.png")
    plot_nearest_neighbors()


if __name__ == "__main__":
    main()
