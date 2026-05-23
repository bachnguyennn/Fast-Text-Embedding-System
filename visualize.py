#!/usr/bin/env python3
"""Generate t-SNE visualizations of learned embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE

from src.utils import load_embeddings_vec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot t-SNE visualization for .vec embeddings.")
    parser.add_argument("--vec-path", required=True)
    parser.add_argument("--output", default="reports/figures/tsne.png")
    parser.add_argument("--max-words", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    word2idx, vectors = load_embeddings_vec(args.vec_path)
    words = [word for word in word2idx if not word.startswith("<")]
    if len(words) > args.max_words:
        rng = np.random.default_rng(args.seed)
        words = list(rng.choice(words, size=args.max_words, replace=False))
    idxs = [word2idx[word] for word in words]
    sample = vectors[idxs]

    tsne = TSNE(n_components=2, random_state=args.seed, init="pca", learning_rate="auto")
    projected = tsne.fit_transform(sample)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 10))
    sns.scatterplot(x=projected[:, 0], y=projected[:, 1], s=12, alpha=0.7)
    for i, word in enumerate(words[:100]):
        plt.annotate(word, (projected[i, 0], projected[i, 1]), fontsize=7, alpha=0.8)
    plt.title(f"t-SNE: {Path(args.vec_path).name}")
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"Saved t-SNE plot to {output}")


if __name__ == "__main__":
    main()
