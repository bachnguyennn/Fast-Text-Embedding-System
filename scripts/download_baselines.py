#!/usr/bin/env python3
"""Download official FastText and GloVe models via gensim for benchmarking."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache official embedding models with gensim.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["fasttext-wiki-news-subwords-300", "glove-wiki-gigaword-300"],
        help="Gensim model names to download.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import gensim.downloader as api

    for name in args.models:
        print(f"Downloading / loading {name} ...")
        model = api.load(name)
        print(f"  Loaded: {len(model)} vectors, dim={model.vector_size}")


if __name__ == "__main__":
    main()
