#!/usr/bin/env python3
"""Run evaluation benchmarks and comparison report."""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.comparison import print_results_table, run_full_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained embeddings.")
    parser.add_argument("--skipgram-vec", default="models/skipgram_final.vec")
    parser.add_argument("--fasttext-vec", default="models/fasttext_final.vec")
    parser.add_argument("--official-fasttext", default="models/cc.en.300.vec")
    parser.add_argument("--glove-vec", default="models/glove.6B.300d.txt")
    parser.add_argument("--output-dir", default="reports")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_full_comparison(
        skipgram_vec=args.skipgram_vec,
        fasttext_vec=args.fasttext_vec,
        official_fasttext=args.official_fasttext,
        glove_vec=args.glove_vec,
        output_dir=args.output_dir,
    )
    print_results_table(results)
    print(f"\nSaved results to {Path(args.output_dir) / 'comparison_results.csv'}")


if __name__ == "__main__":
    main()
