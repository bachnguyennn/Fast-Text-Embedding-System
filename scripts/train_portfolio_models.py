#!/usr/bin/env python3
"""Train Skip-Gram and FastText with portfolio-quality defaults (300d, full text8)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    common = [
        sys.executable,
        str(ROOT / "train.py"),
        "--epochs",
        "10",
        "--embedding-dim",
        "300",
        "--batch-size",
        "1024",
        "--streaming",
        "--subsample",
        "--num-negatives",
        "15",
        "--learning-rate",
        "0.0025",
    ]

    print("=" * 60)
    print("Training Skip-Gram (300d, 10 epochs, full text8)")
    print("=" * 60)
    subprocess.run([*common, "--model-type", "skipgram"], check=True, cwd=ROOT)

    print("=" * 60)
    print("Training FastText (300d, 10 epochs, full text8)")
    print("=" * 60)
    subprocess.run([*common, "--model-type", "fasttext"], check=True, cwd=ROOT)

    print("Both models trained. Run: python evaluate.py && python scripts/generate_report_figures.py")


if __name__ == "__main__":
    main()
