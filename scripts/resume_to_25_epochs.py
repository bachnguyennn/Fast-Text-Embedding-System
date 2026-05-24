#!/usr/bin/env python3
"""Resume both models from best checkpoints to 25 epochs (MPS-optimized)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = 25
BATCH_SIZE = 1024  # faster on M1 Pro MPS (see scripts/benchmark_batch_size.py)


def main() -> None:
    common = [
        sys.executable,
        str(ROOT / "train.py"),
        "--target-epochs",
        str(TARGET),
        "--embedding-dim",
        "300",
        "--batch-size",
        str(BATCH_SIZE),
        "--fast-dataset",
        "--subsample",
        "--num-negatives",
        "15",
        "--learning-rate",
        "0.0025",
        "--use-amp",
        "--num-workers",
        "0",
    ]

    for model in ("skipgram", "fasttext"):
        ckpt = ROOT / "models" / f"{model}_best.pt"
        if not ckpt.exists():
            print(f"Missing {ckpt}, skipping {model}")
            continue
        print("=" * 60)
        print(f"Resume {model} -> epoch {TARGET} (from {ckpt.name})")
        print("=" * 60)
        subprocess.run(
            [*common, "--model-type", model, "--resume-from", str(ckpt)],
            check=True,
            cwd=ROOT,
        )

    print("Done. Run: bash scripts/post_train_pipeline.sh")


if __name__ == "__main__":
    main()
