"""Device selection for Apple Silicon / CUDA / CPU."""

from __future__ import annotations

import torch


def get_device() -> torch.device:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def supports_autocast(device: torch.device) -> bool:
    return device.type in {"mps", "cuda"}
