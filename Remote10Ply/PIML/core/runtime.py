"""Runtime helpers shared by command-line workflows."""

from __future__ import annotations

import random

import numpy as np
import torch


def select_device(name: str) -> torch.device:
    """Resolve an explicit or automatic Torch device safely."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def seed_everything(seed: int) -> None:
    """Seed all random sources used by PIML training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
