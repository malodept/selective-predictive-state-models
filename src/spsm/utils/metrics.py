from __future__ import annotations

import torch


def mean_squared_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean((pred - target) ** 2).item())
