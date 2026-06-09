from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_mse(pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    mask = valid_mask.float().view(-1, *([1] * (pred.ndim - 1)))
    denom = mask.sum().clamp_min(1.0)
    return ((pred - target).pow(2) * mask).sum() / denom / pred.shape[-1]


def cosine_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_n = F.normalize(pred, dim=-1)
    target_n = F.normalize(target, dim=-1)
    return 1.0 - (pred_n * target_n).sum(dim=-1)
