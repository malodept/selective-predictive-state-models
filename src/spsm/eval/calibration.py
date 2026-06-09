from __future__ import annotations

import torch


def expected_calibration_error(labels: torch.Tensor, probabilities: torch.Tensor, n_bins: int = 10) -> float:
    labels = labels.detach().flatten().float().cpu()
    probabilities = probabilities.detach().flatten().float().cpu().clamp(0, 1)
    ece = 0.0
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        mask = (probabilities >= lo) & (probabilities < hi if i < n_bins - 1 else probabilities <= hi)
        if mask.any():
            conf = probabilities[mask].mean()
            acc = labels[mask].mean()
            ece += float(mask.float().mean().item() * abs(conf.item() - acc.item()))
    return ece
