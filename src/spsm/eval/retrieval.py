from __future__ import annotations

import torch
import torch.nn.functional as F


def retrieval_at_k(pred: torch.Tensor, target: torch.Tensor, ks: list[int] | tuple[int, ...]) -> dict[str, float]:
    """Compute retrieval recall@k using cosine similarity.

    Assumes sample i should retrieve target i.
    """

    if pred.ndim != 2 or target.ndim != 2:
        raise ValueError("pred and target must be 2D tensors")
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {target.shape}")

    pred_n = F.normalize(pred, dim=-1)
    target_n = F.normalize(target, dim=-1)
    sim = pred_n @ target_n.T
    ranks = sim.argsort(dim=1, descending=True)
    correct = torch.arange(pred.shape[0], device=pred.device)[:, None]
    metrics = {}
    for k in ks:
        k_eff = min(int(k), pred.shape[0])
        hit = (ranks[:, :k_eff] == correct).any(dim=1).float().mean()
        metrics[f"R@{k}"] = float(hit.item())
    return metrics
