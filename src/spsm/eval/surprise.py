from __future__ import annotations

import torch


def _prepare(labels: torch.Tensor, scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    labels = labels.detach().flatten().float().cpu()
    scores = scores.detach().flatten().float().cpu()
    if labels.numel() != scores.numel():
        raise ValueError("labels and scores must have the same number of elements")
    return labels, scores


def auroc_score(labels: torch.Tensor, scores: torch.Tensor) -> float:
    labels, scores = _prepare(labels, scores)
    positives = labels == 1
    negatives = labels == 0
    n_pos = int(positives.sum().item())
    n_neg = int(negatives.sum().item())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float)
    rank_sum_pos = ranks[positives].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc.item())


def auprc_score(labels: torch.Tensor, scores: torch.Tensor) -> float:
    labels, scores = _prepare(labels, scores)
    order = torch.argsort(scores, descending=True)
    y = labels[order]
    tp = torch.cumsum(y, dim=0)
    fp = torch.cumsum(1 - y, dim=0)
    precision = tp / (tp + fp).clamp_min(1)
    recall = tp / tp[-1].clamp_min(1)
    # Step-wise average precision.
    recall_prev = torch.cat([torch.zeros(1), recall[:-1]])
    ap = ((recall - recall_prev) * precision).sum()
    return float(ap.item())


def brier_score(labels: torch.Tensor, probabilities: torch.Tensor) -> float:
    labels, probabilities = _prepare(labels, probabilities)
    probabilities = probabilities.clamp(0, 1)
    return float(((probabilities - labels) ** 2).mean().item())
