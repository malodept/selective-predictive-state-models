import torch

from spsm.eval.retrieval import retrieval_at_k
from spsm.eval.surprise import auprc_score, auroc_score, brier_score


def test_retrieval_identity():
    z = torch.eye(5)
    metrics = retrieval_at_k(z, z, [1, 3])
    assert metrics["R@1"] == 1.0
    assert metrics["R@3"] == 1.0


def test_surprise_metrics_perfect_ordering():
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.float)
    scores = torch.tensor([0.1, 0.2, 0.8, 0.9])
    assert auroc_score(labels, scores) == 1.0
    assert auprc_score(labels, scores) > 0.99
    assert brier_score(labels, scores) < 0.05
