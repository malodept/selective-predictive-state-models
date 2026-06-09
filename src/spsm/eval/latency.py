from __future__ import annotations

import time

import torch
from torch import nn


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def profile_forward_latency(
    model: nn.Module,
    z_current: torch.Tensor,
    action: torch.Tensor | None = None,
    warmup: int = 5,
    repeats: int = 30,
) -> dict[str, float]:
    model.eval()
    for _ in range(warmup):
        _ = model(z_current, action)
    if z_current.is_cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        _ = model(z_current, action)
    if z_current.is_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return {
        "latency_ms": 1000.0 * elapsed / repeats,
        "batch_size": int(z_current.shape[0]),
        "samples_per_second": float(z_current.shape[0] * repeats / elapsed),
    }
