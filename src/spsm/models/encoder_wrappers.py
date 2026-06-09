from __future__ import annotations

import torch
from torch import nn


class IdentityEncoder(nn.Module):
    """Pass-through encoder for cached or synthetic features."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class FrozenLinearTeacher(nn.Module):
    """Small deterministic frozen projection for toy experiments."""

    def __init__(self, input_dim: int, output_dim: int, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        weight = torch.randn(output_dim, input_dim, generator=g) / (input_dim**0.5)
        self.proj = nn.Linear(input_dim, output_dim, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(weight)
        for param in self.parameters():
            param.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
