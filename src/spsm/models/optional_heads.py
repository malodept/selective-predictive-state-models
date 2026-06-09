from __future__ import annotations

import torch
from torch import nn


class LinearOptionalHead(nn.Module):
    """Generic optional head for future task-specific modules."""

    def __init__(self, latent_dim: int, output_dim: int):
        super().__init__()
        self.proj = nn.Linear(latent_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.proj(z)
