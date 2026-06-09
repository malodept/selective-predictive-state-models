from __future__ import annotations

import torch
from torch import nn

from .reliability import ReliabilityHead


class LatentPredictor(nn.Module):
    """MLP predictor for one-step future latent prediction."""

    def __init__(
        self,
        latent_dim: int,
        action_dim: int = 0,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        in_dim = latent_dim + action_dim
        layers: list[nn.Module] = []
        current = in_dim
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(current, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
            current = hidden_dim
        layers.append(nn.Linear(current, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z_current: torch.Tensor, action: torch.Tensor | None = None) -> torch.Tensor:
        if action is not None:
            x = torch.cat([z_current, action], dim=-1)
        else:
            x = z_current
        return self.net(x)


class PredictiveStateModel(nn.Module):
    """Minimal SPSM model: future predictor + reliability estimator."""

    def __init__(
        self,
        latent_dim: int,
        action_dim: int = 0,
        hidden_dim: int = 256,
        predictor_layers: int = 2,
        reliability_hidden_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.predictor = LatentPredictor(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=predictor_layers,
            dropout=dropout,
        )
        self.reliability = ReliabilityHead(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=reliability_hidden_dim,
            dropout=dropout,
        )

    def forward(self, z_current: torch.Tensor, action: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        z_pred = self.predictor(z_current, action)
        reliability_logit = self.reliability(z_current, z_pred, action)
        return {"z_pred": z_pred, "reliability_logit": reliability_logit}
