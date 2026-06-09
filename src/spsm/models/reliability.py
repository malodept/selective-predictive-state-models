from __future__ import annotations

import torch
from torch import nn


class ReliabilityHead(nn.Module):
    """Predicts pre-observation probability that a transition is unreliable.

    The head receives the current latent, the predicted future latent, and optionally
    the action/context. This matters because reliability should be estimated before
    seeing the future observation, so it must be based on signals available at
    decision time.
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int = 0,
        hidden_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        # Reliability is estimated before observing the future. In physical
        # settings, the reliability signal often depends on action/context
        # magnitude (large egomotion, unusual controls, uncertain metadata,
        # domain flags, etc.). We therefore expose simple action statistics in
        # addition to the raw action vector. This keeps the synthetic task
        # learnable while matching the future real-data design.
        action_feature_dim = 0 if action_dim == 0 else (3 * action_dim + 1)
        input_dim = 2 * latent_dim + action_feature_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        z_current: torch.Tensor,
        z_pred: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = [z_current, z_pred]
        if self.action_dim > 0:
            if action is None:
                raise ValueError("action must be provided when ReliabilityHead.action_dim > 0")
            action_norm = torch.linalg.vector_norm(action, dim=-1, keepdim=True)
            parts.extend([action, action.abs(), action.square(), action_norm])
        return self.net(torch.cat(parts, dim=-1)).squeeze(-1)
