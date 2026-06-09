from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from spsm.models.losses import masked_mse


@dataclass
class TrainConfig:
    epochs: int = 5
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    prediction_loss_weight: float = 1.0
    reliability_loss_weight: float = 0.25
    reliability_pos_weight: float | None = None


class Trainer:
    def __init__(self, model: nn.Module, train_config: TrainConfig, device: torch.device):
        self.model = model.to(device)
        self.cfg = train_config
        self.device = device
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay
        )

    def fit(self, train_loader: DataLoader) -> list[dict[str, float]]:
        history = []
        self.model.train()
        for epoch in range(self.cfg.epochs):
            total_loss = 0.0
            total_pred = 0.0
            total_rel = 0.0
            n = 0
            pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{self.cfg.epochs}", leave=False)
            for batch in pbar:
                z_current = batch["z_current"].to(self.device)
                action = batch["action"].to(self.device)
                z_future = batch["z_future"].to(self.device)
                expected_unreliable = batch.get("expected_unreliable", batch["is_surprise"]).to(self.device)
                observed_surprise = batch.get("observed_surprise", batch["is_surprise"]).to(self.device)
                valid_mask = 1.0 - observed_surprise

                out = self.model(z_current, action)
                pred_loss = masked_mse(out["z_pred"], z_future, valid_mask)
                pos_weight = None
                if self.cfg.reliability_pos_weight is not None:
                    pos_weight = torch.tensor(self.cfg.reliability_pos_weight, device=self.device)
                rel_loss = F.binary_cross_entropy_with_logits(
                    out["reliability_logit"], expected_unreliable, pos_weight=pos_weight
                )
                loss = (
                    self.cfg.prediction_loss_weight * pred_loss
                    + self.cfg.reliability_loss_weight * rel_loss
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                bs = z_current.shape[0]
                total_loss += float(loss.item()) * bs
                total_pred += float(pred_loss.item()) * bs
                total_rel += float(rel_loss.item()) * bs
                n += bs
                pbar.set_postfix(loss=total_loss / max(n, 1))

            history.append(
                {
                    "epoch": float(epoch + 1),
                    "loss": total_loss / n,
                    "prediction_loss": total_pred / n,
                    "reliability_loss": total_rel / n,
                }
            )
        return history
