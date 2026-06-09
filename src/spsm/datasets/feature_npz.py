from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class FeatureNPZPredictiveStateDataset(Dataset):
    """Predictive-state dataset backed by cached visual features.

    Expected arrays in the .npz file:
      - z_current: [N, D]
      - action: [N, A]
      - z_future: [N, D]
      - clean_future: [N, D] optional; defaults to z_future
      - expected_unreliable: [N] optional; defaults to zeros
      - observed_surprise: [N] optional; defaults to zeros

    This class lets us replace synthetic latents with frozen visual features while
    keeping the exact same predictor/reliability/selector pipeline.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Feature file not found: {self.path}")
        data = np.load(self.path)
        self.z_current = torch.as_tensor(data["z_current"], dtype=torch.float32)
        self.action = torch.as_tensor(data["action"], dtype=torch.float32)
        self.z_future = torch.as_tensor(data["z_future"], dtype=torch.float32)
        clean = data["clean_future"] if "clean_future" in data else data["z_future"]
        self.clean_future = torch.as_tensor(clean, dtype=torch.float32)
        n = self.z_current.shape[0]
        expected = data["expected_unreliable"] if "expected_unreliable" in data else np.zeros(n)
        observed = data["observed_surprise"] if "observed_surprise" in data else np.zeros(n)
        self.expected_unreliable = torch.as_tensor(expected, dtype=torch.float32)
        self.observed_surprise = torch.as_tensor(observed, dtype=torch.float32)
        self.is_surprise = self.observed_surprise

        if self.z_current.ndim != 2 or self.z_future.ndim != 2:
            raise ValueError("z_current and z_future must be 2D arrays [N, D].")
        if self.action.ndim != 2:
            raise ValueError("action must be a 2D array [N, A].")
        if self.z_current.shape != self.z_future.shape:
            raise ValueError("z_current and z_future must have the same shape.")
        if self.action.shape[0] != n:
            raise ValueError("action must have the same number of rows as z_current.")

    def __len__(self) -> int:
        return self.z_current.shape[0]

    @property
    def latent_dim(self) -> int:
        return int(self.z_current.shape[1])

    @property
    def action_dim(self) -> int:
        return int(self.action.shape[1])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "z_current": self.z_current[idx],
            "action": self.action[idx],
            "z_future": self.z_future[idx],
            "clean_future": self.clean_future[idx],
            "expected_unreliable": self.expected_unreliable[idx],
            "observed_surprise": self.observed_surprise[idx],
            "is_surprise": self.is_surprise[idx],
        }
