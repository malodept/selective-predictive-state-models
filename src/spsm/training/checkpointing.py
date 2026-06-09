from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def save_checkpoint(model: nn.Module, path: str | Path, extra: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict(), "extra": extra or {}}
    torch.save(payload, path)
