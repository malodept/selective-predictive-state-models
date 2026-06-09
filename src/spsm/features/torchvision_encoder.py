from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image


@dataclass
class TorchvisionFeatureConfig:
    encoder: str = "resnet18"
    weights: str = "imagenet"  # "imagenet" or "random"
    image_size: int = 224
    batch_size: int = 32
    device: str | None = None


class ResNet18FeatureExtractor:
    """Frozen ResNet18 feature extractor for visual predictive-state experiments.

    This module is intentionally imported lazily by scripts. The base SPSM package
    does not require torchvision unless the user wants real visual encoders.
    """

    def __init__(self, cfg: TorchvisionFeatureConfig):
        try:
            from torchvision import models, transforms
        except Exception as exc:  # pragma: no cover - environment-specific
            raise ImportError(
                "torchvision is required for --encoder resnet18. Install it with: "
                "pip install -e .[vision]"
            ) from exc

        self.device = torch.device(
            cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.batch_size = cfg.batch_size

        if cfg.weights == "imagenet":
            weights = models.ResNet18_Weights.DEFAULT
            model = models.resnet18(weights=weights)
            # Keep the official preprocessing associated with the weights.
            self.transform = weights.transforms()
        elif cfg.weights == "random":
            model = models.resnet18(weights=None)
            self.transform = transforms.Compose(
                [
                    transforms.Resize((cfg.image_size, cfg.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
        else:
            raise ValueError("weights must be 'imagenet' or 'random'")

        # Output before final FC: [B, 512, 1, 1].
        model.fc = torch.nn.Identity()
        model.eval().to(self.device)
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model

    @torch.no_grad()
    def encode_paths(self, image_paths: list[str | Path]) -> np.ndarray:
        features: list[np.ndarray] = []
        batch_tensors: list[torch.Tensor] = []

        for path in image_paths:
            img = Image.open(path).convert("RGB")
            batch_tensors.append(self.transform(img))
            if len(batch_tensors) == self.batch_size:
                features.append(self._encode_batch(batch_tensors))
                batch_tensors = []

        if batch_tensors:
            features.append(self._encode_batch(batch_tensors))

        return np.concatenate(features, axis=0).astype(np.float32)

    def _encode_batch(self, batch_tensors: list[torch.Tensor]) -> np.ndarray:
        batch = torch.stack(batch_tensors, dim=0).to(self.device)
        feats = self.model(batch)
        feats = torch.nn.functional.normalize(feats, dim=-1)
        return feats.detach().cpu().numpy()
