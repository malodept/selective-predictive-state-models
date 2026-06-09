from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def patch_mean_features(image_path: str | Path, grid_size: int = 8) -> np.ndarray:
    """Dependency-light frozen visual feature extractor.

    This is intentionally simple: it resizes an RGB image to grid_size x grid_size
    and flattens RGB patch means. It is not meant as a final teacher encoder; it
    is a smoke-test stand-in so the repo can validate the full visual-feature
    pipeline before plugging in DINO/V-JEPA/TartanAir features.
    """

    img = Image.open(image_path).convert("RGB").resize((grid_size, grid_size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    feat = arr.reshape(-1)
    # Center roughly around zero for easier prediction.
    return (feat - 0.5).astype(np.float32)
