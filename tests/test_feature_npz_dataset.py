from __future__ import annotations

import numpy as np

from spsm.datasets.feature_npz import FeatureNPZPredictiveStateDataset


def test_feature_npz_dataset(tmp_path):
    path = tmp_path / "features.npz"
    np.savez(
        path,
        z_current=np.zeros((4, 8), dtype=np.float32),
        action=np.ones((4, 2), dtype=np.float32),
        z_future=np.ones((4, 8), dtype=np.float32),
        expected_unreliable=np.array([0, 1, 0, 1], dtype=np.float32),
        observed_surprise=np.array([0, 0, 1, 1], dtype=np.float32),
    )
    ds = FeatureNPZPredictiveStateDataset(path)
    assert len(ds) == 4
    assert ds.latent_dim == 8
    assert ds.action_dim == 2
    item = ds[1]
    assert item["z_current"].shape[0] == 8
    assert item["action"].shape[0] == 2
    assert float(item["expected_unreliable"]) == 1.0
