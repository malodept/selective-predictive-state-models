import torch

from spsm.datasets.synthetic import SyntheticDatasetConfig, SyntheticPredictiveStateDataset


def test_synthetic_dataset_reproducible():
    cfg = SyntheticDatasetConfig(n_samples=20, latent_dim=8, action_dim=2, seed=123)
    a = SyntheticPredictiveStateDataset(cfg)
    b = SyntheticPredictiveStateDataset(cfg)
    assert torch.allclose(a.z_current, b.z_current)
    assert torch.allclose(a.z_future, b.z_future)
