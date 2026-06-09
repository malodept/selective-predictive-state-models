from spsm.datasets.synthetic import SyntheticDatasetConfig, SyntheticPredictiveStateDataset


def test_synthetic_dataset_item_shapes():
    ds = SyntheticPredictiveStateDataset(SyntheticDatasetConfig(n_samples=10, latent_dim=12, action_dim=3))
    item = ds[0]
    assert item["z_current"].shape == (12,)
    assert item["action"].shape == (3,)
    assert item["z_future"].shape == (12,)
    assert item["is_surprise"].ndim == 0
    assert item["expected_unreliable"].ndim == 0
    assert item["observed_surprise"].ndim == 0
