from __future__ import annotations

from torch.utils.data import Dataset


class TartanAirLatentDataset(Dataset):
    """Planned public-data loader.

    The v0 repository uses synthetic latent transitions. This class is a stable interface
    placeholder for later frozen-feature trajectories.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Public trajectory loader will be implemented after week-1 MRE.")

    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: int):
        raise IndexError(idx)
