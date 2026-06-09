from __future__ import annotations

from torch.utils.data import Dataset


class DroidLatentDataset(Dataset):
    """Planned real-data transfer loader."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("DROID loader is planned for the real-data transfer phase.")

    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: int):
        raise IndexError(idx)
