from __future__ import annotations

from torch.utils.data import Dataset


class TartanDriveLatentDataset(Dataset):
    """Planned real dynamics evaluation loader."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("TartanDrive loader is planned for robustness checks.")

    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: int):
        raise IndexError(idx)
