from typing import Callable, Optional

import numpy as np
from torch.utils.data import Dataset


class DataReader2DFrom3D(Dataset):
    def __init__(self, volume: np.array, transform: Optional[Callable] = None):
        self.items = volume
        self.transform = transform

    def __len__(self):
        return self.items.shape[2]

    def __getitem__(self, idx: int):
        item = {'input': self.items[:, :, idx].copy()}

        if self.transform:
            item = self.transform(item)
        return item

    def __repr__(self):
        msg = f"DatasetReader2DFrom3D(transform={self.transform})"
        return msg
