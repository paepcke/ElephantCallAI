from typing import Tuple
import numpy as np
from torch.utils.data import Dataset
import torch

import parameters
from utils import set_seed


class RandomDataset(Dataset):
    """
    A completely random dataset. All classifiers should perform terribly.
    """

    len: int
    shape: Tuple[int, int]
    labels: np.ndarray
    std: float

    def __init__(self, len, shape, labels, std: float = 10):
        self.len = len
        self.shape = shape
        self.labels = labels
        self.std = std

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data = np.random.normal(0, self.std, self.shape)
        label = np.random.choice(self.labels)

        # return a simple tuple of data and label
        return (torch.from_numpy(data.astype(np.float32)), label)


def get_random_dataloader(
       len,
       shape,
       labels,
       batch_size,
       std: float = 10,
       random_seed=8,
       shuffle=True,
       num_workers=16,
       pin_memory=False):
    print("DataLoader Seed:", parameters.DATA_LOADER_SEED)
    set_seed(parameters.DATA_LOADER_SEED + 1)

    dataset = RandomDataset(len, shape, labels, std)

    # Set the data_loader random seed for reproducibility.
    def _init_fn(worker_id):
        # Assign each worker its own seed
        np.random.seed(int(random_seed) + worker_id)
        # Is this bad??
        # This seems bad as each epoch will be the same order of data!
        # torch.manual_seed(int(random_seed) + worker_id)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory,
                                              worker_init_fn=_init_fn)

    return data_loader

