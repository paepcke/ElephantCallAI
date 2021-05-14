from torch.utils.data import Dataset
from typing import List, Tuple, Optional
import re, os
import numpy as np
from torch.utils.data import Dataset
import torch

from gun_data.DataAugmentor import DataAugmentor
from gun_data.DataMaker import TRAIN_MEAN_FILENAME, TRAIN_STD_FILENAME, FILENAME_REGEX


class GunshotDataset(Dataset):
    # data files should have names ending in "_<class label>.npy"
    data_dir: str
    files: List[Tuple[str, int]]
    augmentor: Optional[DataAugmentor]
    mean: Optional[np.ndarray]
    std: Optional[np.ndarray]
    preprocess_normalization: bool
    # TODO: cache file contents here to avoid constantly reloading from disk? Does torch optimize this?

    def __init__(self, data_dir: str, augmentor: Optional[DataAugmentor] = None, preprocess_normalization: bool = True):
        self.data_dir = data_dir
        self.augmentor = augmentor
        filename_list = os.listdir(data_dir)
        self.files = []
        for fname in filename_list:
            label = self._get_label_from_filename(fname)
            if label is not None:
                self.files.append((fname, label))

        self.preprocess_normalization = preprocess_normalization
        if self.preprocess_normalization:
            # assume that mean and std files exist in parent directory
            self.mean = np.load(f"{data_dir}/../{TRAIN_MEAN_FILENAME}")
            self.std = np.load(f"{data_dir}/../{TRAIN_STD_FILENAME}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename, label = self.files[idx]
        data = np.load(self.data_dir + "/" + filename)

        if self.preprocess_normalization:
            data -= self.mean
            data /= self.std

        # the 'augmentor' object has a configurable aug probability. This may not change the data.
        if self.augmentor is not None:
            data = self.augmentor.augment(data)

        # TODO: add preprocessing logic to convert to RGB and optimize for imagenet pretrained weights?

        # return a simple tuple of data and label
        return (torch.from_numpy(data.astype(np.float32)), label)

    def _get_label_from_filename(self, filename) -> Optional[int]:
        m = re.match(FILENAME_REGEX, filename)
        if m is None:
            # exclude this file from the dataset
            return None
        return int(m.group(1))