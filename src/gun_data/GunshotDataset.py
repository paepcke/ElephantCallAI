from torch.utils.data import Dataset
from typing import List, Tuple, Optional
import re, os
import numpy as np
from torch.utils.data import Dataset
import torch


FILENAME_REGEX = ".*_(\d+)\.npy"


class GunshotDataset(Dataset):
    # data files should have names ending in "_<class label>.npy"
    data_dir: str
    files: List[Tuple[str, int]]
    # TODO: cache file contents here to avoid constantly reloading from disk? Does torch optimize this?

    def __init__(self, data_dir):
        # TODO: add preprocessing logic?
        self.data_dir = data_dir
        filename_list = os.listdir(data_dir)
        self.files = []
        for fname in filename_list:
            label = self._get_label_from_filename(fname)
            if label is not None:
                self.files.append((fname, label))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename, label = self.files[idx]
        data = np.load(self.data_dir + "/" + filename)
        # return a simple tuple of data and label
        return (torch.from_numpy(data.astype(np.float32)), label)

    def _get_label_from_filename(self, filename) -> Optional[int]:
        m = re.match(FILENAME_REGEX, filename)
        if m is None:
            # exclude this file from the dataset
            return None
        return int(m.group(1))