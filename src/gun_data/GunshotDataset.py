from typing import List, Tuple, Optional
import re, os
import numpy as np
from torch.utils.data import Dataset
import torch

from gun_data.DataAugmentor import DataAugmentor
from gun_data.DataMaker import TRAIN_MEAN_FILENAME, TRAIN_STD_FILENAME, TRAIN_MAXMIN_FILENAME, FILENAME_REGEX


class GunshotDataset(Dataset):
    # data files should have names ending in "_<class label>.npy"
    data_dir: str
    files: List[Tuple[str, int]]
    augmentor: Optional[DataAugmentor]
    mean: Optional[np.ndarray]
    std: Optional[np.ndarray]
    preprocess_normalization: bool
    # TODO: cache file contents here to avoid constantly reloading from disk? Does torch optimize this?

    def __init__(self, data_dir: str, augmentor: Optional[DataAugmentor] = None, preprocess_normalization: bool = True,
                 max_time_len: Optional[int] = None, override_mean_std_dir: Optional[str] = None):
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
            if override_mean_std_dir is not None:
                mean_std_dir = override_mean_std_dir
            else:
                mean_std_dir = f"{data_dir}/.."
            # assume that mean and std files exist in parent directory
            self.mean = np.load(f"{mean_std_dir}/{TRAIN_MEAN_FILENAME}")
            self.std = np.load(f"{mean_std_dir}/{TRAIN_STD_FILENAME}")
            if os.path.exists(f"{mean_std_dir}/{TRAIN_MAXMIN_FILENAME}"):
                maxmin = np.load(f"{mean_std_dir}/{TRAIN_MAXMIN_FILENAME}")
                """
                a (2,)-shaped np array where the first element is the max of all amplitudes in the training set
                and the second element is the min
                """
                self.maxmin = maxmin
            else:
                self.maxmin = None

        self.max_time_len = max_time_len

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

        cur_time_len = data.shape[0]
        data_as_tensor = torch.from_numpy(data.astype(np.float32))
        if self.max_time_len is not None:
            if cur_time_len < self.max_time_len:
                """
                zero-pad the rest of the time sequence for this example, it will be discarded in computation of the objective function
                
                That weird tuple argument to 'pad' is, in *reverse order* of dimensions, # dims to pad before tensor, # dims to pad after tensor
                """
                data_as_tensor = torch.nn.functional.pad(data_as_tensor, (0, 0, 0, self.max_time_len - cur_time_len))

        # return a simple tuple of data, label, and time length of data
        return (data_as_tensor, label, cur_time_len)

    def _get_label_from_filename(self, filename) -> Optional[int]:
        m = re.match(FILENAME_REGEX, filename)
        if m is None:
            # exclude this file from the dataset
            return None
        return int(m.group(1))

    def unnormalize(self, spec: torch.Tensor) -> np.ndarray:
        """
        Un-normalizes a spectrogram by applying mean and standard deviation

        :param spec: normalized spectrogram Tensor
        :return: un-normalized spectrogram numpy array
        """
        spec = spec.detach().clone().cpu().numpy()

        # if some frequency bands are suppressed for some reason, un-normalization should still work correctly
        if spec.shape[1] < self.std.shape[1]:
            std = self.std[:, spec.shape[1]]
            mean = self.mean[:, :spec.shape[1]]
        else:
            std = self.std
            mean = self.mean

        spec *= std
        spec += mean
        return spec
