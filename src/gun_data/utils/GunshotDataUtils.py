from typing import Optional

import parameters
from gun_data.DataAugmentor import DataAugmentor
from utils import set_seed
from gun_data.GunshotDataset import GunshotDataset
import torch
import numpy as np
from models import Model17
from torch import nn


# TODO: rename to 'GunshotTrainingUtils'?

def get_loader(data_dir,
               batch_size,
               augmentor: Optional[DataAugmentor] = None,
               preprocess_normalization: bool = True,
               max_time_len: Optional[int] = None,
               random_seed=8,
               shuffle=True,
               num_workers=16,
               pin_memory=False,
               # specifying the following parameter as 'true' will cause this function to return a tuple of (dataloader, dataset)
               return_dataset_too=False,
               override_mean_std_dir: Optional[str] = None):
    """
    Utility function for loading and returning train and valid
    multi-process iterators.
    """
    print("DataLoader Seed:", parameters.DATA_LOADER_SEED)
    set_seed(parameters.DATA_LOADER_SEED)

    dataset = GunshotDataset(data_dir, augmentor, preprocess_normalization, max_time_len=max_time_len, override_mean_std_dir=override_mean_std_dir)

    print('Size of dataset at {} is {} samples'.format(data_dir, len(dataset)))

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

    if not return_dataset_too:
        return data_loader
    return (data_loader, dataset)


def get_model(input_size, dropout_p=0.) -> nn.Module:
    """
    This function is temporary, just to test out the end-to-end workflow here
    """
    model = Model17(input_size, 2340)  # the '2340' is ignored, as this parameter is not used
    model.model.fc = nn.Sequential(
           nn.Dropout(dropout_p),
           nn.Linear(512, 128),
           nn.ReLU(inplace=True),
           nn.Linear(128, 3))  # This is hard coded to the size of the training windows
    return model


def compute_acc(labels: np.ndarray, logits: np.ndarray):
    predictions = np.argmax(logits, axis=1)
    agreement = np.where(labels == predictions, 1, 0)
    return np.mean(agreement) * 100



