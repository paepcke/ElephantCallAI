import matplotlib.pyplot as plt
import numpy as np
import torch
import aifc
from scipy import signal
from torch.utils import data
from torchvision import transforms
from scipy.misc import imresize
import pandas as pd
import os
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import glob

Noise_Stats_Directory = "../elephant_dataset/eleph_dataset/Noise_Stats/"

def get_loader(data_dir,
               batch_size,
               norm="norm",
               scale=False,
               augment=False,
               shuffle=True,
               num_workers=4,
               pin_memory=True):
    """
    Utility function for loading and returning train and valid
    multi-process iterators.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - random_seed: fix seed for reproducibility.
    - augment: whether data augmentation scheme. Only applied on the train split.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    # Note here we could do some data preprocessing!
    # define transform
    dataset = ElephantDataset(data_dir, preprocess=norm, scale=scale)
    
    print('Size of dataset at {} is {} samples'.format(data_dir, len(dataset)))

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

    return data_loader

"""
    Notes
    - Preprocess = Norm, Scale = False ===> seems bad
    - Preprocess = Norm, Scale = True ===> Works well on small dataset!
    - Preprocess = Scale, Scale = False ===> Has quite a bit of trouble over fitting small dataset compared to other but eventually can
    - Preprocess = Scale, Scale = True ===> Has quite a bit of trouble over fitting small dataset compared to other and bad val acc!
    - Preprocess = ChunkNorm, Scale = False ===> Very slow and bad
    - Preprocess = ChunkNorm, Scale = True ===> Similar to Norm with scale
    - Preprocess = None, Scale = True ====> No worky
    - Preprocess = Scale range (-1, 1), Scale = True ===> Overfit but huge variance issue
"""
class ElephantDataset(data.Dataset):
    def __init__(self, data_path, transform=None, preprocess="norm", scale=False):
        # Plan: Load in all feature and label names to create a list
        self.data_path = data_path
        self.user_transforms = transform
        self.preprocess = preprocess
        self.scale = scale

        self.features = glob.glob(data_path + "*_features_*")
        self.labels = []

        for feature_path in self.features:
            feature_parts = feature_path.split("_features_")
            self.labels.append(glob.glob(feature_parts[0] + "_labels_" + feature_parts[1])[0])

        assert len(self.features) == len(self.labels)

        print("ElephantDataset number of features {} and number of labels {}".format(len(self.features), len(self.labels)))
        print('Normalizing with {} and scaling {}'.format(preprocess, scale))


    def __len__(self):
        return len(self.features)

    """
    Return a single element at provided index
    """
    def __getitem__(self, index):
        feature = np.load(self.features[index])
        label = np.load(self.labels[index])

        feature = self.apply_transforms(feature)
        if self.transforms:
            feature = self.user_transforms(feature)
            
        # Honestly may be worth pre-process this
        feature = torch.from_numpy(feature)
        label = torch.from_numpy(label)

        return feature, label

    def apply_transforms(self, data):
        if self.scale:
            data = 10 * np.log10(data)

        # Normalize Features
        if self.preprocess == "norm":
            data = (data - np.mean(data)) / np.std(data)
        elif self.preprocess == "globalnorm":
            data = (data - 132.228) / 726.319 # Calculated these over the training dataset 

        return data

        # elif self.preprocess == "Scale":
        #     scaler = MinMaxScaler()
        #     # Scale features for each training example
        #     # to be within a certain range. Preserves the
        #     # relative distribution of each feature. Here
        #     # each feature is the different frequency band
        #     for i in range(self.features.shape[0]):
        #         self.features[i, :, :] = scaler.fit_transform(self.features[i,:,:].astype(np.float32))
        #     #num_ex = self.features.shape[0]
        #     #seq_len = self.features.shape[1]
        #     #self.features = self.features.reshape(num_ex * seq_len, -1)
        #     #self.features = scaler.fit_transform(self.features)
        #     #self.features = self.features.reshape(num_ex, seq_len, -1)
        # elif self.preprocess == "ChunkNorm":
        #     for i in range(self.features.shape[0]):
        #         self.features[i, :, :] = (self.features[i, :, :] - np.mean(self.features[i, :, :])) / np.std(self.features[i, :, :])
        # elif self.preprocess == "BackgroundS":
        #     # Load in the pre-calculated mean,std,etc.
        #     if not scale:
        #         mean_noise = np.load(Noise_Stats_Directory + "mean.npy")
        #         std_noise = np.load(Noise_Stats_Directory + "std.npy")
        #     else:
        #         mean_noise = np.load(Noise_Stats_Directory + "mean_log.npy")
        #         std_noise = np.load(Noise_Stats_Directory + "std_log.npy")

        #     self.features = (self.features - mean_noise) / std_noise
        # elif self.preprocess == "BackgroundM":
        #     # Load in the pre-calculated mean,std,etc.
        #     if not scale:
        #         mean_noise = np.load(Noise_Stats_Directory + "mean.npy")
        #         median_noise = np.load(Noise_Stats_Directory + "median.npy")
        #     else:
        #         mean_noise = np.load(Noise_Stats_Directory + "mean_log.npy")
        #         median_noise = np.load(Noise_Stats_Directory + "median_log.npy")

        #     self.features = (self.features - mean_noise) / median_noise
        # elif self.preprocess == "FeatureNorm":
        #     self.features = (self.features - np.mean(self.features, axis=(0, 1))) / np.std(self.features, axis=(0,1))

"""
    Dataset for full test length audio
"""
class ElephantDatasetFull(data.Dataset):
    def __init__(self, spectrogram_files, label_files, gt_calls, preprocess="Norm", scale=True):

        self.specs = spectrogram_files
        self.labels = label_files
        self.gt_calls = gt_calls # This is the .txt file that contains start and end times of calls
        self.preprocess = preprocess
        self.scale = scale
        
        print('Normalizing with {} and scaling {}'.format(preprocess, scale))


    def __len__(self):
        return len(self.specs)


    def transform(self, spectrogram):
        # Potentially include other transforms
        if self.scale:
            spectrogram = 10 * np.log10(spectrogram)

        # Normalize Features
        if self.preprocess == "Norm": # Only have one training example so is essentially chunk norm
            spectrogram = (spectrogram - np.mean(spectrogram)) / np.std(spectrogram)
        elif preprocess == "Scale":
            scaler = MinMaxScaler()
            # Scale features for each training example
            # to be within a certain range. Preserves the
            # relative distribution of each feature. Here
            # each feature is the different frequency band
            spectrogram = scaler.fit_transform(spectrogram.astype(np.float32))
        elif self.preprocess == "ChunkNorm":
            spectrogram = (spectrogram - np.mean(spectrogram)) / np.std(spectrogram)
        elif self.preprocess == "BackgroundS":
            # Load in the pre-calculated mean,std,etc.
            if not scale:
                mean_noise = np.load(Noise_Stats_Directory + "mean.npy")
                std_noise = np.load(Noise_Stats_Directory + "std.npy")
            else:
                mean_noise = np.load(Noise_Stats_Directory + "mean_log.npy")
                std_noise = np.load(Noise_Stats_Directory + "std_log.npy")

            spectrogram = (spectrogram - mean_noise) / std_noise
        elif self.preprocess == "BackgroundM":
            # Load in the pre-calculated mean,std,etc.
            if not scale:
                mean_noise = np.load(Noise_Stats_Directory + "mean.npy")
                median_noise = np.load(Noise_Stats_Directory + "median.npy")
            else:
                mean_noise = np.load(Noise_Stats_Directory + "mean_log.npy")
                median_noise = np.load(Noise_Stats_Directory + "median_log.npy")

            spectrogram = (spectrogram - mean_noise) / median_noise
        elif self.preprocess == "FeatureNorm":
            spectrogram = (spectrogram - np.mean(spectrogram, axis=1)) / np.std(spectrogram, axis=1)
        return spectrogram

    """
    Return a single element at provided index
    """
    def __getitem__(self, index):
        spectrogram_path = self.specs[index]
        label_path = self.labels[index]
        gt_call_path = self.gt_calls[index]

        spectrogram = np.load(spectrogram_path).transpose()
        label = np.load(label_path)

        spectrogram = self.transform(spectrogram)
        #spectrogram = np.expand_dims(spectrogram, axis=0) # Add the batch dimension so we can apply our lstm!
            
        # Honestly may be worth pre-process this
        #spectrogram = torch.from_numpy(spectrogram)
        #label = torch.from_numpy(label)

        return spectrogram, label, gt_call_path


