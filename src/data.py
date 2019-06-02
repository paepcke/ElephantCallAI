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

def get_train_valid_loader(data_dir,
                           batch_size,
                           random_seed,
                           augment=False,
                           valid_size=0.1,
                           shuffle=True,
                           num_workers=4,
                           pin_memory=False):
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
    dataset = ElephantDataset(data_dir + 'features.npy', data_dir + 'labels.npy')
    
    num_train = len(dataset)
    print('Num train', num_train)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return train_loader, valid_loader

def get_test_loader(data_dir,
                    batch_size,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=False):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR-10 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - data_loader: test set iterator.
    """
    dataset = ElephantDataset(data_dir + 'features.npy', data_dir + 'labels.npy', transform=transform)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader


class ElephantDataset(data.Dataset):
    def __init__(self, feature_path, label_path, transform=None, preprocess="Norm"):
        # TODO: Do some things depending on how data looks
        self.features = np.load(feature_path) # Shape - (num_train, time, freqs)
        self.labels = np.load(label_path) # Shape - (num_train, time)

        # Potentially include other transforms
        self.transforms = transform

        # Normalize Features
        if preprocess == "Norm":
            standard_norm = StandardScaler()
            # Re-shape the features:
            # (num_ex, seq_len, features) ---> (num_ex * seq_len, features)
            num_ex = self.features.shape[0]
            seq_len = self.features.shape[1]
            self.features = self.features.reshape(num_ex * seq_len, features)
            self.features = standard_norm.fit_transform(self.features)
            self.features = self.features.reshape(num_ex, seq_len, features)
            #self.features = (self.features - np.mean(self.features)) / np.std(self.features)
        elif preprocess == "Scale":
            scaler = MinMaxScaler()
            # Scale features for each training example
            # to be within a certain range. Preserves the
            # relative distribution of each feature. Here
            # each feature is the different frequency band
            for i in range(self.features.shape[0]):
                self.features[i, :, :] = scaler.fit_transform(self.features[i,:,:].astype(np.float32))

    def __len__(self):
        return self.features.shape[0]

    """
    Return a single element at provided index
    """
    def __getitem__(self, index):
        data = self.features[index]
        label = self.labels[index]

        if self.transforms:
            data = self.transforms(data)
            
        # Honestly may be worth pre-process this
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)

        return data, label

class WhaleDataset(data.Dataset):
  # 'Characterizes a dataset for PyTorch'
  def __init__(self, train_csv, train_dir, resize_shape=None):
        # Initialization

        csv = pd.read_csv(train_csv)

        self.file_names = csv.iloc[:, 0]
        self.labels = csv.iloc[:, 1]

        self.train_dir = train_dir

        self.resize_shape = resize_shape

  def __len__(self):
        # Denotes the total number of samples
        return len(self.labels)

  def __getitem__(self, index):
        # 'Generates one sample of data'

        # Select sample
        filename = os.path.join(self.train_dir, self.file_names[index])

        # Load data and get label
        # Here we want to actually load the audio
        # and perform the short time FT
        file = aifc.open(filename)

        nframes = file.getnframes()
        strsig = file.readframes(nframes)

        audio = np.frombuffer(strsig, np.short).byteswap()

        # See documentation on what each of these actually means
        freqs, times, Sx = signal.spectrogram(audio, fs=2000, window='hanning',
                                      nperseg=256, noverlap=236,
                                      detrend=False, scaling='spectrum')

        # Right-whale calls only occur under 250Hz
        Sx = Sx[freqs<250.] 

        if self.resize_shape:
            Sx = imresize(np.log10(Sx),(self.resize_shape[0],self.resize_shape[1]), interp= 'lanczos').astype('float32')
        else:
            # Want to take the log to decrease extremity of values
            Sx = np.log10(Sx)

        X = Sx / 255.# Here could convert to ints if we want
        X = np.repeat(X[np.newaxis, :, :], 3, axis=0)
        y = self.labels[index]

        return X, y



