import matplotlib.pyplot as plt
import numpy as np
import torchaudio
import torch
import aifc
from scipy import signal
from torch.utils import data
from scipy.misc import imresize
import pandas as pd
import os

class WhaleDataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
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



