import matplotlib.pyplot as plt
import numpy as np
import torchaudio
import torch
import aifc
from scipy import signal
from torch.utils import data
from scipy.misc import imresize

class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, file_names, labels, resize_flag=False, new_height=224, new_width=224):
        'Initialization'
        self.labels = labels
        self.file_names = file_names

        self.resize_im = resize_flag
        self.width = new_width
        self.height = new_height

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        # Here we want to actually load the audio
        # and perform the short time FT
        file = aifc.open(filename)

        nframes = file.getnframes()
        strsig = file.readframes(nframes)

        audio = np.frombuffer(strsig, np.short).byteswap()

        # See documentation on what each of these actually means
        freqs, times, Sx = signal.spectrogram(y, fs=sample_rate, window='hanning',
                                      nperseg=256, noverlap=236,
                                      detrend=False, scaling='spectrum')

        print (Sx.shape)
        # Right-whale calls only occur under 250Hz
        Pxx = Pxx[freqs<250.] 

        if self.resize_im:
            Sx = imresize(np.log10(Sx),(self.new_width,self.new_height), interp= 'lanczos').astype('float32')
        else:
            # Want to take the log to decrease extremity of values
            Sx = np.log10(Sx)

        X = Sx # Here could convert to ints if we want
        y = self.labels[ID]

        return X, y