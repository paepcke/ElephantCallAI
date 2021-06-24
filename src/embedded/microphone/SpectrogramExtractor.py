import numpy as np
from matplotlib import mlab as ml


class SpectrogramExtractor:
    """
    An object that encapsulates STFT settings to provide a simple interface for
    performing a frequency transform.
    """

    nfft: int  # Window size used for creating spectrograms
    hop: int  # Hop size used for creating spectrograms
    max_freq: int  # Determines the maximum frequency band we keep
    sampling_freq: int  # The frequency at which the data is sampled
    pad_to: int  # Determines the padded window size that we want to give a particular grid spacing (e.g., 1.95hz)

    def __init__(self, nfft: int = 4096, hop: int = 800,
                 max_freq: int = 150, sampling_freq: int = 8000, pad_to: int = 4096):
        self.nfft = nfft
        self.hop = hop
        self.max_freq = max_freq
        self.sampling_freq = sampling_freq
        self.pad_to = pad_to

    def extract_spectrogram(self, time_amplitudes: np.ndarray) -> np.ndarray:
        [spectrum, freqs, t] = ml.specgram(time_amplitudes,
                                           NFFT=self.nfft, Fs=self.sampling_freq,
                                           noverlap=(self.nfft - self.hop), window=ml.window_hanning,
                                           pad_to=self.pad_to)
        # Cut out the high frequencies that are not of interest
        spectrum = spectrum[(freqs <= self.max_freq)]
        return spectrum.T
