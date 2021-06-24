from typing import Dict
import numpy as np
import pickle


def load_pickled_capture(filepath: str) -> Dict[str, np.ndarray]:
    """
    This is an example to demonstrate how to load captured spectrograms.

    :param filepath: a path to a pickle file containing a captured spectrogram
    :return: a dictionary of captured spectrogram information
    """

    with open(filepath, 'rb') as file:
        return pickle.load(file)
