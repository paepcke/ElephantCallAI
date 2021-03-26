from typing import Dict
import numpy as np
import pickle


def load_pickled_capture(filepath: str) -> Dict[str, np.ndarray]:
    """This is an example to demonstrate how to load captured spectrograms."""
    with open(filepath, 'rb') as file:
        return pickle.load(file)
