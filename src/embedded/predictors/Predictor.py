import numpy as np
from typing import Tuple


class Predictor:
    """An abstract base class for making predictions based on spectrograms"""

    def make_predictions(self, spectrogram_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns two numpy arrays: predictions and overlap_counts."""
        raise NotImplementedError()
