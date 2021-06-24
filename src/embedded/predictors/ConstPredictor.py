import numpy as np
from typing import Tuple

from embedded.predictors.Predictor import Predictor


class ConstPredictor(Predictor):
    """
    This implementation is a 'dummy' implementation meant to be replaced or overridden later. Its purpose
        is to test the harness code rather than be an accurate model.
    """

    const_prediction: int

    def __init__(self, const_prediction: int = 1):
        self.const_prediction = const_prediction

    def make_predictions(self, spectrogram_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        predictions = np.ones((spectrogram_data.shape[0],)) * self.const_prediction
        overlap_counts = np.ones((spectrogram_data.shape[0],))
        return predictions, overlap_counts
