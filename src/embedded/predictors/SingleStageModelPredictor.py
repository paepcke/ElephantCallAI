import numpy as np
from typing import Tuple
import torch
from torch import nn

from embedded.predictors.Predictor import Predictor
from embedded.SpectrogramBuffer import NUM_SAMPLES_IN_TIME_WINDOW
from PredictionUtils import get_batched_predictions_and_overlap_counts


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class SingleStageModelPredictor(Predictor):
    """
    A predictor that uses a single pre-trained PyTorch model. A specific input and output shape is required here,
    (num_batches, 256, 77) and (num_batches, 256) respectively.
    """

    model: nn.Module
    batch_size: int
    jump: int
    half_precision: bool

    def __init__(self, path_to_model: str, batch_size: int = 4,
                 jump: int = NUM_SAMPLES_IN_TIME_WINDOW // 4, half_precision: bool = False):
        self.model = torch.load(path_to_model, map_location=DEVICE)
        self.model.eval()  # put model in eval mode
        self.batch_size = batch_size
        self.jump = jump
        if DEVICE.type == 'cpu' and half_precision:
            raise ValueError("PyTorch does not support many operations, including convolution, for half-precision " +
                             "inputs on CPU devices. Use a GPU device or do not use half-precision.")
        self.half_precision = half_precision
        if self.half_precision:
            self.model = self.model.half()

    def make_predictions(self, spectrogram_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns two numpy arrays: predictions and overlap_counts."""
        return get_batched_predictions_and_overlap_counts(spectrogram_data, self.model, self.jump,
                                                          n_samples_in_time_window=NUM_SAMPLES_IN_TIME_WINDOW,
                                                          batch_size=self.batch_size, half_precision=self.half_precision)
