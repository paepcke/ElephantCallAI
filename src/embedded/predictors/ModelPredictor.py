import numpy as np
from typing import Tuple
import torch
from torch import nn

from src.embedded.predictors.Predictor import Predictor
from src.embedded.SpectrogramBuffer import NUM_SAMPLES_IN_TIME_WINDOW
from src.eval import get_batched_predictions_and_overlap_counts


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ModelPredictor(Predictor):
    """A predictor that uses a pre-trained PyTorch model. A specific input and output shape is required here!"""
    model: nn.Module
    batch_size: int
    jump: int

    def __init__(self, path_to_model: str, batch_size: int = 4, jump: int = NUM_SAMPLES_IN_TIME_WINDOW // 4):
        self.model = torch.load(path_to_model, map_location=DEVICE)
        self.model.eval()  # put model in eval mode
        self.batch_size = batch_size
        self.jump = jump

    def make_predictions(self, spectrogram_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns two numpy arrays: predictions and overlap_counts."""
        return get_batched_predictions_and_overlap_counts(spectrogram_data, self.model, self.jump,
                                                          n_samples_in_time_window=NUM_SAMPLES_IN_TIME_WINDOW,
                                                          batch_size=self.batch_size)
