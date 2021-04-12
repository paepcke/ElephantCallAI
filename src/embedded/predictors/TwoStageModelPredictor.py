import numpy as np
from typing import Tuple
import torch
from torch import nn

from PredictionUtils import get_batched_predictions_and_overlap_counts, get_batched_predictions_and_overlap_counts_for_two_stage_model
from embedded.SpectrogramBuffer import NUM_SAMPLES_IN_TIME_WINDOW
from embedded.FileUtils import assert_is_directory, assert_path_exists


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# middle threshold is always 0.5
class TwoStageModelPredictor:
    first_stage_model: nn.Module
    second_stage_model: nn.Module
    batch_size: int
    jump: int

    """An abstract base class for making predictions based on spectrograms"""
    def __init__(self, path_to_2stage_dir: str, batch_size: int = 4, jump: int = NUM_SAMPLES_IN_TIME_WINDOW // 4):
        assert_is_directory(path_to_2stage_dir,
                            "A two-stage model must be specified with a directory containing each stage's model")
        first_model_path = path_to_2stage_dir + "/first_stage.pt"
        assert_path_exists(first_model_path, "The file 'first_stage.pt' must exist in the specified directory.")
        second_model_path = path_to_2stage_dir + "/second_stage.pt"
        assert_path_exists(second_model_path, "The file 'second_stage.pt' must exist in the specified directory.")
        self.first_stage_model = torch.load(first_model_path, map_location=DEVICE)
        self.first_stage_model.eval()  # put model in eval mode
        self.second_stage_model = torch.load(second_model_path, map_location=DEVICE)
        self.second_stage_model.eval()  # put model in eval mode
        self.batch_size = batch_size
        self.jump = jump

    def make_predictions(self, spectrogram_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns two numpy arrays: predictions and overlap_counts.

        :param spectrogram_data: input to a model to make predictions
        :return: a tuple of predictions, overlap_counts
        """
        return get_batched_predictions_and_overlap_counts_for_two_stage_model(
            spectrogram_data, self.first_stage_model, self.second_stage_model, self.jump,
            n_samples_in_time_window=NUM_SAMPLES_IN_TIME_WINDOW,
            batch_size=self.batch_size)
