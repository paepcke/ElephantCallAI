import numpy as np
from typing import Tuple
import torch
from torch import nn
import sys

from src.embedded.predictors.Predictor import Predictor
from src.embedded.SpectrogramBuffer import NUM_SAMPLES_IN_TIME_WINDOW
from src.eval import forward_inference_on_batch


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ModelPredictor(Predictor):
    """A predictor that uses a pre-trained PyTorch model. A specific input and output shape is required here!"""
    model: nn.Module
    batch_size: int

    def __init__(self, path_to_model: str, batch_size: int = 4):
        self.model = torch.load(path_to_model, map_location=DEVICE)
        self.batch_size = batch_size

    def make_predictions(self, spectrogram_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns two numpy arrays: predictions and overlap_counts."""
        return self.predict_batched(spectrogram_data)

    # TODO: This is copied almost verbatim from eval.py, it can be refactored in the future
    def predict_batched(self, data, jump=NUM_SAMPLES_IN_TIME_WINDOW//2):
        time_idx = 0

        # key assumption: TIME_STEPS_IN_WINDOW is evenly divisible by 'jump'
        assert(NUM_SAMPLES_IN_TIME_WINDOW % jump == 0)
        if jump == 0:
            k = 1
        else:
            k = NUM_SAMPLES_IN_TIME_WINDOW // jump

        # cut off data at end to allow for even divisibility
        raw_end_time = data.shape[0]
        clean_end_time = raw_end_time - (raw_end_time % jump)

        if clean_end_time != raw_end_time:
            print("WARNING: {} time steps were cut off for the sake of even divisibiliity".format(raw_end_time - clean_end_time), file=sys.stderr)

        predictions = np.zeros(clean_end_time)
        overlap_counts = np.zeros(clean_end_time)

        while time_idx + NUM_SAMPLES_IN_TIME_WINDOW*self.batch_size + (k - 1)*jump <= clean_end_time:
            forward_inference_on_batch(self.model, data, time_idx, jump, self.batch_size, predictions, overlap_counts, k)
            time_idx += NUM_SAMPLES_IN_TIME_WINDOW*self.batch_size

        # final batch (if size < BATCH_SIZE)
        final_full_batch_size = (clean_end_time - time_idx - (k - 1)*jump)//NUM_SAMPLES_IN_TIME_WINDOW
        if final_full_batch_size > 0:
            forward_inference_on_batch(self.model, data, time_idx, jump, final_full_batch_size, predictions, overlap_counts, k)
            time_idx += NUM_SAMPLES_IN_TIME_WINDOW*final_full_batch_size

        # remaining jumps (less than k)
        if time_idx + NUM_SAMPLES_IN_TIME_WINDOW <= clean_end_time:
            remaining_jumps = (clean_end_time - time_idx - NUM_SAMPLES_IN_TIME_WINDOW)//jump + 1
            forward_inference_on_batch(self.model, data, time_idx, jump, 1, predictions, overlap_counts, remaining_jumps)

        return predictions, overlap_counts
