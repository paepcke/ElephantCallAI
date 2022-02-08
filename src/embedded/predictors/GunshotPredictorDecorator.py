import torch
from torch import nn, Tensor
import numpy as np
from typing import Tuple

from embedded.FileLogger import FileLogger
from embedded.predictors.Predictor import Predictor
from embedded.ModelingConstants import GUNSHOT_MODEL_DETECTION_THRESHOLD, \
    GUNSHOT_MODEL_TIME_STEPS, GUNSHOT_OVERLAP_FRAC


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class GunshotPredictorDecorator(Predictor):
    """
    A decorator that runs a gunshot model every time the delegate predictor runs, optionally applying
    a spectrogram transform operation between invocation of the gunshot model and the delegate predictor.

    It uses an LED controller to manipulate an LED on the board depending on the behavior of the gunshot model.
    """

    def __init__(self, gunshot_model: nn.Module, delegate: Predictor,
                 log_file_path: str,
                 gunshot_threshold: float = GUNSHOT_MODEL_DETECTION_THRESHOLD):
        super().__init__()
        self.delegate = delegate
        self.gunshot_model = gunshot_model
        self.file_logger = FileLogger(log_file_path=log_file_path)
        self.gunshot_threshold = gunshot_threshold

    def make_predictions(self, spectrogram_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # run gunshot model, handle output
        spectrogram_tensor = torch.from_numpy(spectrogram_data).float().to(DEVICE)

        # extract out individual model inputs, then arrange them into batches
        examples = []
        step_size = int(GUNSHOT_MODEL_TIME_STEPS * (1 - GUNSHOT_OVERLAP_FRAC))
        for i in range(0, spectrogram_data.shape[0] - GUNSHOT_MODEL_TIME_STEPS + 1, step_size):
            examples.append(spectrogram_tensor[i:(i + GUNSHOT_MODEL_TIME_STEPS), :])

        # assemble batches and predict
        predictions = []
        cur_batch = []
        for i in range(len(examples)):
            cur_batch.append(examples[i])
            if (i % self.delegate.batch_size) == (self.delegate.batch_size - 1):
                predictions.append(self.gunshot_model(torch.stack(cur_batch, dim=0)))
                cur_batch = []
        if len(cur_batch) != 0:
            predictions.append(self.gunshot_model(torch.stack(cur_batch, dim=0)))

        predictions = torch.cat(predictions, dim=0)
        self._handle_gunshot_predictions(predictions)

        # apply delegate behavior
        return self.delegate.make_predictions(spectrogram_data)

    def _handle_gunshot_predictions(self, predictions: Tensor):
        found_gunshot = \
            torch.any(predictions >= (self.gunshot_threshold * torch.ones_like(predictions)))
        if found_gunshot:
            # this behavior can be overridden with arbitrary logic
            self.file_logger.log("Gunshot detected!")

    def close(self):
        self.file_logger.close()
