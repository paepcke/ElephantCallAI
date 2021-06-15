import numpy as np
from typing import Tuple
import torch
from torch import nn
from torchvision import models, transforms

from embedded.predictors.Predictor import Predictor
from embedded.SpectrogramBuffer import NUM_SAMPLES_IN_TIME_WINDOW
from PredictionUtils import get_batched_predictions_and_overlap_counts


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class RandomTorchvisionPredictor(Predictor):
    """
    A predictor that uses a single pre-trained PyTorch model. A specific input and output shape is required here,
    (num_batches, 256, 77) and (num_batches, 256) respectively.
    """

    model: nn.Module
    batch_size: int
    jump: int
    half_precision: bool

    def __init__(self, batch_size: int = 4,
                 jump: int = NUM_SAMPLES_IN_TIME_WINDOW // 4, half_precision: bool = False,
                 model_type: str = "mobilenet"):
        self.init_model(model_type)
        self.model.eval()  # put model in eval mode
        self.batch_size = batch_size
        self.jump = jump
        if DEVICE.type == 'cpu' and half_precision:
            raise ValueError("PyTorch does not support many operations, including convolution, for half-precision " +
                             "inputs on CPU devices. Use a GPU device or do not use half-precision.")
        self.half_precision = half_precision
        if self.half_precision:
            self.model = self.model.half()

    def init_model(self, model_type: str):
        if model_type == "mobilenet":
            self.model = RandomMobileNetV2Model()
        elif model_type == "resnet":
            self.model = RandomResNet101Model()
        else:
            raise ValueError(f"Only 'resnet' and 'mobilenet' are supported model types. '{model_type}' is not recognized.")
        self.model.eval()
        self.model.to(DEVICE)

    def make_predictions(self, spectrogram_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns two numpy arrays: predictions and overlap_counts."""
        return get_batched_predictions_and_overlap_counts(spectrogram_data, self.model, self.jump,
                                                          n_samples_in_time_window=NUM_SAMPLES_IN_TIME_WINDOW,
                                                          batch_size=self.batch_size, half_precision=self.half_precision)


class RandomResNet101Model(nn.Module):
    def __init__(self):
        super(RandomResNet101Model, self).__init__()

        self.model = models.resnet101()
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256))  # This is hard coded to the size of the training windows

    def forward(self, inputs):
        inputs = inputs.unsqueeze(1)
        inputs = inputs.repeat(1, 3, 1, 1)
        out = self.model(inputs)
        return out


class RandomMobileNetV2Model(nn.Module):
    def __init__(self):
        super(RandomMobileNetV2Model, self).__init__()

        self.resizer = transforms.Resize((224, 224))

        mobilenet_model = models.mobilenet_v2()
        mobilenet_model.classifier = nn.Sequential(
            nn.Linear(1280, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256)
        )

        self.model = mobilenet_model

    def forward(self, inputs):
        inputs = inputs.unsqueeze(1)
        inputs = inputs.repeat(1, 3, 1, 1)
        out = self.model(inputs)
        return out


