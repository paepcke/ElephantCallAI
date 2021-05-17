import torch
from torch import nn
import numpy as np


class TrainedGunshotModel(nn.Module):
    """
    A convenient wrapper around a sub-model to abstract away details like applying a softmax to the outputs
    and scaling input by the training set mean and std.
    """

    training_set_mean: torch.Tensor
    training_set_std: torch.Tensor
    sub_model: nn.Module  # this is the model that is actually trained

    def __init__(self, sub_model: nn.Module, mean: np.ndarray, std: np.ndarray):
        super(TrainedGunshotModel, self).__init__()
        self.register_buffer("training_set_mean", torch.from_numpy(mean))
        self.register_buffer("training_set_std", torch.from_numpy(std))
        self.sub_model = sub_model
        # TODO: do we have to register this?
        self.sub_model.eval()

    def forward(self, x):
        x_normed = x - self.training_set_mean
        x_normed = x_normed/self.training_set_std
        logits = self.sub_model.forward(x_normed)
        return torch.nn.functional.softmax(logits, dim=-1)

    def predict(self, x):
        probs = self.forward(x)
        return torch.argmax(probs, dim=-1)
