import torch
from torch import nn


class AutoregressiveHead(nn.Module):
    """
    This is an abstract base class for Autoregressive Head modules.
    """
    def __init__(self, input_dim: int, freq_dim: int):
        super(AutoregressiveHead, self).__init__()
        self.input_dim = input_dim
        self.freq_dim = freq_dim

    def forward(self, x: torch.FloatTensor, time_len: torch.LongTensor) -> torch.Tensor:
        """
        The 'forward' function take a vector of logits "x" and computes parameters for some probability density function
            over a space of outputs.
        """
        raise NotImplementedError

    def loss(self, x_data: torch.Tensor, time_len: torch.LongTensor, distribution_parameters: torch.Tensor) -> torch.Tensor:
        """
        The 'loss' function computes the negative log-probability of a particular set of input data on this
            probability density function (parameterized by the tensor "distribution_parameters").
        """
        raise NotImplementedError

    def batched_sample(self, distribution_parameters: torch.Tensor) -> torch.Tensor:
        """
        The 'sample' function returns a batch of samples from each probability density function parameterized by
            the corresponding batch of parameters in the tensor "distribution_parameters".
        """
        raise NotImplementedError
