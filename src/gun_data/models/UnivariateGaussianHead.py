import torch.nn as nn
import torch

from gun_data.models import MaskUtils
from gun_data.models.AutoregressiveHead import AutoregressiveHead


class UnivariateGaussianHead(AutoregressiveHead):
    """
    A module that takes logits and produces means and variances of univariate Gaussian distributions to sample from.
    """

    def __init__(self, input_dim: int, freq_dim: int):
        super(UnivariateGaussianHead, self).__init__(input_dim, freq_dim)

        self.linear_layer_means = nn.Linear(input_dim, freq_dim)
        self.linear_layer_variances = nn.Linear(input_dim, freq_dim)
        self.softplus = nn.Softplus(beta=2)

    def forward(self, x: torch.FloatTensor, time_len: torch.LongTensor):
        """

        :param x: a tensor of shape (batch..., input_dim)

        Returns a tensor of shape (batch..., freq, 2) where [batch..., freq, 0] are means of univariate gaussians and
        [batch, freq, 1] are variances of univariate gaussians.
        """

        means = self.linear_layer_means(x)

        # use softplus to ensure these outputs (variances) are always positive
        variances = self.softplus(self.linear_layer_variances(x)) + 1e-8

        distribution_parameters = torch.stack((means, variances), dim=-1)

        return distribution_parameters

    def loss(self, x_data: torch.Tensor, time_len: torch.LongTensor, distribution_parameters: torch.Tensor):
        x_data = x_data[:, 1:, :]  # compare model outputs to next time step in input data

        means = distribution_parameters[:, :, :, 0]
        vars = distribution_parameters[:, :, :, 1]

        # compute individual log likelihoods
        log_prob = torch.nn.GaussianNLLLoss(full=True, reduction='none')(means, x_data, vars)

        # sum across frequencies
        log_prob = torch.sum(log_prob, dim=-1)

        # mask out terms according to time_len
        time_mask = MaskUtils.create_2d_time_mask(x_data.shape[1], time_len)

        log_prob = log_prob * time_mask

        # sum log likelihood across time dimension
        log_prob = torch.sum(log_prob, dim=-1)

        # return mean of log likelihood, averaging over each example in the batch
        loss = torch.mean(log_prob)

        return loss

    def batched_sample(self, distribution_parameters: torch.Tensor):
        means = distribution_parameters[:, :, 0]
        vars = distribution_parameters[:, :, 1]
        stds = torch.pow(vars, 0.5)

        return torch.normal(means, stds)

def NLLloss(x_data: torch.Tensor, time_len: torch.LongTensor, predicted_means_vars: torch.Tensor):
    x_data = x_data[:, 1:, :]  # compare model outputs to next time step in input data

    means = predicted_means_vars[:, :, :, 0]
    vars = predicted_means_vars[:, :, :, 1]

    # compute individual log likelihoods
    log_prob = torch.nn.GaussianNLLLoss(full=True, reduction='none')(means, x_data, vars)

    # sum across frequencies
    log_prob = torch.sum(log_prob, dim=-1)

    # mask out terms according to time_len
    time_mask = MaskUtils.create_2d_time_mask(x_data.shape[1], time_len)

    log_prob = log_prob * time_mask

    # sum log likelihood across time dimension
    log_prob = torch.sum(log_prob, dim=-1)

    # return mean of log likelihood, averaging over each example in the batch
    loss = torch.mean(log_prob)

    return loss
