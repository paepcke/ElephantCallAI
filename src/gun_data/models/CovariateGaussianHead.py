import torch.nn as nn
import torch

from gun_data.models import MaskUtils
from gun_data.models.AutoregressiveHead import AutoregressiveHead


class CovariateGaussianHead(AutoregressiveHead):
    """
    A module that takes logits and produces gaussian means and variances to sample from. Also features a low-rank
    covariance matrix.

    Specifically, it models a multivariate Gaussian distribution for a vector of F frequencies:
    The mean of this distribution is another vector, mu.
    This head also outputs a diagonal matrix ('diag').
    This head also outputs a matrix C of dimensions F by D, where D is the parameter 'cov_factor_dims'
    seen in the constructor. The covariance matrix of the distribution is C*C^T + 'diag'.
    """

    def __init__(self, input_dim: int, freq_dim: int, cov_factor_dims: int):
        super(CovariateGaussianHead, self).__init__(input_dim, freq_dim)

        self.cov_factor_dims = cov_factor_dims

        self.linear_layer_means = nn.Linear(input_dim, freq_dim)
        self.linear_layer_diag_variances = nn.Linear(input_dim, freq_dim)
        self.linear_layer_cov_factor = nn.Linear(input_dim, freq_dim * cov_factor_dims)
        self.softplus = nn.Softplus(beta=2)

    def forward(self, x: torch.FloatTensor, time_len: torch.LongTensor):
        """

        :param x: a tensor of shape (batch..., input_dim)

        Returns a tensor of shape (batch..., freq, 2) where [batch..., freq, 0] are means of univariate gaussians and
        [batch, freq, 1] are variances of univariate gaussians.
        """

        means = self.linear_layer_means(x)

        # use softplus to ensure these outputs (variances) are always positive
        diag_variances = self.softplus(self.linear_layer_diag_variances(x)) + 1e-8

        # The only requirement here is that cov_factor * cov_factor^T is positive semi-definite, which it always is
        # since it has real elements. This allows *negative* covariances between frequency pairs
        cov_factor = self.linear_layer_cov_factor(x)

        batch_size = means.shape[0]
        time_len = means.shape[1]

        # We need to reshape the means and variances into a single tensor. It will be of shape (N, T, F, F+1)
        means = means.reshape(batch_size, time_len, self.freq_dim, 1)
        diag_variances = diag_variances.reshape(batch_size, time_len, self.freq_dim, 1)
        cov_factor = cov_factor.reshape(batch_size, time_len, self.freq_dim, self.cov_factor_dims)

        distribution_parameters = torch.cat([means, diag_variances, cov_factor], dim=-1)

        return distribution_parameters

    def loss(self, x_data: torch.Tensor, time_len: torch.LongTensor, distribution_parameters: torch.Tensor):
        x_data = x_data[:, 1:, :]  # compare model outputs to next time step in input data

        means = distribution_parameters[:, :, :, 0]
        diag_vars = distribution_parameters[:, :, :, 1]
        cov_factor = distribution_parameters[:, :, :, 2:]

        batch_size = means.shape[0]
        time_dim = means.shape[1]

        # group batch and time dims together to form 'super-batch' dim
        means = means.reshape(batch_size * time_dim, self.freq_dim)
        diag_vars = diag_vars.reshape(batch_size * time_dim, self.freq_dim)
        cov_factor = cov_factor.reshape(batch_size * time_dim, self.freq_dim, -1)
        superbatched_x_data = x_data.reshape(batch_size * time_dim, self.freq_dim)

        distribution = torch.distributions.LowRankMultivariateNormal(loc=means, cov_diag=diag_vars, cov_factor=cov_factor)

        # compute individual log likelihoods
        log_prob = distribution.log_prob(superbatched_x_data)

        # reshape log_prob back into (N, T)
        log_prob = log_prob.reshape(batch_size, time_dim)

        # mask out terms according to time_len
        time_mask = MaskUtils.create_2d_time_mask(x_data.shape[1], time_len)
        log_prob = log_prob * time_mask

        # sum log likelihood across time dimension
        log_prob = torch.sum(log_prob, dim=-1)

        # return mean of log likelihood, averaging over each example in the batch.
        # Multiply by -1 because this is a loss function that will be minimized.
        loss = -1*torch.mean(log_prob)

        return loss

    def batched_sample(self, distribution_parameters: torch.Tensor):
        means = distribution_parameters[:, :, 0]
        diag_vars = distribution_parameters[:, :, 1]
        cov_factor = distribution_parameters[:, :, 2:]

        distribution = torch.distributions.LowRankMultivariateNormal(loc=means, cov_diag=diag_vars, cov_factor=cov_factor)
        samples = distribution.sample()

        return samples
