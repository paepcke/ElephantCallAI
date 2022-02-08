import unittest
import torch

from gun_data.models.UnivariateGaussianHead import UnivariateGaussianHead


class UnivariateGaussianTest(unittest.TestCase):
    def test_gaussian_computation_does_not_crash(self):
        # This is merely a sanity check to ensure the gaussian loss computation and gradient update does not raise an exception
        x_data = torch.arange(4, dtype=torch.float32)
        x_data = torch.unsqueeze(x_data, dim=-1)
        x_data = torch.unsqueeze(x_data, dim=-1)
        x_data = x_data.expand(4, 5, 2)

        means = x_data[:, :-1, :]
        vars = torch.ones_like(means)

        predicted_means_vars = torch.stack((means, vars), dim=-1)
        predicted_means_vars.requires_grad = True

        time_len = 1 + torch.arange(4, dtype=torch.long)

        head = UnivariateGaussianHead(2, 2)

        loss = head.loss(x_data, time_len, predicted_means_vars)
        loss.backward()


if __name__ == '__main__':
    unittest.main()
