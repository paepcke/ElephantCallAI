import unittest
import torch

from gun_data.models.AutoregressiveHeadModel import AutoregressiveHeadModel
from gun_data.models.UnivariateGaussianHead import UnivariateGaussianHead
from gun_data.models.positional.SinusoidalPositionalEncoding import SinusoidalPositionalEncoding


class AutoRegressiveModelTest(unittest.TestCase):
    def test_autoreg_model_gradient_computation(self):
        x_data = torch.arange(4, dtype=torch.float32)
        x_data = torch.unsqueeze(x_data, dim=-1)
        x_data = torch.unsqueeze(x_data, dim=-1)
        x_data = x_data.expand(4, 5, 10)

        zers = torch.zeros((4, 5, 10))
        zers[:, :, :] = x_data
        x_data = zers

        time_len = torch.ones((4,)) * 5

        head = UnivariateGaussianHead(8, 10)
        model = AutoregressiveHeadModel(head=head, max_time_dim=4, num_attention_heads=2, num_attention_blocks=2,
                                        dropout_p=0.,
                                        positional_encoding=SinusoidalPositionalEncoding(max_time_dim=4,
                                                                                         encoding_dim=2))

        distribution_params = model(x_data[:, :-1, :], time_len)

        loss = model.head.loss(x_data, time_len, distribution_params)

        loss.backward()


if __name__ == '__main__':
    unittest.main()
