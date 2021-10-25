import unittest
import torch

from gun_data.models.AttentionBlock import AttentionBlock


class AttentionBlockTest(unittest.TestCase):
    def test_autoregressive_property(self):
        block = AttentionBlock(input_dim=6, max_time_dim=10, num_heads=2, dropout_p=0., is_decoder=True)

        x_data = torch.rand((2, 10, 6))

        time_len = torch.Tensor([10, 10])

        out1 = block(x_data, time_len)

        x_data[:, 5, :] = x_data[:, 5, :] + 1

        out2 = block(x_data, time_len)

        # elements before the 5th time step should not change due to autoregressive property
        self.assertTrue(torch.equal(out1[:, :5, :], out2[:, :5, :]))

        # elements after the 5th time step should have changed
        self.assertFalse(torch.equal(out1[:, 5:, :], out2[:, 5:, :]))


if __name__ == '__main__':
    unittest.main()
