import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    A positional encoding matrix of dimensions (max_time_dim, encoding_dim)
    """

    def __init__(self, max_time_dim: int, encoding_dim: int):
        super(PositionalEncoding, self).__init__()
        self.max_time_dim = max_time_dim
        self.encoding_dim = encoding_dim

    def forward(self):
        raise NotImplementedError()
