import torch.nn as nn
import torch

from gun_data.models.positional.PositionalEncoding import PositionalEncoding


class LearnedPositionalEncoding(PositionalEncoding):
    """
    A positional encoding made of up learned position embeddings
    """

    def __init__(self, max_time_dim: int, encoding_dim: int):
        super().__init__(max_time_dim, encoding_dim)

        initial_pos_encodings = torch.zeros((max_time_dim, encoding_dim))
        torch.nn.init.xavier_uniform(initial_pos_encodings)
        self.positional_encodings = torch.nn.Parameter(data=initial_pos_encodings, requires_grad=True)

    def forward(self):
        return self.positional_encodings
