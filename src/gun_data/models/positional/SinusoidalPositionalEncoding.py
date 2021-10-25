from math import pi
import torch

from gun_data.models.positional.PositionalEncoding import PositionalEncoding


BASE_CONSTANT = 10000


class SinusoidalPositionalEncoding(PositionalEncoding):
    """
    A positional encoding made of up sinusoidal position embeddings (as seen in the 'Attention Is All You Need' paper)
    """

    def __init__(self, max_time_dim: int, encoding_dim: int):
        super().__init__(max_time_dim, encoding_dim)

        if encoding_dim % 2 != 0:
            raise ValueError("Encoding dimension must be divisible by 2!")

        positions = torch.unsqueeze(torch.arange(max_time_dim), dim=1).expand(max_time_dim, encoding_dim)
        exponents = 2*torch.arange(encoding_dim//2)/encoding_dim
        exponents = torch.repeat_interleave(exponents, 2)
        exponents = torch.unsqueeze(exponents, dim=0).expand(max_time_dim, encoding_dim)

        # this term determines whether the operation applied to an element should be 'sine' or 'cosine'
        cosine_indicators = torch.zeros((1, encoding_dim))
        for i in range(cosine_indicators.shape[1]):
            if i % 2 != 0:
                cosine_indicators[0, i] = 1.
        cosine_indicators = cosine_indicators.expand(max_time_dim, encoding_dim)
        positional_encodings = torch.sin(positions/torch.pow(BASE_CONSTANT, exponents) + (pi/2)*cosine_indicators)

        self.positional_encodings = torch.nn.Parameter(data=positional_encodings, requires_grad=False)

    def forward(self):
        return self.positional_encodings
