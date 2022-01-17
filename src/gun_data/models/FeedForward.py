import torch
from torch import nn


class FeedForward(nn.Module):
    """
    A residual-MLP block for use as the feedforward layer in an AttentionBlock
    """

    def __init__(self, dim: int, num_layers: int, residual_connections: bool = True):
        super(FeedForward, self).__init__()

        self.dim = dim
        self.residual_connections = residual_connections

        layers = []
        for i in range(num_layers):
            layer = nn.Sequential(nn.Linear(dim, dim), nn.PReLU())
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.FloatTensor):
        for layer in self.layers:
            if self.residual_connections:
                x = x + layer(x)
            else:
                x = layer(x)
        return x
