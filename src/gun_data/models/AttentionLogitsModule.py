import torch.nn as nn
import torch

from gun_data.models.positional.PositionalEncoding import PositionalEncoding
from gun_data.models.AttentionBlock import AttentionBlock


class AttentionLogitsModule(nn.Module):
    """
    Torch module that produces logits in a style seen in transformer architectures.
    """

    def __init__(self, freq_dim: int, max_time_dim: int, hidden_dim: int, num_attention_heads: int,
                 num_attention_blocks: int, dropout_p: float, positional_encoding: PositionalEncoding,
                 is_decoder: bool = True):
        super(AttentionLogitsModule, self).__init__()

        self.assert_constructor_args(freq_dim, max_time_dim, hidden_dim,
                                     num_attention_heads, num_attention_blocks,
                                     dropout_p, positional_encoding)

        self.freq_dim: int = freq_dim
        self.max_time_dim: int = max_time_dim
        self.hidden_dim: int = hidden_dim
        self.num_attention_heads: int = num_attention_heads
        self.num_attention_blocks: int = num_attention_blocks
        self.dropout_p: float = dropout_p
        self.positional_encoding: PositionalEncoding = positional_encoding
        self.is_decoder = is_decoder

        self.dim_reducer = nn.Linear(freq_dim, hidden_dim - positional_encoding.encoding_dim, bias=False)
        self.attention_blocks = self.init_attention_blocks()

    def forward(self, x: torch.FloatTensor, time_len: torch.LongTensor):
        """

        :param x: a batch of 2D FloatTensors with dimensions (batch, time, freq)
        :param time_len: a 1D LongTensor that specifies the time-length of each sample in the batch (batch,)
        :return: the logits output by this module
        """

        # linear projection of x down to a lower dimensionality BEFORE the position encoding
        if self.positional_encoding.encoding_dim + self.freq_dim != self.hidden_dim:
            x = self.dim_reducer(x)

        # Add positional encodings to x
        x = self.add_positional_encodings(x)

        for block in self.attention_blocks:
          x = block(x, time_len)

        return x

    def add_positional_encodings(self, x):
        """
        concatenates positional encodings into the input tensor
        :param x:
        :return:
        """

        batch_dim = x.shape[0]

        positional_encoding = self.positional_encoding.forward()
        positional_encoding = torch.unsqueeze(positional_encoding, dim=0)
        positional_encoding = positional_encoding.expand(batch_dim, -1, -1)

        # concatenate the inputs and the positional encodings at the frequency dimension
        return torch.cat((x, positional_encoding), dim=-1)

    def init_attention_blocks(self) -> nn.ModuleList:
        attention_block_list = []
        for i in range(self.num_attention_blocks):
            attention_block_list.append(
                AttentionBlock(self.hidden_dim, self.max_time_dim, self.num_attention_heads,
                               self.dropout_p, self.is_decoder))

        return nn.ModuleList(attention_block_list)

    def freeze_for_transfer(self, layers_to_finetune: int):
        layers_to_finetune = min(layers_to_finetune, self.num_attention_blocks)

        for i in range(0, self.num_attention_blocks - layers_to_finetune):
            block = self.attention_blocks[i]
            block.dropout = nn.Dropout(p=0)  # reduce variance in training from dropouts on fully-frozen layers
            for param in block.parameters():
                param.requires_grad = False

    def assert_constructor_args(self, freq_dim: int, max_time_dim: int,
                                hidden_dim: int, num_attention_heads: int, num_attention_blocks: int,
                                dropout_p: float, positional_encoding: PositionalEncoding):
        if dropout_p < 0 or dropout_p > 1:
            raise ValueError(f"Dropout probability configured as {dropout_p} but must be between 0 and 1, inclusive")
        if hidden_dim <= positional_encoding.encoding_dim:
            raise ValueError(f"hidden dimension {hidden_dim} is less than or equal to positional encoding dimension {positional_encoding.encoding_dim}, this configuration makes no sense.")
