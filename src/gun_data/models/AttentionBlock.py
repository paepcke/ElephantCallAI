import torch.nn as nn
import torch

from gun_data.models import MaskUtils
from gun_data.models.FeedForward import FeedForward


class AttentionBlock(nn.Module):
    def __init__(self, input_dim: int, max_time_dim: int, num_heads: int, dropout_p: float,
                 is_decoder: bool = True):
        super(AttentionBlock, self).__init__()

        self.max_time_dim = max_time_dim
        self.num_heads = num_heads

        if is_decoder:
            # causal attention mask
            self.attn_mask: torch.Tensor = torch.tril(torch.ones((max_time_dim, max_time_dim), requires_grad=False))
            self.attn_mask = torch.where(self.attn_mask == 0, torch.Tensor([True]).bool().expand(max_time_dim, max_time_dim),
                                         torch.Tensor([False]).bool().expand(max_time_dim, max_time_dim))

        else:
            # no mask restrictions
            self.attn_mask: torch.Tensor = torch.Tensor([False]).expand(max_time_dim, max_time_dim)
        self.attn_mask = nn.Parameter(self.attn_mask, requires_grad=False)

        self.dropout_p = dropout_p

        #self.attention = nn.MultiheadAttention(input_dim, num_heads, dropout=self.dropout_p, batch_first=True)
        self.attention = nn.MultiheadAttention(input_dim, num_heads, dropout=0., batch_first=True)

        self.dropout = nn.Dropout(p=dropout_p)

        self.layernorm1 = nn.LayerNorm(normalized_shape=input_dim)

        self.feed_forward = FeedForward(dim=input_dim, num_layers=3)

        self.layernorm2 = nn.LayerNorm(normalized_shape=input_dim)

    def forward(self, x: torch.FloatTensor, time_len: torch.LongTensor):
        # this gets recomputed for every attn block in a sequence, kind of inefficient
        key_padding_mask = MaskUtils.create_2d_time_mask(self.max_time_dim, time_len + 1, True)

        attn_out, _ = self.attention(query=x, key=x, value=x, attn_mask=self.attn_mask,
                                    key_padding_mask=key_padding_mask, need_weights=False)

        # first skip connection
        x = x + attn_out

        x = self.layernorm1(x)

        feed_forward_out = self.feed_forward(x)
        feed_forward_out = self.dropout(feed_forward_out)

        x = feed_forward_out

        x = self.layernorm2(x)

        return x
