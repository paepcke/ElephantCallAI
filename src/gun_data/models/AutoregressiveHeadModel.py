import torch.nn as nn

from gun_data.models.AutoregressiveHead import AutoregressiveHead
from gun_data.models.positional.PositionalEncoding import PositionalEncoding
from gun_data.models.AttentionLogitsModule import AttentionLogitsModule
from gun_data.models.UnivariateGaussianHead import UnivariateGaussianHead

# imports for test
from gun_data.models.positional.SinusoidalPositionalEncoding import SinusoidalPositionalEncoding
import torch


class AutoregressiveHeadModel(nn.Module):
    """
    A model that produces means and variances of univariate Gaussian distributions to sample from.
    """

    def __init__(self, head: AutoregressiveHead, max_time_dim: int, num_attention_heads: int,
                 num_attention_blocks: int, dropout_p: float, positional_encoding: PositionalEncoding):
        super(AutoregressiveHeadModel, self).__init__()
        self.max_time_dim = max_time_dim

        self.head = head

        self.freq_dim = head.freq_dim
        self.hidden_dim = head.input_dim

        self.attention_logits_module = AttentionLogitsModule(self.freq_dim, max_time_dim, self.hidden_dim, num_attention_heads,
                                                             num_attention_blocks, dropout_p, positional_encoding)


    def forward(self, x, time_len):
        logits = self.attention_logits_module(x, time_len)
        means_and_vars = self.head(logits, time_len)
        return means_and_vars

    def loss(self, x_data: torch.Tensor, time_len: torch.LongTensor, predicted_means_vars: torch.Tensor):
        return self.head.loss(x_data, time_len, predicted_means_vars)

    def batched_sample(self, start: torch.Tensor, prefix_lens: torch.LongTensor, speculative_mode: bool = True):
        """
        Given a short tensor of prefixes, sample rest of sequence
        :param start: a prefix tensor to start the sequence. (batch, max_time_len + 1, freq)
        :param prefix_lens: the time length of each prefix tensor in the batch.
        :param speculative_mode: if True, uses generated samples as input to predict future time steps of information.
            In practice, this is how autoregressive models are sampled. For qualitative analysis, turning this off may
            help to show the degradation in sample quality across time steps.
        :return: (batch_size, max_time_len, freq_dim) sample tensor
        """
        self.eval()

        batch_size = start.shape[0]
        device = start.device

        # output is shaped like (batch, max_time_dim + 1, freq)
        output = start.clone()

        min_prefix_len = torch.min(prefix_lens)
        cur_timestep = min_prefix_len - 1

        with torch.no_grad():
            while cur_timestep < self.max_time_dim:
                temp_prefix_lens = torch.ones_like(prefix_lens).to(device)*(cur_timestep + 1)

                if speculative_mode:
                    model_input = output
                else:
                    model_input = start

                means_and_vars = self.forward(model_input[:, :-1, :], temp_prefix_lens)[:, cur_timestep, :, :]

                # don't overwrite prefix parts
                replace = torch.where(temp_prefix_lens >= prefix_lens,
                    torch.ones_like(prefix_lens).to(device),
                    torch.zeros_like(prefix_lens).to(device))

                replace = torch.unsqueeze(replace, dim=1)
                replace = torch.unsqueeze(replace, dim=1)
                replace = replace.expand(batch_size, 1, self.freq_dim)

                keep = 1 - replace

                samples = self.head.batched_sample(means_and_vars)

                output[:, (cur_timestep + 1):(cur_timestep + 2), :] = \
                    keep * output[:, (cur_timestep + 1):(cur_timestep + 2), :] + replace * samples

                cur_timestep += 1

        return output

    def freeze_for_transfer(self, layers_to_finetune: int):
        self.attention_logits_module.freeze_for_transfer(layers_to_finetune)
