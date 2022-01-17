import random
import os
import torch

from gun_data.GunshotDataset import GunshotDataset
from gun_data.models.AutoregressiveHeadModel import AutoregressiveHeadModel
from gun_data import VisualizationUtils as viz


MASK_VALUE = -400


def populate_sample_directory(dataset: GunshotDataset, model: AutoregressiveHeadModel,
                              sample_dir: str, prefix_len: int, num_batches: int,
                              batch_size: int, device: torch.device,
                              show_nonspeculative_samples: bool = False):
    # TODO: modify to allow multiple samples from same prefix?

    if not os.path.exists(sample_dir):
        os.mkdir(sample_dir)

    indices = random.sample(range(len(dataset)), num_batches*batch_size)

    for i in range(num_batches):
        data = []
        unnormed_data = []
        time_lens = []

        for j in range(batch_size):
            batch_idx = i * batch_size + j
            example, _, time_len = dataset[indices[batch_idx]]

            time_lens.append(time_len)
            data.append(example)
            unnormed_example = dataset.unnormalize(example)

            unnormed_example[time_len:, :] = MASK_VALUE  # apply this to make the example visually distinct

            unnormed_data.append(unnormed_example)

        prefixes = torch.stack(data, dim=0).to(device)
        prefix_len_tensor = torch.ones((batch_size,), dtype=torch.long).to(device)*prefix_len

        samples = model.batched_sample(prefixes, prefix_len_tensor)

        if show_nonspeculative_samples:
            nonspec_samples = model.batched_sample(prefixes, prefix_len_tensor, speculative_mode=False)

        for j in range(batch_size):
            sample = samples[j, :, :]
            unnormed_sample = dataset.unnormalize(sample)

            if show_nonspeculative_samples:
                nonspec_sample = nonspec_samples[j, :, :]
                unnormed_nonspec_sample = dataset.unnormalize(nonspec_sample)

                unnormed_nonspec_sample[time_lens[j]:, :] = MASK_VALUE  # apply this to make the example visually distinct where the original sample ends

                samples_to_display = [unnormed_sample, unnormed_nonspec_sample]
                sample_subtitle_dict = {
                    0: "Completion with compounding variance",
                    1: "Completion without compounding variance"
                }
            else:
                samples_to_display = unnormed_sample
                sample_subtitle_dict = None

            example_idx = i * batch_size + j

            # create a visualization pairing unnormed data and unnormed sample
            viz.create_visualization_image(unnormed_data[j],
                                           samples_to_display,
                                           f"{sample_dir}/sample{example_idx}.png",
                                           dataset.maxmin,
                                           title=None,
                                           sample_subtitle_dict=sample_subtitle_dict)
