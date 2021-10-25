from typing import Optional, Union, List, Dict

import numpy as np
from matplotlib import pyplot as plt
import matplotlib


YELLOW = np.array([255., 255., 0.]).reshape((1, 1, 3))
BLUE = np.array([0., 0., 255.]).reshape((1, 1, 3))

DURATION_SEC = 10
MAX_FREQ_HZ = 1024


def create_visualization_image(real: np.ndarray,
                               sampled: Union[np.ndarray, List[np.ndarray]],
                               savepath: str,
                               maxmin: Optional[np.ndarray] = None,
                               title: Optional[str] = None,
                               sample_subtitle_dict: Optional[Dict[int, str]] = None,
                               show_diff: bool = False):
    """

    :param real: a (time, freq) spectrogram sampled from an actual dataset
    :param sampled: a (time, freq) spectrogram sampled from a generative model
    :param savepath: a location to save the generated image file
    :param maxmin: a (2,)-shaped np array where the first element is the max of all amplitudes in the training set
        and the second element is the min
    :param title: the title to display on the generated image
    :param show_diff: a boolean indicating whether to show a third image, the difference of the first two. Does nothing
        when multiple samples are passed in.
    :return: nothing
    """

    # transpose samples here
    real = real.T

    single_sample = (type(sampled) != list)

    if single_sample:
        sampled = sampled.T
    else:
        show_diff = False  # we won't show the diff between the real sample and every fake in any circumstance
        for i in range(len(sampled)):
            sampled[i] = sampled[i].T

    if show_diff:
        # we only show_diff in the cases where there's only one sample
        fig, ax = plt.subplots(3)
    else:
        if single_sample:
            num_plots = 2
        else:
            num_plots = 1 + len(sampled)
        fig, ax = plt.subplots(num_plots)

    # 'maxmin' allows colormaps across multiple samples to be the same
    if maxmin is None:
        print("WARNING: using per-sample color scale mode")
        max_real = np.max(real)
        min_real = np.min(real)
        if single_sample:
            max_fake = np.max(sampled)
            min_fake = np.min(sampled)
        else:
            min_fake = min_real
            max_fake = max_real
            for sample in sampled:
                max_fake = max(np.max(sample), max_fake)
                min_fake = min(np.min(sample), min_fake)
        min_intensity = min(min_real, min_fake)
        max_intensity = max(max_fake, max_real)
    else:
        max_intensity = maxmin[0]
        min_intensity = maxmin[1]

    real_image = create_colorscale_image(real, max_intensity, min_intensity)
    ax[0].imshow(real_image, interpolation='nearest', aspect='auto')
    ax[0].set_title("Sample from dataset")
    add_axis_labels(ax[0], real_image)
    add_colorbar(min_intensity, max_intensity, fig, ax[0])

    if single_sample:
        sampled_image = create_colorscale_image(sampled, max_intensity, min_intensity)
        ax[1].imshow(sampled_image, interpolation='nearest', aspect='auto')
        ax[1].set_title("Generated sample from model")
        add_axis_labels(ax[1], sampled_image)
        add_colorbar(min_intensity, max_intensity, fig, ax[1])

    else:
        for i, sample in enumerate(sampled):
            sampled_image = create_colorscale_image(sampled[i], max_intensity, min_intensity)
            ax[i + 1].imshow(sampled_image, interpolation='nearest', aspect='auto')
            if sample_subtitle_dict is None:
                sample_title = "Generated sample from model"
            else:
                sample_title = sample_subtitle_dict[i]
            ax[i + 1].set_title(sample_title)
            add_axis_labels(ax[i + 1], sampled_image)
            add_colorbar(min_intensity, max_intensity, fig, ax[i + 1])

    if show_diff:
        diff_max = max(-1 * min_intensity, max_intensity)
        diff_min = min(-1 * max_intensity, min_intensity)

        diff_image = create_colorscale_image(real - sampled, diff_max, diff_min)
        ax[2].imshow(diff_image, interpolation='nearest', aspect='auto')
        ax[2].set_title("Element-wise difference (real - sample)")
        add_axis_labels(ax[2], sampled_image)

        add_colorbar(diff_max, diff_min, fig, ax[2])

    if title is not None:
        plt.suptitle(title)

    fig.tight_layout()
    plt.savefig(savepath)
    plt.close(fig)

def create_colorscale_image(data: np.ndarray, max: float, min: float):
    data = np.copy(data).reshape((data.shape[0], data.shape[1], 1))

    data -= min
    data /= (max - min)

    # image = np.zeros((data.shape[0], data.shape[1], 3))
    image = data * YELLOW + (1 - data) * BLUE

    return image.astype(np.uint8)


def add_colorbar(min_intensity: float, max_intensity: float, fig, ax):
    norm = matplotlib.colors.Normalize(vmin=min_intensity, vmax=max_intensity)
    colors = np.linspace(0, 1, 256).reshape((256, 1))
    colors = (1 - colors) * BLUE.reshape((1, 3)) / 255 + colors * YELLOW.reshape((1, 3)) / 255
    cmap = matplotlib.colors.ListedColormap(colors)

    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)


def add_axis_labels(ax, image):
    ax.set_xlabel("Time Offset (s)")
    xtick_positions = np.linspace(0, image.shape[1] - 1, 5)
    xtick_values = DURATION_SEC/(image.shape[1] - 1)*xtick_positions
    xtick_values = np.round(xtick_values, decimals=2)
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_values)

    ax.set_ylabel("Frequency (Hz)")
    ytick_positions = np.linspace(0, image.shape[0] - 1, 6)
    ytick_values = MAX_FREQ_HZ/(image.shape[0] - 1)*ytick_positions
    ytick_values = np.round(ytick_values, decimals=2)
    ytick_values = np.flip(ytick_values, axis=0)
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(ytick_values)
