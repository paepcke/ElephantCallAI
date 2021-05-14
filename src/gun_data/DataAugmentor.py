import random
import numpy as np

MAX_REASONABLE_OCCLUSION = 0.6  # past this point, it's not realistic to expect the network to work well
AUG_TYPES = ["freq_mask", "time_mask", "noise"]

DEFAULT_STD = 0.9  # noise should have a stddev of 0.9 times the stddev across the training set (which is 1 after normalization)

class DataAugmentor():
    """
    This object holds data augmentation settings and has a simple interface for probabilistic data augmentation.
    """
    max_freq_occlusion: int
    max_time_occlusion: int
    aug_prob: float

    def __init__(self,
                 max_freq_occlusion: int,
                 max_time_occlusion: int,
                 aug_prob: float):
        self.max_freq_occlusion = max_freq_occlusion
        self.max_time_occlusion = max_time_occlusion
        self.aug_prob = aug_prob

    def augment(self, spec: np.ndarray) -> np.ndarray:
        # TODO: based on some settings like "mix multiple augmentations" and "augmentation probability",
        # augment an input spectrogram. Then, hook this up to the Dataset/DataLoader
        if np.random.rand() <= self.aug_prob:
            # for now, each type of augmentation is mutually exclusive and occurs with the same probability, 1/3
            aug_idx = np.random.randint(low=0, high=3)
            aug_type = AUG_TYPES[aug_idx]

            if aug_type == "freq_mask":
                return self.augment_with_freq_mask(spec)
            elif aug_type == "time_mask":
                return self.augment_with_time_mask(spec)
            else:
                # aug type == noise
                return self.augment_with_noise(spec, DEFAULT_STD)
        else:
            return spec

    # spectrogram dimension are (time, freq)
    def augment_with_freq_mask(self, spec: np.ndarray) -> np.ndarray:
        return self.augment_with_mask(spec, self.max_freq_occlusion)

    def augment_with_time_mask(self, spec: np.ndarray) -> np.ndarray:
        return self.augment_with_mask(spec.T, self.max_time_occlusion).T

    def augment_with_mask(self, spec: np.ndarray, max_height: int) -> np.ndarray:
        max_height = min(max_height, int(spec.shape[1]*MAX_REASONABLE_OCCLUSION))
        height = random.randrange(1, max_height)
        offset = random.randrange(0, spec.shape[1] - height)
        spec_copy = spec.copy()
        spec_mean = np.mean(spec_copy)

        spec_copy[:, offset:(offset + height)] = spec_mean
        return spec_copy

    def augment_with_noise(self, spec: np.ndarray, noise_std: float) -> np.ndarray:
        spec_copy = spec.copy()

        spec_copy += np.random.normal(loc=0.0, scale=noise_std, size=spec_copy.shape)
        return spec_copy

    def augment_with_random_volume_changes(self, spec: np.ndarray):
        # TODO: can we implement this from the spectrogram side of things?
        raise NotImplementedError()