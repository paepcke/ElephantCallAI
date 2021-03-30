from typing import Tuple
import sys
import numpy as np
import torch
from torch.autograd import Variable

# This file exists so we can access the batched prediction logic without having to import
# all of the dependencies in eval.py. It will be needed by the embedded logic, so please keep dependencies to a minimum.

import parameters


# these can be made configurable through argparse if necessary.
FREQS = 77
TIME_STEPS_IN_WINDOW = 256
BATCH_SIZE = 32


def get_batched_predictions_and_overlap_counts(data, model, jump,
                                               n_samples_in_time_window: int = TIME_STEPS_IN_WINDOW,
                                               batch_size: int = BATCH_SIZE) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (prediction array, overlap count array)."""
    time_idx = 0

    # key assumption: TIME_STEPS_IN_WINDOW is evenly divisible by 'jump'
    assert (n_samples_in_time_window % jump == 0)
    if jump == 0:
        k = 1
    else:
        k = n_samples_in_time_window // jump

    # cut off data at end to allow for even divisibility
    raw_end_time = data.shape[0]
    clean_end_time = raw_end_time - (raw_end_time % jump)

    if clean_end_time != raw_end_time:
        print("WARNING: {} time steps were cut off for the sake of even divisibility".format(
            raw_end_time - clean_end_time), file=sys.stderr)

    predictions = np.zeros(clean_end_time)
    overlap_counts = np.zeros(clean_end_time)

    while time_idx + n_samples_in_time_window * batch_size + (k - 1) * jump <= clean_end_time:
        forward_inference_on_batch(model, data, time_idx, jump, batch_size, predictions, overlap_counts, k)
        time_idx += n_samples_in_time_window * batch_size

    # final batch (if size < BATCH_SIZE)
    final_full_batch_size = (clean_end_time - time_idx - (k - 1) * jump) // n_samples_in_time_window
    if final_full_batch_size > 0:
        forward_inference_on_batch(model, data, time_idx, jump, final_full_batch_size, predictions,
                                   overlap_counts, k)
        time_idx += n_samples_in_time_window* final_full_batch_size

    # remaining jumps (less than k)
    if time_idx + n_samples_in_time_window <= clean_end_time:
        remaining_jumps = (clean_end_time - time_idx - n_samples_in_time_window) // jump + 1
        forward_inference_on_batch(model, data, time_idx, jump, 1, predictions, overlap_counts,
                                   remaining_jumps)

    return predictions, overlap_counts


def forward_inference_on_batch(
        model,  # the ML model we use to generate predictions
        data,  # input data, of dims number of time segments, number of frequencies
        time_idx,  # the beginning time index for this inference
        jump,  # number of time steps between starts of frames to perform inference on (must be a clean divisor of TIME_STEPS_IN_WINDOW)
        batchsize,  # number of frames to perform inference on at once. Be careful not to exceed VRAM limits! This is going to be highly hardware-dependent.
        predictions,  # full ndarray of the sum of all predictions at each individual time step
        overlap_counts,  # full ndarray of the number of predictions applied to each invididual time step

        # the number of different offsets that should be used. This method will process this many batches of input.
        # If this number is 1, no 'jumps' will actually be evaluated, just the standard start of the array (the offset of 0).
        max_jumps):

    # select the region of the data to perform inference on
    input_batch = data[time_idx:(time_idx + batchsize * TIME_STEPS_IN_WINDOW + (max_jumps - 1) * jump), :]

    # one batch of spectrogram 'frames' (each representing a full input to the model) to be processed in parallel are consecutive and non-overlapping.
    # each iteration of this loop performs this approach with a different offset into the first frame, allowing the evaluation
    # of overlapping frames. Overlapping frames are NOT evaluated in parallel, but non-overlapping consecutive frames can be.
    for num_jumps_to_offset in range(0, max_jumps):
        # Used for indexing into the input batch
        local_begin_idx = num_jumps_to_offset * jump
        local_end_idx = local_begin_idx + batchsize * TIME_STEPS_IN_WINDOW

        # Used for indexing into the prediction arrays
        global_begin_idx = local_begin_idx + time_idx
        global_end_idx = local_end_idx + time_idx

        reshaped_input_batch = input_batch[local_begin_idx:local_end_idx, :].reshape(batchsize, TIME_STEPS_IN_WINDOW, FREQS)

        # apply per-frame normalization
        # TODO: figure out how to apply per-frame normalization on the GPU/device so we can get rid of this annoying perf hit
        # TODO: If we can do this, we won't have to copy the same data to the GPU/device 'max_jumps' times...
        means = np.mean(reshaped_input_batch, axis=(1,2)).reshape(-1, 1, 1)
        stds = np.std(reshaped_input_batch, axis=(1,2)).reshape(-1, 1, 1)

        reshaped_input_batch -= means
        reshaped_input_batch /= stds

        reshaped_input_batch_var = Variable(torch.from_numpy(reshaped_input_batch).float().to(parameters.device))

        raw_outputs = model(reshaped_input_batch_var)
        outputs = raw_outputs.view(batchsize, TIME_STEPS_IN_WINDOW)

        relevant_predictions = predictions[global_begin_idx:global_end_idx].reshape((batchsize, TIME_STEPS_IN_WINDOW))
        relevant_overlap_counts = overlap_counts[global_begin_idx:global_end_idx].reshape((batchsize, TIME_STEPS_IN_WINDOW))

        relevant_predictions += outputs.cpu().detach().numpy()
        relevant_overlap_counts += 1

        # now we restore the segment of the input data
        reshaped_input_batch *= stds
        reshaped_input_batch += means