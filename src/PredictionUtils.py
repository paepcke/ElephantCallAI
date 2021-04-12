from typing import Tuple, Optional, Dict, Deque
import sys
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
import collections

# This file exists so we can access the batched prediction logic without having to import
# all of the dependencies in eval.py. It will be needed by the embedded logic, so please keep dependencies to a minimum.

import parameters


# these can be made configurable through argparse if necessary.
FREQS = 77
TIME_STEPS_IN_WINDOW = 256
BATCH_SIZE = 32

FIRST_STAGE_THRESHOLD = 0.5  # this is not configurable for now
FIRST_STAGE_NUM_INTERESTING = 15  # this is not configurable for now


def get_batched_predictions_and_overlap_counts(data: np.ndarray, model: nn.Module, jump: int,
                                               n_samples_in_time_window: int = TIME_STEPS_IN_WINDOW,
                                               batch_size: int = BATCH_SIZE, first_stage: Optional[bool] = None,
                                               second_stage_interest_queue: Optional[Deque[np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray]:
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
        forward_inference_on_batch(model, data, time_idx, jump, batch_size, predictions, overlap_counts, k, first_stage,
                                   second_stage_interest_queue)
        time_idx += n_samples_in_time_window * batch_size

    # final batch (if size < BATCH_SIZE)
    final_full_batch_size = (clean_end_time - time_idx - (k - 1) * jump) // n_samples_in_time_window
    if final_full_batch_size > 0:
        forward_inference_on_batch(model, data, time_idx, jump, final_full_batch_size, predictions,
                                   overlap_counts, k, first_stage, second_stage_interest_queue)
        time_idx += n_samples_in_time_window* final_full_batch_size

    # remaining jumps (less than k)
    if time_idx + n_samples_in_time_window <= clean_end_time:
        remaining_jumps = (clean_end_time - time_idx - n_samples_in_time_window) // jump + 1
        forward_inference_on_batch(model, data, time_idx, jump, 1, predictions, overlap_counts,
                                   remaining_jumps, first_stage, second_stage_interest_queue)

    return predictions, overlap_counts


def get_batched_predictions_and_overlap_counts_for_two_stage_model(
        data: np.ndarray, first_model: nn.Module,
        second_model: nn.Module, jump: int,
       n_samples_in_time_window: int = TIME_STEPS_IN_WINDOW,
       batch_size: int = BATCH_SIZE) -> Tuple[np.ndarray, np.ndarray]:
    second_stage_interest_queue = collections.deque()

    """
    Try to run *all* computations requiring first model before accessing the second model in
    an attempt to benefit from GPU caches
    """

    first_stage_predictions, first_stage_overlaps = get_batched_predictions_and_overlap_counts(
        data, first_model,
        jump, n_samples_in_time_window,
        batch_size, True, second_stage_interest_queue)

    second_stage_predictions, second_stage_overlaps = get_batched_predictions_and_overlap_counts(
        data, second_model,
        jump, n_samples_in_time_window,
        batch_size, False, second_stage_interest_queue)

    predictions = first_stage_predictions + second_stage_predictions
    overlap_counts = first_stage_overlaps + second_stage_overlaps

    return predictions, overlap_counts


def forward_inference_on_batch(
        model: nn.Module,
        data: np.ndarray,
        time_idx: int,
        jump: int,
        batchsize: int,
        predictions: np.ndarray,
        overlap_counts: np.ndarray,
        max_jumps: int,
        first_stage: Optional[bool] = None,
        second_stage_interest_queue: Optional[Deque[np.ndarray]] = None):
    """
    Run forward inference on a batch of examples. Supports 2-stage model paradigm.

    :param model: the ML model we use to generate predictions
    :param data: input data, of dims number of time segments, number of frequencies
    :param time_idx: the beginning time index for this inference
    :param jump: number of time steps between starts of frames to perform inference on (must be a clean divisor of TIME_STEPS_IN_WINDOW)
    :param batchsize: number of frames to perform inference on at once. Be careful not to exceed VRAM limits! This is going to be highly hardware-dependent.
    :param predictions: full ndarray of the sum of all predictions at each individual time step
    :param overlap_counts: full ndarray of the number of predictions applied to each invididual time step
    :param max_jumps: the number of different offsets that should be used. This method will process this many batches of input.
        If this number is 1, no 'jumps' will actually be evaluated, just the standard start of the array (the offset of 0).
    :param first_stage: if not None, run special processing for 2-stage model. 'True' indicates this invocation is for
        the first of the two model stages.
    :param second_stage_interest_queue: if not None, use this to filter things after running through second stage model
    :return: Nothing, output is written to the 2 param arrays 'predictions' and 'overlap_counts'
    """
    # select the region of the data to perform inference on
    input_batch = data[time_idx:(time_idx + batchsize * TIME_STEPS_IN_WINDOW + (max_jumps - 1) * jump), :]

    # one batch of spectrogram 'frames' (each representing a full input to the model) to be processed in parallel are consecutive and non-overlapping.
    # each iteration of this loop performs this approach with a different offset into the first frame, allowing the evaluation
    # of overlapping frames. Overlapping frames are NOT evaluated in parallel, but non-overlapping consecutive frames can be.
    for num_jumps_to_offset in range(0, max_jumps):
        # Will always return 'False' for a single-stage model
        if should_skip_batch(first_stage, second_stage_interest_queue):
            continue

        # Used for indexing into the input batch
        local_begin_idx = num_jumps_to_offset * jump
        local_end_idx = local_begin_idx + batchsize * TIME_STEPS_IN_WINDOW

        # Used for indexing into the prediction arrays
        global_begin_idx = local_begin_idx + time_idx
        global_end_idx = local_end_idx + time_idx

        reshaped_input_batch = input_batch[local_begin_idx:local_end_idx, :].reshape(batchsize, TIME_STEPS_IN_WINDOW,
                                                                                     FREQS)

        # apply per-frame normalization
        # TODO: figure out how to apply per-frame normalization on the GPU/device so we can get rid of this annoying perf hit
        # TODO: If we can do this, we won't have to copy the same data to the GPU/device 'max_jumps' times...
        means = np.mean(reshaped_input_batch, axis=(1, 2)).reshape(-1, 1, 1)
        stds = np.std(reshaped_input_batch, axis=(1, 2)).reshape(-1, 1, 1)

        reshaped_input_batch -= means
        reshaped_input_batch /= stds

        reshaped_input_batch_var = Variable(torch.from_numpy(reshaped_input_batch).float().to(parameters.device))

        raw_outputs = model(reshaped_input_batch_var)  # TODO: take subset of batches and re-expand?
        outputs = raw_outputs.view(batchsize, TIME_STEPS_IN_WINDOW)

        relevant_predictions = predictions[global_begin_idx:global_end_idx].reshape((batchsize, TIME_STEPS_IN_WINDOW))
        relevant_overlap_counts = overlap_counts[global_begin_idx:global_end_idx].reshape(
            (batchsize, TIME_STEPS_IN_WINDOW))

        prediction_outputs = outputs.cpu().detach().numpy()
        consolidate_inference_output(prediction_outputs, relevant_predictions,
                                     relevant_overlap_counts, first_stage,
                                     second_stage_interest_queue)

        relevant_predictions += outputs.cpu().detach().numpy()
        relevant_overlap_counts += 1

        # now we restore the segment of the input data
        reshaped_input_batch *= stds
        reshaped_input_batch += means


def consolidate_inference_output(
        prediction_outputs: np.ndarray,
        relevant_predictions: np.ndarray, relevant_overlap_counts: np.ndarray,
        first_stage: Optional[bool] = None,
        second_stage_interest_queue: Optional[Deque[np.ndarray]] = None):
    """
    Writes output of model to prediction arrays differently depending on the kind of model that is being used.

    :param prediction_outputs: What the model has just predicted for the relevant batch
    :param relevant_predictions: Array to write prediction output into.
    :param relevant_overlap_counts: Array to write overlap counts into.
    :param first_stage: Whether the first piece of a two-stage model is being used for inference
    :param second_stage_interest_queue: Must not be None if a two-stage model is being used for
        inference. It keeps track of which spectrogram frames should be passed to the second piece of the
        two-stage model.
    :return: No return value
    """

    if second_stage_interest_queue is not None:
        if first_stage:
            binarized_predictions = np.where(sigmoid(prediction_outputs) > FIRST_STAGE_THRESHOLD, 1, 0)
            prediction_sums_by_batch = np.sum(binarized_predictions, axis=1)
            use_batches_for_stage_2 = np.where(prediction_sums_by_batch >= FIRST_STAGE_NUM_INTERESTING, 1, 0).reshape(-1, 1)
            second_stage_interest_queue.append(use_batches_for_stage_2)

            use_batches_for_stage_1 = 1 - use_batches_for_stage_2
            prediction_outputs = prediction_outputs * use_batches_for_stage_1
            relevant_overlap_counts += use_batches_for_stage_1

        else:
            is_input_interesting = second_stage_interest_queue.popleft()
            prediction_outputs = prediction_outputs * is_input_interesting
            # TODO: Re-evaluate use of model1 predictions in outputs?
            relevant_overlap_counts += is_input_interesting
    else:
        relevant_overlap_counts += 1

    relevant_predictions += prediction_outputs


def should_skip_batch(first_stage: Optional[bool], second_stage_interest_queue: Deque[np.ndarray]) -> bool:
    """
    Used to encapsulate logic for skipping unnecessary invocations of a second-stage model to save on compute

    :param first_stage: None if using a single-stage model, otherwise, True if processing the first stage, else False
    :param second_stage_interest_queue: A queue of np arrays indicating whether the second model should be invoked for
        the corresponding examples in the batch
    :return: a bool indicating whether the invocation of the model should be skipped for this batch of examples
    """
    if first_stage is not None and not first_stage:
        batch_interest = second_stage_interest_queue[0]
        if np.sum(batch_interest) == 0:
            """
            If none of the examples are 'interesting', remove the indicator from the queue and return 'True'
            so that the invocation of the second stage model is skipped for this batch
            """
            second_stage_interest_queue.popleft()
            return True
    return False


def sigmoid(arg: np.ndarray):
        exp_arg = np.exp(arg)
        return exp_arg/(exp_arg + 1)