from typing import Optional, Deque, Tuple, List
import numpy as np
from datetime import datetime
from collections import deque
import sys, os, pickle
from threading import Lock

from embedded import FileUtils
from embedded.Closeable import Closeable
from embedded.IntervalRecorder import IntervalRecorder
from embedded.SpectrogramBuffer import SpectrogramBuffer, NUM_SAMPLES_IN_TIME_WINDOW
from embedded.PredictionBuffer import PredictionBuffer
from embedded.TransitionState import TransitionState, non_detected_transition_state
from embedded.predictors import Predictor

PREDICTION_THRESHOLD = 0.5
BYTES_PER_GIGABYTE = 1024*1024*1024


class DataCoordinator(Closeable):
    """An object that abstracts out the unsightly ring-buffer indexing used when storing spectrograms and predictions.
    Extracts intervals of continuously-detected positive predictions and saves these intervals to a file.

    This is thread-safe, with some conditions: No more than one thread should concurrently append data.
    No more than one thread should concurrently make predictions or consume unprocessed data.
    No more than one thread should concurrently consume data for post-processing or free space."""
    spectrogram_buffer: SpectrogramBuffer
    prediction_buffer: PredictionBuffer
    prediction_transition_state: TransitionState
    prediction_interval_recorder: IntervalRecorder
    blackout_interval_recorder: IntervalRecorder
    # This parameter determines the offset between evaluated overlapping spectrogram frames.
    # It is computed as 'min_appendable_time_steps - jump'.
    overlap_allowance: int
    prediction_load: float  # If predictions take up at least this proportion of the buffer, allow them to be collected
    min_free_space_for_input: int

    # A directory to save positively-predicted spectrograms to
    spectrogram_capture_dir: Optional[str]

    # The total number of gigabytes of disk space occupied by captured spectrograms saved by this process at some point.
    captured_disk_usage: float

    # A configurable number of gigabytes. If `captured_disk_usage` exceeds this, no more spectrograms will be captured.
    max_captured_disk_usage: float

    # If a prediction is greater than or equal to this value, it is classified as positive.
    prediction_threshold: float

    # Forbid the collection of fewer than this number of predictions. Useful for spectrogram capturing.
    # A value of '1' does not impose any restrictions.
    min_collectable_predictions: int

    # The minimum number of windows to process through a model simultaneously
    min_batch_size: int


    # Thread synchronization states

    # This will be unlocked if and only if there is enough available space in the buffer for input.
    # It allows an input-buffering thread to sleep until such time.
    space_available_for_input_lock: Lock

    # This will be unlocked if and only if there is enough available data in the buffer for a prediction to be made.
    # It allows a prediction-making thread to sleep until such time.
    data_available_for_prediction_lock: Lock

    # This will be unlocked if and only if there are predictions available for collection.
    # It allows a prediction-collecting thread to sleep until such time.
    predictions_available_for_collection_lock: Lock

    # A small piece of state that determines whether the data_coordinator thread that produces this data is expected to unlock the input lock
    _holding_input_lock: bool

    # A small piece of state that determines whether the data_coordinator thread that produces this data is expected to unlock the prediction lock
    _holding_prediction_lock: bool

    # A small piece of state that determines whether the data_coordinator thread that produces this data is expected to unlock the collection lock
    _holding_collection_lock: bool

    # manipulating the input lock inside the DataCoordinator logic must be done while holding this
    input_sync_lock: Lock

    # manipulating the prediction lock inside the DataCoordinator logic must be done while holding this
    prediction_sync_lock: Lock

    # manipulating the collection lock inside the DataCoordinator logic must be done while holding this
    collection_sync_lock: Lock

    def __init__(self, prediction_interval_output_path: str,
                 # If the data is being inserted faster than it can be processed, we may be forced to drop
                 # some incoming data. This creates a time discontinuity for which we cannot make predictions.
                 # This parameter specifies a path to a txt file where those intervals of 'blackout' are recorded.
                 blackout_interval_output_path: str,
                 # The number of seconds a single time step of a spectrogram frame represents
                 # (the length of the non-overlapping segments of each time window used in the STFT)
                 time_delta_per_time_step: Optional[float] = None,
                 jump: Optional[int] = None, override_buffer_size: Optional[int] = None,
                 # We assume that 'min_appendable_time_steps' is sufficient to make a prediction.
                 min_appendable_time_steps: Optional[int] = None,
                 # The minimum number of windows that can be processed through a model simultaneously
                 min_batch_size: int = 1,
                 # Do not accept input if fewer than this number of rows are unallocated
                 min_free_space_for_input: Optional[int] = None,
                 # Predictions will not be collected unless at least the following proportion of the prediction buffer is full
                 prediction_load: float = 0.005,
                 # A directory to save positively-predicted spectrograms to. The spectrograms are not saved if this is not specified.
                 spectrogram_capture_dir: Optional[str] = None,
                 # prediction outputs for each time step are between 0 and 1. If any are greater than or equal to
                 # this threshold, those will be classified as positive.
                 prediction_threshold: float = PREDICTION_THRESHOLD,
                 # Forbid the collection of fewer than this number of predictions. Useful for spectrogram capturing.
                 min_collectable_predictions: int = NUM_SAMPLES_IN_TIME_WINDOW,
                 # A configurable number of gigabytes. If `captured_disk_usage` exceeds this, no more spectrograms will be captured.
                 max_captured_disk_usage: float = 1):
        """
        DataCoordinator constructor

        :param prediction_interval_output_path: path to the file where prediction intervals are saved
        :param blackout_interval_output_path: If the data is being inserted faster than it can be processed, we may be
            forced to drop some incoming data. This creates a time discontinuity for which we cannot make predictions.
            This parameter specifies a path to a txt file where those intervals of 'blackout' are recorded.
        :param time_delta_per_time_step: The number of seconds a single time step of a spectrogram frame represents
            (the length of the non-overlapping segments of each time window used in the STFT)
        :param jump:
        :param override_buffer_size:
        :param min_appendable_time_steps: We assume that 'min_appendable_time_steps' is sufficient to make a prediction
        :param min_free_space_for_input: Do not accept input if fewer than this number of rows are unallocated
        :param prediction_load: Predictions will not be collected unless at least the following proportion of the
            prediction buffer is full
        :param spectrogram_capture_dir: A directory to save positively-predicted spectrograms to. The spectrograms are
            not saved if this is not specified.
        :param prediction_threshold: prediction outputs for each time step are between 0 and 1. If any are greater than
            or equal to this threshold, those will be classified as positive.
        :param min_collectable_predictions: Forbid the collection of fewer than this number of predictions.
            Useful for spectrogram capturing.
        :param max_captured_disk_usage: A configurable number of gigabytes. If `captured_disk_usage` exceeds this,
            no more spectrograms will be captured.
        """
        super().__init__()
        self.prediction_interval_recorder = IntervalRecorder(prediction_interval_output_path)
        self.spectrogram_buffer = SpectrogramBuffer(override_buffer_size=override_buffer_size,
                                                    min_appendable_time_steps=min_appendable_time_steps,
                                                    time_delta_per_time_step=time_delta_per_time_step)
        self.prediction_buffer = PredictionBuffer(self.spectrogram_buffer.buffer.shape[0])
        self.prediction_transition_state = non_detected_transition_state()
        self.prediction_threshold = prediction_threshold
        self.min_collectable_predictions = max(1, min_collectable_predictions)

        self.blackout_interval_recorder = IntervalRecorder(blackout_interval_output_path)

        self.spectrogram_capture_dir = spectrogram_capture_dir
        self._assert_spectrogram_capture_dir()
        self.captured_disk_usage = 0.
        self.max_captured_disk_usage = max_captured_disk_usage

        self.min_batch_size = min_batch_size

        self.min_free_space_for_input = min_free_space_for_input
        if min_free_space_for_input is None or self.min_free_space_for_input < self.spectrogram_buffer.min_appendable_time_steps:
            self.min_free_space_for_input = self.spectrogram_buffer.min_appendable_time_steps

        if jump is None or jump == 0:
            print("WARNING: DataCoordinator overlap_allowance parameter is 0. There will not be any overlap between "
                  "evaluated spectrogram frames. Please make sure this is intentional.", file=sys.stderr)
            self.overlap_allowance = 0
        else:
            if jump > self.spectrogram_buffer.min_appendable_time_steps:
                raise ValueError("jump must be less than or equal to min appendable time steps")
            elif self.spectrogram_buffer.min_appendable_time_steps % jump != 0:
                raise ValueError("jump must evenly divide min appendable time steps")
            self.overlap_allowance = self.spectrogram_buffer.min_appendable_time_steps - jump

        if prediction_load < 0 or prediction_load >= 1:
            raise ValueError("Prediction load must be a value between 0 and 1, excluding 1.")
        self.prediction_load = prediction_load

        # Initialize thread synchronization resources
        self.space_available_for_input_lock = Lock()
        self.data_available_for_prediction_lock = Lock()
        self.predictions_available_for_collection_lock = Lock()
        self.data_available_for_prediction_lock.acquire()
        self.predictions_available_for_collection_lock.acquire()
        self._holding_input_lock = False
        self._holding_prediction_lock = True
        self._holding_collection_lock = True
        self.input_sync_lock = Lock()
        self.prediction_sync_lock = Lock()
        self.collection_sync_lock = Lock()

    # Appends new spectrogram data to the spectrogram buffer
    def write(self, spectrogram: np.ndarray, timestamp: Optional[datetime] = None):
        most_recent_timestamp_entry = None
        prev_unprocessed_end = None
        if timestamp is not None:
            with self.spectrogram_buffer.metadata_mutex:
                if self.spectrogram_buffer.rows_unprocessed > 0:
                    with self.spectrogram_buffer.timestamp_mutex:
                        most_recent_timestamp_entry = self.spectrogram_buffer.unprocessed_timestamp_deque[-1]
                        prev_unprocessed_end = self.spectrogram_buffer.unprocessed_end

        jump = self.spectrogram_buffer.min_appendable_time_steps - self.overlap_allowance
        if spectrogram.shape[0] % jump != 0:
            raise ValueError("Must append a number of time steps equal to an integer multiple of jump size")
        self.spectrogram_buffer.append_data(spectrogram, time=timestamp)

        # Update thread synchronization states
        self.update_input_lock()
        self.update_prediction_lock()

        if most_recent_timestamp_entry is not None:
            prev_continuity_len =\
                self.spectrogram_buffer.circular_distance(prev_unprocessed_end, most_recent_timestamp_entry[0])
            blackout_start = most_recent_timestamp_entry[1] + self.spectrogram_buffer.time_delta_per_time_step * prev_continuity_len
            blackout_end = timestamp
            self.blackout_interval_recorder.write_interval((blackout_start, blackout_end))


    # returns number of rows for which a prediction has been made, useful for backoff logic
    def make_predictions(self, predictor: Predictor, time_window: int) -> int:
        metadata_snapshot = self.spectrogram_buffer.get_metadata_snapshot()
        if metadata_snapshot.rows_unprocessed < self.spectrogram_buffer.min_appendable_time_steps:
            # We can't process enough data yet
            return 0

        if self.overlap_allowance >= time_window:
            raise ValueError("Time window must exceed overlap allowance")

        jump = self.spectrogram_buffer.min_appendable_time_steps - self.overlap_allowance
        if self.overlap_allowance != 0:
            if time_window % jump != 0:
                raise ValueError("Time window must be evenly divisible by jump")
        else:
            if time_window % self.spectrogram_buffer.min_appendable_time_steps:
                raise ValueError(f"With no overlap, time window must be an integer multiple of {self.spectrogram_buffer.min_appendable_time_steps}")

        if time_window < self.spectrogram_buffer.min_appendable_time_steps:
            raise ValueError(f"Predictions must be made on at least {self.spectrogram_buffer.min_appendable_time_steps}"
                             + " rows of contiguous spectrogram data")

        if time_window > metadata_snapshot.rows_unprocessed:
            time_window = metadata_snapshot.rows_unprocessed - metadata_snapshot.rows_unprocessed % jump
            if time_window == 0:
                return 0

        # if 'enough' data would be left for one time seq window, allow it. Else, if NO data from this time seq would be left (there is a follow-up timestamp), allow it.
        # else, disallow it.
        found_follow_up_timestamp: bool
        with self.spectrogram_buffer.timestamp_mutex:
            unprocessed_timestamp_deque = self.spectrogram_buffer.unprocessed_timestamp_deque
            if len(unprocessed_timestamp_deque) < 2:
                found_follow_up_timestamp = False
            else:
                found_follow_up_timestamp = True
                first_timestamp = unprocessed_timestamp_deque[0]
                second_timestamp = unprocessed_timestamp_deque[1]

        if not found_follow_up_timestamp:
            if (self.spectrogram_buffer.rows_unprocessed - time_window + self.overlap_allowance) >= self.spectrogram_buffer.min_appendable_time_steps:
                actual_time_window = time_window
            else:
                candidate_time_window = self.spectrogram_buffer.rows_unprocessed - self.spectrogram_buffer.min_appendable_time_steps + self.overlap_allowance
                candidate_time_window -= (candidate_time_window%jump)
                if candidate_time_window >= self.spectrogram_buffer.min_appendable_time_steps:
                    actual_time_window = candidate_time_window
                else:
                    # We can't risk having 'hanging' data that isn't large enough to process all at once
                    return 0
            # process the first time_window samples
            spect_data, begin_idx = self.spectrogram_buffer.get_unprocessed_data(time_steps=actual_time_window,
                                                                                 mark_for_postprocessing=False)
            predictions, overlap_counts = predictor.make_predictions(spect_data)
            self.prediction_buffer.write(begin_idx, predictions, overlap_counts)
            self.spectrogram_buffer.mark_for_post_processing(time_steps=actual_time_window - self.overlap_allowance)

            num_predictions = predictions.shape[0]
        else:
            time_dist = self.spectrogram_buffer.circular_distance(second_timestamp[0], first_timestamp[0])
            if (time_dist - time_window + self.overlap_allowance) >= self.spectrogram_buffer.min_appendable_time_steps:
                # process the first time_window samples, there is enough data for there to be another full min_appendable_time_steps after this
                spect_data, begin_idx = self.spectrogram_buffer.get_unprocessed_data(time_steps=time_window,
                                                                                     mark_for_postprocessing=False)
                predictions, overlap_counts = predictor.make_predictions(spect_data)
                self.prediction_buffer.write(begin_idx, predictions, overlap_counts)
                self.spectrogram_buffer.mark_for_post_processing(time_steps=time_window - self.overlap_allowance)
                num_predictions = predictions.shape[0]
            else:
                # process all of this remaining data
                spect_data, begin_idx = self.spectrogram_buffer.get_unprocessed_data(time_steps=time_dist,
                                                                                     mark_for_postprocessing=False)
                predictions, overlap_counts = predictor.make_predictions(spect_data)
                self.prediction_buffer.write(begin_idx, predictions, overlap_counts)
                self.spectrogram_buffer.mark_for_post_processing(time_steps=time_dist)
                num_predictions = predictions.shape[0]

        # Update thread synchronization states
        self.update_prediction_lock()
        self.update_collection_lock()

        return num_predictions

    # returns a tuple of:
    # 1. number of rows for which predictions were finalized
    # 2. finalized predictions
    def finalize_predictions(self, time_window: int) -> Tuple[int, Optional[np.ndarray]]:
        metadata_snapshot = self.spectrogram_buffer.get_metadata_snapshot()
        rows_pending_post_proc = metadata_snapshot.rows_allocated - metadata_snapshot.rows_unprocessed
        if rows_pending_post_proc == 0 or time_window <= 0:
            # There's nothing to finalize here
            return 0, None
        if time_window > rows_pending_post_proc:
            time_window = rows_pending_post_proc
        begin_idx = metadata_snapshot.allocated_begin
        end_idx = (begin_idx + time_window)%self.spectrogram_buffer.buffer.shape[0]
        final_predictions = self.prediction_buffer.finalize(begin_idx, end_idx)

        # copy elements off the timestamp deque until getting to one that is beyond the end_idx and before the begin idx or the other way around
        timestamps = deque()
        with self.spectrogram_buffer.timestamp_mutex:
            buffer_timestamps = self.spectrogram_buffer.post_processing_timestamp_deque
            buf_timestamp = buffer_timestamps[0]
            start_timestamp = buffer_timestamps[0][1]
            buf_idx = 0
            buf_timestamp_dist = 0
            while buf_timestamp is not None and buf_timestamp_dist < time_window:
                timestamps.append((buf_timestamp_dist, buf_timestamp[1]))
                buf_idx += 1
                if len(buffer_timestamps) >= buf_idx:
                    buf_timestamp = None
                else:
                    buf_timestamp = buffer_timestamps[buf_idx]
                    buf_timestamp_dist = self.spectrogram_buffer.circular_distance(end_idx, buf_timestamp[0])

        # create time intervals for detection events, save them to a file
        intervals, found_discontinuity = self.get_detection_intervals(final_predictions, timestamps)

        # Capture spectrogram only if capture_dir is configured and there is a prediction event in `final_predictions`
        if self.spectrogram_capture_dir is not None \
                and (len(intervals) > 0 or self.prediction_transition_state.is_detected())\
                and self.captured_disk_usage < self.max_captured_disk_usage:

            '''
            In the case of a time discontinuity, do not capture a spectrogram.
            The expectation is that time discontinuities will be rare and additional logic to
            handle this is probably not worth the complexity it adds to the capture process.
            '''
            if not found_discontinuity:
                # spectrogram data for use in capturing
                spectrogram, _ = self.spectrogram_buffer.consume_data(len(final_predictions), True)
                end_timestamp = start_timestamp + len(final_predictions) * self.spectrogram_buffer.time_delta_per_time_step
                file_size_gb = self._capture_spectrogram((start_timestamp, end_timestamp), spectrogram=spectrogram,
                                          predictions=final_predictions)
                self.captured_disk_usage += file_size_gb
                if self.captured_disk_usage >= self.max_captured_disk_usage:
                    print("Disk usage limit for spectrogram capture reached, no more spectrograms will be captured.",
                          file=sys.stderr)
            else:
                print("Omitting spectrogram capture because of time discontinuity in data - check blackout intervals " +
                      "file to see if your device is struggling to process data quickly enough", file=sys.stderr)

        # free the underlying memory in the spectrogrambuffer
        self.spectrogram_buffer.mark_post_processing_complete(time_window)

        # update thread synchronization states
        self.update_input_lock()
        self.update_collection_lock()

        # save detection intervals to interval file
        self.save_detection_intervals(intervals)

        return time_window, final_predictions

    # Start with the left-over transition state and a deque of timestamps, return a list of event intervals and a boolean
    # specifying whether there was a time discontinuity in this interval of predictions.
    # If a spectrogram_capture_dir is defined, this method will also save positively-classified spectrograms to a file.
    def get_detection_intervals(self, finalized_predictions: np.ndarray, timestamps: Deque[Tuple[int, datetime]]) ->\
            Tuple[List[Tuple[datetime, datetime]], bool]:
        intervals = list()
        found_discontinuity = False
        cur_timestamp = timestamps.popleft()

        if self.prediction_transition_state.is_detected():
            tentative_interval_end = self.prediction_transition_state.start_time +\
                                     self.prediction_transition_state.num_consecutive_ones * self.spectrogram_buffer.time_delta_per_time_step
            if tentative_interval_end != cur_timestamp[1]:
                # This is a time discontinuity
                found_discontinuity = True
                interval = (self.prediction_transition_state.start_time, tentative_interval_end)
                intervals.append(interval)
                self.prediction_transition_state = non_detected_transition_state()

        for i in range(0, finalized_predictions.shape[0]):
            if len(timestamps) > 0 and timestamps[0][0] <= i:
                prev_timestamp = cur_timestamp
                cur_timestamp = timestamps.popleft()
                if cur_timestamp[1] != ((cur_timestamp[0] - prev_timestamp[0]) * self.spectrogram_buffer.time_delta_per_time_step + prev_timestamp[1]):
                    # This is a time discontinuity. We can only detect call events that cross this boundary as two separate events; one on each side of the discontinuity.
                    found_discontinuity = True
                    if self.prediction_transition_state.is_detected():
                        interval = (self.prediction_transition_state.start_time,
                                    self.prediction_transition_state.start_time +
                                    self.prediction_transition_state.num_consecutive_ones * self.spectrogram_buffer.time_delta_per_time_step)
                        intervals.append(interval)
                        self.prediction_transition_state = non_detected_transition_state()

            # handle next time increment of prediction
            if self.prediction_transition_state.is_detected():
                if finalized_predictions[i] >= self.prediction_threshold:
                    # expand the detected event by one time step
                    self.prediction_transition_state.num_consecutive_ones += 1
                else:
                    # finalize the detected event
                    interval = (self.prediction_transition_state.start_time,
                                self.prediction_transition_state.start_time +
                                self.prediction_transition_state.num_consecutive_ones * self.spectrogram_buffer.time_delta_per_time_step)
                    intervals.append(interval)
                    self.prediction_transition_state = non_detected_transition_state()
            elif finalized_predictions[i] >= self.prediction_threshold:
                # begin a new detection event
                start_time = cur_timestamp[1] + (i - cur_timestamp[0]) * self.spectrogram_buffer.time_delta_per_time_step
                self.prediction_transition_state = TransitionState(start_time, 1)

        return intervals, found_discontinuity

    # Returns size of created file in GB.
    def _capture_spectrogram(self, time_bounds: Tuple[datetime, datetime],
                             spectrogram: np.ndarray, predictions: np.ndarray) -> float:
        interval_begin_time_formatted = time_bounds[0].strftime("%Y-%m-%d-%H-%M-%S.%f")
        interval_end_time_formatted = time_bounds[1].strftime("%Y-%m-%d-%H-%M-%S.%f")
        filename = f"{interval_begin_time_formatted}_to_{interval_end_time_formatted}.pkl"
        filepath = self.spectrogram_capture_dir + "/" + filename
        arrays_of_interest = {"spectrogram": spectrogram, "predictions": predictions}
        with open(filepath, 'wb') as file:
            pickle.dump(arrays_of_interest, file)

        return os.stat(filepath).st_size/BYTES_PER_GIGABYTE

    def _assert_spectrogram_capture_dir(self):
        if self.spectrogram_capture_dir is None:
            return
        FileUtils.assert_is_directory(self.spectrogram_capture_dir,
                                      "The spectrogram capture directory specified does not exist!")

    def update_input_lock(self):
        with self.input_sync_lock:
            metadata_snapshot = self.spectrogram_buffer.get_metadata_snapshot()
            available_space = self.spectrogram_buffer.buffer.shape[0] - metadata_snapshot.rows_allocated
            if available_space >= self.min_free_space_for_input:
                # unlock it if possible
                if self._holding_input_lock:
                    self._holding_input_lock = False
                    self.space_available_for_input_lock.release()
            else:
                # make sure it's locked
                if not self._holding_input_lock:
                    self.space_available_for_input_lock.acquire()
                    self._holding_input_lock = True

    def update_prediction_lock(self):
        with self.prediction_sync_lock:
            metadata_snapshot = self.spectrogram_buffer.get_metadata_snapshot()
            with self.spectrogram_buffer.timestamp_mutex:
                unprocessed_discontinuities = len(self.spectrogram_buffer.unprocessed_timestamp_deque)
            if unprocessed_discontinuities >= 2:
                # unlock it if possible
                if self._holding_prediction_lock:
                    self._holding_prediction_lock = False
                    self.data_available_for_prediction_lock.release()
            else:
                jump = self.spectrogram_buffer.min_appendable_time_steps - self.overlap_allowance
                if metadata_snapshot.rows_unprocessed >= (self.min_batch_size * self.spectrogram_buffer.min_appendable_time_steps + jump):
                    # unlock it if possible
                    if self._holding_prediction_lock:
                        self._holding_prediction_lock = False
                        self.data_available_for_prediction_lock.release()
                else:
                    # make sure it's locked
                    if not self._holding_prediction_lock:
                        self.data_available_for_prediction_lock.acquire()
                        self._holding_prediction_lock = True

    def update_collection_lock(self):
        with self.collection_sync_lock:
            metadata_snapshot = self.spectrogram_buffer.get_metadata_snapshot()
            num_predictions_threshold = self.spectrogram_buffer.buffer.shape[0] * self.prediction_load
            pending_predictions = metadata_snapshot.rows_allocated - metadata_snapshot.rows_unprocessed
            # 1. If there's some configurably large amount of pending predictions, unlock it
            if pending_predictions >= num_predictions_threshold and pending_predictions > 0:
                # unlock it if possible
                if self._holding_collection_lock:
                    self._holding_collection_lock = False
                    self.predictions_available_for_collection_lock.release()
                return
            # 2. If there is very little unprocessed data in the buffer, we're probably just finishing off the data we have, so unlock it
            if metadata_snapshot.rows_unprocessed <= 2*self.spectrogram_buffer.min_appendable_time_steps \
                    and pending_predictions >= self.min_collectable_predictions:
                if self._holding_collection_lock:
                    self._holding_collection_lock = False
                    self.predictions_available_for_collection_lock.release()
            # 3. Else, make sure it's locked
            else:
                if not self._holding_collection_lock:
                    self.predictions_available_for_collection_lock.acquire()
                    self._holding_collection_lock = True

    # appends detection intervals to a file
    def save_detection_intervals(self, intervals: List[Tuple[datetime, datetime]]):
        for interval in intervals:
            self.prediction_interval_recorder.write_interval(interval)

    def close(self):
        self.prediction_interval_recorder.close()
        self.blackout_interval_recorder.close()
