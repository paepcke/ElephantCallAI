from typing import Optional, Deque, Tuple, List
import numpy as np
from datetime import datetime
from collections import deque
import sys, os
from threading import Lock

from embedded.Closeable import Closeable
from embedded.IntervalRecorder import IntervalRecorder
from embedded.SpectrogramBuffer import SpectrogramBuffer, TIME_DELTA_PER_TIME_STEP
from embedded.PredictionBuffer import PredictionBuffer
from embedded.TransitionState import TransitionState, non_detected_transition_state
from embedded.predictors import Predictor

PREDICTION_THRESHOLD = 0.5


class DataCoordinator(Closeable):
    """An object that abstracts out the unsightly ring-buffer indexing used when storing spectrograms and predictions.
    Extracts intervals of continuously-detected positive predictions and saves these intervals to a file."""
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

    # A piece of state that corresponds to a potentially unfinished positively-detected interval.
    current_spectrogram_capture: Optional[np.ndarray]

    # If a prediction is greater than or equal to this value, it is classified as positive.
    prediction_threshold: float


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
                 jump: Optional[int] = None, override_buffer_size: Optional[int] = None,
                 # We assume that 'min_appendable_time_steps' is sufficient to make a prediction.
                 min_appendable_time_steps: Optional[int] = None,
                 # Do not accept input if fewer than this number of rows are unallocated
                 min_free_space_for_input: Optional[int] = None,
                 # Predictions will not be collected unless at least the following proportion of the prediction buffer is full
                 prediction_load: float = 0.005,
                 # A directory to save positively-predicted spectrograms to. The spectrograms are not saved if this is not specified.
                 spectrogram_capture_dir: Optional[str] = None,
                 # prediction outputs for each time step are between 0 and 1. If any are greater than or equal to
                 # this threshold, those will be classified as positive.
                 prediction_threshold: float = PREDICTION_THRESHOLD):
        super().__init__()
        self.prediction_interval_recorder = IntervalRecorder(prediction_interval_output_path)
        self.spectrogram_buffer = SpectrogramBuffer(override_buffer_size=override_buffer_size,
                                                    min_appendable_time_steps=min_appendable_time_steps)
        self.prediction_buffer = PredictionBuffer(self.spectrogram_buffer.buffer.shape[0])
        self.prediction_transition_state = non_detected_transition_state()
        self.prediction_threshold = prediction_threshold

        self.blackout_interval_recorder = IntervalRecorder(blackout_interval_output_path)

        self.spectrogram_capture_dir = spectrogram_capture_dir
        self.current_spectrogram_capture = None
        self._assert_spectrogram_capture_dir()

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
            blackout_start = most_recent_timestamp_entry[1] + TIME_DELTA_PER_TIME_STEP*prev_continuity_len
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
                raise ValueError("With no overlap, time window must be an integer multiple of {}".format(self.spectrogram_buffer.min_appendable_time_steps))

        if time_window < self.spectrogram_buffer.min_appendable_time_steps:
            raise ValueError("Predictions must be made on at least {} rows of contiguous spectrogram data"
                             .format(self.spectrogram_buffer.min_appendable_time_steps))

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

        spect_data = None
        if self.spectrogram_capture_dir is not None:
            # spectrogram data for use in capturing
            spect_data, _ = self.spectrogram_buffer.consume_data(len(final_predictions), True)

        # create time intervals for detection events, save them to a file
        intervals = self.get_detection_intervals(final_predictions, timestamps, spect_data)

        # free the underlying memory in the spectrogrambuffer
        self.spectrogram_buffer.mark_post_processing_complete(time_window)

        # update thread synchronization states
        self.update_input_lock()
        self.update_collection_lock()

        # save detection intervals to interval file
        self.save_detection_intervals(intervals)

        return time_window, final_predictions

    # Start with the left-over transition state and a deque of timestamps, return a list of event intervals
    # If a spectrogram_capture_dir is defined, this method will also save positively-classified spectrograms to a file.
    def get_detection_intervals(self, finalized_predictions: np.ndarray, timestamps: Deque[Tuple[int, datetime]],
                                spect_data: Optional[np.ndarray]) -> List[Tuple[datetime, datetime]]:
        intervals = list()
        cur_timestamp = timestamps.popleft()
        extending_current_capture = False
        detect_start_idx = 0

        if self.prediction_transition_state.is_detected():
            tentative_interval_end = self.prediction_transition_state.start_time + self.prediction_transition_state.num_consecutive_ones * TIME_DELTA_PER_TIME_STEP
            if tentative_interval_end != cur_timestamp[1]:
                interval = (self.prediction_transition_state.start_time, tentative_interval_end)
                intervals.append(interval)
                if self.spectrogram_capture_dir is not None:
                    self._capture_spectrogram(interval, ignore_spectrogram=True)
                self.prediction_transition_state = non_detected_transition_state()
            elif self.spectrogram_capture_dir is not None:
                extending_current_capture = True

        for i in range(0, finalized_predictions.shape[0]):
            if len(timestamps) > 0 and timestamps[0][0] <= i:
                prev_timestamp = cur_timestamp
                cur_timestamp = timestamps.popleft()
                if cur_timestamp[1] != ((cur_timestamp[0] - prev_timestamp[0])*TIME_DELTA_PER_TIME_STEP + prev_timestamp[1]):
                    # This is a time discontinuity. We can only detect call events that cross this boundary as two separate events; one on each side of the discontinuity.
                    if self.prediction_transition_state.is_detected():
                        interval = (self.prediction_transition_state.start_time,
                                          self.prediction_transition_state.start_time +
                                    self.prediction_transition_state.num_consecutive_ones * TIME_DELTA_PER_TIME_STEP)
                        intervals.append(interval)
                        self._capture_spectrogram(interval, begin_idx=detect_start_idx, end_idx=i, spectrogram=spect_data,
                                                  extending_current_capture=extending_current_capture)
                        self.prediction_transition_state = non_detected_transition_state()
                        extending_current_capture = False

            # handle next time increment of prediction
            if self.prediction_transition_state.is_detected():
                if finalized_predictions[i] >= self.prediction_threshold:
                    # expand the detected event by one time step
                    self.prediction_transition_state.num_consecutive_ones += 1
                else:
                    # finalize the detected event
                    interval = (self.prediction_transition_state.start_time,
                                      self.prediction_transition_state.start_time +
                                self.prediction_transition_state.num_consecutive_ones * TIME_DELTA_PER_TIME_STEP)
                    intervals.append(interval)
                    self._capture_spectrogram(interval, begin_idx=detect_start_idx, end_idx=i, spectrogram=spect_data,
                                              extending_current_capture=extending_current_capture)
                    self.prediction_transition_state = non_detected_transition_state()
                    extending_current_capture = False
            elif finalized_predictions[i] >= self.prediction_threshold:
                # begin a new detection event
                start_time = cur_timestamp[1] + (i - cur_timestamp[0])*TIME_DELTA_PER_TIME_STEP
                detect_start_idx = i
                self.prediction_transition_state = TransitionState(start_time, 1)

        if self.spectrogram_capture_dir is not None:
            if extending_current_capture:
                self._combine_captured_spectrogram(spect_data)
            elif self.prediction_transition_state.is_detected():
                self.current_spectrogram_capture = np.copy(spect_data[detect_start_idx:, :])

        return intervals

    def _capture_spectrogram(self, time_bounds: Tuple[datetime, datetime], begin_idx: int = -1, end_idx: int = -1,
                             spectrogram: Optional[np.ndarray] = None, ignore_spectrogram: bool = False,
                             extending_current_capture: bool = False):
        if self.spectrogram_capture_dir is None:
            return

        if ignore_spectrogram and self.current_spectrogram_capture is not None:
            spect_to_save = self.current_spectrogram_capture
        elif extending_current_capture and self.current_spectrogram_capture is not None:
            spect_to_save = np.concatenate([self.current_spectrogram_capture, spectrogram[:end_idx, :]])
        else:
            spect_to_save = spectrogram[begin_idx:end_idx, :]

        filename = "{}_to_{}.npy".format(time_bounds[0].strftime("%Y-%m-%d-%H-%M-%S.%f"),
                                         time_bounds[1].strftime("%Y-%m-%d-%H-%M-%S.%f"))
        np.save(self.spectrogram_capture_dir + "/" + filename, spect_to_save)
        self.current_spectrogram_capture = None

    def _combine_captured_spectrogram(self, new_spectrogram: np.ndarray):
        self.current_spectrogram_capture = np.concatenate([self.current_spectrogram_capture, new_spectrogram])

    def _assert_spectrogram_capture_dir(self):
        if self.spectrogram_capture_dir is None:
            return
        if os.path.isdir(self.spectrogram_capture_dir):
            return
        raise ValueError("The spectrogram capture directory specified, '{}', does not exist!"
                         .format(self.spectrogram_capture_dir))

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
                if metadata_snapshot.rows_unprocessed >= (self.spectrogram_buffer.min_appendable_time_steps + jump):
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
            if metadata_snapshot.rows_unprocessed <= 2*self.spectrogram_buffer.min_appendable_time_steps and pending_predictions > 0:
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
