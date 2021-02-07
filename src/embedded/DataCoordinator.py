from typing import Optional, Deque, Tuple, List
import numpy as np
from datetime import datetime
from collections import deque

from src.embedded.SpectrogramBuffer import SpectrogramBuffer, TIME_DELTA_PER_TIME_STEP
from src.embedded.PredictionBuffer import PredictionBuffer
from src.embedded.TransitionState import TransitionState, non_detected_transition_state
from embedded.predictors import Predictor

PREDICTION_THRESHOLD = 0.5


class DataCoordinator:
    """An object that abstracts out the unsightly ring-buffer indexing used when storing spectrograms and predictions.
    Extracts intervals of continuously-detected positive predictions and saves these intervals to a file."""
    spectrogram_buffer: SpectrogramBuffer
    prediction_buffer: PredictionBuffer
    interval_output_path: str
    transition_state: TransitionState

    def __init__(self, interval_output_path: str, override_buffer_size: Optional[int] = None,
                 min_appendable_time_steps: Optional[int] = None):
        self.interval_output_path = interval_output_path
        self.spectrogram_buffer = SpectrogramBuffer(override_buffer_size=override_buffer_size,
                                                    min_appendable_time_steps=min_appendable_time_steps)
        self.prediction_buffer = PredictionBuffer(self.spectrogram_buffer.buffer.shape[0])
        self.transition_state = non_detected_transition_state()

    # Appends new spectrogram data to the spectrogram buffer
    def write(self, spectrogram: np.ndarray, timestamp: Optional[datetime] = None):
        self.spectrogram_buffer.append_data(spectrogram, time=timestamp)

    # returns number of rows for which a prediction has been made, useful for backoff logic
    def make_predictions(self, predictor: Predictor, time_window: int, overlap_allowance: int) -> int:
        metadata_snapshot = self.spectrogram_buffer.get_metadata_snapshot()
        if metadata_snapshot.rows_unprocessed <= self.spectrogram_buffer.min_appendable_time_steps:
            # We can't process enough data yet
            return 0

        if overlap_allowance >= time_window:
            raise ValueError("Time window must exceed overlap allowance")

        if time_window < self.spectrogram_buffer.min_appendable_time_steps:
            raise ValueError("Predictions must be made on at least {} rows of contiguous spectrogram data"
                             .format(self.spectrogram_buffer.min_appendable_time_steps))

        if time_window > metadata_snapshot.rows_unprocessed:
            time_window = metadata_snapshot.rows_unprocessed

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
            if self.spectrogram_buffer.rows_unprocessed >= (2 * time_window - overlap_allowance):
                # process the first time_window samples
                spect_data, begin_idx = self.spectrogram_buffer.get_unprocessed_data(time_steps=time_window,
                                                                                     mark_for_postprocessing=False)
                self.spectrogram_buffer.mark_for_post_processing(time_steps=time_window - overlap_allowance)
                predictions, overlap_counts = predictor.make_predictions(spect_data)
                self.prediction_buffer.write(begin_idx, predictions, overlap_counts)
                return predictions.shape[0]
            else:
                # We can't risk having 'hanging' data that isn't large enough to process all at once
                return 0
        else:
            time_dist = self.spectrogram_buffer.circular_distance(second_timestamp[0], first_timestamp[0])
            if time_dist >= (2*time_window - overlap_allowance):
                # process the first time_window samples, there is enough data for there to be another full time_window after this
                spect_data, begin_idx = self.spectrogram_buffer.get_unprocessed_data(time_steps=time_window,
                                                                                     mark_for_postprocessing=False)
                self.spectrogram_buffer.mark_for_post_processing(time_steps=time_window - overlap_allowance)
                predictions, overlap_counts = predictor.make_predictions(spect_data)
                self.prediction_buffer.write(begin_idx, predictions, overlap_counts)
                return predictions.shape[0]
            else:
                # process all of this remaining data
                spect_data, begin_idx = self.spectrogram_buffer.get_unprocessed_data(time_steps=time_dist,
                                                                                     mark_for_postprocessing=True)
                predictions, overlap_counts = predictor.make_predictions(spect_data)
                self.prediction_buffer.write(begin_idx, predictions, overlap_counts)
                return predictions.shape[0]

    # returns number of rows for which predictions were finalized
    def finalize_predictions(self, time_window: int) -> int:
        metadata_snapshot = self.spectrogram_buffer.get_metadata_snapshot()
        rows_pending_post_proc = metadata_snapshot.rows_allocated - metadata_snapshot.rows_unprocessed
        if rows_pending_post_proc == 0 or time_window <= 0:
            # There's nothing to finalize here
            return 0
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

        # free the underlying memory in the spectrogrambuffer
        self.spectrogram_buffer.mark_post_processing_complete(time_window)

        # create time intervals for detection events, save them to a file
        intervals = self.get_detection_intervals(final_predictions, timestamps)
        self.save_detection_intervals(intervals)

        return time_window

    # Start with the left-over transition state and a deque of timestamps, return a list of event intervals
    def get_detection_intervals(self, finalized_predictions: np.ndarray, timestamps: Deque[Tuple[int, datetime]]) -> List[Tuple[datetime, datetime]]:
        intervals = list()
        cur_timestamp = timestamps.popleft()

        if self.transition_state.is_detected():
            tentative_interval_end = self.transition_state.start_time + self.transition_state.num_consecutive_ones * TIME_DELTA_PER_TIME_STEP
            if tentative_interval_end != cur_timestamp[1]:
                intervals.append((self.transition_state.start_time, tentative_interval_end))
                self.transition_state = non_detected_transition_state()

        for i in range(0, finalized_predictions.shape[0]):
            if len(timestamps) > 0 and timestamps[0][0] <= i:
                prev_timestamp = cur_timestamp
                cur_timestamp = timestamps.popleft()
                if cur_timestamp[1] != ((cur_timestamp[0] - prev_timestamp[0])*TIME_DELTA_PER_TIME_STEP + prev_timestamp[1]):
                    # This is a time discontinuity. We can only detect call events that cross this boundary as two separate events; one on each side of the discontinuity.
                    if self.transition_state.is_detected():
                        intervals.append((self.transition_state.start_time,
                                          self.transition_state.start_time + self.transition_state.num_consecutive_ones * TIME_DELTA_PER_TIME_STEP))
                        self.transition_state = non_detected_transition_state()

            # handle next time increment of prediction
            if self.transition_state.is_detected():
                if finalized_predictions[i] >= PREDICTION_THRESHOLD:
                    # expand the detected event by one time step
                    self.transition_state.num_consecutive_ones += 1
                else:
                    # finalize the detected event
                    intervals.append((self.transition_state.start_time,
                                      self.transition_state.start_time + self.transition_state.num_consecutive_ones * TIME_DELTA_PER_TIME_STEP))
                    self.transition_state = non_detected_transition_state()
            elif finalized_predictions[i] >= PREDICTION_THRESHOLD:
                # begin a new detection event
                start_time = cur_timestamp[1] + (i - cur_timestamp[0])*TIME_DELTA_PER_TIME_STEP
                self.transition_state = TransitionState(start_time, 1)

        return intervals

    # appends detection intervals to a file
    def save_detection_intervals(self, intervals: List[Tuple[datetime, datetime]]):
        with open(self.interval_output_path, "a+") as writer:
            for interval in intervals:
                writer.write("{} - {}\n".format(interval[0].isoformat(), interval[1].isoformat()))