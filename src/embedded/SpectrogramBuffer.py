from typing import Optional, Tuple, Deque
from threading import Lock

import numpy as np
from collections import deque
from datetime import datetime, timezone, timedelta


# defaults
FREQ_BINS = 77
NUM_SAMPLES_IN_TIME_WINDOW = 256
BUFFER_SIZE_MB = 512  # This is configurable, it's how much dedicated storage space we want for our spectrogram buffer

BYTES_PER_ELEMENT = 8  # We're using float64
BUFFER_SIZE_IN_TIME_STEPS = BUFFER_SIZE_MB * 1024 * 1024 // (FREQ_BINS * BYTES_PER_ELEMENT)

# It won't make sense to run the model on spectrogram data with fewer than this number of time steps.
# This means that appending fewer than this number of contiguous time steps is disallowed.
# This constant is the default value, but other values can be configured in the constructor.
MIN_APPENDABLE_TIME_STEPS = NUM_SAMPLES_IN_TIME_WINDOW

# This represents the length of the non-overlapping segments of each time window.
TIME_DELTA_PER_TIME_STEP = timedelta(seconds=0.1)

"""A thread-safe spectrogram buffer."""
class SpectrogramBuffer:
    buffer: np.ndarray  # The first dimension is time steps, the second is frequency bins

    # an index into the first dimension of the buffer indicating the data between 'processed' and this has not been evaluated by the model
    # the data at this time row may only be unprocessed if the entire buffer is full.
    unprocessed_end: int

    # an index into the first dimension of the buffer indicating the data between 'end' and this has been evaluated by the model
    # but should not be discarded yet
    # the data at this time row is only pending post processing if the entire buffer is full and all of it is pending post processing.
    pending_post_processing_end: int

    # data before this point is garbage and may be freely overwritten.
    allocated_begin: int

    # The number of rows currently allocated to storing unprocessed data
    rows_unprocessed: int

    # The number of rows currently allocated to storing data
    rows_allocated: int

    # A queue of tuples that allow us to keep track of the timestamps corresponding to unprocessed examples in the queue.
    unprocessed_timestamp_deque: Deque[Tuple[int, datetime]]

    # A queue of tuples that allow us to keep track of the timestamps corresponding to examples in the queue that are awaiting post-processing.
    post_processing_timestamp_deque: Deque[Tuple[int, datetime]]

    # NOTE: this synchronization scheme is only guaranteed to work if there is only one thread that writes data in,
    # one thread that consumes unprocessed data, and one thread that consumes data for post-processing.
    # A synchronization object that synchronizes changes to the metadata (buffer pointers)
    metadata_mutex: Lock

    # A synchronization object that synchronizes changes to the timestamp deques
    timestamp_mutex: Lock

    def __init__(self, override_buffer_size: Optional[int] = None, min_appendable_time_steps: Optional[int] = None):
        buf_size = BUFFER_SIZE_IN_TIME_STEPS if override_buffer_size is None else override_buffer_size
        self.buffer = np.zeros((buf_size, FREQ_BINS))
        self.unprocessed_end = 0
        self.pending_post_processing_end = 0
        self.allocated_begin = 0
        self.rows_allocated = 0
        self.rows_unprocessed = 0
        self.unprocessed_timestamp_deque = deque()
        self.post_processing_timestamp_deque = deque()
        if min_appendable_time_steps is None:
            self.min_appendable_time_steps = MIN_APPENDABLE_TIME_STEPS
        else:
            self.min_appendable_time_steps = min_appendable_time_steps
        self.metadata_mutex = Lock()
        self.timestamp_mutex = Lock()

    def get_metadata_snapshot(self):
        with self.metadata_mutex:
            metadata_snapshot = SpectrogramBufferMetadata(self)
        return metadata_snapshot

    # TODO: an approach where the STFT writes directly into this buffer?
    def append_data(self, new_data: np.ndarray, time: Optional[datetime] = None):
        """Append new unprocessed data to the buffer."""
        with self.metadata_mutex:
            metadata_snapshot = SpectrogramBufferMetadata(self)
        new_data_time_len = new_data.shape[0]
        available_time_distance_in_buf = self.buffer.shape[0] - metadata_snapshot.rows_allocated

        if available_time_distance_in_buf < new_data_time_len:
            # TODO: handle this more gracefully?
            raise ValueError("Not enough space in the buffer for new data!")
        self._assert_sufficient_time_steps(new_data, time)

        # remember this for timestamp updates
        data_start_idx = metadata_snapshot.unprocessed_end

        if self.unprocessed_end + new_data_time_len > self.buffer.shape[0]:
            first_len = self.buffer.shape[0] - metadata_snapshot.unprocessed_end
            self.buffer[metadata_snapshot.unprocessed_end:self.buffer.shape[0], :] = new_data[0:first_len, :]
            second_len = new_data_time_len - first_len
            self.buffer[0:second_len, :] = new_data[first_len:new_data_time_len, :]
            next_unprocessed_end = second_len
        else:
            self.buffer[metadata_snapshot.unprocessed_end:(metadata_snapshot.unprocessed_end + new_data_time_len), :] = new_data
            next_unprocessed_end = metadata_snapshot.unprocessed_end + new_data_time_len

        with self.metadata_mutex:
            self.unprocessed_end = next_unprocessed_end
            self.rows_allocated += new_data_time_len
            self.rows_unprocessed += new_data_time_len

        with self.timestamp_mutex:
            # handle the timestamp
            if time is None and len(self.unprocessed_timestamp_deque) == 0:
                now = datetime.now(timezone.utc)
                self.unprocessed_timestamp_deque.append((data_start_idx, now))
            elif time is not None:
                self.unprocessed_timestamp_deque.append((data_start_idx, time))
            # If time is None, this class assumes that the recording of this data occurred immediately after
            # the recording of the previous data, without interruption.

    def _assert_sufficient_time_steps(self, new_data: np.ndarray, time: Optional[datetime]):
        if new_data.shape[0] < self.min_appendable_time_steps:
            raise ValueError(("Cannot put less than {} contiguous time steps of data " +
                             "into the buffer in a single append operation.").format(self.min_appendable_time_steps))

    # data should always be accessed from the *return value* of this method
    def get_unprocessed_data(self, time_steps: int, target: Optional[np.ndarray] = None,
                             mark_for_postprocessing: bool = True) -> Tuple[np.ndarray, int]:
        """Read unprocessed data from the buffer. By default, this read marks the data as 'processed'."""
        with self.metadata_mutex:
            metadata_snapshot = SpectrogramBufferMetadata(self)
        data_start_idx = metadata_snapshot.pending_post_processing_end  # return this value to allow simple indexing of a prediction buffer
        data, new_begin_idx = self.consume_data(time_steps, False, target=target, force_copy=False)
        if mark_for_postprocessing:
            with self.metadata_mutex:
                old_post_proc_end = self.pending_post_processing_end
                self.pending_post_processing_end = new_begin_idx
                self.rows_unprocessed -= time_steps
                self._transfer_timestamp_data(old_post_proc_end)
        return data, data_start_idx

    def mark_for_post_processing(self, time_steps: int):
        """Can be used to manually reclassify a region of the buffer without reading it"""
        with self.metadata_mutex:
            available_time_steps = self.rows_unprocessed
            if time_steps > available_time_steps:
                time_steps = available_time_steps
            self.rows_unprocessed -= time_steps
            old_post_proc_end = self.pending_post_processing_end
            self.pending_post_processing_end = (self.pending_post_processing_end + time_steps) % self.buffer.shape[0]
            self._transfer_timestamp_data(old_post_proc_end)

    def get_processed_data(self, time_steps: int, target: Optional[np.ndarray] = None, free_buffer_region: bool = True) -> np.ndarray:
        """Read data from the buffer for post-processing. This copies the data into a separate location
        and, by default, frees the region of the buffer for other use."""
        data, new_begin_idx = self.consume_data(time_steps, True, target=target, force_copy=True)
        if free_buffer_region:
            with self.metadata_mutex:
                self.rows_allocated -= time_steps
                old_allocated_begin = self.allocated_begin
                self.allocated_begin = new_begin_idx
                self._free_rows_update_timestamp(old_allocated_begin)
        return data

    def mark_post_processing_complete(self, time_steps: int) -> int:
        """Manually free the oldest *time_steps* time steps of the buffer for reuse. Allows data here to be overwritten.
        Returns an integer corresponding to the number of rows actually freed for reuse."""
        with self.metadata_mutex:
            available_time_steps = self.rows_allocated - self.rows_unprocessed
            if time_steps > available_time_steps:
                time_steps = available_time_steps
            self.rows_allocated -= time_steps
            old_allocated_begin = self.allocated_begin
            self.allocated_begin = (self.allocated_begin + time_steps) % self.buffer.shape[0]
            self._free_rows_update_timestamp(old_allocated_begin)
            return time_steps

    # update 'rows_allocated' and 'allocated_begin' BEFORE calling this method
    # Call this method while holding the metadata_mutex
    def _free_rows_update_timestamp(self, old_allocated_begin: int):
        """Updates the timestamp deques after a region of the buffer is deallocated"""
        if (self.rows_allocated - self.rows_unprocessed) == 0:
            with self.timestamp_mutex:
                self.post_processing_timestamp_deque.clear()
            return

        with self.timestamp_mutex:
            new_allocated_begin = self.allocated_begin
            # take elements off the timestamp queue until we go past or hit the new allocated beginning, then, assuming
            # the timestamp isn't exact, add the time delta for how many rows after the most recently deleted timestamp were
            # freed, then re-queue that timestamp.
            max_distance = (new_allocated_begin - old_allocated_begin) % self.buffer.shape[0]
            cur_timestamp_entry = self.post_processing_timestamp_deque.popleft()
            next_timestamp_distance = 0
            if len(self.post_processing_timestamp_deque) != 0:
                next_timestamp_distance = self.circular_distance(self.post_processing_timestamp_deque[0][0], old_allocated_begin)
            while len(self.post_processing_timestamp_deque) != 0 and next_timestamp_distance <= max_distance:
                cur_timestamp_entry = self.post_processing_timestamp_deque.popleft()
                if len(self.post_processing_timestamp_deque) == 0:
                    break
                next_timestamp_distance = self.circular_distance(self.post_processing_timestamp_deque[0][0], old_allocated_begin)

            new_timestamp = (new_allocated_begin,
                             cur_timestamp_entry[1]
                             + TIME_DELTA_PER_TIME_STEP * (self.circular_distance(new_allocated_begin, cur_timestamp_entry[0])))
            self.post_processing_timestamp_deque.appendleft(new_timestamp)

    # update 'rows_unprocessed' and 'pending_post_processing_end' BEFORE calling this method.
    # Call this method while holding the metadata mutex.
    def _transfer_timestamp_data(self, old_post_proc_end: int):
        """Moves timestamps from the unprocessed deque to the pending_post_processing deque as data in the buffer is reclassified"""
        with self.timestamp_mutex:
            new_post_proc_end = self.pending_post_processing_end
            max_distance = (new_post_proc_end - old_post_proc_end) % self.buffer.shape[0]
            cur_timestamp_entry = self.unprocessed_timestamp_deque.popleft()
            next_timestamp_distance = 0
            if len(self.unprocessed_timestamp_deque) != 0:
                next_timestamp_distance = self.circular_distance(self.unprocessed_timestamp_deque[0][0], old_post_proc_end)
            while len(self.unprocessed_timestamp_deque) != 0 and next_timestamp_distance < max_distance:
                self._transfer_timestamp_entry_without_duplication(cur_timestamp_entry)  # transfer the element to the post-processing timestamp deque
                cur_timestamp_entry = self.unprocessed_timestamp_deque.popleft()
                if len(self.unprocessed_timestamp_deque) == 0:
                    break
                next_timestamp_distance = self.circular_distance(self.unprocessed_timestamp_deque[0][0], old_post_proc_end)

            if self.rows_unprocessed != 0:
                circular_distance = self.circular_distance(new_post_proc_end, cur_timestamp_entry[0])
                new_earliest_unprocessed_timestamp = (new_post_proc_end,
                                                      cur_timestamp_entry[1]
                                                      + TIME_DELTA_PER_TIME_STEP * circular_distance)
                if len(self.unprocessed_timestamp_deque) != 0:
                    first_index = self.unprocessed_timestamp_deque[0][0]
                    if cur_timestamp_entry[0] + circular_distance != first_index:
                        self.unprocessed_timestamp_deque.appendleft(new_earliest_unprocessed_timestamp)
                else:
                    self.unprocessed_timestamp_deque.appendleft(new_earliest_unprocessed_timestamp)
            self._transfer_timestamp_entry_without_duplication(cur_timestamp_entry)

    def _transfer_timestamp_entry_without_duplication(self, cur_timestamp_entry: Tuple[int, datetime]):
        """A helper method that ensures timestamps only exist in the case of discontinuities or manual labels via 'append_data()'"""
        if len(self.post_processing_timestamp_deque) != 0:
            prev_time_idx, prev_timestamp = self.post_processing_timestamp_deque[-1]
            if cur_timestamp_entry[1] != (prev_timestamp + self.circular_distance(cur_timestamp_entry[0], prev_time_idx) * TIME_DELTA_PER_TIME_STEP):
                self.post_processing_timestamp_deque.append(cur_timestamp_entry)
        else:
            self.post_processing_timestamp_deque.append(cur_timestamp_entry)

    def circular_distance(self, right_idx: int, left_idx: int) -> int:
        """A helper method that returns the distance between right_idx and left_idx, going past the end of the array if necessary"""
        # Note: be careful when using this for the case in which right_idx == left_idx, there's not enough information
        # to tell whether the appropriate return value should be 0 or the entire size of the buffer
        return (right_idx - left_idx) % self.buffer.shape[0]

    def consume_data(self, time_steps: int, processed: bool, target: Optional[np.ndarray] = None,
                     force_copy: Optional[bool] = False) -> Tuple[np.ndarray, int]:
        """copy data from the buffer into the target if necessary, otherwise, return a view.
        returns (read data, new value for starting index of read-from chunk)"""
        with self.metadata_mutex:
            metadata_snapshot = SpectrogramBufferMetadata(self)
        begin_idx: int
        end_idx: int
        if processed:
            begin_idx = metadata_snapshot.allocated_begin
            available_time_steps = metadata_snapshot.rows_allocated - metadata_snapshot.rows_unprocessed
        else:
            begin_idx = metadata_snapshot.pending_post_processing_end
            available_time_steps = metadata_snapshot.rows_unprocessed
        if time_steps > available_time_steps:
            time_steps = available_time_steps
            if available_time_steps == 0:
                raise ValueError("No data to read")
        if metadata_snapshot.pending_post_processing_end + time_steps > self.buffer.shape[0]:
            if target is None:
                target = np.zeros((time_steps, FREQ_BINS))
            first_len = self.buffer.shape[0] - begin_idx
            target[0:first_len, :] = self.buffer[begin_idx:(begin_idx + first_len), :]
            second_len = time_steps - first_len
            target[first_len:time_steps, :] = self.buffer[0:second_len, :]
            return target, second_len
        else:
            # return a view of the array to avoid making a copy
            view = self.buffer[begin_idx:(begin_idx + time_steps), :]
            if force_copy:
                if target is None:
                    target = np.zeros((time_steps, FREQ_BINS))
                target[:, :] = view[:, :]
                view = target
            new_starting_idx = begin_idx + time_steps
            return view, new_starting_idx

    def clear(self):
        """Overwrites the entire buffer with 0 and deallocates all of it."""
        with self.metadata_mutex:
            with self.timestamp_mutex:
                self.buffer[:, :] = 0
                self.unprocessed_end = 0
                self.pending_post_processing_end = 0
                self.allocated_begin = 0
                self.rows_allocated = 0
                self.rows_unprocessed = 0
                self.unprocessed_timestamp_deque.clear()
                self.post_processing_timestamp_deque.clear()


class SpectrogramBufferMetadata:
    unprocessed_end: int
    pending_post_processing_end: int
    allocated_begin: int
    rows_allocated: int
    rows_unprocessed: int

    def __init__(self, buffer):
        self.unprocessed_end = buffer.unprocessed_end
        self.pending_post_processing_end = buffer.pending_post_processing_end
        self.allocated_begin = buffer.allocated_begin
        self.rows_allocated = buffer.rows_allocated
        self.rows_unprocessed = buffer.rows_unprocessed
