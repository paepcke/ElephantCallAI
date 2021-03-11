from threading import Lock
from typing import Deque, Tuple, Optional

import numpy as np
from collections import deque
from datetime import datetime, timezone


# defaults
NUM_SAMPLES_IN_TIME_WINDOW = 4096
BUFFER_SIZE_MB = 4  # This is configurable, it's how much dedicated storage space we want for our audio buffer

BYTES_PER_ELEMENT = 2  # We're using int16
BUFFER_SIZE_IN_TIME_STEPS = BUFFER_SIZE_MB * 1024 * 1024 // BYTES_PER_ELEMENT

# It won't make sense to run the model on spectrogram data with fewer than this number of time steps.
# This means that appending fewer than this number of contiguous time steps is disallowed.
# This constant is the default value, but other values can be configured in the constructor.
MIN_APPENDABLE_TIME_STEPS = NUM_SAMPLES_IN_TIME_WINDOW


class AudioBuffer:
    """A thread-safe audio buffer."""

    buffer: np.ndarray  # The first dimension is time steps, the second is frequency bins

    # an index into the first dimension of the buffer indicating the data between 'processed' and this has not been evaluated by the model
    # the data at this time row may only be unprocessed if the entire buffer is full.
    unprocessed_end: int

    # data before this point is garbage and may be freely overwritten.
    allocated_begin: int

    # The number of rows currently allocated to storing data
    rows_allocated: int

    # The minimum number of rows that can be consumed at a time
    min_consumable_rows: int

    # A queue of tuples that allow us to keep track of the timestamps corresponding to unprocessed examples in the queue.
    timestamp_deque: Deque[Tuple[int, datetime]]

    # NOTE: this synchronization scheme is only guaranteed to work if there is only one thread that writes data in,
    # one thread that consumes unprocessed data, and one thread that consumes data for post-processing.
    # A synchronization object that synchronizes changes to the metadata (buffer pointers)
    metadata_mutex: Lock

    # A synchronization object that synchronizes changes to the timestamp deques
    timestamp_mutex: Lock

    # Thread synchronization

    output_available_lock: Lock

    _holding_output_lock: bool

    output_sync_lock: Lock

    def __init__(self, min_consumable_rows: Optional[int] = None, override_buffer_size: Optional[int] = None, min_appendable_time_steps: Optional[int] = None):
        buf_size = BUFFER_SIZE_IN_TIME_STEPS if override_buffer_size is None else override_buffer_size
        self.buffer = np.zeros((buf_size,), dtype=np.int16)
        self.unprocessed_end = 0
        self.pending_post_processing_end = 0
        self.allocated_begin = 0
        self.rows_allocated = 0
        self.timestamp_deque = deque()
        if min_appendable_time_steps is None:
            self.min_appendable_time_steps = MIN_APPENDABLE_TIME_STEPS
        else:
            self.min_appendable_time_steps = min_appendable_time_steps
        self.min_consumable_rows = min_consumable_rows
        if self.min_consumable_rows is None:
            self.min_consumable_rows = 2*self.min_appendable_time_steps
        self.metadata_mutex = Lock()
        self.timestamp_mutex = Lock()

        # Thread synchronization
        self.output_sync_lock = Lock()
        self.output_available_lock = Lock()
        self.output_available_lock.acquire()
        self._holding_output_lock = True

    def get_metadata_snapshot(self):
        with self.metadata_mutex:
            metadata_snapshot = AudioBufferMetadata(self)
        return metadata_snapshot

    def append_data(self, new_data: np.ndarray, time: Optional[datetime] = None):
        """Append new unprocessed data to the buffer."""
        with self.metadata_mutex:
            metadata_snapshot = AudioBufferMetadata(self)
        new_data_time_len = new_data.shape[0]
        available_time_distance_in_buf = self.buffer.shape[0] - metadata_snapshot.rows_allocated

        if available_time_distance_in_buf < new_data_time_len:
            raise ValueError("Not enough space in the buffer for new data!")
        self._assert_sufficient_time_steps(new_data, time)

        # remember this for timestamp updates
        data_start_idx = metadata_snapshot.unprocessed_end

        if self.unprocessed_end + new_data_time_len > self.buffer.shape[0]:
            first_len = self.buffer.shape[0] - metadata_snapshot.unprocessed_end
            self.buffer[metadata_snapshot.unprocessed_end:self.buffer.shape[0]] = new_data[0:first_len]
            second_len = new_data_time_len - first_len
            self.buffer[0:second_len] = new_data[first_len:new_data_time_len]
            next_unprocessed_end = second_len
        else:
            self.buffer[metadata_snapshot.unprocessed_end:(metadata_snapshot.unprocessed_end + new_data_time_len)] = new_data
            next_unprocessed_end = metadata_snapshot.unprocessed_end + new_data_time_len

        with self.metadata_mutex:
            self.unprocessed_end = next_unprocessed_end
            self.rows_allocated += new_data_time_len

        with self.timestamp_mutex:
            # handle the timestamp
            if time is not None:
                self.timestamp_deque.append((data_start_idx, time))
            # If time is None, this class assumes that the recording of this data occurred immediately after
            # the recording of the previous data, without interruption.
        self.update_output_lock()

    def _assert_sufficient_time_steps(self, new_data: np.ndarray, time: Optional[datetime]):
        if new_data.shape[0] < self.min_appendable_time_steps:
            raise ValueError(("Cannot put less than {} contiguous time steps of data " +
                             "into the buffer in a single append operation.").format(self.min_appendable_time_steps))

    def free_rows(self, num_rows_to_free: int):
        with self.metadata_mutex:
            old_allocated_begin = self.allocated_begin
            if self.rows_allocated < num_rows_to_free:
                num_rows_to_free = self.rows_allocated
            next_allocated_begin = (self.allocated_begin + num_rows_to_free) % self.buffer.shape[0]
            self.rows_allocated -= num_rows_to_free
            self.allocated_begin = next_allocated_begin
            self._free_rows_update_timestamp(old_allocated_begin)
        self.update_output_lock()

    # data should always be accessed from the *return value* of this method
    def get_unprocessed_data(self, time_steps: int, target: Optional[np.ndarray] = None) -> Tuple[np.ndarray, int]:
        """Read unprocessed data from the buffer. This method does not free any data."""
        with self.metadata_mutex:
            metadata_snapshot = AudioBufferMetadata(self)
        data_start_idx = metadata_snapshot.allocated_begin
        data, new_begin_idx = self.consume_data(time_steps, target=target, force_copy=False)
        return data, data_start_idx

    # update 'rows_allocated' and 'allocated_begin' BEFORE calling this method
    # Call this method while holding the metadata_mutex
    def _free_rows_update_timestamp(self, old_allocated_begin: int):
        """Updates the timestamp deque after a region of the buffer is deallocated"""
        if self.rows_allocated == 0:
            with self.timestamp_mutex:
                self.timestamp_deque.clear()
            return

        with self.timestamp_mutex:
            new_allocated_begin = self.allocated_begin
            # take elements off the timestamp queue until we go past or hit the new allocated beginning.
            # Unlike with the spectrogram buffer, these timestamps only exist to mark the start of a recording.
            # It's entirely possible for this deque to be totally empty.
            max_distance = (new_allocated_begin - old_allocated_begin) % self.buffer.shape[0]
            next_timestamp_distance = 0
            if len(self.timestamp_deque) != 0:
                next_timestamp_distance = self.circular_distance(self.timestamp_deque[0][0], old_allocated_begin)
            while len(self.timestamp_deque) != 0 and next_timestamp_distance < max_distance:
                self.timestamp_deque.popleft()
                if len(self.timestamp_deque) == 0:
                    break
                next_timestamp_distance = self.circular_distance(self.timestamp_deque[0][0], old_allocated_begin)

    def circular_distance(self, right_idx: int, left_idx: int) -> int:
        """A helper method that returns the distance between right_idx and left_idx, going past the end of the array if necessary"""
        # Note: be careful when using this for the case in which right_idx == left_idx, there's not enough information
        # to tell whether the appropriate return value should be 0 or the entire size of the buffer
        return (right_idx - left_idx) % self.buffer.shape[0]

    def consume_data(self, time_steps: int, target: Optional[np.ndarray] = None,
                     force_copy: Optional[bool] = False) -> Tuple[np.ndarray, int]:
        """copy data from the buffer into the target if necessary, otherwise, return a view.
        returns (read data, new value for starting index of read-from chunk)"""
        with self.metadata_mutex:
            metadata_snapshot = AudioBufferMetadata(self)
        begin_idx = metadata_snapshot.allocated_begin
        available_time_steps = metadata_snapshot.rows_allocated
        if time_steps > available_time_steps:
            # If you want some amount of data from a particular recording interval to remain in the audio buffer,
            # you must ensure it with logic outside this class
            time_steps = available_time_steps
            if available_time_steps == 0:
                raise ValueError("No data to read")
        if metadata_snapshot.allocated_begin + time_steps > self.buffer.shape[0]:
            if target is None:
                target = np.zeros((time_steps,))
            first_len = self.buffer.shape[0] - begin_idx
            target[0:first_len] = self.buffer[begin_idx:(begin_idx + first_len)]
            second_len = time_steps - first_len
            target[first_len:time_steps] = self.buffer[0:second_len]
            return target, second_len
        else:
            # return a view of the array to avoid making a copy
            view = self.buffer[begin_idx:(begin_idx + time_steps)]
            if force_copy:
                if target is None:
                    target = np.zeros((time_steps,))
                target[:] = view[:]
                view = target
            new_starting_idx = begin_idx + time_steps
            return view, new_starting_idx

    def update_output_lock(self):
        with self.output_sync_lock:
            metadata_snapshot = self.get_metadata_snapshot()

            with self.timestamp_mutex:
                follow_up_interval = False
                if len(self.timestamp_deque) > 0:
                    if metadata_snapshot.allocated_begin != self.timestamp_deque[0][0]:
                        follow_up_interval = True
                    else:
                        follow_up_interval = len(self.timestamp_deque) > 1
            if metadata_snapshot.rows_allocated >= self.min_consumable_rows or follow_up_interval:
                # unlock it if possible
                if self._holding_output_lock:
                    self._holding_output_lock = False
                    self.output_available_lock.release()
            else:
                if not self._holding_output_lock:
                    self.output_available_lock.acquire()
                    self._holding_output_lock = True

    def clear(self):
        """Overwrites the entire buffer with 0 and deallocates all of it."""
        # TODO: reset locks on clear?
        with self.metadata_mutex:
            with self.timestamp_mutex:
                self.buffer[:] = 0
                self.unprocessed_end = 0
                self.allocated_begin = 0
                self.rows_allocated = 0
                self.timestamp_deque.clear()


class AudioBufferMetadata:
    unprocessed_end: int
    allocated_begin: int
    rows_allocated: int

    def __init__(self, buffer):
        self.unprocessed_end = buffer.unprocessed_end
        self.allocated_begin = buffer.allocated_begin
        self.rows_allocated = buffer.rows_allocated