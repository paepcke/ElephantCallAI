from typing import Optional, Tuple, Deque

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

# TODO: figure this out by looking at other places in the codebase, current value is a placeholder
TIME_DELTA_PER_TIME_STEP = timedelta(seconds=1)

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


    # TODO: an approach where the STFT writes directly into this buffer?
    def append_data(self, new_data: np.ndarray, time: Optional[datetime] = None):
        """Append new unprocessed data to the buffer."""
        new_data_time_len = new_data.shape[0]
        # TODO: synchronization between different threads
        available_time_distance_in_buf = self.buffer.shape[0] - self.rows_allocated

        if available_time_distance_in_buf < new_data_time_len:
            # TODO: handle this more gracefully?
            raise ValueError("Not enough space in the buffer for new data!")
        self._assert_sufficient_time_steps(new_data, time)

        # remember this for timestamp updates
        data_start_idx = self.unprocessed_end

        if self.unprocessed_end + new_data_time_len > self.buffer.shape[0]:
            first_len = self.buffer.shape[0] - self.unprocessed_end
            self.buffer[self.unprocessed_end:self.buffer.shape[0], :] = new_data[0:first_len, :]
            second_len = new_data_time_len - first_len
            self.buffer[0:second_len, :] = new_data[first_len:new_data_time_len, :]
            self.unprocessed_end = second_len
        else:
            self.buffer[self.unprocessed_end:(self.unprocessed_end + new_data_time_len), :] = new_data
            self.unprocessed_end += new_data_time_len
        self.rows_allocated += new_data_time_len
        self.rows_unprocessed += new_data_time_len

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
    def get_unprocessed_data(self, time_steps: int, target: Optional[np.ndarray] = None, mark_for_postprocessing: bool = True) -> np.ndarray:
        """Read unprocessed data from the buffer. By default, this read marks the data as 'processed'."""
        data, new_begin_idx = self.consume_data(time_steps, False, target=target, force_copy=False)
        if mark_for_postprocessing:
            old_post_proc_end = self.pending_post_processing_end
            self.pending_post_processing_end = new_begin_idx
            self.rows_unprocessed -= time_steps
            self._transfer_timestamp_data(old_post_proc_end)
        return data

    def get_processed_data(self, time_steps: int, target: Optional[np.ndarray] = None, free_buffer_region: bool = True) -> np.ndarray:
        """Read data from the buffer for post-processing. This copies the data into a separate location
        and, by default, frees the region of the buffer for other use."""
        # TODO: block this operation until there is a meaningful number of rows left for processing
        data, new_begin_idx = self.consume_data(time_steps, True, target=target, force_copy=True)
        if free_buffer_region:
            self.rows_allocated -= time_steps
            old_allocated_begin = self.allocated_begin
            self.allocated_begin = new_begin_idx
            self._free_rows_update_timestamp(old_allocated_begin)
        return data

    def mark_post_processing_complete(self, time_steps: int) -> int:
        """Manually free the oldest *time_steps* time steps of the buffer for reuse. Allows data here to be overwritten.
        Returns an integer corresponding to the number of rows actually freed for reuse."""
        available_time_steps = self.rows_allocated - self.rows_unprocessed
        if time_steps > available_time_steps:
            time_steps = available_time_steps
        self.rows_allocated -= time_steps
        old_allocated_begin = self.allocated_begin
        self.allocated_begin = (self.allocated_begin + time_steps) % self.buffer.shape[0]
        self._free_rows_update_timestamp(old_allocated_begin)
        return time_steps

    # update 'rows_allocated' and 'allocated_begin' BEFORE calling this method
    def _free_rows_update_timestamp(self, old_allocated_begin: int):
        if (self.rows_allocated - self.rows_unprocessed) == 0:
            self.post_processing_timestamp_deque.clear()
            return

        new_allocated_begin = self.allocated_begin
        # take elements off the timestamp queue until we go past or hit the new allocated beginning, then, assuming
        # the timestamp isn't exact, add the time delta for how many rows after the most recently deleted timestamp were
        # freed, then re-queue that timestamp.
        max_distance = (new_allocated_begin - old_allocated_begin) % self.buffer.shape[0]
        cur_timestamp_entry = self.post_processing_timestamp_deque.popleft()
        next_timestamp_distance = 0
        if len(self.post_processing_timestamp_deque) != 0:
            next_timestamp_distance = self._circular_distance(self.post_processing_timestamp_deque[0][0], old_allocated_begin)
        while len(self.post_processing_timestamp_deque) != 0 and next_timestamp_distance <= max_distance:
            cur_timestamp_entry = self.post_processing_timestamp_deque.popleft()
            if len(self.post_processing_timestamp_deque) == 0:
                break
            next_timestamp_distance = self._circular_distance(self.post_processing_timestamp_deque[0][0], old_allocated_begin)

        new_timestamp = (new_allocated_begin,
                         cur_timestamp_entry[1]
                         + TIME_DELTA_PER_TIME_STEP * (self._circular_distance(new_allocated_begin, cur_timestamp_entry[0])))
        self.post_processing_timestamp_deque.appendleft(new_timestamp)

    # update 'rows_unprocessed' and 'pending_post_processing_end' BEFORE calling this method
    def _transfer_timestamp_data(self, old_post_proc_end: int):
        new_post_proc_end = self.pending_post_processing_end
        max_distance = (new_post_proc_end - old_post_proc_end) % self.buffer.shape[0]
        cur_timestamp_entry = self.unprocessed_timestamp_deque.popleft()
        next_timestamp_distance = 0
        if len(self.unprocessed_timestamp_deque) != 0:
            next_timestamp_distance = self._circular_distance(self.unprocessed_timestamp_deque[0][0], old_post_proc_end)
        while len(self.unprocessed_timestamp_deque) != 0 and next_timestamp_distance < max_distance:
            self.post_processing_timestamp_deque.append(cur_timestamp_entry)  # transfer the element to the post-processing timestamp deque
            cur_timestamp_entry = self.unprocessed_timestamp_deque.popleft()
            if len(self.unprocessed_timestamp_deque) == 0:
                break
            next_timestamp_distance = self._circular_distance(self.unprocessed_timestamp_deque[0][0], old_post_proc_end)

        if self.rows_unprocessed != 0:
            circular_distance = self._circular_distance(new_post_proc_end, cur_timestamp_entry[0])
            new_earliest_unprocessed_timestamp = (new_post_proc_end,
                                                  cur_timestamp_entry[1]
                                                  + TIME_DELTA_PER_TIME_STEP * circular_distance)
            if len(self.unprocessed_timestamp_deque) != 0:
                first_index = self.unprocessed_timestamp_deque[0][0]
                if cur_timestamp_entry[0] + circular_distance != first_index:
                    self.unprocessed_timestamp_deque.appendleft(new_earliest_unprocessed_timestamp)
            else:
                self.unprocessed_timestamp_deque.appendleft(new_earliest_unprocessed_timestamp)
        self.post_processing_timestamp_deque.append(cur_timestamp_entry)

    def _circular_distance(self, right_idx: int, left_idx: int) -> int:
        return (right_idx - left_idx) % self.buffer.shape[0]

    # copy data from the buffer into the target if necessary, otherwise, return a view.
    # returns (read data, new value for starting index of read-from chunk)
    def consume_data(self, time_steps: int, processed: bool, target: Optional[np.ndarray] = None,
                     force_copy: Optional[bool] = False) -> Tuple[np.ndarray, int]:
        begin_idx: int
        end_idx: int
        if processed:
            begin_idx = self.allocated_begin
            available_time_steps = self.rows_allocated - self.rows_unprocessed
        else:
            begin_idx = self.pending_post_processing_end
            available_time_steps = self.rows_unprocessed
        if time_steps > available_time_steps:
            time_steps = available_time_steps
            if available_time_steps == 0:
                raise ValueError("No data to read")
        if self.pending_post_processing_end + time_steps > self.buffer.shape[0]:
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
        self.buffer[:, :] = 0
        self.unprocessed_end = 0
        self.pending_post_processing_end = 0
        self.allocated_begin = 0
        self.rows_allocated = 0
        self.rows_unprocessed = 0
        self.unprocessed_timestamp_deque.clear()
        self.post_processing_timestamp_deque.clear()
