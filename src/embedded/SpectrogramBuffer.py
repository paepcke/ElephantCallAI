from typing import Optional, Tuple

import numpy as np


# defaults
FREQ_BINS = 77
NUM_SAMPLES_IN_TIME_WINDOW = 256
BUFFER_SIZE_MB = 512  # This is configurable, it's how much dedicated storage space we want for our spectrogram buffer

BYTES_PER_ELEMENT = 8  # We're using float64
BUFFER_SIZE_IN_TIME_STEPS = BUFFER_SIZE_MB * 1024 * 1024 // (FREQ_BINS * BYTES_PER_ELEMENT)

"""A thread-safe spectrogram buffer."""
class SpectrogramBuffer:
    buffer: np.ndarray  # The first dimension is time steps, the second is frequency bins

    # an index into the first dimension of the buffer indicating the data between 'processed' and this has not been evaluated by the model
    unprocessed_end: int

    # an index into the first dimension of the buffer indicating the data between 'end' and this has been evaluated by the model
    # but should not be discarded yet
    pending_post_processing_end: int

    # data before this point is garbage and may be freely overwritten.
    allocated_begin: int

    # The number of rows currently allocated to storing unprocessed data
    rows_unprocessed: int

    # The number of rows currently allocated to storing data
    rows_allocated: int

    # TODO: create a timestamp <-> array index scheme, perhaps with an AVL tree

    def __init__(self, override_buffer_size: Optional[int] = None):
        buf_size = BUFFER_SIZE_IN_TIME_STEPS if override_buffer_size is None else override_buffer_size
        self.buffer = np.zeros((buf_size, FREQ_BINS))
        self.unprocessed_end = 0
        self.pending_post_processing_end = 0
        self.allocated_begin = 0
        self.rows_allocated = 0
        self.rows_unprocessed = 0


    # TODO: an approach where the STFT writes directly into this buffer?
    # TODO: take timestamp as a parameter?
    def append_data(self, new_data: np.ndarray):
        """Append new unprocessed data to the buffer."""
        new_data_time_len = new_data.shape[0]
        # TODO: synchronization between different threads
        available_time_distance_in_buf = self.buffer.shape[0] - self.rows_allocated

        if available_time_distance_in_buf < new_data_time_len:
            # TODO: handle this more gracefully
            raise ValueError("Not enough space in the buffer for new data!")

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

    # data should always be accessed from the *return value* of this method
    def get_unprocessed_data(self, time_steps: int, target: Optional[np.ndarray] = None, mark_for_postprocessing: bool = True) -> np.ndarray:
        """Read unprocessed data from the buffer. By default, this read marks the data as 'processed'."""
        data, new_begin_idx = self.consume_data(time_steps, False, target=target, force_copy=False)
        if mark_for_postprocessing:
            self.pending_post_processing_end = new_begin_idx
            self.rows_unprocessed -= time_steps
        return data

    def get_processed_data(self, time_steps: int, target: Optional[np.ndarray] = None, free_buffer_region: bool = True) -> np.ndarray:
        """Read data from the buffer for post-processing. This copies the data into a separate location
        and, by default, frees the region of the buffer for other use."""
        data, new_begin_idx = self.consume_data(time_steps, True, target=target, force_copy=True)
        if free_buffer_region:
            self.rows_allocated -= time_steps
            self.allocated_begin = new_begin_idx
        return data

    def mark_post_processing_complete(self, time_steps: int) -> int:
        """Manually free the oldest *time_steps* time steps of the buffer for reuse. Allows data here to be overwritten.
        Returns an integer corresponding to the number of rows actually freed for reuse."""
        available_time_steps = self.rows_allocated - self.rows_unprocessed
        if time_steps > available_time_steps:
            time_steps = available_time_steps
        self.rows_allocated -= time_steps
        self.allocated_begin = (self.allocated_begin + time_steps) % self.buffer.shape[0]
        return time_steps

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
