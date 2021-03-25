from threading import Thread
import numpy as np
from typing import Optional
from datetime import datetime, timezone
import sys

from embedded.Closeable import Closeable
from embedded.DataCoordinator import DataCoordinator


CHUNK_SIZE = 256*8
INPUT_LOCK_TIMEOUT_IN_SECONDS = 0.01
MIN_EXPECTED_SHAPE = 100  # We know there are more time steps than this...


class FileSpectrogramStream(Closeable):
    """An object which governs a thread that inserts spectrogram data from a file into a
    DataCoordinator object in chunks. Used for testing."""
    max_time_steps: Optional[int]
    spectrogram_data: np.ndarray
    stream_thread: Thread
    # If there isn't room in the buffer and 'drop_data' is true, we don't re-attempt to buffer this data, instead moving on to the next batch
    drop_data: bool
    closed: bool

    def __init__(self, path_to_spectrogram_file: str, max_time_steps: Optional[int] = None, drop_data: bool = True):
        super().__init__()
        self.closed = False
        self.spectrogram_data = np.load(path_to_spectrogram_file)
        if self.spectrogram_data.shape[0] < MIN_EXPECTED_SHAPE:
            self.spectrogram_data = self.spectrogram_data.T
        self.max_time_steps = max_time_steps
        self.drop_data = drop_data

    def start(self, data_coordinator: DataCoordinator):
        # 'daemon' threads are killed when the parent process dies
        self.stream_thread = Thread(target=self.stream, args=(data_coordinator,), daemon=True)
        self.stream_thread.start()

    def stream(self, data_coordinator: DataCoordinator):
        max_chunks = self.spectrogram_data.shape[0] // CHUNK_SIZE
        if self.max_time_steps is not None:
            max_chunks = min(max_chunks, self.max_time_steps//CHUNK_SIZE)

        if self.max_time_steps is None or max_chunks != self.max_time_steps//CHUNK_SIZE:
            print("WARNING: processing all {} time steps of data".format(self.spectrogram_data.shape[0]))

        need_new_timestamp = True
        i = 0

        while not self.closed and i < max_chunks:
            if not data_coordinator.space_available_for_input_lock.acquire(timeout=INPUT_LOCK_TIMEOUT_IN_SECONDS):
                if self.drop_data:
                    i += 1
                    need_new_timestamp = True
                    print("Dropped a chunk", file=sys.stderr)
                continue
            else:
                data_coordinator.space_available_for_input_lock.release()
            if need_new_timestamp:
                now = datetime.now(timezone.utc)
            else:
                now = None
            try:
                data_coordinator.write(self.transform(self.spectrogram_data[(i*CHUNK_SIZE):((i+1)*CHUNK_SIZE), :]), timestamp=now)
                need_new_timestamp = False
                i += 1
            except ValueError:
                if self.drop_data:
                    i += 1
                    need_new_timestamp = True
                    print("Dropped a chunk", file=sys.stderr)

        print("Done streaming spectrogram data, inserted {} rows".format(i*CHUNK_SIZE))

    def transform(self, spectrogram_data: np.ndarray):
        """A transformation applied to spectrogram data specific to the model with which
        this code was developed. It may be changed for other models."""
        return 10*np.log10(spectrogram_data)

    def close(self):
        self.closed = True

    def join(self):
        self.stream_thread.join()
