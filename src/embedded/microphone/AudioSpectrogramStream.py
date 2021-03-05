from threading import Thread
from datetime import datetime
from typing import Optional, Tuple
import numpy as np
import math

from embedded.DataCoordinator import DataCoordinator
from embedded.microphone.AudioBuffer import AudioBuffer
from embedded.microphone.SpectrogramExtractor import SpectrogramExtractor

GIVE_UP_THRESHOLD = 100
INPUT_LOCK_TIMEOUT_IN_SECONDS = 1
AUDIO_OUTPUT_LOCK_TIMEOUT_IN_SECONDS = 0.05
N_OVERLAPS_PER_CHUNK = 16


class AudioSpectrogramStream:
    audio_buf: AudioBuffer
    spectrogram_extractor: SpectrogramExtractor
    stream_thread: Thread
    prev_timestamp: Optional[datetime]

    def __init__(self, audio_buffer: AudioBuffer, spect_extractor: SpectrogramExtractor):
        self.audio_buf = audio_buffer
        self.spectrogram_extractor = spect_extractor

    def start(self, data_coordinator: DataCoordinator):
        self.stream_thread = Thread(target=self.stream, args=(data_coordinator,))
        self.stream_thread.start()

    def stream(self, data_coordinator: DataCoordinator):
        num_consecutive_times_audio_buf_empty = 0
        # TODO: some debug/failsafe logic about long-term datacoordinator blockages
        while num_consecutive_times_audio_buf_empty < GIVE_UP_THRESHOLD:
            # step 1: block on datacoordinator availability
            if not data_coordinator.space_available_for_input_lock.acquire(timeout=INPUT_LOCK_TIMEOUT_IN_SECONDS):
                # TODO: failsafe logic
                continue
            else:
                data_coordinator.space_available_for_input_lock.release()

            # step 2: block on an audio buffer lock, receive input, process it into spectrogram
            if not self.audio_buf.output_available_lock.acquire(timeout=AUDIO_OUTPUT_LOCK_TIMEOUT_IN_SECONDS):
                num_consecutive_times_audio_buf_empty += 1
                continue
            else:
                self.audio_buf.output_available_lock.release()
                num_consecutive_times_audio_buf_empty = 0
            data, timestamp = self._read_audio_data()

            spectrogram = self.spectrogram_extractor.extract_spectrogram(data)

            # step 4: append input
            data_coordinator.write(spectrogram, timestamp)
        # TODO: print something about how much data we processed before giving up

    def join(self):
        self.stream_thread.join()

    def _read_audio_data(self) -> Tuple[np.ndarray, Optional[datetime]]:
        n_rows_to_read = 0
        n_windows_to_free = 0
        free_all = False
        new_timestamp = None

        nfft = self.spectrogram_extractor.nfft
        hop = self.spectrogram_extractor.hop
        with self.audio_buf.metadata_mutex:
            with self.audio_buf.timestamp_mutex:
                follow_up_interval = False
                if len(self.audio_buf.timestamp_deque) > 0:
                    next_interval_start_idx = self.audio_buf.timestamp_deque[0][0]
                    if next_interval_start_idx == self.audio_buf.allocated_begin:
                        new_timestamp = self.audio_buf.timestamp_deque[0][1]
                        if len(self.audio_buf.timestamp_deque) > 1:
                            next_interval_start_idx = self.audio_buf.timestamp_deque[1][0]
                            follow_up_interval = True
                        else:
                            follow_up_interval = False
                    else:
                        follow_up_interval = True
                if follow_up_interval:
                    cur_interval_dist_remaining = self.audio_buf.circular_distance(next_interval_start_idx, self.audio_buf.allocated_begin)
                else:
                    cur_interval_dist_remaining = self.audio_buf.rows_allocated

                if cur_interval_dist_remaining >= nfft + N_OVERLAPS_PER_CHUNK * hop:
                    # read the max
                    n_windows_to_free = N_OVERLAPS_PER_CHUNK
                    n_rows_to_read = hop * (n_windows_to_free - 1) + nfft
                elif not follow_up_interval:
                    # read everything except the last nfft samples (they'll be padded if need be)
                    if cur_interval_dist_remaining <= nfft:
                        # This should not happen with a good locking scheme
                        n_windows_to_free = 0
                        n_rows_to_read = 0
                    else:
                        n_windows_to_free = math.ceil((cur_interval_dist_remaining - nfft)/hop)
                        n_rows_to_read = hop * (n_windows_to_free - 1) + nfft
                else:
                    # read and free the remaining contents of the time interval
                    n_rows_to_read = cur_interval_dist_remaining
                    free_all = True

        data, _ = self.audio_buf.get_unprocessed_data(n_rows_to_read)
        if free_all:
            self.audio_buf.free_rows(n_rows_to_read)
        else:
            self.audio_buf.free_rows(n_windows_to_free*hop)
        return data, new_timestamp
