from threading import Thread
from datetime import datetime, timezone
from typing import Optional, Tuple
import numpy as np
import sys

from embedded.Closeable import Closeable
from embedded.DataCoordinator import DataCoordinator
from embedded.microphone.AudioBuffer import AudioBuffer
from embedded.microphone.SpectrogramExtractor import SpectrogramExtractor

GIVE_UP_THRESHOLD_FOR_AUDIO_BUF = 100
GIVE_UP_THRESHOLD_FOR_DATA_COORDINATOR = 200
INPUT_LOCK_TIMEOUT_IN_SECONDS = 1
AUDIO_OUTPUT_LOCK_TIMEOUT_IN_SECONDS = 0.3
DEFAULT_N_OVERLAPS_PER_CHUNK = 256


class AudioSpectrogramStream(Closeable):
    """
    An object that takes audio data from an AudioBuffer, performs an STFT conversion on it,
    and appends it to a DataCoordinator.
    """

    audio_buf: AudioBuffer
    spectrogram_extractor: SpectrogramExtractor
    stream_thread: Thread
    prev_timestamp: Optional[datetime]
    n_overlaps_per_chunk: int
    timeout: bool
    closed: bool

    def __init__(self, audio_buffer: AudioBuffer, spect_extractor: SpectrogramExtractor,
                 n_overlaps_per_chunk: int = DEFAULT_N_OVERLAPS_PER_CHUNK, timeout: bool = False):
        super().__init__()
        self.closed = False
        self.audio_buf = audio_buffer
        self.spectrogram_extractor = spect_extractor
        self.n_overlaps_per_chunk = n_overlaps_per_chunk
        self.timeout = timeout

    def start(self, data_coordinator: DataCoordinator):
        # 'daemon' threads are killed when the parent process dies
        self.stream_thread = Thread(target=self.stream, args=(data_coordinator,), daemon=True)
        self.stream_thread.start()

    def stream(self, data_coordinator: DataCoordinator):
        num_consecutive_times_audio_buf_empty = 0
        num_consecutive_times_locked_out_of_data_coordinator = 0
        while not self.closed and (not self.timeout or
                (num_consecutive_times_audio_buf_empty < GIVE_UP_THRESHOLD_FOR_AUDIO_BUF
                    and num_consecutive_times_locked_out_of_data_coordinator < GIVE_UP_THRESHOLD_FOR_DATA_COORDINATOR)):
            # step 1: block on datacoordinator availability
            if not data_coordinator.space_available_for_input_lock\
                    .acquire(timeout=INPUT_LOCK_TIMEOUT_IN_SECONDS if self.timeout else -1):
                num_consecutive_times_locked_out_of_data_coordinator += 1
                continue
            else:
                data_coordinator.space_available_for_input_lock.release()
                num_consecutive_times_locked_out_of_data_coordinator = 0

            # step 2: block on an audio buffer lock, receive input, process it into spectrogram
            if not self.audio_buf.output_available_lock\
                    .acquire(timeout=AUDIO_OUTPUT_LOCK_TIMEOUT_IN_SECONDS if self.timeout else -1):
                num_consecutive_times_audio_buf_empty += 1
                continue
            else:
                self.audio_buf.output_available_lock.release()
            data, timestamp = self._read_audio_data()

            if data is not None:
                num_consecutive_times_audio_buf_empty = 0

                spectrogram = self.spectrogram_extractor.extract_spectrogram(data)

                # step 4: append input
                data_coordinator.write(self.transform(spectrogram), timestamp)
            else:
                num_consecutive_times_audio_buf_empty += 1

        if self.timeout:
            now = datetime.now(timezone.utc)
            error_msg: str
            if num_consecutive_times_locked_out_of_data_coordinator >= GIVE_UP_THRESHOLD_FOR_DATA_COORDINATOR:
                error_msg = "Spectrogram Buffer too full for additional input for {} consecutive seconds." \
                            " Spectrogram stream giving up at {}.".format(
                    num_consecutive_times_locked_out_of_data_coordinator * INPUT_LOCK_TIMEOUT_IN_SECONDS, now)
            else:
                error_msg = "Audio input unavailable for {} consecutive seconds." \
                            " Spectrogram stream giving up at {}.".format(
                    num_consecutive_times_audio_buf_empty * AUDIO_OUTPUT_LOCK_TIMEOUT_IN_SECONDS, now)
            print(error_msg, file=sys.stderr)

    def transform(self, spectrogram_data: np.ndarray):
        """
        A transformation applied to spectrogram data specific to the model with which
        this code was developed. It may be changed for other models.

        :param spectrogram_data: the spectrogram data to be transformed
        :return: transformed spectrogram data
        """
        return 10*np.log10(spectrogram_data)

    def close(self):
        self.closed = True

    def join(self):
        self.stream_thread.join()

    '''
    To comply with SpectrogramBuffer input requirements, *Throw away excess audiodata* if it's not large enough
    to make 256 + k*jump time steps of spectral data (provided there's a follow-up timestamp).
    For now, it's slightly less efficient than this, only keeping integer multiples of 256 time steps. This is not
    expected to have much practical impact, but it can be changed.
    '''
    def _read_audio_data(self) -> Tuple[Optional[np.ndarray], Optional[datetime]]:
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

                if cur_interval_dist_remaining >= nfft + (self.n_overlaps_per_chunk - 1) * hop:
                    # read the max
                    n_windows_to_free = self.n_overlaps_per_chunk
                    n_rows_to_read = hop * (n_windows_to_free - 1) + nfft
                elif not follow_up_interval:
                    # Don't read anything; hope that next time this method is called, there's enough data to use
                    return None, None
                else:
                    # free the remaining contents of the time interval without reading them. This data is effectively dropped.
                    free_all = True

        if free_all:
            self.audio_buf.free_rows(cur_interval_dist_remaining)
            return None, None

        data, _ = self.audio_buf.get_unprocessed_data(n_rows_to_read)
        self.audio_buf.free_rows(n_windows_to_free*hop)
        return data, new_timestamp
