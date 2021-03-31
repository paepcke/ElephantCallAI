import pyaudio
from datetime import datetime, timezone, timedelta
import numpy as np

from embedded.Closeable import Closeable
from embedded.microphone.AudioBuffer import AudioBuffer


DEFAULT_SAMPLE_FREQ = 8000
DEFAULT_FRAMES_PER_BUFFER = 4096  # about half a second of audio

TIME_KEY = "input_buffer_adc_time"


class AudioCapturer(Closeable):
    """
    An object that listens to a USB microphone and asynchronously appends gathered data
    to an AudioBuffer.
    """

    audio_buf: AudioBuffer
    sampling_freq: int
    stream: pyaudio.Stream
    frames_per_buffer: int
    start_time_seconds: float
    start_timestamp: datetime
    dropped_prev_segment: bool

    def __init__(self, audio_buf: AudioBuffer, sampling_freq: int = DEFAULT_SAMPLE_FREQ, frames_per_buffer: int = DEFAULT_FRAMES_PER_BUFFER):
        super().__init__()
        self.audio_buf = audio_buf
        self.sampling_freq = sampling_freq
        self.frames_per_buffer = frames_per_buffer

    def start(self):
        pyaudio_obj = pyaudio.PyAudio()
        # Careful! The following line assumes you have exactly 1 USB mic plugged in (with a working driver!)
        self.stream = pyaudio_obj.open(rate=self.sampling_freq, channels=1, format=pyaudio.paInt16, input=True,
                                       frames_per_buffer=self.frames_per_buffer, stream_callback=self._stream_callback)
        self.start_timestamp = datetime.now(timezone.utc)
        self.start_time_seconds = self.stream.get_time()
        self.dropped_prev_segment = True

    def _stream_callback(self, in_data, frame_count, time_info, status_flags):
        """This is invoked by PyAudio every time a configurable amount of audio data
        is collected by the microphone"""
        if frame_count != self.frames_per_buffer:
            raise ValueError("Frame_count and frames_per_buffer not equal")

        metadata_snapshot = self.audio_buf.get_metadata_snapshot()
        rows_free = self.audio_buf.buffer.shape[0] - metadata_snapshot.rows_allocated
        if rows_free >= self.frames_per_buffer:
            new_data = np.fromstring(in_data, dtype=np.int16)
            timestamp = None
            if self.dropped_prev_segment:
                absolute_time_seconds = time_info[TIME_KEY]
                timestamp = self._compute_timestamp(absolute_time_seconds)
            self.audio_buf.append_data(new_data, timestamp)
            self.dropped_prev_segment = False
        else:
            self.dropped_prev_segment = True

        return in_data, pyaudio.paContinue

    def _compute_timestamp(self, absolute_time: float):
        relative_time = absolute_time - self.start_time_seconds
        return self.start_timestamp + timedelta(seconds=relative_time)

    def close(self):
        self.stream.close()
