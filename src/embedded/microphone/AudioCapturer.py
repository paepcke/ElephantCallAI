from typing import Optional
import sounddevice
from datetime import datetime, timezone
import numpy as np
import sys

from embedded.Closeable import Closeable
from embedded.microphone.AudioBuffer import AudioBuffer


TARGET_DEVICE_NAME = "pulse"
DEFAULT_SAMPLE_FREQ = 8000
DEFAULT_FRAMES_PER_BUFFER = 4096  # about half a second of audio


class AudioCapturer(Closeable):
    """
    An object that listens to a USB microphone and asynchronously appends gathered data
    to an AudioBuffer.
    """

    audio_buf: AudioBuffer
    sampling_freq: int
    stream: sounddevice.InputStream
    frames_per_buffer: int
    start_timestamp: datetime
    dropped_prev_segment: bool

    def __init__(self, audio_buf: AudioBuffer, sampling_freq: int = DEFAULT_SAMPLE_FREQ, frames_per_buffer: int = DEFAULT_FRAMES_PER_BUFFER):
        super().__init__()
        self.audio_buf = audio_buf
        self.sampling_freq = sampling_freq
        self.frames_per_buffer = frames_per_buffer

    def start(self):
        audio_device_idx = self._identify_pulse_device_id()
        if audio_device_idx is None:
            print(f"WARNING: '{TARGET_DEVICE_NAME}' audio device not found. Audio capture may not work as expected.",
                  file=sys.stderr)
        # Careful! The following line assumes you have exactly 1 USB mic plugged in (with a working driver!)
        self.stream = sounddevice.InputStream(samplerate=self.sampling_freq, channels=1, dtype=np.int16,
                                              blocksize=self.frames_per_buffer, callback=self._stream_callback,
                                              device=audio_device_idx)
        self.start_timestamp = datetime.now(timezone.utc)
        self.dropped_prev_segment = True
        self.stream.start()

    def _stream_callback(self, in_data, frame_count, time_info, status_flags):
        """
        This is invoked by sounddevice every time a configurable amount of audio data
        is collected by the microphone.
        """

        if frame_count != self.frames_per_buffer:
            raise ValueError("Frame_count and frames_per_buffer not equal")

        metadata_snapshot = self.audio_buf.get_metadata_snapshot()
        rows_free = self.audio_buf.buffer.shape[0] - metadata_snapshot.rows_allocated
        if rows_free >= self.frames_per_buffer:
            new_data = np.fromstring(in_data, dtype=np.int16)
            timestamp = None
            if self.dropped_prev_segment:
                '''
                there may be a delay between the actual time of recording and this call,
                but it should be about 1 second or less, probably not enough to make a practical difference
                for the intended use case of this module.
                '''
                timestamp = datetime.now(timezone.utc)
            self.audio_buf.append_data(new_data, timestamp)
            self.dropped_prev_segment = False
        else:
            self.dropped_prev_segment = True

    def _identify_pulse_device_id(self) -> Optional[int]:
        """
        Due to some audio quirks on Linux devices, we want a device called 'pulse' to be used as audio input.
        If pulseaudio is installed, it should appear in the list of sound devices.

        :return: a device ID for the pulse device or 'None' if the pulse device cannot be found
        """
        for device_id, device_info in enumerate(sounddevice.query_devices()):
            if device_info['name'] == TARGET_DEVICE_NAME:
                return device_id

        return None

    def close(self):
        self.stream.close()
