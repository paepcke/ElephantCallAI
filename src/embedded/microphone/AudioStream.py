import numpy as np
import pyaudio

from embedded.microphone.AudioBuffer import AudioBuffer


class AudioStream:
    buffer: AudioBuffer
    dropped_prev_chunk: bool

    def __init__(self):
        self.buffer = AudioBuffer()
        self.dropped_prev_chunk = True
        self.setup_microphone()

    def setup_microphone(self):
        # TODO: init stream, init time
        pass

    def _callback(self, in_data, frame_count, time_info, status_flags):
        in_data = np.fromstring(in_data, dtype=np.int16)

        try:
            # TODO: if dropped_prev_chunk, pass new time based on time_info
            self.buffer.append_data(new_data=in_data, time=None)
        except ValueError:  # An exception is thrown when the buffer does not have space to accommodate the new data
            self.dropped_prev_chunk = True

        return None, pyaudio.paContinue