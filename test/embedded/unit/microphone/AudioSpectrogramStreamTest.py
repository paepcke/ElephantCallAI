import unittest
import numpy as np
from datetime import datetime, timezone, timedelta

from embedded.microphone.AudioBuffer import AudioBuffer
from embedded.microphone.AudioSpectrogramStream import AudioSpectrogramStream
from embedded.microphone.SpectrogramExtractor import SpectrogramExtractor


class AudioSpectrogramStreamTest(unittest.TestCase):
    def test_read_from_buffer_with_plenty_of_data_and_no_follow_up(self):
        audio_buf = AudioBuffer(override_buffer_size=4096*16, min_appendable_time_steps=1)
        spect_ex = SpectrogramExtractor()
        aud_stream = AudioSpectrogramStream(audio_buf, spect_ex, n_overlaps_per_chunk=4)

        now = datetime.now(timezone.utc)

        data = np.random.randint(low=0, high=10, size=(4096 + 20*800,), dtype=np.int16)

        audio_buf.append_data(data, now)

        data_out, timestamp = aud_stream._read_audio_data()

        expected_data_len = (4096 + (aud_stream.n_overlaps_per_chunk - 1) * 800)

        self.assertEqual(expected_data_len, len(data_out))
        self.assertTrue(np.array_equal(data[0:expected_data_len], data_out))
        self.assertEqual(now, timestamp)

        self.assertEqual(4096 + 16*800, audio_buf.rows_allocated)

    def test_read_from_buffer_when_less_than_full_remaining_data_and_no_follow_up_timestamp(self):
        audio_buf = AudioBuffer(override_buffer_size=4096*16, min_appendable_time_steps=1)
        spect_ex = SpectrogramExtractor()
        aud_stream = AudioSpectrogramStream(audio_buf, spect_ex, n_overlaps_per_chunk=10)

        now = datetime.now(timezone.utc)

        data = np.random.randint(low=0, high=10, size=(4096 + 8*800,), dtype=np.int16)

        audio_buf.append_data(data, now)

        data_out, timestamp = aud_stream._read_audio_data()

        self.assertIsNone(data_out)
        self.assertIsNone(timestamp)

        self.assertEqual(len(data), audio_buf.rows_allocated)

    def test_read_from_buffer_when_slightly_more_than_enough_data_and_no_follow_up_timestamp(self):
        audio_buf = AudioBuffer(override_buffer_size=4096*8, min_appendable_time_steps=1)
        spect_ex = SpectrogramExtractor()
        aud_stream = AudioSpectrogramStream(audio_buf, spect_ex, n_overlaps_per_chunk=4)

        now = datetime.now(timezone.utc)

        data = np.random.randint(low=0, high=10, size=(4096 + 200 + aud_stream.n_overlaps_per_chunk * 800,), dtype=np.int16)

        audio_buf.append_data(data, now)

        data_out, timestamp = aud_stream._read_audio_data()

        expected_data_len = 4096 + (aud_stream.n_overlaps_per_chunk - 1) * 800

        self.assertEqual(expected_data_len, len(data_out))
        self.assertTrue(np.array_equal(data[0:expected_data_len], data_out))
        self.assertEqual(now, timestamp)

        self.assertEqual(4096 + 200, audio_buf.rows_allocated)

    def test_read_from_buffer_recognizes_follow_up_timestamp(self):
        audio_buf = AudioBuffer(override_buffer_size=4096*8, min_appendable_time_steps=1)
        spect_ex = SpectrogramExtractor()
        aud_stream = AudioSpectrogramStream(audio_buf, spect_ex)

        now = datetime.now(timezone.utc)
        time1 = datetime.now(timezone.utc)

        aud_stream.prev_timestamp = now

        data = np.random.randint(low=0, high=10, size=(4096 + 200,), dtype=np.int16)

        audio_buf.append_data(data, time1)

        data_out, timestamp = aud_stream._read_audio_data()

        self.assertIsNone(data_out)
        self.assertIsNone(timestamp)

        self.assertEqual(4096 + 200, audio_buf.rows_allocated)

    def test_read_from_buffer_when_slightly_more_than_enough_data_and_follow_up_timestamp(self):
        audio_buf = AudioBuffer(override_buffer_size=4096*8, min_appendable_time_steps=1)
        spect_ex = SpectrogramExtractor()
        aud_stream = AudioSpectrogramStream(audio_buf, spect_ex, n_overlaps_per_chunk=4)

        now = datetime.now(timezone.utc)
        time1 = now + timedelta(seconds=10)

        data = np.random.randint(low=0, high=10, size=(4096 + 200 + aud_stream.n_overlaps_per_chunk * 800,), dtype=np.int16)

        audio_buf.append_data(data, now)
        audio_buf.append_data(data, time1)

        data_out, timestamp = aud_stream._read_audio_data()

        expected_data_len = 4096 + (aud_stream.n_overlaps_per_chunk - 1)*800

        self.assertEqual(expected_data_len, len(data_out))
        self.assertTrue(np.array_equal(data[0:expected_data_len], data_out))
        self.assertEqual(now, timestamp)

        self.assertEqual(len(data) + 200 + 4096, audio_buf.rows_allocated)

    def test_read_from_buffer_with_plenty_of_data_and_follow_up(self):
        audio_buf = AudioBuffer(override_buffer_size=4096*16, min_appendable_time_steps=1)
        spect_ex = SpectrogramExtractor()
        aud_stream = AudioSpectrogramStream(audio_buf, spect_ex, n_overlaps_per_chunk=4)

        now = datetime.now(timezone.utc)
        time1 = now + timedelta(seconds=10)

        data = np.random.randint(low=0, high=10, size=(4096 + 20*800,), dtype=np.int16)

        audio_buf.append_data(data, now)
        audio_buf.append_data(data, time1)

        data_out, timestamp = aud_stream._read_audio_data()

        expected_data_len = (4096 + (aud_stream.n_overlaps_per_chunk - 1) * 800)

        self.assertEqual(expected_data_len, len(data_out))
        self.assertTrue(np.array_equal(data[0:expected_data_len], data_out))
        self.assertEqual(now, timestamp)

        self.assertEqual(4096*2 + (20*2 - aud_stream.n_overlaps_per_chunk)*800, audio_buf.rows_allocated)


if __name__ == '__main__':
    unittest.main()
