import unittest
import numpy as np
from datetime import datetime, timezone, timedelta

from embedded.microphone.AudioBuffer import AudioBuffer


ACQUIRE_LOCK_TIMEOUT_SECONDS = 0.2


class AudioBufferTest(unittest.TestCase):

    def test_basic_write(self):
        buffer = AudioBuffer(override_buffer_size=16, min_appendable_time_steps=1)

        data = np.arange(3)
        buffer.append_data(data)

        self.assertTrue(np.array_equal(data, buffer.buffer[0:3]))

    def test_write_too_much(self):
        buffer = AudioBuffer(override_buffer_size=16, min_appendable_time_steps=1)

        data = np.arange(20)
        try:
            buffer.append_data(data)
        except ValueError:
            return
        self.fail("Expected exception but none thrown")

    def test_write_fails_when_new_data_too_short(self):
        buffer = AudioBuffer(override_buffer_size=16, min_appendable_time_steps=4)

        data = np.arange(3)
        try:
            buffer.append_data(data)
        except ValueError:
            return
        self.fail("Expected exception but none thrown")

    def test_basic_read(self):
        buffer = AudioBuffer(override_buffer_size=16, min_appendable_time_steps=1)

        data = np.arange(3)
        buffer.append_data(data)

        data_out, _ = buffer.get_unprocessed_data(3)

        self.assertTrue(np.array_equal(data, data_out))

    def test_read_too_much(self):
        buffer = AudioBuffer(override_buffer_size=16, min_appendable_time_steps=1)

        data = np.arange(3)
        buffer.append_data(data)

        data_out, _ = buffer.get_unprocessed_data(6)

        self.assertTrue(np.array_equal(data, data_out))

    def test_free_rows(self):
        buffer = AudioBuffer(override_buffer_size=16, min_appendable_time_steps=1)

        data = np.arange(3)
        buffer.append_data(data)

        buffer.free_rows(3)

        self.assertEqual(0, buffer.rows_allocated)

    def test_free_too_many_rows(self):
        buffer = AudioBuffer(override_buffer_size=16, min_appendable_time_steps=1)

        data = np.arange(3)
        buffer.append_data(data)

        buffer.free_rows(6)

        self.assertEqual(0, buffer.rows_allocated)

    def test_basic_write_wraparound(self):
        buffer = AudioBuffer(override_buffer_size=16, min_appendable_time_steps=1)
        buffer.allocated_begin = 14
        buffer.unprocessed_end = 14

        data = np.arange(5)
        buffer.append_data(data)

        self.assertTrue(np.array_equal(data[0:2], buffer.buffer[14:]))
        self.assertTrue(np.array_equal(data[2:], buffer.buffer[0:3]))

    def test_basic_read_wraparound(self):
        buffer = AudioBuffer(override_buffer_size=16, min_appendable_time_steps=1)
        buffer.allocated_begin = 14
        buffer.unprocessed_end = 14

        data = np.arange(5)
        buffer.append_data(data)

        data_out, _ = buffer.consume_data(5)

        self.assertTrue(np.array_equal(data, data_out))

    def test_free_wraparound(self):
        buffer = AudioBuffer(override_buffer_size=16, min_appendable_time_steps=1)
        buffer.allocated_begin = 14
        buffer.unprocessed_end = 3
        buffer.rows_allocated = 5

        buffer.free_rows(5)

        self.assertEqual(0, buffer.rows_allocated)
        self.assertEqual(3, buffer.allocated_begin)
        self.assertEqual(3, buffer.unprocessed_end)

    def test_output_available_lock_basic_unlocked_state(self):
        buffer = AudioBuffer(override_buffer_size=16, min_appendable_time_steps=1)

        data = np.arange(3)
        buffer.append_data(data)

        got_lock = buffer.output_available_lock.acquire(timeout=ACQUIRE_LOCK_TIMEOUT_SECONDS)
        self.assertTrue(got_lock)

    def test_output_available_lock_locked_state(self):
        buffer = AudioBuffer(override_buffer_size=16, min_appendable_time_steps=1)

        got_lock = buffer.output_available_lock.acquire(timeout=ACQUIRE_LOCK_TIMEOUT_SECONDS)
        self.assertFalse(got_lock)

    def test_output_available_lock_unlocked_if_lt_min_consumable_with_follow_up(self):
        buffer = AudioBuffer(override_buffer_size=16, min_appendable_time_steps=1, min_consumable_rows=4)

        now = datetime.now(timezone.utc)
        time1 = now + timedelta(seconds=3)

        data = np.arange(2)

        buffer.append_data(data, now)
        buffer.append_data(data, time1)
        buffer.append_data(data)

        got_lock = buffer.output_available_lock.acquire(timeout=ACQUIRE_LOCK_TIMEOUT_SECONDS)
        self.assertTrue(got_lock)

    def test_output_available_lock_unlocked_if_lt_min_consumable_with_non_explicit_follow_up(self):
        buffer = AudioBuffer(override_buffer_size=16, min_appendable_time_steps=1, min_consumable_rows=4)

        now = datetime.now(timezone.utc)

        data = np.arange(2)

        buffer.append_data(data)
        buffer.append_data(data, now)
        buffer.append_data(data)

        got_lock = buffer.output_available_lock.acquire(timeout=ACQUIRE_LOCK_TIMEOUT_SECONDS)
        self.assertTrue(got_lock)

    def test_output_available_lock_locked_if_min_consumable_without_follow_up(self):
        buffer = AudioBuffer(override_buffer_size=16, min_appendable_time_steps=1, min_consumable_rows=4)

        now = datetime.now(timezone.utc)

        data = np.arange(3)

        buffer.append_data(data, now)

        got_lock = buffer.output_available_lock.acquire(timeout=ACQUIRE_LOCK_TIMEOUT_SECONDS)
        self.assertFalse(got_lock)

    def test_output_available_lock_locked_if_min_consumable_without_any_timestamps(self):
        buffer = AudioBuffer(override_buffer_size=16, min_appendable_time_steps=1, min_consumable_rows=4)

        data = np.arange(3)

        buffer.append_data(data)

        got_lock = buffer.output_available_lock.acquire(timeout=ACQUIRE_LOCK_TIMEOUT_SECONDS)
        self.assertFalse(got_lock)

    # Test timestamp propagation
    def test_timestamps_written_properly(self):
        buffer = AudioBuffer(override_buffer_size=16, min_appendable_time_steps=1)

        now = datetime.now(timezone.utc)
        time1 = now + timedelta(seconds=3)

        data = np.arange(2)

        buffer.append_data(data, now)
        buffer.append_data(data, time1)
        buffer.append_data(data)

        self.assertEqual(2, len(buffer.timestamp_deque))
        self.assertEqual((0, now), buffer.timestamp_deque[0])
        self.assertEqual((2, time1), buffer.timestamp_deque[1])

    def test_some_timestamps_consumed_when_freed(self):
        buffer = AudioBuffer(override_buffer_size=16, min_appendable_time_steps=1)

        now = datetime.now(timezone.utc)
        time1 = now + timedelta(seconds=3)

        data = np.arange(2)

        buffer.append_data(data, now)
        buffer.append_data(data, time1)
        buffer.append_data(data)

        buffer.free_rows(2)

        self.assertEqual(1, len(buffer.timestamp_deque))
        self.assertEqual((2, time1), buffer.timestamp_deque[0])

    def test_all_timestamps_consumed_when_freed(self):
        buffer = AudioBuffer(override_buffer_size=16, min_appendable_time_steps=1)

        now = datetime.now(timezone.utc)
        time1 = now + timedelta(seconds=3)

        data = np.arange(2)

        buffer.append_data(data, now)
        buffer.append_data(data, time1)
        buffer.append_data(data)

        buffer.free_rows(3)

        self.assertEqual(0, len(buffer.timestamp_deque))


if __name__ == '__main__':
    unittest.main()
