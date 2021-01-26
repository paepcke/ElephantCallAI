import unittest
import numpy as np
from datetime import datetime, timezone, timedelta

from src.embedded.SpectrogramBuffer import SpectrogramBuffer, TIME_DELTA_PER_TIME_STEP


class SpectrogramBufferTest(unittest.TestCase):
    """Unit tests for the SpectrogramBuffer class."""

    def test_basic_write_into_buffer(self):
        buffer = SpectrogramBuffer(override_buffer_size=8, min_appendable_time_steps=1)
        new_data = np.arange(3, dtype=float).reshape(-1, 1)
        buffer.append_data(new_data)
        self.assertEqual(3, buffer.unprocessed_end)
        self.assertEqual(0, buffer.pending_post_processing_end)
        self.assertEqual(0, buffer.allocated_begin)
        self.assertTrue(np.array_equal(buffer.buffer[0:3, 12].reshape(-1, 1), new_data))

    def test_basic_read_from_buffer(self):
        buffer = SpectrogramBuffer(override_buffer_size=8, min_appendable_time_steps=1)
        new_data = np.arange(3, dtype=float).reshape(-1, 1)
        buffer.append_data(new_data)
        self.assertEqual(3, buffer.unprocessed_end)
        self.assertEqual(0, buffer.pending_post_processing_end)
        self.assertEqual(0, buffer.allocated_begin)
        out_data = buffer.get_unprocessed_data(3)
        self.assertTrue(np.array_equal(out_data[:, 12].reshape(-1, 1), new_data))
        self.assertEqual(3, buffer.unprocessed_end)
        self.assertEqual(3, buffer.pending_post_processing_end)
        self.assertEqual(0, buffer.allocated_begin)

    def test_basic_read_from_buffer_without_marking_data_as_processed(self):
        buffer = SpectrogramBuffer(override_buffer_size=8, min_appendable_time_steps=1)
        new_data = np.arange(3, dtype=float).reshape(-1, 1)
        buffer.append_data(new_data)
        self.assertEqual(3, buffer.unprocessed_end)
        self.assertEqual(0, buffer.pending_post_processing_end)
        self.assertEqual(0, buffer.allocated_begin)
        out_data = buffer.get_unprocessed_data(3, mark_for_postprocessing=False)
        self.assertTrue(np.array_equal(out_data[:, 12].reshape(-1, 1), new_data))
        self.assertEqual(3, buffer.unprocessed_end)
        self.assertEqual(0, buffer.pending_post_processing_end)
        self.assertEqual(0, buffer.allocated_begin)

    def test_wraparound_writing(self):
        buffer = SpectrogramBuffer(override_buffer_size=8, min_appendable_time_steps=1)
        buffer.unprocessed_end = 6
        buffer.pending_post_processing_end = 6
        buffer.allocated_begin = 6
        new_data = np.arange(3, dtype=float).reshape(-1, 1)
        buffer.append_data(new_data)
        self.assertEqual(1, buffer.unprocessed_end)
        self.assertEqual(6, buffer.pending_post_processing_end)
        self.assertEqual(6, buffer.allocated_begin)
        self.assertTrue(np.array_equal(buffer.buffer[6:7, 12].reshape(-1, 1), new_data[0:1]))
        self.assertEqual(2, buffer.buffer[0, 12])

    def test_wraparound_read(self):
        buffer = SpectrogramBuffer(override_buffer_size=8, min_appendable_time_steps=1)
        buffer.unprocessed_end = 6
        buffer.pending_post_processing_end = 6
        buffer.allocated_begin = 6
        new_data = np.arange(3, dtype=float).reshape(-1, 1)
        buffer.append_data(new_data)
        self.assertEqual(1, buffer.unprocessed_end)
        self.assertEqual(6, buffer.pending_post_processing_end)
        self.assertEqual(6, buffer.allocated_begin)

        out_data = buffer.get_unprocessed_data(3)
        self.assertTrue(np.array_equal(out_data[:, 12].reshape(-1, 1), new_data))

        self.assertEqual(1, buffer.unprocessed_end)
        self.assertEqual(1, buffer.pending_post_processing_end)
        self.assertEqual(6, buffer.allocated_begin)

    def test_free_memory(self):
        buffer = SpectrogramBuffer(override_buffer_size=8, min_appendable_time_steps=1)
        buffer.unprocessed_end = 5
        buffer.pending_post_processing_end = 3
        buffer.rows_unprocessed = 2
        buffer.rows_allocated = 5
        buffer.timestamp_deque.append((0, datetime.now(timezone.utc)))

        buffer.mark_post_processing_complete(2)
        self.assertEqual(2, buffer.allocated_begin)
        self.assertEqual(5, buffer.unprocessed_end)
        self.assertEqual(3, buffer.pending_post_processing_end)
        self.assertEqual(3, buffer.rows_allocated)
        self.assertEqual(2, buffer.rows_unprocessed)

    def test_free_memory_wraparound(self):
        buffer = SpectrogramBuffer(override_buffer_size=8, min_appendable_time_steps=1)
        buffer.unprocessed_end = 4
        buffer.pending_post_processing_end = 2
        buffer.allocated_begin = 4
        buffer.rows_unprocessed = 2
        buffer.rows_allocated = 8
        buffer.timestamp_deque.append((4, datetime.now(timezone.utc)))

        buffer.mark_post_processing_complete(4)
        self.assertEqual(0, buffer.allocated_begin)
        self.assertEqual(4, buffer.unprocessed_end)
        self.assertEqual(2, buffer.pending_post_processing_end)
        self.assertEqual(4, buffer.rows_allocated)
        self.assertEqual(2, buffer.rows_unprocessed)

    def test_exception_on_write_too_big(self):
        buffer = SpectrogramBuffer(override_buffer_size=8, min_appendable_time_steps=1)
        new_data = np.arange(10, dtype=float).reshape(-1, 1)

        try:
            buffer.append_data(new_data)
        except ValueError:
            return
        self.fail("Expected exception but none raised")

    # Note: In the case of a wraparound, the output data will be a copy. This test does not cover that case.
    def test_read_unprocessed_data_doesnt_copy(self):
        buffer = SpectrogramBuffer(override_buffer_size=8, min_appendable_time_steps=1)
        buffer.unprocessed_end = 2
        buffer.rows_unprocessed = 2
        buffer.rows_allocated = 2
        data_out = buffer.get_unprocessed_data(2)

        # these should be the same memory location
        data_out[0, 0] = 4
        self.assertEqual(4, buffer.buffer[0, 0])

    def test_read_for_postprocessing_makes_copy(self):
        buffer = SpectrogramBuffer(override_buffer_size=8, min_appendable_time_steps=1)
        buffer.buffer[0, 0] = 3
        buffer.unprocessed_end = 2
        buffer.pending_post_processing_end = 2
        buffer.rows_allocated = 2
        data_out = buffer.get_processed_data(2)

        # these should NOT be the same memory location.
        data_out[0, 0] = 4
        self.assertNotEqual(data_out[0, 0], buffer.buffer[0, 0])

    def test_completely_full_buffer_disallows_writes(self):
        buffer = SpectrogramBuffer(override_buffer_size=8, min_appendable_time_steps=1)
        new_data = np.arange(8, dtype=float).reshape(-1, 1)
        buffer.append_data(new_data)

        try:
            buffer.append_data(new_data)
        except ValueError:
            return
        self.fail("Expected exception but none occurred")

    def test_cannot_free_unprocessed_data(self):
        buffer = SpectrogramBuffer(override_buffer_size=8, min_appendable_time_steps=1)
        new_data = np.arange(8, dtype=float).reshape(-1, 1)
        buffer.append_data(new_data)

        rows_freed = buffer.mark_post_processing_complete(2)
        self.assertEqual(0, rows_freed)

    def test_free_memory_allows_overwrite(self):
        buffer = SpectrogramBuffer(override_buffer_size=8, min_appendable_time_steps=1)
        new_data = np.arange(8, dtype=float).reshape(-1, 1)
        buffer.append_data(new_data)
        buffer.get_unprocessed_data(3)
        rows_freed = buffer.mark_post_processing_complete(2)
        self.assertEqual(2, rows_freed)
        buffer.append_data(np.zeros((2, 1)))

    def test_exception_thrown_when_reading_unavailable_processed_data(self):
        buffer = SpectrogramBuffer(override_buffer_size=8, min_appendable_time_steps=1)
        new_data = np.arange(8, dtype=float).reshape(-1, 1)
        buffer.append_data(new_data)
        try:
            buffer.get_processed_data(3)
        except ValueError:
            return
        self.fail("Expected exception but none thrown.")

    def test_exception_thrown_when_reading_unavailable_unprocessed_data(self):
        buffer = SpectrogramBuffer(override_buffer_size=8, min_appendable_time_steps=1)
        try:
            buffer.get_unprocessed_data(3)
        except ValueError:
            return
        self.fail("Expected exception but none thrown.")

    def test_ex_thrown_when_queueing_insufficient_data(self):
        buffer = SpectrogramBuffer(override_buffer_size=8, min_appendable_time_steps=3)
        new_data = np.arange(2, dtype=float).reshape(-1, 1)

        try:
            buffer.append_data(new_data)
        except ValueError:
            return
        self.fail("Expected exception but none thrown")

    def test_free_all_memory_empties_timestamp_deque(self):
        buffer = SpectrogramBuffer(override_buffer_size=8, min_appendable_time_steps=1)
        buffer.unprocessed_end = 3
        buffer.pending_post_processing_end = 3
        buffer.rows_unprocessed = 0
        buffer.rows_allocated = 3
        buffer.timestamp_deque.append((0, datetime.now(timezone.utc)))

        buffer.mark_post_processing_complete(3)
        self.assertEqual(3, buffer.allocated_begin)
        self.assertEqual(3, buffer.unprocessed_end)
        self.assertEqual(3, buffer.pending_post_processing_end)
        self.assertEqual(0, buffer.rows_allocated)
        self.assertEqual(0, buffer.rows_unprocessed)

        self.assertEqual(0, len(buffer.timestamp_deque))

    def test_timestamp_queue_updates_freeing_data(self):
        buffer = SpectrogramBuffer(override_buffer_size=8, min_appendable_time_steps=1)
        buffer.unprocessed_end = 5
        buffer.pending_post_processing_end = 3
        buffer.rows_unprocessed = 2
        buffer.rows_allocated = 5
        now = datetime.now(timezone.utc)
        buffer.timestamp_deque.append((0, now))
        buffer.mark_post_processing_complete(3)

        self.assertEqual(1, len(buffer.timestamp_deque))

        top_timestamp = buffer.timestamp_deque[0]
        self.assertEqual(3, top_timestamp[0])
        expected_time = now + 3 * TIME_DELTA_PER_TIME_STEP
        self.assertEqual(expected_time, top_timestamp[1])

    def test_timestamp_queue_updates_freeing_data_pruning_multiple_timestamps(self):
        buffer = SpectrogramBuffer(override_buffer_size=8, min_appendable_time_steps=1)
        buffer.unprocessed_end = 7
        buffer.pending_post_processing_end = 6
        buffer.rows_unprocessed = 1
        buffer.rows_allocated = 7
        now = datetime.now(timezone.utc)
        time2 = now + 10*timedelta(seconds=2)
        time3 = time2 + 18*timedelta(seconds=3)
        time4 = time3 + 7*timedelta(seconds=1)
        buffer.timestamp_deque.append((0, now))
        buffer.timestamp_deque.append((2, time2))
        buffer.timestamp_deque.append((4, time3))
        buffer.timestamp_deque.append((6, time4))

        buffer.mark_post_processing_complete(5)

        self.assertEqual(2, len(buffer.timestamp_deque))
        top_timestamp = buffer.timestamp_deque[0]
        self.assertEqual(5, top_timestamp[0])
        expected_time = time3 + 1 * TIME_DELTA_PER_TIME_STEP
        self.assertEqual(expected_time, top_timestamp[1])

        self.assertEqual((6, time4), buffer.timestamp_deque[1])

    def test_timestamp_queue_updates_freeing_data_pruning_multiple_timestamps_with_wraparound(self):
        buffer = SpectrogramBuffer(override_buffer_size=8, min_appendable_time_steps=1)
        buffer.unprocessed_end = 3
        buffer.pending_post_processing_end = 2
        buffer.allocated_begin = 4
        buffer.rows_unprocessed = 1
        buffer.rows_allocated = 7
        now = datetime.now(timezone.utc)
        time2 = now + 10 * timedelta(seconds=2)
        time3 = time2 + 18 * timedelta(seconds=3)
        buffer.timestamp_deque.append((5, now))
        buffer.timestamp_deque.append((0, time2))
        buffer.timestamp_deque.append((2, time3))

        buffer.mark_post_processing_complete(5)

        self.assertEqual(2, len(buffer.timestamp_deque))
        top_timestamp = buffer.timestamp_deque[0]
        self.assertEqual(1, top_timestamp[0])
        expected_time = time2 + 1 * TIME_DELTA_PER_TIME_STEP
        self.assertEqual(expected_time, top_timestamp[1])

        self.assertEqual((2, time3), buffer.timestamp_deque[1])

    def test_timestamp_queue_updates_freeing_data_exact_new_timestamp(self):
        buffer = SpectrogramBuffer(override_buffer_size=8, min_appendable_time_steps=1)
        buffer.unprocessed_end = 3
        buffer.pending_post_processing_end = 2
        buffer.allocated_begin = 4
        buffer.rows_unprocessed = 1
        buffer.rows_allocated = 7
        now = datetime.now(timezone.utc)
        time2 = now + 10 * timedelta(seconds=2)
        time3 = time2 + 18 * timedelta(seconds=3)
        buffer.timestamp_deque.append((5, now))
        buffer.timestamp_deque.append((0, time2))
        buffer.timestamp_deque.append((2, time3))

        buffer.mark_post_processing_complete(4)

        self.assertEqual(2, len(buffer.timestamp_deque))
        top_timestamp = buffer.timestamp_deque[0]
        self.assertEqual(0, top_timestamp[0])
        expected_time = time2
        self.assertEqual(expected_time, top_timestamp[1])

        self.assertEqual((2, time3), buffer.timestamp_deque[1])


if __name__ == '__main__':
    unittest.main()
