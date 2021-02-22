import os
import unittest
from datetime import datetime, timezone
from collections import deque
import numpy as np
from typing import List

from src.embedded.DataCoordinator import DataCoordinator
from src.embedded.TransitionState import TransitionState
from src.embedded.SpectrogramBuffer import TIME_DELTA_PER_TIME_STEP, FREQ_BINS
from src.embedded.predictors.ConstPredictor import ConstPredictor


INTERVAL_OUTPUT_PATH = "/tmp/intervals.txt"
TEST_LOCK_TIMEOUT_SECONDS = 0.2


class DataCoordinatorTest(unittest.TestCase):
    """Unit tests for the DataCoordinator class."""

    # This test is sort of an integration test for the entire spectrogram-to-predicted-intervals pipeline
    def test_data_pipeline(self):
        clear_interval_file()

        coordinator = DataCoordinator(INTERVAL_OUTPUT_PATH, override_buffer_size=16, min_appendable_time_steps=1, jump=1)
        now = datetime.now(timezone.utc)
        data = np.zeros((12, FREQ_BINS))
        ones_predictor = ConstPredictor(1)
        zeros_predictor = ConstPredictor(-1)

        coordinator.write(data, timestamp=now)
        coordinator.make_predictions(ones_predictor, 4)
        coordinator.make_predictions(zeros_predictor, 4)

        coordinator.finalize_predictions(4)
        coordinator.finalize_predictions(4)

        coordinator.write(data)
        coordinator.make_predictions(ones_predictor, 3)
        coordinator.make_predictions(zeros_predictor, 4)
        coordinator.make_predictions(zeros_predictor, 3)
        coordinator.finalize_predictions(9)
        coordinator.wrap_up()

        # Does the internal state of the spectrogram buffer look right?
        self.assertEqual(8, coordinator.spectrogram_buffer.unprocessed_end)
        self.assertEqual(2, coordinator.spectrogram_buffer.pending_post_processing_end)
        self.assertEqual(7, coordinator.spectrogram_buffer.rows_allocated)
        self.assertEqual(6, coordinator.spectrogram_buffer.rows_unprocessed)

        # Does the output file look right?
        lines = get_lines_of_interval_file()
        self.assertEqual(2, len(lines))

        begin_interval_0 = now
        end_interval_0 = now + 4*TIME_DELTA_PER_TIME_STEP
        begin_interval_1 = now + 8*TIME_DELTA_PER_TIME_STEP
        end_interval_1 = now + 11*TIME_DELTA_PER_TIME_STEP
        self.assertEqual("{},{}\n".format(begin_interval_0.isoformat(), end_interval_0.isoformat()), lines[0])
        self.assertEqual("{},{}\n".format(begin_interval_1.isoformat(), end_interval_1.isoformat()), lines[1])

    def test_dont_predict_without_another_timestamp_if_not_leaving_min_appendable_time_steps(self):
        coordinator = DataCoordinator(INTERVAL_OUTPUT_PATH, override_buffer_size=16, min_appendable_time_steps=6, jump=3)
        data = np.zeros((6, FREQ_BINS))
        predictor = ConstPredictor(1)

        coordinator.write(data)

        processed = coordinator.make_predictions(predictor, 6)
        coordinator.wrap_up()

        self.assertEqual(0, processed)

    def test_can_predict_without_another_timestamp_if_leaving_min_appendable_time_steps_with_overlap_allowance(self):
        coordinator = DataCoordinator(INTERVAL_OUTPUT_PATH, override_buffer_size=16, min_appendable_time_steps=4, jump=2)
        data = np.zeros((8, FREQ_BINS))
        predictor = ConstPredictor(1)

        coordinator.write(data)

        processed = coordinator.make_predictions(predictor, 4)
        coordinator.wrap_up()

        self.assertEqual(4, processed)
        self.assertEqual(2, coordinator.spectrogram_buffer.rows_allocated - coordinator.spectrogram_buffer.rows_unprocessed)

    def test_time_window_must_be_a_multiple_of_overlap_allowance(self):
        coordinator = DataCoordinator(INTERVAL_OUTPUT_PATH, override_buffer_size=16, min_appendable_time_steps=4, jump=2)
        data = np.zeros((8, FREQ_BINS))
        predictor = ConstPredictor(1)

        coordinator.write(data)

        try:
            coordinator.make_predictions(predictor, 5)
        except ValueError:
            coordinator.wrap_up()
            return
        coordinator.wrap_up()
        self.fail("Expected exception but none thrown")

    def test_time_window_must_be_a_multiple_of_min_appendable_if_no_overlap_allowance(self):
        coordinator = DataCoordinator(INTERVAL_OUTPUT_PATH, override_buffer_size=16, min_appendable_time_steps=3)
        data = np.zeros((9, FREQ_BINS))
        predictor = ConstPredictor(1)

        coordinator.write(data)

        try:
            coordinator.make_predictions(predictor, 5)
        except ValueError:
            coordinator.wrap_up()
            return
        coordinator.wrap_up()
        self.fail("Expected exception but none thrown")

    def test_can_make_fewer_predictions_than_requested_to_allow_future_predictions(self):
        coordinator = DataCoordinator(INTERVAL_OUTPUT_PATH, override_buffer_size=16, min_appendable_time_steps=6, jump=3)
        data = np.zeros((12, FREQ_BINS))
        predictor = ConstPredictor(1)

        coordinator.write(data)

        processed = coordinator.make_predictions(predictor, 12)
        coordinator.wrap_up()

        self.assertEqual(9, processed)

    def test_predict_entire_outstanding_data_if_another_timestamp_exists(self):
        coordinator = DataCoordinator(INTERVAL_OUTPUT_PATH, override_buffer_size=16, min_appendable_time_steps=4, jump=2)
        data = np.zeros((12, FREQ_BINS))
        predictor = ConstPredictor(1)
        now = datetime.now(timezone.utc)

        coordinator.write(data)

        coordinator.spectrogram_buffer.unprocessed_timestamp_deque.append((5, now + 40*TIME_DELTA_PER_TIME_STEP))

        processed = coordinator.make_predictions(predictor, 4)
        coordinator.wrap_up()

        self.assertEqual(5, processed)

    def test_predict_subset_of_outstanding_data_if_another_timestamp_exists(self):
        coordinator = DataCoordinator(INTERVAL_OUTPUT_PATH, override_buffer_size=16, min_appendable_time_steps=2, jump=1)
        data = np.zeros((12, FREQ_BINS))
        predictor = ConstPredictor(1)
        now = datetime.now(timezone.utc)

        coordinator.write(data)

        coordinator.spectrogram_buffer.unprocessed_timestamp_deque.append((5, now + 40*TIME_DELTA_PER_TIME_STEP))

        processed = coordinator.make_predictions(predictor, 2)
        coordinator.wrap_up()

        self.assertEqual(2, processed)

    def test_exception_thrown_if_jump_geq_min_appendable_len(self):
        try:
            DataCoordinator(INTERVAL_OUTPUT_PATH, override_buffer_size=16, min_appendable_time_steps=1, jump=5)
        except ValueError:
            return
        self.fail("Expected exception but none thrown. 'overlap_allowance' >= 'time_window' should not be allowed.")

    def test_cant_predict_less_than_min_appendable_time_steps(self):
        coordinator = DataCoordinator(INTERVAL_OUTPUT_PATH, override_buffer_size=16, min_appendable_time_steps=3, jump=1)
        data = np.zeros((8, FREQ_BINS))
        predictor = ConstPredictor(1)

        coordinator.write(data)

        try:
            coordinator.make_predictions(predictor, 2)
        except ValueError:
            coordinator.wrap_up()
            return
        coordinator.wrap_up()
        self.fail("Expected exception but none thrown. Should not be able to predict less than min_appendable_time_steps at once.")

    def test_prediction_lock_allows_entry_when_appropriate_without_discontinuity(self):
        coordinator = DataCoordinator(INTERVAL_OUTPUT_PATH, override_buffer_size=16, min_appendable_time_steps=4, jump=2)
        data = np.zeros((6, FREQ_BINS))

        coordinator.write(data)

        got_lock = coordinator.data_available_for_prediction_lock.acquire(timeout=TEST_LOCK_TIMEOUT_SECONDS)
        coordinator.wrap_up()

        self.assertTrue(got_lock)

    def test_prediction_lock_disallows_entry_when_appropriate_without_discontinuity(self):
        coordinator = DataCoordinator(INTERVAL_OUTPUT_PATH, override_buffer_size=16, min_appendable_time_steps=4, jump=2)
        data = np.zeros((4, FREQ_BINS))

        coordinator.write(data)

        got_lock = coordinator.data_available_for_prediction_lock.acquire(timeout=TEST_LOCK_TIMEOUT_SECONDS)
        coordinator.wrap_up()

        self.assertFalse(got_lock)

    def test_prediction_lock_allows_entry_when_appropriate_with_discontinuity(self):
        coordinator = DataCoordinator(INTERVAL_OUTPUT_PATH, override_buffer_size=16, min_appendable_time_steps=4,
                                      jump=2)
        data = np.zeros((10, FREQ_BINS))
        now = datetime.now(timezone.utc)

        coordinator.write(data)

        coordinator.spectrogram_buffer.unprocessed_timestamp_deque.append((4, now + 40 * TIME_DELTA_PER_TIME_STEP))

        got_lock = coordinator.data_available_for_prediction_lock.acquire(timeout=TEST_LOCK_TIMEOUT_SECONDS)
        coordinator.wrap_up()

        self.assertTrue(got_lock)

    def test_get_detection_intervals(self):
        coordinator = DataCoordinator(INTERVAL_OUTPUT_PATH, override_buffer_size=16)
        now = datetime.now(timezone.utc)
        time1 = now + 40*TIME_DELTA_PER_TIME_STEP
        time2 = time1 + 40*TIME_DELTA_PER_TIME_STEP

        timestamps = deque()
        timestamps.append((0, now + 2*TIME_DELTA_PER_TIME_STEP))
        timestamps.append((3, time1))
        timestamps.append((4, time2))

        finalized_predictions = np.ones((10,))
        finalized_predictions[6] = 0
        finalized_predictions[8] = 0

        coordinator.transition_state = TransitionState(now, 2)

        intervals = coordinator.get_detection_intervals(finalized_predictions, timestamps)
        coordinator.wrap_up()

        self.assertEqual(4, len(intervals))
        self.assertEqual((now, now + 5*TIME_DELTA_PER_TIME_STEP), intervals[0])
        self.assertEqual((time1, time1 + 1*TIME_DELTA_PER_TIME_STEP), intervals[1])
        self.assertEqual((time2, time2 + 2 * TIME_DELTA_PER_TIME_STEP), intervals[2])
        self.assertEqual((time2 + 3*TIME_DELTA_PER_TIME_STEP, time2 + 4*TIME_DELTA_PER_TIME_STEP), intervals[3])

        self.assertEqual(time2 + 5*TIME_DELTA_PER_TIME_STEP, coordinator.transition_state.start_time)
        self.assertEqual(1, coordinator.transition_state.num_consecutive_ones)

    def test_get_detection_interval_without_initial_state_or_leftover_state(self):
        coordinator = DataCoordinator(INTERVAL_OUTPUT_PATH, override_buffer_size=16)
        now = datetime.now(timezone.utc)

        timestamps = deque()
        timestamps.append((0, now))

        finalized_predictions = np.ones((10,))
        finalized_predictions[9] = 0

        intervals = coordinator.get_detection_intervals(finalized_predictions, timestamps)
        coordinator.wrap_up()

        self.assertEqual(1, len(intervals))
        self.assertEqual((now, now + 9*TIME_DELTA_PER_TIME_STEP), intervals[0])

        self.assertIsNone(coordinator.transition_state.start_time)
        self.assertIsNone(coordinator.transition_state.num_consecutive_ones)

    def test_write_intervals_to_file(self):
        clear_interval_file()

        coordinator = DataCoordinator(INTERVAL_OUTPUT_PATH, override_buffer_size=16)
        now = datetime.now(timezone.utc)
        start = now
        end = now + 2*TIME_DELTA_PER_TIME_STEP
        intervals = [(start, end)]
        coordinator.save_detection_intervals(intervals)
        coordinator.wrap_up()

        lines = get_lines_of_interval_file()

        self.assertEqual(1, len(lines))
        self.assertEqual("{},{}\n".format(start.isoformat(), end.isoformat()), lines[0])


def clear_interval_file():
    os.system("rm {}".format(INTERVAL_OUTPUT_PATH))


def get_lines_of_interval_file() -> List[str]:
    with open(INTERVAL_OUTPUT_PATH, "r") as file:
        lines = file.readlines()
    return lines


if __name__ == '__main__':
    unittest.main()