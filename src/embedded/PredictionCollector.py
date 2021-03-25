from threading import Thread
from typing import List, Optional
import numpy as np

from embedded.Closeable import Closeable
from embedded.DataCoordinator import DataCoordinator

GIVE_UP_THRESHOLD = 250
TIME_WINDOW = 256
LOCK_TIMEOUT_IN_SECONDS = 0.1


class PredictionCollector(Closeable):
    """An object which governs a thread that gathers predictions made by a model and converts them
    into useful output."""
    collector_thread: Thread
    timeout: bool
    verbose: bool
    predictions_list: Optional[List[np.ndarray]]
    give_up_threshold: int
    closed: bool

    def __init__(self, timeout: bool = False, verbose: bool = False, keep_predictions: bool = False,
                 give_up_threshold: int = GIVE_UP_THRESHOLD):
        super().__init__()
        self.closed = False
        self.timeout = timeout
        self.verbose = verbose
        if keep_predictions:
            self.predictions_list = []
        else:
            self.predictions_list = None
        self.give_up_threshold = give_up_threshold

    def start(self, data_coordinator: DataCoordinator):
        # 'daemon' threads are killed when the parent process dies
        self.collector_thread = Thread(target=self.collect_predictions, args=(data_coordinator,), daemon=True)
        self.collector_thread.start()
        self.predictions_list = []

    def collect_predictions(self, data_coordinator: DataCoordinator):
        num_consecutive_times_buffer_empty = 0
        total_time_steps_collected = 0

        while not self.closed and (not self.timeout or num_consecutive_times_buffer_empty < self.give_up_threshold):
            if not data_coordinator.predictions_available_for_collection_lock\
                    .acquire(timeout=LOCK_TIMEOUT_IN_SECONDS if self.timeout else -1):
                num_consecutive_times_buffer_empty += 1
                continue
            else:
                data_coordinator.predictions_available_for_collection_lock.release()
            num_time_steps_collected, predictions = data_coordinator.finalize_predictions(TIME_WINDOW)
            total_time_steps_collected += num_time_steps_collected
            if num_time_steps_collected != 0:
                if self.verbose:
                    print("PredictionCollector.py: {} time steps worth of predictions collected so far"
                          .format(total_time_steps_collected))
                num_consecutive_times_buffer_empty = 0
            else:
                num_consecutive_times_buffer_empty += 1
            if predictions is not None and self.predictions_list is not None:
                self.predictions_list.append(predictions)

        if self.verbose:
            print("Total time steps collected by collector thread: {}".format(total_time_steps_collected))

    def close(self):
        self.closed = True

    def join(self) -> Optional[List[np.ndarray]]:
        self.collector_thread.join()
        return self.predictions_list
