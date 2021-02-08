from threading import Thread
from time import sleep
from typing import List
import numpy as np

from src.embedded.DataCoordinator import DataCoordinator

GIVE_UP_THRESHOLD = 1000
TIME_WINDOW = 256
SLEEP_BETWEEN_COLLECTIONS_IN_SECONDS = 0.01


class PredictionCollector:
    collector_thread: Thread
    predictions_list: List[np.ndarray]

    def __init__(self):
        pass

    def start(self, data_coordinator: DataCoordinator):
        self.collector_thread = Thread(target=self.collect_predictions, args=(data_coordinator,))
        self.collector_thread.start()
        self.predictions_list = []

    def collect_predictions(self, data_coordinator: DataCoordinator):
        num_consecutive_times_buffer_empty = 0
        total_time_steps_collected = 0

        while num_consecutive_times_buffer_empty < GIVE_UP_THRESHOLD:
            sleep(SLEEP_BETWEEN_COLLECTIONS_IN_SECONDS)
            num_time_steps_collected, predictions = data_coordinator.finalize_predictions(TIME_WINDOW)
            total_time_steps_collected += num_time_steps_collected
            if num_time_steps_collected != 0:
                num_consecutive_times_buffer_empty = 0
            else:
                num_consecutive_times_buffer_empty += 1
            if predictions is not None:
                self.predictions_list.append(predictions)

        print("Total time steps collected by collector thread: {}".format(total_time_steps_collected))

    def join(self) -> List[np.ndarray]:
        self.collector_thread.join()
        return self.predictions_list
