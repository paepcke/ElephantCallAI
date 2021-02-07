from threading import Thread
from time import sleep

from src.embedded.DataCoordinator import DataCoordinator

GIVE_UP_THRESHOLD = 10
TIME_WINDOW = 256
SLEEP_BETWEEN_COLLECTIONS_IN_SECONDS = 0.1


class PredictionCollector:
    collector_thread: Thread

    def __init__(self):
        pass

    def start(self, data_coordinator: DataCoordinator):
        self.collector_thread = Thread(target=self.collect_predictions, args=(data_coordinator,))
        self.collector_thread.start()

    def collect_predictions(self, data_coordinator: DataCoordinator):
        num_consecutive_times_buffer_empty = 0
        total_time_steps_collected = 0

        while num_consecutive_times_buffer_empty < GIVE_UP_THRESHOLD:
            sleep(SLEEP_BETWEEN_COLLECTIONS_IN_SECONDS)
            num_time_steps_collected = data_coordinator.finalize_predictions(TIME_WINDOW)
            total_time_steps_collected += num_time_steps_collected
            if num_time_steps_collected != 0:
                num_consecutive_times_buffer_empty = 0
            else:
                num_consecutive_times_buffer_empty += 1

        print("Total time steps collected by collector thread: {}".format(total_time_steps_collected))

    def join(self):
        self.collector_thread.join()
