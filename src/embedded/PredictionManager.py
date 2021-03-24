from threading import Thread

from embedded.predictors.Predictor import Predictor
from embedded.DataCoordinator import DataCoordinator

GIVE_UP_THRESHOLD = 250
TIME_WINDOW = 256*4 + 3*64
LOCK_TIMEOUT_IN_SECONDS = 0.1


class PredictionManager:
    predictor: Predictor
    predictor_thread: Thread
    timeout: bool
    verbose: bool
    give_up_threshold: int

    def __init__(self, predictor: Predictor, timeout: bool = False, verbose: bool = False,
                 give_up_threshold: int = GIVE_UP_THRESHOLD):
        self.predictor = predictor
        self.timeout = timeout
        self.verbose = verbose
        self.give_up_threshold = give_up_threshold

    def start(self, data_coordinator: DataCoordinator):
        self.predictor_thread = Thread(target=self.predict, args=(data_coordinator,))
        self.predictor_thread.start()

    def predict(self, data_coordinator: DataCoordinator):
        num_consecutive_times_buffer_empty = 0
        total_time_steps_predicted = 0

        while not self.timeout or num_consecutive_times_buffer_empty < self.give_up_threshold:
            if not data_coordinator.data_available_for_prediction_lock.acquire(timeout=LOCK_TIMEOUT_IN_SECONDS):
                num_consecutive_times_buffer_empty += 1
                continue
            else:
                data_coordinator.data_available_for_prediction_lock.release()
            num_time_steps_predicted = data_coordinator.make_predictions(self.predictor, TIME_WINDOW)
            total_time_steps_predicted += num_time_steps_predicted
            if num_time_steps_predicted != 0:
                if self.verbose:
                    print("PredictionManager.py: {} total time steps predicted so far".format(total_time_steps_predicted))
                num_consecutive_times_buffer_empty = 0
            else:
                num_consecutive_times_buffer_empty += 1

        if self.verbose:
            print("Total time steps predicted by prediction thread: {}".format(total_time_steps_predicted))

    def join(self):
        self.predictor_thread.join()
