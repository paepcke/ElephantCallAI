from threading import Thread
from time import sleep

from embedded.predictors.Predictor import Predictor
from src.embedded.DataCoordinator import DataCoordinator

GIVE_UP_THRESHOLD = 1000
TIME_WINDOW = 256*4  # TODO: Make this more precise to optimize the prediction parallelism
DEFAULT_OVERLAP_ALLOWANCE = 64
SLEEP_BETWEEN_PREDICTIONS_IN_SECONDS = 0.01


class PredictionManager:
    predictor: Predictor
    predictor_thread: Thread
    overlap_allowance: int

    def __init__(self, predictor: Predictor, overlap_allowance: int = DEFAULT_OVERLAP_ALLOWANCE):
        self.predictor = predictor
        self.overlap_allowance = overlap_allowance

    def start(self, data_coordinator: DataCoordinator):
        self.predictor_thread = Thread(target=self.predict, args=(data_coordinator,))
        self.predictor_thread.start()

    def predict(self, data_coordinator: DataCoordinator):
        num_consecutive_times_buffer_empty = 0
        total_time_steps_predicted = 0

        while num_consecutive_times_buffer_empty < GIVE_UP_THRESHOLD:
            sleep(SLEEP_BETWEEN_PREDICTIONS_IN_SECONDS)
            num_time_steps_predicted = data_coordinator.make_predictions(self.predictor, TIME_WINDOW, self.overlap_allowance)
            total_time_steps_predicted += num_time_steps_predicted
            if num_time_steps_predicted != 0:
                num_consecutive_times_buffer_empty = 0
            else:
                num_consecutive_times_buffer_empty += 1

        print("Total time steps predicted by prediction thread: {}".format(total_time_steps_predicted))

    def join(self):
        self.predictor_thread.join()
