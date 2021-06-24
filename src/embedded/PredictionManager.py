from threading import Thread

from embedded.Closeable import Closeable
from embedded.predictors.Predictor import Predictor
from embedded.DataCoordinator import DataCoordinator
from embedded.SynchronizationUtils import yield_to_thread_scheduler

GIVE_UP_THRESHOLD = 250
TIME_WINDOW = 256*4 + 3*64
LOCK_TIMEOUT_IN_SECONDS = 0.1


class PredictionManager(Closeable):
    """
    An object which governs a thread that makes model predictions on spectrogram data and stores those
    predictions for later collection.
    """

    predictor: Predictor
    predictor_thread: Thread
    timeout: bool
    verbose: bool
    give_up_threshold: int
    closed: bool

    def __init__(self, predictor: Predictor, timeout: bool = False, verbose: bool = False,
                 give_up_threshold: int = GIVE_UP_THRESHOLD):
        super().__init__()
        self.closed = False
        self.predictor = predictor
        self.timeout = timeout
        self.verbose = verbose
        self.give_up_threshold = give_up_threshold

    def start(self, data_coordinator: DataCoordinator):
        # 'daemon' threads are killed when the parent process dies
        self.predictor_thread = Thread(target=self.predict, args=(data_coordinator,), daemon=True)
        self.predictor_thread.start()

    def predict(self, data_coordinator: DataCoordinator):
        num_consecutive_times_buffer_empty = 0
        total_time_steps_predicted = 0

        while not self.closed and (not self.timeout or num_consecutive_times_buffer_empty < self.give_up_threshold):
            if not data_coordinator.data_available_for_prediction_lock\
                    .acquire(timeout=LOCK_TIMEOUT_IN_SECONDS if self.timeout else -1):
                num_consecutive_times_buffer_empty += 1
                continue
            else:
                data_coordinator.data_available_for_prediction_lock.release()
            num_time_steps_predicted = data_coordinator.make_predictions(self.predictor, TIME_WINDOW)
            total_time_steps_predicted += num_time_steps_predicted
            if num_time_steps_predicted != 0:
                if self.verbose:
                    print(f"PredictionManager.py: {total_time_steps_predicted} total time steps predicted so far")
                num_consecutive_times_buffer_empty = 0
            else:
                num_consecutive_times_buffer_empty += 1
            yield_to_thread_scheduler()

        if self.verbose:
            print(f"Total time steps predicted by prediction thread: {total_time_steps_predicted}")

    def close(self):
        self.closed = True

    def join(self):
        self.predictor_thread.join()
