import torch
import numpy as np
from threading import Thread, Lock
from collections import deque

from embedded.Closeable import Closeable
from embedded.SynchronizationUtils import yield_to_thread_scheduler
from embedded.led.LedController import LedController
from embedded.microphone.SpectrogramExtractor import SpectrogramExtractor
from embedded.ModelingConstants import GUNSHOT_MODEL_DETECTION_THRESHOLD


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
INPUT_LOCK_TIMEOUT_SECONDS = 4


class GunshotDetectionModule(Closeable):
    def __init__(self, model: torch.nn.Module,
                 led_controller: LedController,
                 spec_extractor: SpectrogramExtractor,
                 detection_threshold: float = GUNSHOT_MODEL_DETECTION_THRESHOLD):
        super().__init__()
        model = model.to(DEVICE)
        self.model = model
        self.led_controller = led_controller
        self.spec_ex = spec_extractor
        self.detection_threshold = detection_threshold

        self.closed: bool = False
        self.model_thread: Thread = Thread(target=self._model_thread, args=(), daemon=True)
        self.task_deque = deque()
        self.deque_lock: Lock = Lock()

        self.model_thread.start()

    def _model_thread(self):
        while not self.closed:
            if self.deque_lock.acquire(timeout=INPUT_LOCK_TIMEOUT_SECONDS):
                if len(self.task_deque) > 0:
                    time_domain_data = self.task_deque.popleft()
                    self.deque_lock.release()

                    spectrogram_data = self.spec_ex.extract_spectrogram(time_domain_data)
                    spectrogram_data = self.transform_data(spectrogram_data)
                    self.detect(spectrogram_data)
                else:
                    self.deque_lock.release()
            yield_to_thread_scheduler()

        return

    def transform_data(self, spectrogram_data: np.ndarray) -> np.ndarray:
        """
        Transforms spectrogram data into something the gunshot model has been trained to work with

        :param spectrogram_data: raw spectrogram data
        :return: transformed spectrogram data
        """
        return 10*np.log10(spectrogram_data)

    def queue_time_domain_data(self, time_domain_data: np.ndarray):
        with self.deque_lock:
            self.task_deque.append(time_domain_data)

    def join(self):
        self.model_thread.join()

    def detect(self, spectrogram_data: np.ndarray):
        spectrogram_data = torch.tensor(spectrogram_data).to(DEVICE)
        spectrogram_data = spectrogram_data.unsqueeze(dim=0).float()
        out = self.model(spectrogram_data)
        if out > self.detection_threshold:
            self.led_controller.illuminate()

    def close(self):
        self.closed = True
