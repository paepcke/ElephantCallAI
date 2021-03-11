import os
import numpy as np
from datetime import datetime, timezone
from time import sleep

from embedded import DataCoordinator, PredictionManager, PredictionCollector
from embedded.microphone.AudioBuffer import AudioBuffer
from embedded.microphone.AudioCapturer import AudioCapturer
from embedded.microphone.AudioSpectrogramStream import AudioSpectrogramStream
from embedded.microphone.SpectrogramExtractor import SpectrogramExtractor
from embedded.predictors import ModelPredictor

# these hard-coded resource paths can be swapped out with environment variables later
SPECTROGRAM_NPY_FILE = "../../../elephant_dataset/Test_Spectrograms/nn10b_20180604_spec.npy"
MODEL_PATH = "../../../models/remote_model.pt"
PREDICTION_INTERVALS_OUTPUT_PATH = "/tmp/prediction_intervals.txt"
BLACKOUT_INTERVALS_OUTPUT_PATH = "/tmp/blackout_intervals.txt"
PREDS_SAVE_PATH = "/tmp/preds.npy"
LABELS_PATH = "../../../elephant_dataset/Test_Spectrograms/nn10b_20180604_label.npy"


def integration_test_with_model_and_audio(small_buffer: bool = False):
    """This test requires a microphone to be connected to your computer. It will collect audio, transform it, and
    run it through the model, and it will do so indefinitely."""
    start = datetime.now(timezone.utc)

    os.system("rm {}".format(PREDICTION_INTERVALS_OUTPUT_PATH))
    os.system("rm {}".format(BLACKOUT_INTERVALS_OUTPUT_PATH))

    jump = 64
    predictor = ModelPredictor.ModelPredictor(MODEL_PATH, jump=jump)

    if small_buffer:
        data_coordinator = DataCoordinator.DataCoordinator(
            PREDICTION_INTERVALS_OUTPUT_PATH, BLACKOUT_INTERVALS_OUTPUT_PATH, override_buffer_size=2048 * 8, jump=jump)
    else:
        data_coordinator = DataCoordinator.DataCoordinator(
            PREDICTION_INTERVALS_OUTPUT_PATH, BLACKOUT_INTERVALS_OUTPUT_PATH, jump=jump)

    audio_buffer = AudioBuffer(min_appendable_time_steps=4096, min_consumable_rows=4096 + 255*800)
    spec_extractor = SpectrogramExtractor()


    audio_capturer = AudioCapturer(audio_buffer, frames_per_buffer=4096)
    spec_stream = AudioSpectrogramStream(audio_buffer, spec_extractor)
    pred_mgr = PredictionManager.PredictionManager(predictor)
    pred_collector = PredictionCollector.PredictionCollector()

    audio_capturer.start()
    sleep(27)

    spec_stream.start(data_coordinator)
    pred_mgr.start(data_coordinator)
    pred_collector.start(data_coordinator)

    spec_stream.join()
    pred_mgr.join()
    preds = pred_collector.join()

    data_coordinator.wrap_up()

    preds_cat = np.concatenate(preds, 0)
    np.save(PREDS_SAVE_PATH, preds_cat)

    finish = datetime.now(timezone.utc)

    print("Done in {}".format(finish - start))


if __name__ == "__main__":
    integration_test_with_model_and_audio(True)
