import os
import numpy as np
from datetime import datetime, timezone
from time import sleep

from embedded import DataCoordinator, PredictionManager, PredictionCollector, SignalUtils
from embedded.FileUtils import assert_path_exists
from embedded.microphone.AudioBuffer import AudioBuffer
from embedded.microphone.AudioCapturer import AudioCapturer
from embedded.microphone.AudioSpectrogramStream import AudioSpectrogramStream
from embedded.microphone.SpectrogramExtractor import SpectrogramExtractor
from embedded.predictors import SingleStageModelPredictor, TwoStageModelPredictor

# these hard-coded resource paths can be swapped out with environment variables later
SINGLE_STAGE_MODEL_PATH = "../../../../Integration_Test_Data/models/remote_model.pt"
TWO_STAGE_MODEL_PATH = "../../../../Integration_Test_Data/models/2stage"
PREDICTION_INTERVALS_OUTPUT_PATH = "/tmp/prediction_intervals.txt"
BLACKOUT_INTERVALS_OUTPUT_PATH = "/tmp/blackout_intervals.txt"
PREDS_SAVE_PATH = "/tmp/preds.npy"


def integration_test_with_model_and_audio(small_buffer: bool = False, two_stage_model: bool = False):
    """
    This test requires a microphone to be connected to your computer. It will collect audio, transform it, and
    run it through the model, and it will do so indefinitely.

    :param small_buffer: set this to 'True' if you want the test to run with a very small buffer size to test the system
    under memory pressure
    :return:
    """

    start = datetime.now(timezone.utc)

    os.system("rm {}".format(PREDICTION_INTERVALS_OUTPUT_PATH))
    os.system("rm {}".format(BLACKOUT_INTERVALS_OUTPUT_PATH))

    jump = 64
    if two_stage_model:
        assert_path_exists(TWO_STAGE_MODEL_PATH, "You must provide a directory containing two PyTorch models.")
        predictor = TwoStageModelPredictor.TwoStageModelPredictor(SINGLE_STAGE_MODEL_PATH, jump=jump)
    else:
        assert_path_exists(SINGLE_STAGE_MODEL_PATH, "You must provide a PyTorch model.")
        predictor = SingleStageModelPredictor.SingleStageModelPredictor(SINGLE_STAGE_MODEL_PATH, jump=jump)

    if small_buffer:
        data_coordinator = DataCoordinator.DataCoordinator(
            PREDICTION_INTERVALS_OUTPUT_PATH, BLACKOUT_INTERVALS_OUTPUT_PATH, override_buffer_size=2048 * 8, jump=jump)
    else:
        data_coordinator = DataCoordinator.DataCoordinator(
            PREDICTION_INTERVALS_OUTPUT_PATH, BLACKOUT_INTERVALS_OUTPUT_PATH, jump=jump)

    audio_buffer = AudioBuffer(min_appendable_time_steps=4096, min_consumable_rows=4096 + 255*800)
    spec_extractor = SpectrogramExtractor()


    audio_capturer = AudioCapturer(audio_buffer, frames_per_buffer=4096)
    spec_stream = AudioSpectrogramStream(audio_buffer, spec_extractor, timeout=True)
    pred_mgr = PredictionManager.PredictionManager(predictor, timeout=True, verbose=True)
    pred_collector = PredictionCollector.PredictionCollector(timeout=True, verbose=True, keep_predictions=True)

    SignalUtils.set_signal_handler([audio_capturer, spec_stream, pred_mgr, pred_collector, data_coordinator],
                                   start_time=start, timeout=True)

    audio_capturer.start()
    sleep(27)

    spec_stream.start(data_coordinator)
    pred_mgr.start(data_coordinator)
    pred_collector.start(data_coordinator)

    spec_stream.join()
    pred_mgr.join()
    preds = pred_collector.join()

    preds_cat = np.concatenate(preds, 0)
    np.save(PREDS_SAVE_PATH, preds_cat)


if __name__ == "__main__":
    integration_test_with_model_and_audio(True)
