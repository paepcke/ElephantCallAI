import os
import numpy as np
from datetime import datetime, timezone, date
from scipy.io import wavfile

from embedded import DataCoordinator, PredictionManager, PredictionCollector, SignalUtils
from embedded.microphone.AudioBuffer import AudioBuffer
from embedded.microphone.AudioSpectrogramStream import AudioSpectrogramStream
from embedded.microphone.SpectrogramExtractor import SpectrogramExtractor
from embedded.predictors import ModelPredictor

# these hard-coded resource paths can be swapped out with environment variables later
WAV_PATH = "../../../../elephant_dataset/Test/rawaudio/nn04c_20180308_000000.wav"
MODEL_PATH = "../../../../models/remote_model.pt"
PREDICTION_INTERVALS_OUTPUT_PATH = "/tmp/prediction_intervals.txt"
BLACKOUT_INTERVALS_OUTPUT_PATH = "/tmp/blackout_intervals.txt"
PREDS_SAVE_PATH = "/tmp/preds.npy"


def integration_test_with_model_and_wav(small_buffer: bool = False):
    """
    This test does not require a microphone to be connected to your computer. It will instead use audio from a WAV file
    and insert it into the processing pipeline where mic-captured audio would normally be found.

    :param small_buffer: set this to 'True' if you want the test to run with a very small buffer size to test the system
    under memory pressure
    :return:
    """

    start = datetime.now(timezone.utc)

    os.system("rm {}".format(PREDICTION_INTERVALS_OUTPUT_PATH))
    os.system("rm {}".format(BLACKOUT_INTERVALS_OUTPUT_PATH))

    jump = 64
    predictor = ModelPredictor.ModelPredictor(MODEL_PATH, jump=jump)

    if small_buffer:
        data_coordinator = DataCoordinator.DataCoordinator(
            PREDICTION_INTERVALS_OUTPUT_PATH, BLACKOUT_INTERVALS_OUTPUT_PATH, override_buffer_size=2048 * 8, jump=jump,
            min_collectable_predictions=1)
    else:
        data_coordinator = DataCoordinator.DataCoordinator(
            PREDICTION_INTERVALS_OUTPUT_PATH, BLACKOUT_INTERVALS_OUTPUT_PATH, jump=jump, min_collectable_predictions=1)

    audio_buffer = AudioBuffer(min_appendable_time_steps=4096, min_consumable_rows=4096 + 255*800, override_buffer_size=(8000*4000 + 256))
    spec_extractor = SpectrogramExtractor()

    # Pre-load the audio buffer with data instead of streaming it from a microphone
    wav_data = read_wav(WAV_PATH)[:(8000*4000)]  # Samples we're interested in occur before t = 4000 seconds for this file

    time = datetime.combine(date.today(), datetime.min.time())  # Sets the time to midnight, which makes reading the intervals conveniently easy

    audio_buffer.append_data(wav_data, time)

    spec_stream = AudioSpectrogramStream(audio_buffer, spec_extractor, timeout=True)
    pred_mgr = PredictionManager.PredictionManager(predictor, timeout=True, verbose=True, give_up_threshold=100)
    pred_collector = PredictionCollector.PredictionCollector(timeout=True, verbose=True, give_up_threshold=100,
                                                             keep_predictions=True)

    SignalUtils.set_signal_handler([spec_stream, pred_mgr, pred_collector, data_coordinator],
                                   start_time=start, timeout=True)

    spec_stream.start(data_coordinator)
    pred_mgr.start(data_coordinator)
    pred_collector.start(data_coordinator)

    spec_stream.join()
    pred_mgr.join()
    preds = pred_collector.join()

    preds_cat = np.concatenate(preds, 0)
    np.save(PREDS_SAVE_PATH, preds_cat)


def read_wav(filepath: str) -> np.ndarray:
    wavobj = wavfile.read(filepath)
    arr = np.array(wavobj[1], dtype=np.int16)
    return arr


if __name__ == "__main__":
    integration_test_with_model_and_wav(True)
