import os

from src.embedded import DataCoordinator, SpectrogramStream, PredictionManager, PredictionCollector
from src.embedded.predictors import ConstPredictor

SPECTROGRAM_NPY_FILE = "../../elephant_dataset/Test_Spectrograms/nn10b_20180604_spec.npy"
INTERVAL_OUTPUT_PATH = "/tmp/intervals.txt"


def main():
    os.system("rm {}".format(INTERVAL_OUTPUT_PATH))

    predictor = ConstPredictor.ConstPredictor(-1)

    data_coordinator = DataCoordinator.DataCoordinator(INTERVAL_OUTPUT_PATH)
    spec_stream = SpectrogramStream.SpectrogramStream(SPECTROGRAM_NPY_FILE, max_time_steps=100)
    pred_mgr = PredictionManager.PredictionManager(predictor)
    pred_collector = PredictionCollector.PredictionCollector()

    spec_stream.start(data_coordinator)
    pred_mgr.start(data_coordinator)
    pred_collector.start(data_coordinator)

    spec_stream.join()
    pred_mgr.join()
    pred_collector.join()

    print("Done!")


if __name__ == "__main__":
    main()

