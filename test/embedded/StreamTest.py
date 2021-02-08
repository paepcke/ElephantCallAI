import os
import numpy as np

from src.embedded import DataCoordinator, SpectrogramStream, PredictionManager, PredictionCollector
from src.embedded.predictors import ModelPredictor

# TODO: these hard-coded resource paths can be swapped out with environment variables later
SPECTROGRAM_NPY_FILE = "../../elephant_dataset/Test_Spectrograms/nn10b_20180604_spec.npy"
INTERVAL_OUTPUT_PATH = "/tmp/intervals.txt"
PREDS_SAVE_PATH = "/tmp/preds.npy"
LABELS_PATH = "../../elephant_dataset/Test_Spectrograms/nn10b_20180604_label.npy"


def main():
    os.system("rm {}".format(INTERVAL_OUTPUT_PATH))

    predictor = ModelPredictor.ModelPredictor("../../models/remote_model.pt")

    data_coordinator = DataCoordinator.DataCoordinator(INTERVAL_OUTPUT_PATH)
    spec_stream = SpectrogramStream.SpectrogramStream(SPECTROGRAM_NPY_FILE, max_time_steps=10000)
    pred_mgr = PredictionManager.PredictionManager(predictor)
    pred_collector = PredictionCollector.PredictionCollector()

    spec_stream.start(data_coordinator)
    pred_mgr.start(data_coordinator)
    pred_collector.start(data_coordinator)

    spec_stream.join()
    pred_mgr.join()
    preds = pred_collector.join()

    preds_cat = np.concatenate(preds, 0)
    np.save(PREDS_SAVE_PATH, preds_cat)

    print("Done!")


def compare_preds():
    labels = np.load(LABELS_PATH)
    preds = np.load(PREDS_SAVE_PATH)

    labels_clipped = labels[:preds.shape[0]]
    preds_threshed = np.where(preds > 0.5, 1, 0)
    diffs = labels_clipped - preds_threshed
    errors = np.sum(np.abs(diffs))
    total_acc = 100 - errors/preds.shape[0] * 100
    print("Total accuracy: {}%".format(total_acc))
    # TODO: more stats, like FP, TP, FN, TN, analysis


if __name__ == "__main__":
    main()
    compare_preds()


