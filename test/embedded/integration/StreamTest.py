import os
import numpy as np
from datetime import datetime, timezone

from embedded import DataCoordinator, FileSpectrogramStream, PredictionManager, PredictionCollector, SignalUtils
from embedded.predictors import ModelPredictor

# TODO: these hard-coded resource paths can be swapped out with environment variables later
SPECTROGRAM_NPY_FILE = "../../elephant_dataset/Test_Spectrograms/nn10b_20180604_spec.npy"
MODEL_PATH = "../../models/remote_model.pt"
PREDICTION_INTERVALS_OUTPUT_PATH = "/tmp/prediction_intervals.txt"
BLACKOUT_INTERVALS_OUTPUT_PATH = "/tmp/blackout_intervals.txt"
PREDS_SAVE_PATH = "/tmp/preds.npy"
LABELS_PATH = "../../elephant_dataset/Test_Spectrograms/nn10b_20180604_label.npy"


def integration_test_with_model(small_buffer: bool = False):
    """
    This test will bypass the audio portion of the pipeline, streaming spectrogram data
    from a file into the DataCoordinator. Useful for testing this part in isolation.

    :param small_buffer:
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
        drop_data = False
    else:
        data_coordinator = DataCoordinator.DataCoordinator(
            PREDICTION_INTERVALS_OUTPUT_PATH, BLACKOUT_INTERVALS_OUTPUT_PATH, jump=jump,
            min_collectable_predictions=1)
        drop_data = True

    spec_stream = FileSpectrogramStream.FileSpectrogramStream(SPECTROGRAM_NPY_FILE, drop_data=drop_data)
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


def compare_preds():
    labels = np.load(LABELS_PATH)
    preds = np.load(PREDS_SAVE_PATH)

    labels_clipped = labels[:preds.shape[0]]
    preds_threshed = np.where(preds > 0.5, 1, 0)
    diffs = labels_clipped - preds_threshed
    errors = np.sum(np.abs(diffs))
    total_acc = 100 - errors/preds.shape[0] * 100
    print("Total accuracy: {}%".format(total_acc))

    true_pos = np.dot(labels_clipped.T, preds_threshed)
    false_pos = np.dot((1 - labels_clipped).T, preds_threshed)
    true_neg = np.dot((1 - labels_clipped).T, 1 - preds_threshed)
    false_neg = np.dot(labels_clipped.T, 1 - preds_threshed)


    class_balance = (true_pos + false_neg)/(true_neg + false_pos)  # ratio of positive examples to negative examples
    pos_acc = true_pos / (true_pos + false_neg)
    neg_acc = true_neg / (true_neg + false_pos)
    precision = true_pos / (true_pos + false_pos)

    print("Class Balance (ratio of # positive examples to # negative examples): {}\n"
          "Positive Accuracy (% of positive examples correctly classified): {}%\n"
          "Negative Accuracy (% of negative examples correctly classified): {}%\n"
          "Precision (% correctness on examples predicted positive): {}%".format(class_balance, 100*pos_acc, 100*neg_acc, 100*precision))


if __name__ == "__main__":
    integration_test_with_model()
    compare_preds()


