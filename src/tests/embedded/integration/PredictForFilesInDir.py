import argparse
import os
import numpy as np

from embedded.FileUtils import assert_path_exists
from tests.embedded.integration.StreamTest import integration_test_with_model, compare_preds
from sklearn.metrics import balanced_accuracy_score


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input-data-dir', type=str,
                        help="The directory containing the input *_spectro.py files", required=True)
    parser.add_argument('--prediction-output-dir', type=str,
                        help="The directory to store the prediction output files", required=True)
    parser.add_argument('--model-path', type=str, help="path to the 2-stage-model directory with the model to run")
    parser.add_argument('--prediction-threshold', type=float, default=0.5,
                        help="Threshold for deciding a positive prediction from model outputs.")
    parser.add_argument('--overwrite-existing-predictions', action='store_true',
                        help="Specify this to overwrite existing predictions. Otherwise, they will be skipped.")
    parser.add_argument('--verbose', action='store_true',
                        help="Specify this to print detailed information during inference. It's a lot, so it may overwhelm your console.")

    return parser.parse_args()


def main():
    args = get_args()

    assert_path_exists(args.input_data_dir, "Please provide a correct input directory.")
    if not os.path.exists(args.prediction_output_dir):
        os.mkdir(args.prediction_output_dir)

    spectro_files = os.listdir(args.input_data_dir)

    prefixes = []

    for spectro_filename in spectro_files:
        if not spectro_filename.endswith("_spectro.npy"):
            continue
        prefix = spectro_filename[:(-1*len("_spectro.npy"))]
        prefixes.append(prefix)
        pred_filename = prefix + "_predictions.npy"
        pred_filepath = f"{args.prediction_output_dir}/{pred_filename}"
        if os.path.exists(pred_filepath) and not args.overwrite_existing_predictions:
            print(f"Skipping {spectro_filename} because {pred_filename} already exists.")
            continue

        print(f"Generating predictions for {spectro_filename}")
        integration_test_with_model(False, True, model_path=args.model_path, preds_save_path=pred_filepath,
                                    spectrogram_file_path=f"{args.input_data_dir}/{spectro_filename}", verbose=args.verbose)

    label_arrays = []
    pred_arrays = []
    for prefix in prefixes:
        label_arr = np.load(f"{args.input_data_dir}/{prefix}_label_mask.npy")
        pred_arr = np.load(f"{args.prediction_output_dir}/{prefix}_predictions.npy")

        """
        Clip end of label array if necessary, this is pretty small (<= 1 model input window) and unlikely to influence
        performance metrics much.

        This has to be done because the predictor won't produce results for time steps it hasn't fully slid its window
        over. In a streaming setting, this isn't a problem, because more data is always expected. In a finite-length
        input setting, we have to do this clipping.
        """
        label_arr = label_arr[:pred_arr.shape[0]]

        label_arrays.append(label_arr)
        pred_arrays.append(pred_arr)

    labels = np.concatenate(label_arrays, 0)
    preds = np.concatenate(pred_arrays, 0)

    # binarize predictions
    preds = np.where(preds > args.prediction_threshold, 1, 0)

    results = compare_preds(labels_arr=labels, preds_arr=preds)

    balanced_acc = balanced_accuracy_score(y_true=labels, y_pred=preds, adjusted=True)
    results += f"\nBalanced Accuracy Score: {balanced_acc}"
    with open(f"{args.prediction_output_dir}/prediction_summary.txt", "w") as outfile:
        outfile.write(results)

    print(results)


if __name__ == "__main__":
    main()
