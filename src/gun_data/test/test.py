import numpy as np
import torch
import argparse
from tqdm import tqdm
import os

from gun_data.utils.GunshotDataUtils import get_model, get_loader
from gun_data.DataMaker import TRAIN_MEAN_FILENAME, TRAIN_STD_FILENAME, CLASS_LABELS
from gun_data.TrainedGunshotModel import TrainedGunshotModel
from gun_data.utils import PerformanceMetrics


DEFAULT_PR_CURVE_FILENAME = "pr_curve_test.png"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, help="path to directory containing data")
    parser.add_argument('--device', type=str, help="device for torch to use. GPUs are faster and can be specified like 'cuda:0'",
                        default="cuda:2")

    parser.add_argument('--batch-size', type=int, help="batch size to use", default=16)
    parser.add_argument('--model-weights', type=str, help="path to file containing model weights (directly from training)")
    parser.add_argument('--pr-curve-savepath', type=str,
                        help="path to a location to save the image of the generated pr curve to."
                             f" Not providing this argument will save the PR curve as '{DEFAULT_PR_CURVE_FILENAME}' in"
                             " the same directory as the model weights are stored.")
    parser.add_argument('--skip-pr-curve', action='store_true', help="specify this to skip generating a PR curve")

    parser.add_argument('--use-rapidfire-category', action="store_true",
                        help="Specify this to treat rapid-fire and non-rapid-fire shots as separate categories for prediction")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    model = get_model((95, 525))
    model.load_state_dict(torch.load(args.model_weights))

    mean = np.load(f"{args.data_dir}/../{TRAIN_MEAN_FILENAME}")
    std = np.load(f"{args.data_dir}/../{TRAIN_STD_FILENAME}")

    test_model = TrainedGunshotModel(model, mean, std)

    test_loader = get_loader(args.data_dir, args.batch_size, preprocess_normalization=False)
    test_model.to(device)

    generate_pr_curve = (not args.skip_pr_curve) and (not args.use_rapidfire_category)

    all_predictions = np.zeros((len(test_loader.dataset),))
    all_labels = np.zeros_like(all_predictions)
    if generate_pr_curve:
        all_probs = np.zeros_like(all_predictions)

        pr_curve_savepath = args.pr_curve_savepath
        if pr_curve_savepath is None:
            dir_containing_model = os.path.dirname(args.model_weights)
            if len(dir_containing_model) == 0:
                # model weights are in the current directory
                dir_containing_model = "."
            pr_curve_savepath = f"{dir_containing_model}/{DEFAULT_PR_CURVE_FILENAME}"

    cumu_examples = 0
    for batch in tqdm(test_loader):
        data, labels = batch[0], batch[1]
        data = data.to(device)
        labels = labels.to(device)

        if generate_pr_curve:
            probs = test_model.forward(data)
            preds = torch.round(probs)

            all_probs[cumu_examples:(cumu_examples + len(labels))] = np.squeeze(probs.detach().cpu().numpy())
        else:
            preds = test_model.predict(data)

        all_predictions[cumu_examples:(cumu_examples + len(labels))] = np.squeeze(preds.detach().cpu().numpy())
        all_labels[cumu_examples:(cumu_examples + len(labels))] = np.squeeze(labels.detach().cpu().numpy())

        examples = len(labels)
        cumu_examples += examples

    test_acc = PerformanceMetrics.all_class_accuracy(all_predictions, all_labels)
    print(f"{test_acc}% accuracy on the test set.")
    total_examples = len(all_labels)

    if args.use_rapidfire_category:
        for label, classname in CLASS_LABELS.items():
            class_acc = PerformanceMetrics.class_accuracy(all_predictions, all_labels, label)
            precision = PerformanceMetrics.precision(all_predictions, all_labels, label)
            recall = PerformanceMetrics.recall(all_predictions, all_labels, label)
            f1 = 2*precision*recall/(precision + recall)
            num_examples = np.sum(np.where(all_labels == label, 1, 0))
            print(f"class stats for '{classname}' ({num_examples}/{total_examples} examples): " +
                  f"accuracy - {class_acc}, precision - {precision}, recall - {recall}, F1 - {f1}")

        print("Confusion matrix [no gunshot, non-rapidfire, rapidfire]:")
    else:
        # use target class label of 1 for gunshots
        precision = PerformanceMetrics.precision(all_predictions, all_labels, 1)
        recall = PerformanceMetrics.recall(all_predictions, all_labels, 1)
        f1 = 2*precision*recall/(precision + recall)
        num_gunshot_examples = np.sum(np.where(all_labels == 1, 1, 0))
        print(f"{num_gunshot_examples} of {total_examples} were gunshots.\n"
              f"precision - {precision}\n"
              f"recall - {recall}\n"
              f"F1 score - {f1}\n")
        print("Confusion matrix [no gunshot, gunshot]:")
    print(PerformanceMetrics.confusion_matrix(all_predictions, all_labels))

    if generate_pr_curve:
        print(f"Generating image of PR curve and saving to {pr_curve_savepath}...")
        PerformanceMetrics.pr_curve(all_labels, all_probs, pr_curve_savepath)

    # TODO: consider adding an arg to allow user to specify the name of this file, remove hacky code reuse of 'dir_containing_model'
    dir_containing_model = os.path.dirname(args.model_weights)
    if len(dir_containing_model) == 0:
        # model weights are in the current directory
        dir_containing_model = "."
    torch.save(test_model, f"{dir_containing_model}/model_and_preprocessing.pt")
