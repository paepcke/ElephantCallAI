import numpy as np
import torch
import argparse

from gun_data.utils.GunshotDataUtils import get_model, get_loader
from gun_data.DataMaker import TRAIN_MEAN_FILENAME, TRAIN_STD_FILENAME, CLASS_LABELS
from gun_data.TrainedGunshotModel import TrainedGunshotModel
from gun_data.utils import PerformanceMetrics


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, help="path to directory containing data")
    parser.add_argument('--device', type=str, help="device for torch to use. GPUs are faster and can be specified like 'cuda:0'",
                        default="cuda:2")

    parser.add_argument('--batch-size', type=int, help="batch size to use", default=16)
    parser.add_argument('--model-weights', type=str, help="path to file containing model weights (directly from training)")

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

    all_predictions = np.zeros((len(test_loader.dataset),))
    all_labels = np.zeros_like(all_predictions)

    cumu_examples = 0
    for batch in test_loader:
        data, labels = batch[0], batch[1]
        data = data.to(device)
        labels = labels.to(device)

        probs = test_model.forward(data)

        preds = np.argmax(probs.detach().cpu().numpy(), axis=-1)

        all_predictions[cumu_examples:(cumu_examples + len(labels))] = preds
        all_labels[cumu_examples:(cumu_examples + len(labels))] = labels.detach().cpu().numpy()

        examples = len(labels)
        cumu_examples += examples

    test_acc = PerformanceMetrics.all_class_accuracy(all_predictions, all_labels)
    print(f"{test_acc}% accuracy on the test set.")
    total_examples = len(all_labels)

    for label, classname in CLASS_LABELS.items():
        class_acc = PerformanceMetrics.class_accuracy(all_predictions, all_labels, label)
        precision = PerformanceMetrics.precision(all_predictions, all_labels, label)
        recall = PerformanceMetrics.recall(all_predictions, all_labels, label)
        f1 = 2*precision*recall/(precision + recall)
        num_examples = np.sum(np.where(all_labels == label, 1, 0))
        print(f"class stats for '{classname}' ({num_examples}/{total_examples} examples): " +
              f"accuracy - {class_acc}, precision - {precision}, recall - {recall}, F1 - {f1}")

    print("Confusion matrix [no gunshot, non-rapidfire, rapidfire]:")
    print(PerformanceMetrics.confusion_matrix(all_predictions, all_labels))
