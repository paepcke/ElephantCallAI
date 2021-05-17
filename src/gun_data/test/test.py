import numpy as np
import torch
import argparse

from gun_data.GunshotDataUtils import get_model, get_loader, compute_acc
from gun_data.DataMaker import TRAIN_MEAN_FILENAME, TRAIN_STD_FILENAME
from gun_data.TrainedGunshotModel import TrainedGunshotModel


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

    cumu_acc = 0.
    cumu_examples = 0
    for batch in test_loader:
        data, labels = batch[0], batch[1]
        data = data.to(device)
        labels = labels.to(device)

        examples = len(labels)
        cumu_examples += examples

        probs = test_model.forward(data)
        cumu_acc += examples * compute_acc(labels.detach().cpu().numpy(), probs.detach().cpu().numpy())

    test_acc = cumu_acc / cumu_examples
    print(f"{test_acc}% accuracy on the test set.")
    # TODO: include other metrics, like per-class accuracies
