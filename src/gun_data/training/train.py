import argparse

from torch.utils.tensorboard import SummaryWriter
import torch

from gun_data.DataAugmentor import DataAugmentor
from gun_data.GunshotDataUtils import get_loader, get_model
from gun_data.training.GunshotTrainer import GunshotTrainer
from gun_data.training.GunshotTrainingSettings import GunshotTrainingSettings


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data-dir', type=str, help="path to directory containing training data")
    parser.add_argument('--val-data-dir', type=str, help="path to directory containing validation data")
    parser.add_argument('--tbx-log-dir', type=str, help="path to location for tensorboard to write logs to")
    parser.add_argument('--device', type=str, help="device for torch to use. GPUs are faster and can be specified like 'cuda:0'",
                        default="cuda:2")

    parser.add_argument('--batch-size', type=int, help="batch size to use during training and validation", default=16)
    parser.add_argument('--max-epochs', type=int, help="maximum number of epochs to train for", default=5)
    parser.add_argument('--dropout', type=float, help="dropout probability to use", default=0.)

    # training settings
    parser.add_argument('--initial-lr', type=float, help="learning rate to use for the first epoch", default=1e-4)
    parser.add_argument('--model-out-dir', type=str, help="path to directory containing model weights")
    parser.add_argument('--weight-decay', type=float, help="weight decay coefficient to use during training", default=0.)
    parser.add_argument('--train-log-frequency', type=int, help="log training metrics to tensorboard after this many batches",
                        default=20)
    parser.add_argument('--val-log-frequency', type=int, help="log val metrics to tensorboard after this many batches",
                        default=150)
    parser.add_argument('--early-stop-epochs', type=int,
                        help="if validation performance does not beat the current best for this many epochs, stop training early",
                        default=6)

    # data augmentation
    parser.add_argument('--max-freq-occlusion', type=int, help="maximum number of frequency bands to block out", default=60)
    parser.add_argument('--max-time-occlusion', type=int, help="maximum number of time steps to block out", default=15)
    parser.add_argument('--aug-prob', type=float, help="probability of an individual sample being augmented", default=0.5)

    return parser.parse_args()


if __name__ == "__main__":
    # TODO: add support for pretrained weights, different model architectures

    args = get_args()

    augmentor = DataAugmentor(args.max_freq_occlusion, args.max_time_occlusion, args.aug_prob)

    train_loader = get_loader(args.train_data_dir, args.batch_size, augmentor)
    val_loader = get_loader(args.val_data_dir, args.batch_size)
    tbx_writer = SummaryWriter(log_dir=args.tbx_log_dir)

    model = get_model((95, 525), args.dropout)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    train_settings = GunshotTrainingSettings(device, args.initial_lr, args.weight_decay,
                                             args.model_out_dir, args.train_log_frequency,
                                             args.val_log_frequency, args.early_stop_epochs)

    gunshot_trainer = GunshotTrainer(train_loader, val_loader, tbx_writer, model, train_settings)
    gunshot_trainer.train(args.max_epochs)
