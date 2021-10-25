import argparse
import torch
from torch.utils.tensorboard import SummaryWriter

from gun_data.DataAugmentor import DataAugmentor
from gun_data.models.UnivariateGaussianHead import UnivariateGaussianHead
from gun_data.models.AutoregressiveHeadModel import AutoregressiveHeadModel
from gun_data.models.positional.SinusoidalPositionalEncoding import SinusoidalPositionalEncoding
from gun_data.utils.GunshotDataUtils import get_loader, get_model
from gun_data.training.GunshotTrainer import GunshotTrainer
from gun_data.training.GenerativeGunshotTrainer import GenerativeGunshotTrainer
from gun_data.training.GunshotTrainingSettings import GunshotTrainingSettings


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data-dir', type=str, help="path to directory containing training data", required=True)
    parser.add_argument('--val-data-dir', type=str, help="path to directory containing validation data", required=True)
    parser.add_argument('--tbx-log-dir', type=str, help="path to location for tensorboard to write logs to", required=True)
    parser.add_argument('--device', type=str, help="device for torch to use. GPUs are faster and can be specified like 'cuda:0'",
                        default="cuda:2")

    parser.add_argument('--batch-size', type=int, help="batch size to use during training and validation", default=16)
    parser.add_argument('--max-epochs', type=int, help="maximum number of epochs to train for", default=5)
    parser.add_argument('--dropout', type=float, help="dropout probability to use", default=0.)
    parser.add_argument('--generative', action='store_true',
                        help="specify this to train a generative (likelihood-based) model")

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
    parser.add_argument('--lr-decay-factor', type=float, default=0.2,
                        help="after lr-step-size epochs, set lr = lr-decay-factor * lr")
    parser.add_argument('--lr-step-size', type=int, default=2,
                        help="After this many epochs, reduce the learning rate by a factor (lr-decay-factor)")
    parser.add_argument('--num-dataloader-workers', type=int, default=16,
                        help="Number of workers each dataloader will use")
    parser.add_argument('--dataset-mean-std-dir', type=str,
                        help="Specify a path to a folder containing the mean, std, and maxmin files intended for use."
                             " This will override the defaults used by the DataLoader."
                             " This option is appropriate for models being fine-tuned on a different dataset"
                             " than the one they were trained on.")
    parser.add_argument('--transferred-model-path', type=str,
                        help="path to a .pt file to fine-tune on")

    # data augmentation
    parser.add_argument('--max-freq-occlusion', type=int, help="maximum number of frequency bands to block out", default=60)
    parser.add_argument('--max-time-occlusion', type=int, help="maximum number of time steps to block out", default=15)
    parser.add_argument('--aug-prob', type=float, help="probability of an individual sample being augmented", default=0.5)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    if not args.generative:
        augmentor = DataAugmentor(args.max_freq_occlusion, args.max_time_occlusion, args.aug_prob)
    else:
        augmentor = None

    train_loader = get_loader(args.train_data_dir, args.batch_size, augmentor,
                                             num_workers=args.num_dataloader_workers, max_time_len=95,
                                             override_mean_std_dir=args.dataset_mean_std_dir)
    val_loader, val_dataset = get_loader(args.val_data_dir, args.batch_size,
                                         num_workers=args.num_dataloader_workers, max_time_len=95,
                                         return_dataset_too=True, override_mean_std_dir=args.dataset_mean_std_dir)
    tbx_writer = SummaryWriter(log_dir=args.tbx_log_dir)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    train_settings = GunshotTrainingSettings(device, args.initial_lr, args.weight_decay,
                                             args.model_out_dir, args.train_log_frequency,
                                             args.val_log_frequency, args.lr_decay_factor,
                                             args.lr_step_size, args.early_stop_epochs)

    if args.transferred_model_path is not None:
        model = torch.load(args.transferred_model_path)
    else:
        if args.generative:
            hidden_dim = 1025
            freq_dim = 525
            head = UnivariateGaussianHead(hidden_dim, freq_dim)
            positional_encoding = SinusoidalPositionalEncoding(94, 500)
            model = AutoregressiveHeadModel(head, 94, 5, 6, args.dropout, positional_encoding=positional_encoding)
        else:
            model = get_model((95, 525), args.dropout)

    if args.generative:
        gunshot_trainer = GenerativeGunshotTrainer(train_loader, val_loader, tbx_writer, model, train_settings)
    else:
        gunshot_trainer = GunshotTrainer(train_loader, val_loader, tbx_writer, model, train_settings)
    gunshot_trainer.train(args.max_epochs)
