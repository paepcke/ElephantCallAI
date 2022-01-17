import argparse
import torch

from gun_data.DataAugmentor import DataAugmentor
from gun_data.utils.GunshotDataUtils import get_loader
from gun_data.training.GunshotTrainingSettings import GunshotTrainingSettings


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, help="device for torch to use. GPUs are faster and can be specified like 'cuda:0'",
                        default="cuda:2")

    parser.add_argument('--batch-size', type=int, help="batch size to use during training and validation", default=16)
    parser.add_argument('--generative', action='store_true',
                        help="specify this to train a generative (likelihood-based) model")
    parser.add_argument('--model-dir', type=str, help="path to directory containing model weights")
    parser.add_argument('--num-dataloader-workers', type=int, default=16,
                        help="Number of workers each dataloader will use")
    parser.add_argument('--dataset-mean-std-dir', type=str,
                        help="Specify a path to a folder containing the mean, std, and maxmin files intended for use."
                             " This will override the defaults used by the DataLoader."
                             " This option is appropriate for models being fine-tuned on a different dataset"
                             " than the one they were trained on.")
    parser.add_argument('--target-eval-dir', type=str,
                        help="directory to evaluate metrics for")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    if not args.generative:
        raise NotImplementedError("eval.py is only supported for generative models right now")
    if args.target_eval_dir is None:
        raise ValueError("you must specify '--target-eval-dir' in eval-only mode!")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    data_loader = get_loader(args.target_eval_dir, args.batch_size,
                             num_workers=args.num_dataloader_workers, max_time_len=95,
                             override_mean_std_dir=args.dataset_mean_std_dir)

    model = torch.load(f"{args.model_dir}/best.pt")

    model.to(device)
    model.eval()

    running_loss = 0.
    num_examples = 0
    with torch.no_grad():
        for batch in data_loader:
            data, time_len = batch[0], batch[2]
            data = data.to(device)
            time_len = time_len.to(device)

            logits = model.forward(data[:, :-1, :], time_len)

            loss = model.loss(data, time_len, logits)

            running_loss += len(data) * loss.item()
            num_examples += len(data)

    avg_loss = running_loss / num_examples

    print(f"Loss value is {avg_loss}")
