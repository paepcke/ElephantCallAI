import argparse
import torch

from gun_data import SampleCollection
from gun_data.utils.GunshotDataUtils import get_loader


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, help="path to directory containing data to visualize completions of", required=True)
    parser.add_argument('--device', type=str, help="device for torch to use. GPUs are faster and can be specified like 'cuda:0'",
                        default="cuda:2")

    # training settings
    parser.add_argument('--model-dir', type=str, help="path to directory containing model weights")
    parser.add_argument('--num-dataloader-workers', type=int, default=16,
                        help="Number of workers each dataloader will use")
    parser.add_argument('--dataset-mean-std-dir', type=str,
                        help="Specify a path to a folder containing the mean, std, and maxmin files intended for use."
                             " This will override the defaults used by the DataLoader."
                             " This option is appropriate for models being fine-tuned on a different dataset"
                             " than the one they were trained on.")

    # sample generation settings
    parser.add_argument('--num-batches', type=int, default=2,
                        help='number of batches of data to visualize')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='size of generation batches')
    parser.add_argument('--prefix-len', type=int, default=4,
                        help='number of timesteps of real data to condition generative model on')

    # visualization of results
    parser.add_argument('--visualize-results-dir', type=str, required=True,
                        help="an absolute path to a directory to store generated sample visualizations")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    num_batches = args.num_batches
    prefix_len = args.prefix_len
    batch_size = args.batch_size

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    _, dataset = get_loader(args.data_dir, 2,
                                         num_workers=args.num_dataloader_workers, max_time_len=95,
                                         return_dataset_too=True, override_mean_std_dir=args.dataset_mean_std_dir)

    model = torch.load(f"{args.model_dir}/best.pt")  # load 'best.pt' from model save dir

    model = model.to(args.device)
    model.eval()
    # create a directory structure where some samples are stored, then populate it with dataset samples and model samples
    SampleCollection.populate_sample_directory(dataset, model,
                                               args.visualize_results_dir,
                                               prefix_len, num_batches, batch_size, device,
                                               show_nonspeculative_samples=True)
