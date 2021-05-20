import argparse
import os


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--wav-dir', type=str, default=".", help="Path to a directory containing wav files to downsample")

    return parser.parse_args()


if __name__ == "__main__":
    """
    Small script to downsample wav files to 8kHz and convert them to mono (1 channel).
    You must have the command-line utility 'sox' installed to use this.
    """
    args = get_args()

    files = os.listdir(args.wav_dir)
    for file in files:
        if file.endswith(".wav"):
            # correct for spaces in filename:
            file = file.replace(" ", "\\ ")
            os.system(f"sox {args.wav_dir}/{file} -c 1 -r 8000 {args.wav_dir}/temp.wav")
            os.system(f"mv {args.wav_dir}/temp.wav {args.wav_dir}/{file}")
