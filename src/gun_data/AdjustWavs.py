import argparse
import os


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--wav-dir', type=str, default=".", help="Path to a directory containing wav files to downsample")
    parser.add_argument('--max-secs', type=int, default=10, help="only the the first `max-secs` seconds of audio")

    return parser.parse_args()


if __name__ == "__main__":
    """
    Small script to downsample wav files to 8kHz and convert them to mono (1 channel). Also trims them, only keeping the
    first `args.max_secs` seconds of audio.

    You must have the command-line utility 'sox' installed to use this.
    """
    args = get_args()

    files = os.listdir(args.wav_dir)
    for file in files:
        if file.endswith(".wav") or file.endswith(".mp3"):
            # correct for spaces in filename:
            file = file.replace(" ", "\\ ")
            os.system(f"sox {args.wav_dir}/{file} -c 1 -r 8000 -b 16 {args.wav_dir}/temp.wav trim 0 {args.max_secs}")
            os.system(f"rm {args.wav_dir}/{file}")
            file = file[:-4] + ".wav"
            os.system(f"mv {args.wav_dir}/temp.wav {args.wav_dir}/{file}")
