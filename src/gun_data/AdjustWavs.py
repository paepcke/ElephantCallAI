import argparse
import os
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--wav-superdir', type=str,
                        help="Directory containing wav directories to process. Overrides --wav-dir argument if specified.")
    parser.add_argument('--wav-dir', type=str, default=".", help="Path to a directory containing wav files to downsample")
    parser.add_argument('--max-secs', type=int, default=10, help="only the the first `max-secs` seconds of audio")

    return parser.parse_args()


def process_wavs_in_dir(wav_dir_path: str, max_secs: int, leave_progress_bar: bool = True):
    files = os.listdir(wav_dir_path)
    for file in tqdm(files, leave=leave_progress_bar):
        if file.endswith(".wav") or file.endswith(".mp3"):  # MP3 files can be converted into WAVs as well
            # correct for spaces in filename:
            file = file.replace(" ", "\\ ")
            os.system(f"sox {wav_dir_path}/{file} -c 1 -r 8000 -b 16 {wav_dir_path}/temp.wav trim 0 {max_secs}")
            os.system(f"rm {wav_dir_path}/{file}")
            file = file[:-4] + ".wav"
            os.system(f"mv {wav_dir_path}/temp.wav {wav_dir_path}/{file}")


if __name__ == "__main__":
    """
    Small script to downsample wav files to 8kHz and convert them to mono (1 channel). Also trims them, only keeping the
    first `args.max_secs` seconds of audio.

    You must have the command-line utility 'sox' installed to use this.
    """
    args = get_args()

    if args.wav_superdir is not None:
        for wav_dir in tqdm(os.listdir(args.wav_superdir)):
            wav_dir_path = f"{args.wav_superdir}/{wav_dir}"
            if os.path.isdir(wav_dir_path):
                process_wavs_in_dir(wav_dir_path, args.max_secs, leave_progress_bar=False)
    else:
        process_wavs_in_dir(args.wav_dir, args.max_secs)
