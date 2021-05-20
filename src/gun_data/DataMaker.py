import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Callable, Dict
from scipy.io import wavfile
import re
import os
import random
import argparse
from tqdm import tqdm
from collections import defaultdict

from embedded.microphone.SpectrogramExtractor import SpectrogramExtractor

ECOGUNS_REGEX = "ecoguns\.\d+\.(\d+\.\d+(?:\.\d+){0,1})\.\d+\.wav"
PNNGUNS_REGEX = "pnnnGuns\.\d+\.\w+\.(\d+\.\d+\.\d+)\.\w+\.wav"
BACKGROUND_FILENAME_PREFIX = "bgAudio"
TRAIN_MEAN_FILENAME = "train_mean.npy"
TRAIN_STD_FILENAME = "train_std.npy"
FILENAME_REGEX = ".*_(\d+)\.npy"  # input data to the model will be of this form

# These variables are for quick iteration on this script. You should override them with command-line arguments.
REMOTE_PATH_PREFIX = "/home/deschwa2/gun_data"

ECOGUNS_WAV_PATH = REMOTE_PATH_PREFIX + "/ecoguns"
ECOGUNS_GUIDE_PATH = REMOTE_PATH_PREFIX + "/Guns_Training_ecoGuns_SST_mac.txt"
PNNN_GUNS_GUIDE_PATH = REMOTE_PATH_PREFIX + "/nn_Grid50_guns_dep1-7_train.txt"
PNNN_GUNS_WAV_PATH = REMOTE_PATH_PREFIX + "/pnnn_dep1-7"
RAW_BG_WAV_PATH = REMOTE_PATH_PREFIX + "/rawBgNoise"

"""
This is a script and a collection of related utilities that takes in a variety of weakly-labeled gunshot clips and
mashes them in with some long-form background noise (in multi-hour-long WAV files).

The output will be spectrograms (after a 10*log10 transform). Perhaps there should be some normalization, but it may not
matter all that much. The output clips should be the same length and some care should be taken to randomly shift each
shot against different background noise (include g1 + b where g1 is close to the start of the clip, then where it's
closer to the end/middle, etc).

This script is suited for input data arranged in a specific file structure, but it could be adapted to fit a more
general use case.

This script has a feature called "scenarios". "scenarios" are wav files from a certain sound scene category
(e.g., 'thunderstorm'). Snippets from scenarios can be overlaid onto generated training examples.

The directory structure of 'scenarios' should look as follows with the categories 'highway' and 'thunderstorm':
scenarios
├── highway
│   └── ...
└── thunderstorm
    ├── storm1.wav
    └── storm2.wav
    

Label guide:
0 is no gunshot
1 is non-rapidfire shot(s)
2 is rapidfire shots
"""
CLASS_LABELS = {0: "no-gunshot", 1: "non-rapidfire_shot", 2: "rapidfire_shot"}
RAPIDFIRE_SHOT_THRESHOLD = 2


def reformat_ecoguns_df(in_df: pd.DataFrame, file_dir: str) -> pd.DataFrame:
    # desired output df cols:
    # filename, distance (m), duration, numshots
    new_df = pd.DataFrame()

    # load in list of file names
    files = os.listdir(file_dir)

    # use regex to capture unique ids
    file_id_dict = {}
    for file in files:
        m = re.match(ECOGUNS_REGEX, file)
        uid = m.group(1)
        file_id_dict[uid] = file

    # map unique ids to file names
    new_df['filename'] = in_df['uniqueID'].transform(lambda id: file_id_dict[id])

    new_df['distance'] = in_df['distance(m)']
    new_df['duration'] = in_df['duration (s)']
    new_df['numshots'] = in_df['numshots']

    return new_df


def reformat_pnnguns_df(in_df: pd.DataFrame, file_dir: str) -> pd.DataFrame:
    new_df = pd.DataFrame()

    # prune in_df entries where # of shots not known
    in_df = in_df[in_df['total shots'] != '?']

    files = os.listdir(file_dir)

    # use regex to capture unique ids
    file_id_dict = {}
    for file in files:
        m = re.match(PNNGUNS_REGEX, file)
        uid = m.group(1)
        file_id_dict[uid] = file

    # map unique ids to file names
    new_df['filename'] = in_df['uniqueID'].transform(lambda id: file_id_dict[id])

    new_df['distance'] = -1 # a placeholder for 'unknown'
    new_df['numshots'] = in_df['total shots'].transform(lambda row: int(row))
    new_df['duration'] = in_df['End Time (s)'] - in_df['Begin Time (s)']
    return new_df


def assert_fraction(x):
    assert(0 <= x <= 1)


def simple_split_helper(df, columnname, split) -> List[pd.DataFrame]:
    greater_df = df[df[columnname] > split]
    less_equal_df = df[df[columnname] <= split]

    return [greater_df, less_equal_df]


def categorical_split_helper(df, columnname, values: Optional[List] = None) -> List[pd.DataFrame]:
    if values is None:
        values = df[columnname].unique()

    cat_dfs = []
    for value in values:
        cat_dfs.append(df[df[columnname] == value])

    return cat_dfs


def train_test_split(df, frac_train, frac_val) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    df = df.copy(deep=False)  # avoid 'chained indexing' problem when setting new values
    df_len = len(df)

    assignments = np.zeros((df_len,))
    end_train_idx = int(df_len * frac_train)
    end_val_idx = df_len - int((df_len * (1 - frac_train - frac_val)))
    assignments[:end_train_idx] = 0
    assignments[end_train_idx:end_val_idx] = 1
    assignments[end_val_idx:] = 2

    df['assignment'] = assignments
    split_dfs = categorical_split_helper(df, 'assignment', values=np.arange(3))
    train_df, val_df, test_df = split_dfs[0], split_dfs[1], split_dfs[2]
    train_df = train_df.drop('assignment', axis=1)
    val_df = val_df.drop('assignment', axis=1)
    test_df = test_df.drop('assignment', axis=1)
    return train_df, val_df, test_df


def split_df_for_train_val_test(combined_df, frac_train, frac_val) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # all of this specific splitting is to help maintain similar distributions between train, val, and test sets
    shots_split = simple_split_helper(combined_df, 'numshots', 2)
    df_list = []
    for df in shots_split:
        df_list.extend(categorical_split_helper(df, 'distance'))
    # TODO: split by sorting positive clip durations into 'buckets' too?

    train_list = []
    val_list = []
    test_list = []
    for df in df_list:
        train_df, val_df, test_df = train_test_split(df, frac_train, frac_val)
        train_list.append(train_df)
        val_list.append(val_df)
        test_list.append(test_df)

    train_df = pd.concat(train_list)
    val_df = pd.concat(val_list)
    test_df = pd.concat(test_list)

    return train_df, val_df, test_df


# returns a tuple of 3 dataframes, one for the training set, one for the validation set, and one for the test set
# these are all un-augmented *positive* samples
def get_positive_clips(args) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(args.ecoguns_tsv_path, sep="\t")
    ecoguns_df = reformat_ecoguns_df(df, args.ecoguns_wav_path)
    df = pd.read_csv(args.pnnn_guns_tsv_path, sep="\t")
    pnn_df = reformat_pnnguns_df(df, args.pnnn_guns_wav_path)
    combined_df = pd.concat([ecoguns_df, pnn_df])

    train_df, val_df, test_df = split_df_for_train_val_test(combined_df, args.frac_train, args.frac_val)
    return train_df, val_df, test_df


def read_wav(filepath: str, force_sample_rate: Optional[int] = None) -> np.ndarray:
    sample_rate, arr = wavfile.read(filepath)
    if force_sample_rate is not None:
        if sample_rate != force_sample_rate:
            raise ValueError(f"{filepath} has a sample rate of {sample_rate}, but a sample rate of {force_sample_rate} was expected.")
    return arr


def write_pos_clips(data_df: pd.DataFrame, out_dir: str, ecoguns_dir: str, pnnnguns_dir: str):
    example_num = 0
    for _, row in data_df.iterrows():
        filename = row['filename']
        if filename.startswith("ecoguns"):
            arr = read_wav(ecoguns_dir + "/" + filename, force_sample_rate=8000)
        else:
            arr = read_wav(pnnnguns_dir + "/" + filename, force_sample_rate=8000)
        numshots = row['numshots']
        if numshots > RAPIDFIRE_SHOT_THRESHOLD:
            label = 2
        else:
            label = 1
        out_filepath = out_dir + f"/positiveExample{example_num}_{label}.npy"
        np.save(out_filepath, arr)

        example_num += 1


def chop_wav(wav_path: str, out_dir: str, created_filename_prefix: str, interval_len_s: float, overlap_fraction: float = 0.):
    arr = read_wav(wav_path, force_sample_rate=8000)
    samples_per_interval = int(interval_len_s * 8000)
    if len(arr) < samples_per_interval:
        print(f"Skipping chop_wav for {wav_path}, file is too short")
        return
    non_overlapping_samples = int(samples_per_interval*(1 - overlap_fraction))
    if non_overlapping_samples < 1:
        print(f"Skipping overlap in chopping {wav_path}, it cannot be chopped as finely as requested")
        non_overlapping_samples = samples_per_interval
    intervals_in_arr = 1 + (len(arr) - samples_per_interval)//non_overlapping_samples

    for i in tqdm(range(intervals_in_arr), delay=2.5):
        left_idx = i*non_overlapping_samples
        right_idx = left_idx + samples_per_interval

        interval = arr[left_idx:right_idx]
        np.save(out_dir + "/" + created_filename_prefix + f"{i}.npy", interval)


def gen_mixed_data(bg_dir: str, positive_dir: str, out_dir: str, num_samples: int, frac_nogunshot: float,
                   spec_extractor: SpectrogramExtractor, transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                   scenario_map: Optional[Dict[str, List[Tuple[str, int]]]] = None, scenario_prob: float = 0.):
    """

    :param bg_dir: path to directory storing background audio snippets
    :param positive_dir: path to directory storing positive audio snippets
    :param out_dir: path to target output directory
    :param num_samples: desired number of data samples to generate
    :param frac_nogunshot: proportion of data that should be generated with no gunshots
    :param spec_extractor: SpectrogramExtractor object
    :param transform: a callable to manipulate the output of the spectrogram
    :param scenario_map: output subdictionary from split_scenarios(). See its documentation for description.
    :param scenario_prob: probability of augmenting an example with scenario audio
    """

    bg_list = os.listdir(bg_dir)
    pos_list_raw = os.listdir(positive_dir)

    pos_list = []
    pos_labels = []
    for filename in pos_list_raw:
        m = re.match(FILENAME_REGEX, filename)
        if m is not None:
            label = int(m.group(1))
            pos_list.append(filename)
            pos_labels.append(label)

    random.shuffle(bg_list)
    random.shuffle(pos_list)

    if scenario_map is not None and len(scenario_map) >= 1:
        """
        pre-compute a probability distribution for each scenario (for use in np.random.choice) 
        to avoid doing this multiple times during generation
        """
        scenarios = list(scenario_map.keys())
        scenario_distributions = compute_scenario_distributions(scenario_map, scenarios)

    for i in tqdm(range(num_samples)):
        bg_idx = i % len(bg_list)

        bg = np.load(bg_dir + "/" + bg_list[bg_idx])

        nogunshot_test = np.random.uniform()
        if nogunshot_test <= frac_nogunshot:
            out_filename = out_dir + f"/generatedExample{i}_0.npy"
            example_audio = bg
        else:
            pos_idx = i % len(pos_list)
            pos = np.load(positive_dir + "/" + pos_list[pos_idx])

            bg_len = len(bg)
            pos_len = len(pos)

            if pos_len > bg_len:
                # for now, do not cut off any of the positive data unless pos_len > bg_len
                pos = pos[:bg_len]
                pos_len = pos.shape[0]
                pos_offset = 0
            else:
                pos_offset = np.random.randint(low=0, high=(bg_len - pos_len))

            # For now, no fancy processing to smooth these together or balance magnitudes
            example_audio = bg
            example_audio[pos_offset:(pos_offset + pos_len)] += pos

            # TODO: construct example name so we can tell how it was assembled?
            out_filename = out_dir + f"/generatedExample{i}_" + str(pos_labels[pos_idx]) + ".npy"

        if scenario_map is not None and len(scenario_map) >= 1:
            scenario_test = np.random.uniform()
            if scenario_test <= scenario_prob:
                example_audio = superimpose_scenario(example_audio, scenario_map, scenarios, scenario_distributions)

        stft = spec_extractor.extract_spectrogram(example_audio)

        if transform is not None:
            stft = transform(stft)

        # convert to float32, stft computation returns float64 by default
        stft = stft.astype(np.float32)
        np.save(out_filename, stft)


def compute_scenario_distributions(scenario_map: Dict[str, List[Tuple[str, int]]], scenarios: List[str]) -> Dict[str, np.ndarray]:
    """
    Computes probability distributions for selecting snippets from individual scenario source files.

    :param scenario_map: output subdictionary from split_scenarios(). See its documentation for description.
    :param scenarios: A list of scenarios in the scenario_map
    :return: a dictionary mapping from scenario name to a probability distribution for selecting
        snippets from individual source scenario files
    """

    scenario_distributions = dict()
    for scenario in scenarios:
        scenario_distributions[scenario] = np.zeros((len(scenario_map[scenario]),))

        total_snippets = 0.
        for i, pair in enumerate(scenario_map[scenario]):
            num_snippets = pair[1]
            scenario_distributions[scenario][i] = num_snippets
            total_snippets += num_snippets

        scenario_distributions[scenario] /= total_snippets

    return scenario_distributions


def superimpose_scenario(example_audio: np.ndarray,
                         scenario_map: Dict[str, List[Tuple[str, int]]],
                         scenarios: List[str],
                         scenario_distributions: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Samples a scenario snippet and superimposes it onto the parameter 'example_audio'.

    :param example_audio: The audio sample to superimpose a scenario audio sample onto
    :param scenario_map: output subdictionary from split_scenarios(). See its documentation for description.
    :param scenarios: A list of scenarios in the scenario_map
    :param scenario_distributions: a dictionary mapping from scenario name to a probability distribution for selecting
        snippets from individual source scenario files
    :return: a numpy array that the scenario audio sample has been superimposed onto
    """

    # choose a scenario uniformly at random from those available
    scenario_idx = np.random.randint(low=0, high=len(scenarios))
    scenario = scenarios[scenario_idx]

    # select a scenario source file to pick a snippet from, weighted by number of snippets
    pairs = scenario_map[scenario]
    pair_idx = np.random.choice(np.arange(len(pairs)), p=scenario_distributions[scenario])
    snippet_dir, num_snippets = pairs[pair_idx]

    # pick a random snippet from the selected source file
    snippet_idx = np.random.randint(low=0, high=num_snippets)
    snippet_filepath = f"{snippet_dir}/{scenario}_{str(snippet_idx)}.npy"

    scenario_audio = np.load(snippet_filepath)

    return example_audio + scenario_audio


def chop_scenarios(args):
    """
    Chops scenario WAV files into snippets
    """

    scenario_types = os.listdir(f"{args.path_prefix}/scenarios")
    if len(scenario_types) == 0:
        return
    make_dir_if_not_exists(f"{args.path_prefix}/{args.output_dir}/scenario_snips")
    for scenario in scenario_types:
        in_scenario_prefix = f"{args.path_prefix}/scenarios/{scenario}"
        files = os.listdir(in_scenario_prefix)
        if len(files) == 0:
            break

        out_scenario_prefix = f"{args.path_prefix}/{args.output_dir}/scenario_snips/{scenario}"
        make_dir_if_not_exists(out_scenario_prefix)
        for file in files:
            snip_out_dir = f"{out_scenario_prefix}/{file[:-4]}"
            make_dir_if_not_exists(snip_out_dir)  # exclude '.wav' suffix from new dir name
            filepath = f"{in_scenario_prefix}/{file}"
            # created files will be named like '<scenarioName>_<snippetNumber>.npy'
            chop_wav(filepath, snip_out_dir, f"{scenario}_", args.seconds_per_example, args.scenario_overlap_frac)


def split_scenarios(args) -> Dict[str, Optional[Dict[str, List[Tuple[str, int]]]]]:
    """

    :param args: command-line arguments for the script
    :return: a dictionary mapping a set name ('train', 'val', 'test') to a subdictionary.
        The subdictionary maps a scenario name (e.g., 'thunderstorm') to a list of 2-item tuples:
        the first item is a path to a directory containing audio snippets and the second is the number of snippets
        in that directory. This number will be used to weight the probability of selecting this directory at generation time.
    """
    split_map: Dict[str, Optional[Dict[str, List[Tuple[str, int]]]]] = defaultdict(lambda: None)  # do this to avoid keyerror

    scenario_sets = ['train', 'val', 'test'] if args.use_scenarios_for_test_set else ['train', 'val']
    for dataset in scenario_sets:
        split_map[dataset] = defaultdict(lambda: [])

    scenario_types = os.listdir(f"{args.path_prefix}/{args.output_dir}/scenario_snips")
    for scenario in scenario_types:
        file_len_pairs = []
        files = os.listdir(f"{args.path_prefix}/{args.output_dir}/scenario_snips/{scenario}")
        for file in files:
            snipspath = f"{args.path_prefix}/{args.output_dir}/scenario_snips/{scenario}/{file}"
            snips = os.listdir(snipspath)
            file_len_pairs.append((snipspath, len(snips)))

        # sort files by most to least snippets, then round-robin allocate files to train, val, and test, in that order
        file_len_pairs = sorted(file_len_pairs, key=lambda p: p[1], reverse=True)
        for i, pair in enumerate(file_len_pairs):
            target_dataset = scenario_sets[i % len(scenario_sets)]
            split_map[target_dataset][scenario].append(pair)

    return split_map


def prepare_freq_mean_std(args):
    """
    Computes training set mean and std (for each frequency band).
    Stores these npy arrays as files in the ".../datasets" directory.
    """
    datasets_path = f"{args.path_prefix}/{args.output_dir}/datasets"
    train_path = f"{datasets_path}/train"
    train_files = os.listdir(train_path)
    first_example = np.load(train_path + "/" + train_files[0])
    mean = np.mean(first_example, axis=0).reshape(1, -1)

    print("Computing training set mean...")
    for filename in tqdm(train_files[1:]):
        filepath = train_path + "/" + filename
        example = np.load(filepath)
        mean += np.mean(example, axis=0).reshape(1, -1)

    mean /= len(train_files)
    var = np.mean(np.power(first_example - mean, 2), axis=0).reshape(1, -1)

    print("Computing training set standard deviation...")
    for filename in tqdm(train_files[1:]):
        filepath = train_path + "/" + filename
        example = np.load(filepath)
        var += np.mean(np.power(example - mean, 2), axis=0).reshape(1, -1)

    var /= len(first_example)
    std = np.sqrt(var)

    np.save(f"{datasets_path}/{TRAIN_MEAN_FILENAME}", mean)
    np.save(f"{datasets_path}/{TRAIN_STD_FILENAME}", std)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--ecoguns-wav-path', type=str, default=ECOGUNS_WAV_PATH,
                        help="path to the directory where ecoguns wav files are stored")
    parser.add_argument('--pnnn-guns-wav-path', type=str, default=PNNN_GUNS_WAV_PATH,
                        help="path to the directory where pnnnguns wav files are stored")
    parser.add_argument('--ecoguns-tsv-path', type=str, default=ECOGUNS_GUIDE_PATH,
                        help="path to the ecoguns file that contains metadata")
    parser.add_argument('--pnnn-guns-tsv-path', type=str, default=PNNN_GUNS_GUIDE_PATH,
                        help="path to the pnnnguns file that contains metadata")
    parser.add_argument('--raw-bg-wav-path', type=str, default=RAW_BG_WAV_PATH,
                        help="path to raw background noise wav files to be chopped up")

    parser.add_argument('--path-prefix', type=str, default=REMOTE_PATH_PREFIX, help="path to the directory where relevant data should be stored")
    parser.add_argument('--frac-train', type=float, default=0.8, help="proportion of the positive examples to use in the training set")
    parser.add_argument('--frac-val', type=float, default=0.1, help="proportion of the positive examples to use in the validation set")
    parser.add_argument('--num-train-samples', type=int, default=10000, help="total number of samples desired in the training set")
    parser.add_argument('--num-val-samples', type=int, default=750, help="total number of samples desired in the validation set")
    parser.add_argument('--num-test-samples', type=int, default=500, help="total number of samples desired in the test set")
    parser.add_argument('--frac-noguns', type=float, default=0.5, help="fraction of samples to create that contain no gunshots")
    parser.add_argument('--seconds-per-example', type=float, default=10., help="total number of seconds encompassed by model input")
    parser.add_argument('--output-dir', type=str, default="processed",
                        help="the name of the directory that output data should be stored in. Look for the directories"
                             " 'train', 'val', and 'test'.")
    parser.add_argument('--bg-overlap-frac', type=float, default=0.,
                        help="fraction of sample overlap between adjacent background audio snippets. Defaults to no overlap.")
    parser.add_argument('--scenario-overlap-frac', type=float, default=0.25,
                        help="fraction of sample overlap between adjacent scenario audio snippets. Defaults to 0.25.")
    parser.add_argument('--scenario-prob', type=float, default=0.15,
                        help="Fraction of the generated examples that should have scenarios superimposed onto them")
    parser.add_argument('--use-scenarios-for-test-set', action='store_true',
                        help="Specify this to use scenarios to generate the test set.")

    # STFT configuration
    parser.add_argument('--nfft', type=int, default=4096,
                        help="Window size used for creating spectrograms. " +
                             "This should match the setting used to train the model.")
    parser.add_argument('--hop', type=int, default=800,
                        help="Hop size used for creating spectrograms (hop = nfft - n_overlap). " +
                             "This should match the setting used to train the model.")
    parser.add_argument('--sampling-freq', type=int, default=8000,
                        help="The frequency at which the data is sampled, in Hz. " +
                             "This should match the setting used to train the model.")
    parser.add_argument('--max-freq', type=int, default=1024,
                        help="Frequencies above this are omitted from generated spectrograms. " +
                             "This should match the setting used to train the model.")

    parser.add_argument('--clear-dirs-first', action='store_true',
                        help="remove all files in directories this script will output to before creating new outputs")

    return parser.parse_args()


def make_dir_if_not_exists(dir_path: str):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


def assert_args_valid(args):
    """
    Validation for command-line args. Call this at the start of the script to avoid making a user wait to find out their run has failed.
    """

    if not os.path.exists(args.path_prefix):
        raise ValueError(f"{args.path_prefix} not found in filesystem. Use a different path prefix?")

    if len(os.listdir(args.raw_bg_wav_path)) < 3:
        raise ValueError("Need at least 3 separate clips of background noise.")

    assert_fraction(args.frac_train)
    assert_fraction(args.frac_val)
    assert_fraction(args.frac_train + args.frac_val)

    assert_fraction(args.bg_overlap_frac)

    assert_fraction(args.scenario_overlap_frac)

    assert_fraction(args.scenario_prob)


if __name__ == "__main__":
    args = get_args()

    assert_args_valid(args)

    make_dir_if_not_exists(f"{args.path_prefix}/{args.output_dir}")
    make_dir_if_not_exists(f"{args.path_prefix}/{args.output_dir}/pos_train")
    make_dir_if_not_exists(f"{args.path_prefix}/{args.output_dir}/pos_val")
    make_dir_if_not_exists(f"{args.path_prefix}/{args.output_dir}/pos_test")
    make_dir_if_not_exists(f"{args.path_prefix}/{args.output_dir}/bg_snips")
    make_dir_if_not_exists(f"{args.path_prefix}/{args.output_dir}/datasets")
    make_dir_if_not_exists(f"{args.path_prefix}/{args.output_dir}/datasets/train")
    make_dir_if_not_exists(f"{args.path_prefix}/{args.output_dir}/datasets/val")
    make_dir_if_not_exists(f"{args.path_prefix}/{args.output_dir}/datasets/test")

    if args.clear_dirs_first:
        print("Clearing existing data at the target location...")
        # TODO: suppress ugly output from these calls?

        # positive outputs
        os.system(f"rm {args.path_prefix}/{args.output_dir}/pos_train/*.npy")
        os.system(f"rm {args.path_prefix}/{args.output_dir}/pos_val/*.npy")
        os.system(f"rm {args.path_prefix}/{args.output_dir}/pos_test/*.npy")

        # background chops
        bg_snip_dirs = os.listdir(f"{args.path_prefix}/{args.output_dir}/bg_snips")
        for bg_snip_dir in bg_snip_dirs:
            os.system(f"rm {args.path_prefix}/{args.output_dir}/bg_snips/{bg_snip_dir}/*.npy")

        # scenarios
        os.system(f"rm -rf {args.path_prefix}/{args.output_dir}/scenario_snips")

        # datasets
        os.system(f"rm {args.path_prefix}/{args.output_dir}/datasets/train/*.npy")
        os.system(f"rm {args.path_prefix}/{args.output_dir}/datasets/val/*.npy")
        os.system(f"rm {args.path_prefix}/{args.output_dir}/datasets/test/*.npy")
        os.system(f"rm {args.path_prefix}/{args.output_dir}/datasets/*.npy")

    train_df, val_df, test_df = get_positive_clips(args)

    print("Extracting positive examples...")
    write_pos_clips(train_df, f"{args.path_prefix}/{args.output_dir}/pos_train", args.ecoguns_wav_path, args.pnnn_guns_wav_path)
    write_pos_clips(val_df, f"{args.path_prefix}/{args.output_dir}/pos_val", args.ecoguns_wav_path, args.pnnn_guns_wav_path)
    write_pos_clips(test_df, f"{args.path_prefix}/{args.output_dir}/pos_test", args.ecoguns_wav_path, args.pnnn_guns_wav_path)

    # for now, only use one background noise wav file per dataset
    bg_filenames = os.listdir(args.raw_bg_wav_path)

    # chop background files
    for filename in bg_filenames[:3]:
        # assume all bg files end with '.wav'
        file_id = filename[:-4]
        bg_snip_dir = f"{args.path_prefix}/{args.output_dir}/bg_snips/{file_id}"
        make_dir_if_not_exists(bg_snip_dir)
        print(f"Chopping {filename} into {args.seconds_per_example}-second intervals...")
        chop_wav(f"{args.raw_bg_wav_path}/{filename}", bg_snip_dir, BACKGROUND_FILENAME_PREFIX,
                 args.seconds_per_example, args.bg_overlap_frac)

    print("Chopping scenario wav files...")
    chop_scenarios(args)
    scenario_split = split_scenarios(args)

    transform = lambda stft: 10*np.log10(stft)
    spec_ex = SpectrogramExtractor(nfft=args.nfft, hop=args.hop, max_freq=args.max_freq,
                                   sampling_freq=args.sampling_freq, pad_to=args.nfft)

    for bg_filename, setname in zip(bg_filenames[:3], ["train", "val", "test"]):
        bg_file_id = bg_filename[:-4]
        bg_snip_dir = f"{args.path_prefix}/{args.output_dir}/bg_snips/{bg_file_id}"

        if setname == "train":
            n_samples = args.num_train_samples
        elif setname == "val":
            n_samples = args.num_val_samples
        else:
            # test set
            n_samples = args.num_test_samples

        print(f"Generating {setname} dataset...")
        gen_mixed_data(bg_snip_dir, f"{args.path_prefix}/{args.output_dir}/pos_{setname}",
                       f"{args.path_prefix}/{args.output_dir}/datasets/{setname}", n_samples,
                       args.frac_noguns, spec_ex, transform, scenario_split[setname], args.scenario_prob)

    prepare_freq_mean_std(args)
