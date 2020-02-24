import json
import os
import numpy as np
import random

from functools import reduce
from multiprocessing import Pool
from bisect import bisect_left

from jsonize_data import OUTPUT_DIR, POOL_SIZE, OUTPUT_DIR_MFCC

# typing
from typing import List, Dict
np_arr_2d = np.array



SEC_PER_H = 3600
FRAME_LENGTH = 64
TO_CROP_EACH_SIDE = 8
FRAME_LENGTH_AFTER_CROP = 64
# derived parameter
#FRAME_LENGTH = FRAME_LENGTH_AFTER_CROP - TO_CROP_EACH_SIDE



NUM_EMPTY_PER_FILE = 120
PLACE_NAMES = ['ceb1', 'ceb4', 'jobo', 'dzan']



def is_call(file_hour : int, sound_file_time_start : float, sound_file_time_end : float, label_dict) -> bool:
    """
    detects if there is overlap with an elephant call
    NOTE : sound_file_time_start is assumed to be relative to when spectrogram starts
    """

    # if no calls
    if len(label_dict) == 0: return False


    for begin_hour, begin_time, end_time in zip(label_dict['begin_hour'], label_dict['begin_time'], label_dict['end_time']): 

        if file_hour != begin_hour:
            continue

        begin_time -= begin_hour*SEC_PER_H
        end_time -= begin_hour*SEC_PER_H

        if sound_file_time_end >= begin_time or sound_file_time_start <= end_time:
            return True

    return False





def extract_empty(sound_jsons : List[str], label_json : str, num_per_file : int) -> List[np_arr_2d]:
    """
    uniformyl randomly selects spectrograms containing no elephant sound
    """

    print("constructing empty for", label_json)

    label_dict = json.load(open(label_json, "r"))
    empty_calls = []

    for sound_file in sound_jsons:

        data = json.load(open(sound_file, "r"))
        spectrum = np.array(data["spectrum"])
        times = np.array(data["times"])
        file_hour = int(sound_file.split('_')[4][0:2]) # CHekc thiss!!!

        num_sampled = 0
        #start_inds = range(len(times) - FRAME_LENGTH - 2*TO_CROP_EACH_SIDE)
        start_inds = range(len(times) - FRAME_LENGTH)

        while num_sampled < num_per_file:
            start_ind = random.choice(start_inds)
            end_ind = start_ind + FRAME_LENGTH
            if not is_call(file_hour, times[start_ind], times[end_ind], label_dict):
                empty_calls.append(spectrum[:, start_ind:end_ind])
                num_sampled += 1


    return empty_calls





def extract_calls(sound_jsons : List[str] , label_json : str) -> List[np_arr_2d]:
    """
    extracts all elephant calls
    """

    print("constructing calls for", label_json)
    
    label_dict = json.load(open(label_json, "r"))
    sound_jsons = sorted(sound_jsons, key = lambda sound_file : int(sound_file.split('_')[3][0:2]))
    print (sound_jsons)
    # if no calls
    if len(label_dict) == 0: return []

    # setup loop invariants
    all_calls = []
    num_sound_files_processed = 0
    curr_sound_file = sound_jsons[0]
    file_hour = int(curr_sound_file.split('_')[4][0:2]) # Note check this
    data = json.load(open(curr_sound_file, "r"))
    curr_spectrum = np.array(data["spectrum"])
    curr_times = np.array(data["times"])

    for begin_hour, begin_time, end_time in zip(label_dict['begin_hour'], label_dict['begin_time'], label_dict['end_time']):
        begin_hour, begin_time, end_time = map(float, (begin_hour, begin_time, end_time))

        # get correct file
        while file_hour < begin_hour:
            num_sound_files_processed +=1
            curr_sound_file = sound_jsons[num_sound_files_processed]
            file_hour = int(curr_sound_file.split('_')[4][0:2])

            if file_hour < begin_hour : continue
            data = json.load(open(curr_sound_file, "r"))
            curr_spectrum = np.array(data["spectrum"])
            curr_times = np.array(data["times"])



        begin_time -= begin_hour*SEC_PER_H
        end_time -= begin_hour*SEC_PER_H

        # figure out right bisection
        start_index = bisect_left(curr_times, begin_time)
        end_index = bisect_left(curr_times, end_time)
        call_len = end_index - start_index

        # skip data at very end, very start or calls longer than FRAME_LENGTH
        if call_len > FRAME_LENGTH or start_index + FRAME_LENGTH+TO_CROP_EACH_SIDE >= len(curr_times) or start_index <= FRAME_LENGTH - call_len + TO_CROP_EACH_SIDE  : continue

        #start_index -= random.randint(0, FRAME_LENGTH - call_len)
        #start_index -= TO_CROP_EACH_SIDE
        #all_calls.append(curr_spectrum[:,start_index:start_index+FRAME_LENGTH+2*TO_CROP_EACH_SIDE]) 
        all_calls.append(curr_spectrum[:,start_index:start_index+FRAME_LENGTH])


    return all_calls



def get_call_args(name='ceb1', use_mfcc=False):

    
    assert name in PLACE_NAMES


    outdir = OUTPUT_DIR_MFCC if use_mfcc else OUTPUT_DIR

    call_args = []
    # find all relevant files, put into call_args
    for lower_data_dir in filter(lambda fname : os.path.isdir(outdir + "/" + fname) and name in fname, os.listdir(outdir)): # for all directories
        input_path = outdir + "/" + lower_data_dir
        sound_jsons = [input_path + "/" + fname for fname in os.listdir(input_path) if ".json" in fname and "labels" not in fname]
        label_json = [input_path + "/" + fname for fname in os.listdir(input_path) if "labels" in fname]
        assert len(label_json) == 1
        label_json = label_json[0]
        call_args.append((sound_jsons, label_json))


    return call_args


        
def create_dataset_from_json(name="ceb1", use_mfcc=False):


    call_args = get_call_args(name=name, use_mfcc=use_mfcc)
    if len(call_args) == 0:
        return [], []

    # extract data from files
    with Pool(POOL_SIZE) as thread_pool:
        concat_func = lambda x,y : x+y
        all_calls = reduce(concat_func , thread_pool.starmap(extract_calls, call_args))
        all_empty = reduce(concat_func, thread_pool.starmap(extract_empty, [(t1, t2, NUM_EMPTY_PER_FILE) for t1,t2 in call_args]))

    return all_calls, all_empty




if __name__ == '__main__':
    # Flag to analyze the mean
    analyze = True

    if not analyze:
        all_calls = []
        all_empty = []
        for region in PLACE_NAMES:
            calls, empty = create_dataset_from_json(name=region)
            all_calls.extend(calls)
            all_empty.extend(empty)

        print("in total we have", len(all_calls), "calls", "and", len(all_empty), "empty")
        all_calls = np.stack(all_calls)
        print (all_calls.shape)
        all_empty = np.stack(all_empty)

        print ("Mean: ", np.mean(all_empty))
        print ("Std: ", np.std(all_empty))
        print ("Median: ", np.median(np.max(all_calls, axis=(1,2))))
        print ('')
        print ("Mean Log: ", np.mean(10 * np.log10(all_empty)))
        print ("Std Log: ", np.std(10 * np.log10(all_empty)))
        print ("Median: ", (np.median(np.max(10 * np.log10(all_calls), axis=(1,2)))))

        # Save the noise and call data files
        np.save("empty.npy", all_empty)
        np.save("call.npy", all_calls)
    else:
        all_empty = np.load("empty.npy")
        all_calls = np.load("call.npy")

        # Do the means now over the feature axis!
        all_calls = np.transpose(all_calls, (0, 2, 1)) # Flip the time and freq axis
        all_empty = np.transpose(all_empty, (0, 2, 1))

        mean = np.mean(all_empty, axis=(0, 1))
        std = np.std(all_empty, axis=(0, 1))
        median = np.median(np.max(all_calls, axis=1), axis = 0)

        mean_log = np.mean(10 * np.log10(all_empty), axis=(0, 1))
        std_log = np.std(10 * np.log10(all_empty), axis=(0, 1))
        median_log = np.median(np.max(10 * np.log10(all_empty), axis=1), axis = 0)

        #Noise data
        Noise_Directory = "./Noise_Stats/"
        np.save(Noise_Directory + "mean.npy", mean)
        np.save(Noise_Directory + "std.npy", std)
        np.save(Noise_Directory + "median.npy", median)
        np.save(Noise_Directory + "mean_log.npy", mean_log)
        np.save(Noise_Directory + "std_log.npy", std_log)
        np.save(Noise_Directory + "median_log.npy", median_log)







