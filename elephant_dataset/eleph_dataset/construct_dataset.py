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

    # extract data from files
    with Pool(POOL_SIZE) as thread_pool:
        concat_func = lambda x,y : x+y
        all_calls = reduce(concat_func , thread_pool.starmap(extract_calls, call_args))
        all_empty = reduce(concat_func, thread_pool.starmap(extract_empty, [(t1, t2, NUM_EMPTY_PER_FILE) for t1,t2 in call_args]))

    return all_calls, all_empty




if __name__ == '__main__':


    all_calls, all_empty = create_dataset_from_json()

    print("in total we have", len(all_calls), "calls", "and", len(all_empty), "empty")
    all_calls = np.dstack(all_calls)
    all_empty = np.dstack(all_empty)
    print (np.mean(10 * np.log10(all_empty)))
    print (np.std(10 * np.log10(all_empty)))
    print (np.median(np.max(10 * np.log10(all_calls), axis=(1,2))))


    '''
    from .utils import plot_call
    for i in range(5):
        call = all_calls[i]
        plot_call(call, "plotDir/call_{}.png".format(i))

    for i in range(5):
        call = all_empty[i]
        plot_call(call, "plotDir/empty_{}.png".format(i))
    '''






