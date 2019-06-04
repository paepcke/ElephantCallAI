from matplotlib import mlab as ml
import numpy as np  
import soundfile as sf
import json
import csv
import os

from bisect import bisect_left
from multiprocessing import Pool


#DATA_DIR = '/home/shared/elephant/rawdata/' # Dir containing all raw data in seperate files like 'ceb1_XXXXXXX'
DATA_DIR = '../Data'
#'/home/shared/elephant/rawdata/'
#'/home/shared/Elephants'
OUTPUT_DIR = '../Json_data'
#OUTPUT_DIR = '/home/shared/ElephantOutput/' # Dir that will contain all of the output data
OUTPUT_DIR_MFCC = '/home/shared/ElephantOutputMFCC/' # Dir that will contain all of the output data
NUM_FFT        = 512 # Number of points to do FFT over 


CUTOFF_BOT = 0
CUTOFF_FREQ = 150
POOL_SIZE = 16


import librosa


def get_freqs(fname_in):
    """ returns the freqs used """
    data, samplerate = sf.read(fname_in, dtype='float32')
    spectrum, freqs, time = ml.specgram(data, NFFT=NUM_FFT, Fs=samplerate)
    cutoff = bisect_left(freqs, CUTOFF_FREQ)
    spectrum = spectrum[CUTOFF_BOT:cutoff]
    print("spectrum have shape", spectrum.shape)    
    freqs = freqs[CUTOFF_BOT:cutoff]
    return freqs


def jsonize_sound(fname_in, fname_out, use_mfcc):
    """
    dumps the sounds
    """

    print("jsonizing sound", fname_in, "->", fname_out)

    data, samplerate = sf.read(fname_in, dtype='float32')
    spectrum, freqs, time = ml.specgram(data, NFFT=NUM_FFT, Fs=samplerate);
    

    if use_mfcc:
        spect_2 = librosa.feature.mfcc(y=data, sr=samplerate, hop_length=128*3, n_fft=512)
        print("using mfcc")
        # fix slightly different padding
        num_new_points = spect_2.shape[1] - spectrum.shape[1]
        if num_new_points == 0:
            pass
        elif num_new_points == 1:
            spect_2 = spect_2[1:]
        elif num_new_points == 2:
            spect_2 = spect_2[1:-1]
        else:
            raise Exception("weird number: {}".format(num_new_points)) 

        spectrum = spect_2

    else:
        cutoff = bisect_left(freqs, CUTOFF_FREQ)
        spectrum = spectrum[CUTOFF_BOT:cutoff]
        freqs = freqs[CUTOFF_BOT:cutoff]

    data = {'spectrum' : spectrum.tolist(), 'times' : time.tolist()}
    json.dump(data, open(fname_out, "w"))


def jsonize_labels(fname_in, fname_out):
    """

    will dump a struct of the following type
    {start_times : [t1, t2 ...], stop_times : [t1, t2...]}
    """

    print("jsonizing label", fname_in, "->", fname_out)
 
    calls = {}
    label_file = csv.DictReader(open(fname_in,'r'))

    for row in label_file:
        for key, val in row.items():
            if key not in calls:
              calls[key] = []
            calls[key].append(val)

    json.dump(calls, open(fname_out, "w"))

    

def jsonize_all_data(use_mfcc=False):
    """
    jsonizes all data by traversing directories
    """

    sound_args = []
    label_args = []

    for lower_data_dir in filter(lambda fname : os.path.isdir(DATA_DIR + "/" + fname), os.listdir(DATA_DIR)):

        input_path = DATA_DIR + "/" + lower_data_dir
        data_files = os.listdir(input_path)

        # set output dir
        output_path = OUTPUT_DIR_MFCC if use_mfcc else OUTPUT_DIR
        output_path += "/" + lower_data_dir
        if not os.path.isdir(output_path) : os.mkdir(output_path)

        for data_file in data_files:
            if data_file.split('.')[1] == 'csv':
                label_args.append((input_path + "/" + data_file, output_path + "/" + data_file.split('.')[0] + "_labels.json"))
            elif data_file.split('.')[1] == 'flac':
                sound_args.append((input_path + "/" + data_file, output_path + "/" + data_file.split('.')[0] + ".json", use_mfcc))


    with Pool(POOL_SIZE) as thread_pool:
        thread_pool.starmap(jsonize_labels, label_args)
        thread_pool.starmap(jsonize_sound, sound_args)


if __name__ == '__main__':

    jsonize_all_data()





