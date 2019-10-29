from matplotlib import pyplot as plt
from matplotlib import mlab as ml
import numpy as np
import csv
import os
import time
import multiprocessing
from scipy.io import wavfile
from visualization import visualize
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', dest='dataDir', default='../elephant_dataset/New_Data/Truth_Logs/', 
    type=str, help='The top level directory with the data (e.g. Truth_Logs)')
parser.add_argument('--out', dest='outputDir', default='../elephant_dataset/Processed_data_new/',
     help='The output directory')
parser.add_argument('--NFFT', type=int, default=4096, help='Window size used for creating spectrograms')
parser.add_argument('--hop', type=int, default=641, help='Hop size used for creating spectrograms')
parser.add_argument('--window', type=int, default=256, 
    help='Deterimes the window size in frames of the resulting spectrogram') # Default corresponds to 21s
parser.add_argument('--max_f', dest='max_freq', type=int, default=100, help='Deterimes the maximum frequency band')

np.random.seed(8)
VERBOSE = False

def generate_labels(labels, len_spect, time_indeces, spectrogram_info):
    '''
        Given ground truth label file 'label' create the full 
        segmentation labeling for a corresponding audio 
        spectrogram. Namely, return a vector containing a 0/1 labeling 
        for each spectrogram time slice
    '''
    labelFile = csv.DictReader(open(labels,'rt'), delimiter='\t')
    
    labelMatrix = np.zeros(shape=(len_labels),dtype=int)

    samplerate = spectrogram_info['samplerate']
    # Iterates through labels and marks the segments with elephant calls
    for row in labelFile:
        # Use the file offset to determine the start of the call
        start_time = float(row['File Offset (s)'])
        call_length = float(row['End Time (s)']) - float(row['Begin Time (s)'])
        end_time = start_time + call_length
        
        # Figure out which spectrogram slices we are on
        # to get columns that we want to span with the given
        # slice. This math transforms .wav indeces to spectrogram
        # indices
        start_spec = max(math.ceil((start_time * samplerate - spectrogram_info['NFFT'] / 2.) / spectrogram_info['hop']), 0)
        end_spec = min(math.ceil((end_time * samplerate - spectrogram_info['NFFT'] / 2.) / spectrogram_info['hop']), labelMatrix.shape[0])
        labelMatrix[start_spec : end_spec] = 1

    return labelMatrix




