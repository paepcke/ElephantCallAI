from matplotlib import pyplot as plt
from matplotlib import mlab as ml
import numpy as np
import csv
import os
import librosa
import time
import multiprocessing
from scipy.io import wavfile

dataDir = '../elephant_dataset/New_Data/Truth_Logs/'

def get_call_stats(label_file_path):
    '''
        Given the truth logs for a recording, calculate
        some basic statistics on the call

        Return - Dictionary with following keys

        num_calls: number of total calls

        num_overlap: number of calls that are overlapping
        (for example, if two calls overlap than this value is 2)

        min_call_length: minimum call duration

        max_call_length: maximum call duration

        lowest_boxed_freq: lowest frequency recorded when boxing in 
        a call by a volunteer

        highest_boxed_freq: highest frequency recorded when boxing in 
        a call by a volunteer
    '''

    labelFile = csv.DictReader(open(label_file_path,'rt'), delimiter='\t')

    num_calls = 0
    num_overlap = 0
    min_call_length = float("inf")
    max_call_length = 0
    lowest_boxed_freq = float("inf")
    highest_boxed_freq = 0
    num_above_100 = 0
    num_complete_above_100 = 0 # How many calls are completely above 100
    avg_call_length = 0

    # Tracks the call with the
    # latest ending time seen so far
    last_call_ending = 0

    for row in labelFile:
        # Get basic information about the call
        start_time = float(row['File Offset (s)'])
        call_length = float(row['End Time (s)']) - float(row['Begin Time (s)'])
        end_time = start_time + call_length
        low_freq = float(row['Low Freq (Hz)'])
        high_freq = float(row['High Freq (Hz)'])

        # Update statistics
        num_calls += 1
        # If this new call started before the 
        # ending of the latest ending call seen
        # so far, we have an overlap
        if start_time <= last_call_ending:
            num_overlap += 1
        
        last_call_ending = max(end_time, last_call_ending)

        min_call_length = min(call_length, min_call_length)
        max_call_length = max(call_length, max_call_length)
        lowest_boxed_freq = min(low_freq, lowest_boxed_freq)
        highest_boxed_freq = max(high_freq, highest_boxed_freq)
        avg_call_length += call_length

        if (high_freq > 100):
            num_above_100 += 1

        if (low_freq > 100):
            num_complete_above_100 += 1

    avg_call_length = avg_call_length / float(num_calls)

    stats = {'num_calls': num_calls,
             'num_overlap': num_overlap,
             'min_call_length': min_call_length,
             'max_call_length': max_call_length,
             'lowest_boxed_freq': lowest_boxed_freq,
             'highest_boxed_freq': highest_boxed_freq,
             'num_above_100': num_above_100, 
             'num_complete_above_100': num_complete_above_100,
             'avg_call_length': avg_call_length}

    return stats

def main():
    # Iterate through all data directories
    allDirs = [];
    # Get the directories that contain the data files
    for (dirpath, dirnames, filenames) in os.walk(dataDir):
        allDirs.extend(dirnames);
        break

    stats = {'num_calls': 0,
             'num_overlap': 0,
             'min_call_length': float("inf"),
             'max_call_length': 0,
             'lowest_boxed_freq': float("inf"),
             'highest_boxed_freq': 0,
             'num_above_100': 0, 
             'num_complete_above_100': 0,
             'avg_call_length': 0}
    # Iterate through all files with in data directories
    file_counter = 0
    for dirName in allDirs:
        #Iterate through each dir and get files within
        currentDir = dataDir + '/' + dirName;
        for(dirpath, dirnames, filenames) in os.walk(dataDir+'/'+dirName):
            # Iterate through the files only analyzing the .txt label files
            data_pairs = {}
            for eachFile in filenames:
                file_type = eachFile.split('.')[1]
                
                if (file_type == 'txt'):
                    file_counter += 1
                    temp_stats = get_call_stats(currentDir + '/' + eachFile)
                    stats['num_calls'] += temp_stats['num_calls']
                    stats['num_overlap'] += temp_stats['num_overlap']
                    stats['min_call_length'] = min(temp_stats['min_call_length'], stats['min_call_length'])
                    stats['max_call_length'] = max(temp_stats['max_call_length'], stats['max_call_length'])
                    stats['lowest_boxed_freq'] = min(temp_stats['lowest_boxed_freq'], stats['lowest_boxed_freq'])
                    stats['highest_boxed_freq'] = max(temp_stats['highest_boxed_freq'], stats['highest_boxed_freq'])
                    stats['num_above_100'] += temp_stats['num_above_100']
                    stats['num_complete_above_100'] += temp_stats['num_complete_above_100']
                    stats['avg_call_length'] += temp_stats['avg_call_length']

    # Update the avg call length
    stats['avg_call_length'] /= float(file_counter)
                
    for item in stats.items():
        print (item[0], ':', item[1])

if __name__ == '__main__':
    main()





