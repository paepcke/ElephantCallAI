from matplotlib import pyplot as plt
from matplotlib import mlab as ml
import numpy as np
import csv
import os
import time
import multiprocessing
from multiprocessing import Value
from scipy.io import wavfile
from visualization import visualize
import math
import argparse
from random import shuffle
import random
from functools import partial

parser = argparse.ArgumentParser()
parser.add_argument('--data_dirs', dest='data_dirs', nargs='+', type=str,
    help='Provide the data_dirs with the files that you want to be processed')


args = parser.parse_args()
data_dirs = args.data_dirs

data_pairs = {}
for currentDir in data_dirs:
    #Iterate through each dir and get files within
    for(dirpath, dirnames, filenames) in os.walk(currentDir):
        # Create data/label 
        # pairs (i.e. (.wav, .txt))
        for eachFile in filenames:
            # Strip off the location and time tags
            tags = eachFile.split('_')
            data_id = tags[0] + '_' + tags[1]
            file_type = eachFile.split('.')[1]

            if (file_type not in ['wav', 'txt']):
                continue

            # Insert the file name into the dictionary
            # with the file type tag for a given id
            if not data_id in data_pairs:
                data_pairs[data_id] = {}
                data_pairs[data_id]['id'] = data_id
                data_pairs[data_id]['dir'] = currentDir

            data_pairs[data_id][file_type] = eachFile

# data_pairs now contains all of the wav/txt data file pairs.
# Let us create a list of these pairs to randomly split into
# train/val if the flag is set

# Exclude strange pairs where there is not a wav or gt file
# Some wav files do not have a gt because there are no calls!
file_pairs = []
for _, pair in data_pairs.items():
    # We must be careful in the case where there is no
    # label file basically there are no calls!
    if 'wav' in pair:
        gt_labels = None if 'txt' not in pair else pair['txt']
        file_pairs.append(((pair['wav'], gt_labels, pair['id'], pair['dir'])))

count = 0
for file in file_pairs:
    in_count = 0
    # Count the number of elephant calls!
    if file[1] != None:
        label_file = file[3] + '/' + file[1]
        labels = csv.DictReader(open(label_file,'rt'), delimiter='\t')

        last_call_num = 0
        for call in labels:
            count += 1
            in_count += 1
            last_call_num = call['Selection']

        if (int(last_call_num) != in_count):
            print (last_call_num)
            print (in_count)
            print ("FUCKKKKKKK")



print ("number of calls:", count)



