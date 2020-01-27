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
parser.add_argument('file', type=str,
    help='how to use')
parser.add_argument('--wav_files', dest='wav_files', default='./test/', 
    type=str, help='The top level directory with the data (e.g. Truth_Logs)')


args = parser.parse_args()

# Extract the header for the csv file
with open(args.file, 'r') as f:
    reader = csv.reader(f, delimiter='\t')
    lines = list(reader)

header = lines[0] # Check this

# Now step through the wav file
current_file = None 
wav_path = args.wav_files
file_lines = []
file_lines.append(header)
for i in range(1, len(lines)):
    line = lines[i]
    wav_file = line[9]
    if current_file != wav_file:
        # Output the gt for the processed
        # wav file
        if current_file != None:
            out_path = wav_path + current_file[:-4] + '.txt'
            with open(out_path, 'w') as write_file:
                writer = csv.writer(write_file, delimiter='\t')
                # Fix the numbering of the lines
                for i in range(1, len(file_lines)):
                    # Re-adjust the start and end time
                    call_length = float(file_lines[i][4]) - float(file_lines[i][3])
                    start_time = float(file_lines[i][8]) 
                    file_lines[i][3] = start_time
                    file_lines[i][4] = start_time + call_length
                    file_lines[i][0] = str(i)
                    
                writer.writerows(file_lines)
            

        file_lines = file_lines[:1]
        current_file = wav_file

    file_lines.append(line)

# Do the last wav file
out_path = wav_path + current_file[:-4] + '.txt'
with open(out_path, 'w') as write_file:
    writer = csv.writer(write_file, delimiter='\t')
    # Fix the numbering of the lines
    for i in range(1, len(file_lines)):
        # Re-adjust the start and end time
        call_length = float(file_lines[i][4]) - float(file_lines[i][3])
        start_time = float(file_lines[i][8]) 
        file_lines[i][3] = start_time
        file_lines[i][4] = start_time + call_length
        file_lines[i][0] = str(i)
        
    writer.writerows(file_lines)

