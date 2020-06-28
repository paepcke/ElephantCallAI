#!/usr/bin/env python
'''
Created on Nov 1, 2019

@author: paepcke
'''
from _datetime import date
import argparse
import csv
import math
import os
import re
import sys
from _collections import OrderedDict


class DayNightCounter(object):
    '''
    classdocs
    '''
    DATE_PAT       = re.compile(r'[0-9]{8}')
    # 7am: seconds since midnight to 7am
    NIGHT_END_SECS   = 25200
    # 8pm: seconds since midnight to 8pm
    NIGHT_START_SECS = 72000

    #-------------------------
    # Constructor 
    #--------------

    def __init__(self, label_file_path):
        '''
        Constructor
        '''
        self.read_labels_to_dict(label_file_path)

    #-------------------------
    # read_labels_to_dict 
    #--------------

    def read_labels_to_dict(self, label_file_path):
        with open(label_file_path, 'r') as in_fd:
            reader = csv.DictReader(in_fd,
                                    delimiter = '\t'
                                    )
            #col_names     = reader.fieldnames
            col_names = ['Observation Row', 'Begin Time', 'Location', 'start_call_24hr_clock', 'is_night']
            writer = csv.DictWriter(sys.stdout,
                                    col_names,
                                    delimiter='\t')
            
            # Col header:
            sys.stdout.write('\t'.join(writer.fieldnames) + '\n')
            row_num = 0
            for row in reader:
                row_num += 1
                
                begin_time_secs = row['Begin Time (s)']
                # Get location date from nn01_20190320_....txt
                label_loc  = row['Begin File'].split('_')[0]                
                label_date = row['Begin File'].split('_')[1]

                
                if DayNightCounter.DATE_PAT.search(label_date) is None:
                    raise ValueError(f"Could not extract date from row {row_num}'s begin-file.")
                
                # Get a date obj from the date:
                label_date_obj = date(int(label_date[0:4]), int(label_date[4:6]), int(label_date[6:8]))
                
                # Get begin time as seconds since midnight e.g.: 2543.4433
                begin_time_secs = float(row['Begin Time (s)'])
                row['is_night'] = 1 if (begin_time_secs <= DayNightCounter.NIGHT_END_SECS
                                      or begin_time_secs >= DayNightCounter.NIGHT_START_SECS) \
                                    else 0

                start_time_24   = f"{label_date_obj} {math.floor(begin_time_secs / 3600)}:" +\
                                  f"{math.floor((begin_time_secs % 3600) / 60)}:" +\
                                  f"{math.floor((begin_time_secs % 3600) % 60)}"

                #row['start_call_24hr_clock'] = start_time_24
                #row['rounded_time'] = str(round(begin_time_secs / 3600)) + ':00hrs'
                row_extract_values = [row_num, begin_time_secs, label_loc, start_time_24, row['is_night']]
                row_extract = OrderedDict(zip(col_names, row_extract_values))
                writer.writerow(row_extract)
            
            


# --------------------------- Main ------------------        
if __name__ == '__main__':


    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Add info to elephant label files; write tsv to stdout"
                                     )

    parser.add_argument('label_file',
                    help='Path to elephant call tsv labels path',
                    )
    
    args = parser.parse_args()

    label_file = args.label_file
    if not os.path.exists(label_file):
        print(f"File {label_file} does not exist.")
        sys.exit(1)

    DayNightCounter(label_file)
    sys.exit(0)
    # TESTING
    #DayNightCounter('/Users/paepcke/Project/Wildlife/Data/Elephants/labels_2018_all.tsv')        