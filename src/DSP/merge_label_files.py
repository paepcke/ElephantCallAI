#!/usr/bin/env python
'''
Created on Feb 25, 2020

@author: paepcke
'''
import argparse
import csv
import os
from pathlib import Path
import pathlib
import sys


class LabelMerger(object):
    '''
    Combines all elephant labels into one 
    file with a single column header. Files are
    assumed to have .txt extension in the passed-in
    directory. 
    '''


    def __init__(self, dir_paths, outfile=None):
        '''
        Constructor
        '''
        
        all_label_files = []
        for label_dir in dir_paths:
            # Find all label files
            if not os.path.exists(label_dir):
                print(f"Label dir {label_dir} does not exist; skipping.")
                continue
                
            label_files = os.listdir(label_dir)
            label_files = [label_file for label_file in label_files 
                           if Path(label_file).suffix in ['.txt', '.tsv', '.csv']
                           ]
            full_label_paths = [os.path.join(label_dir,
                                             label_file) for label_file in label_files]
            all_label_files.extend(full_label_paths)
        
        # Get array of label row, with .wav file localized,
        # and time/freq ranges added: 
        all_label_rows = self.merge_paths(full_label_paths)

        if outfile is None:
            self.out_result(all_label_rows, sys.stdout)
        else:
            with open(outfile, 'w') as fd:
                self.out_result(all_label_rows, fd)
        
    #------------------------------------
    # out_result
    #-------------------
    
    def out_result(self, rows, fd):
        writer = csv.writer(fd, delimiter='\t')
        for row in rows:
            writer.writerow(row)
            

    #------------------------------------
    # merge_paths
    #-------------------    

    def merge_paths(self, full_label_paths):
        '''
        Takes list of full paths to label files. 
        Returns an array of rows from all the labels
        combined. The 'Begin Path' to the corresponding
            .wav file is updated to be the local path.
            Columns 'Time Range' and 'Freq Range' are added
            to each row.
        
        @param full_label_paths: paths to label files to combine
        @type full_label_paths: (str)
        '''
        
        all_label_rows = []
        
        for label_path in full_label_paths:
            curr_path_dir = os.path.dirname(label_path)
            with open(label_path,'r') as csv_fd:
                reader = csv.reader(csv_fd, delimiter='\t')
                # Read the col headers, and keep them only 
                # from the first label file:
                header = next(reader)
                if len(all_label_rows) == 0:
                    # Anticipate adding time and freq ranges:
                    header.extend(['Time Range', 'Freq Range'])
                    all_label_rows.append(header)
                    try:
                        self.begin_time_pos    = header.index('Begin Time (s)')
                        self.end_time_pos      = header.index('End Time (s)')
                        self.low_freq_pos      = header.index('Low Freq (Hz)')
                        self.high_freq_pos     = header.index('High Freq (Hz)')
                        self.full_wav_path_pos = header.index('Begin Path')
                    except ValueError as e:
                        # For this label file the headers aren't what
                        # we expect:
                        print(f"Header file of {label_path} missing a column: {repr(e)}; ignoring file.")
                        continue
                    
                for row in reader:
                    # Localize the file path, and add frequency
                    # and time spans: 
                    self.enrich_row(row, curr_path_dir)
                    all_label_rows.append(row)

        return all_label_rows

    #------------------------------------
    # enrich_row
    #-------------------    

    def enrich_row(self, label_row, label_dir):
        '''
        Given a label row:
            1. Change the Cornell file path to local file path
            2. Add frequency range column ('Freq Range')
            3. Add time range column ('Time Range')
            
        Assumptions: the following instance vars are 
        indexes into the row:
        
            self.begin_time_pos
            self.end_time_pos
            self.low_freq_pos
            self.high_freq_pos
            self.full_wav_path_pos

        @param label_row: label data
        @type label_row: [str]
        @param label_dir: directory of the label file from where row 
            originated
        @type label_dir: str
        @return: new label row
        @rtype: [str]
        '''
        cornell_file_path = label_row[self.full_wav_path_pos]
        if cornell_file_path.find('\\') > -1:
            # Windows path:
            cornell_file_path = pathlib.PureWindowsPath(cornell_file_path)
        else:
            cornell_file_path = pathlib.PurePath(cornell_file_path)
            
        file_name  = cornell_file_path.name
        local_path = os.path.join(label_dir, file_name)
        label_row[self.full_wav_path_pos] = local_path
        time_range = str(float(label_row[self.end_time_pos]) - float(label_row[self.begin_time_pos]))
        freq_range = str(float(label_row[self.high_freq_pos]) - float(label_row[self.low_freq_pos]))
        label_row.extend([time_range, freq_range])
        return label_row
        
# ----------------------------- Main ----------------

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Combine elephant labels in multiple dirs into one file."
                                     )

    parser.add_argument('-o', '--outfile',
                        help='fully qualified outfile name. Default: stdout',
                        dest='outfile',
                        default=None);
    parser.add_argument('dirs',
                        nargs='+',
                        help='Repeatable: paths to directories with labels.')

    args = parser.parse_args();
    
    LabelMerger(args.dirs, args.outfile)

    
    
        