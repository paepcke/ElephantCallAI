'''
Created on Feb 25, 2020

@author: paepcke
'''
import argparse
import csv
import os
from pathlib import Path
import sys
from csv import DictReader


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
        
        all_label_rows = []
        
        for label_path in full_label_paths:
            with open(label_path,'r') as csv_fd:
                reader = csv.reader(csv_fd, delimiter='\t')
                # Read the col headers, and keep them only 
                # from the first label file:
                header = next(reader)
                if len(all_label_rows) == 0:
                    all_label_rows.append(header)
                all_label_rows.extend([row for row in reader])
        return all_label_rows
    
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

    
    
        