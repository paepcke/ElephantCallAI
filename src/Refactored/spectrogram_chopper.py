#!/usr/bin/env python
'''
Created on Feb. 11, 2020

@author: Jonathan Gomes Selman
'''
import argparse
import math
import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_utils import FileFamily
from data_utils import DATAUtils
from data_utils import AudioType
from logging_service import LoggingService


class SpectrogramChopper(object):
    '''
    Given a single spectrogram and the corresponding labeling, 
    chop the spectrograms into the specified window size! 
    
    NOTE: For now avoid doing any parallelization to just get this done functionality wise
    '''

    #------------------------------------
    # Constructor
    #-------------------

    def __init__(self,
                 spect_file,
                 spect_label_file,
                 snippet_outdir,
                 window_size=256,
                 logfile=None,
                 ):
        '''
        
        @param spect_file: The spectrogram file we want to chop
        @type spect_file: str
        @param spect_label_file: The spectrogram label file with 0/1 binary labels
        @type spect_label_file: str
        @param snippet_outdir: where the snippets are to be placed.
        This is likely to be either in the training / test folder
        of data
        @type snippet_outdir: str
        @param logfile: where to log
        @type logfile: {None|str}
        @param test_snippet_width: if set to a positive int,
            set the snippet width to that value. Used by unittests
            to work with smaller dataframes
        @type test_snippet_width: int
        '''
        '''
        Constructor
        '''
                        
        # Again we may just ditch this!!!
        self.log = LoggingService(logfile=logfile,
                                  msg_identifier=f"chop_spectrograms")
               
        # If snippet_outdir's path does not exist
        # yet, create all dirs along the path:
        if not os.path.exists(snippet_outdir):
            os.makedirs(snippet_outdir)
            
        
        """
            Open the spect file and corresponding label and then CHOP, CHOP, CHOP
        """
        try:
            spectrogram = np.load(spect_file)
        except:
            raise ValueError("Invalid spectrogram file")

        try:
            spect_labels = np.load(spect_label_file)
        except:
            raise ValueError("Invalid spectrogram label file")

        # Extract the spectrogram name so that we can label the chopped spectrograms
        file_family = FileFamily(spect_file)
        spect_root_name = file_family.file_root

        # Remove the .npy tag
        self.chop_spectrogram(spectrogram, spect_labels, window_size, spect_root_name, snippet_outdir)



    def chop_spectrogram(self, spectrogram, spect_labels, window_size, spect_root_name, snippet_outdir):
        '''
            Given a spectrogram and its corresponding labeling,
            chop the spectrogram into window_size pieces. Save each
            chopped window to the snipped_outdir based on the 
            spect_root_name (i.e. data file name without any other tags, 
            corresponding chop id, and finally 
            a label (pos/neg) indicating whether the window contains an
            elephant rumble (or part of one). 

            Note 1: For now if we cannot evenly chop the spectrogram
            just discard the final piece

            Note 2: In the future, we should also consider generating windows that
            are larger than the "CNN" window size. This will allow us to do
            window shifting!
        '''
        print ("####################################")
        print ("####### Chopping Spectrogram #######")
        print ("####################################")
        start_idx = 0
        spect_id = 0
        while start_idx + window_size < spectrogram.shape[0]:
            # Chop chop
            chopped_window = spectrogram[start_idx: start_idx + window_size, :]
            chopped_labels = spect_labels[start_idx: start_idx + window_size]

            # Check if the window contains elephants rumbles
            slice_name = spect_root_name
            rumble_flag = np.sum(chopped_labels) > 0
            if rumble_flag:
                slice_name += '_pos'
            else:
                slice_name += '_neg'

            # Save the chopped window, incorperating the spect id
            np.save(os.path.join(snippet_outdir, slice_name + '-features_' + str(spect_id) + ".npy"), chopped_window)
            np.save(os.path.join(snippet_outdir, slice_name + '-labels_' + str(spect_id) + ".npy"), chopped_labels)

            # Incrememt indeces
            start_idx += window_size
            spect_id += 1

        print ("#############################")
        print ("####### Done Chopping #######")
        print ("#############################")





# ---------------- Main -------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Spectrogram creation."
                                     )

    parser.add_argument('-l', '--logfile',
                        help='fully qualified log file name to which info and error messages \n' +\
                             'are directed. Default: stdout.',
                        default=None);
    parser.add_argument('-o', '--outdir',
                        help='directory for outfiles; default /tmp',
                        default='/tmp');
    parser.add_argument('--num_workers',
                        type=int,
                        default=0,
                        help='total number of workers started via gnu parallel'
                        );
    parser.add_argument('--this_worker',
                        type=int,
                        default=0,
                        help="this worker's rank within num_workers started via gnu parallel"
                        );
    parser.add_argument('infiles',
                        nargs='+',
                        help='Repeatable: spectrogram input files and directories')

    args = parser.parse_args();
    
    SpectrogramChopper(args.infiles,
        snippet_outdir=args.outdir,
        num_workers=args.num_workers,
        this_worker=args.this_worker,
        logfile=args.logfile)
    
