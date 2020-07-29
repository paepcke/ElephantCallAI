#!/usr/bin/env python
'''
Created on May 4, 2020

Given a list of elephant recording .wav files, 
batch-creates noise gated elephant files, associated
24-hr spectrograms, and mask files (if requested).

Results are placed into a command line specified
destination dir. On request, associated text files
are copied to that destination as well. 

Input paths may be a mix of .wav files, and directories.
All .wav files in the subdirs are recursively included.

@author: paepcke
'''
import os
import shutil
import sys
import math

import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dsp_utils import AudioType
from dsp_utils import FileFamily
from elephant_utils.logging_service import LoggingService
from spectrogrammer import Spectrogrammer

class WavMaker(object):
    '''
    classdocs
    '''

    #------------------------------------
    # Constructor
    #-------------------

    def __init__(self, 
                 infiles,
                 outdir='/tmp/',
                 normalize=True,
                 threshold_db=-40,
                 copy_label_files=True,
                 low_freq=10,
                 high_freq=50,
                 spectrogram_freq_cap=150, # Hz
                 limit=None,
                 num_workers=0,
                 this_worker=0,
                 logfile=None
                 ):
        '''
        Constructor
        '''
        # Make sure the full path to the 
        # outdir exists:
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        self.log = LoggingService(logfile=logfile,
                                  msg_identifier=f"wave_maker#{this_worker}")


        # Get a list of FileFamily instances. The
        # list includes all recursively found files:
        file_families = self.get_files_todo_info(infiles)
        num_wav_files = len(file_families)
        
        # If this instance is one of several gnu parallel
        # workers, find which of the files this worker
        # is supposed to do:
        
        if num_workers > 0:
            my_file_families = self.select_my_infiles(file_families, num_workers, this_worker)
        else:
            my_file_families = file_families
        
        if copy_label_files:
            # We are to copy txt files where available:
            num_lab = len([True for file_family in my_file_families \
                                if os.path.exists(file_family.fullpath(AudioType.LABEL))
                                ])
            self.log.debug(f"Todo: {num_wav_files} .wav files; copy {num_lab} label files.")
            if num_wav_files > num_lab:
                self.log.debug(f"Missing label file(s) for {num_wav_files - num_lab} file(s).")
        else:
            self.log.info(f"Todo: {num_wav_files} .wav files")

        files_done = 0
        for file_family in my_file_families:
            
            # Reconstruct the full file path
            infile = file_family.fullpath(file_family.file_type)
            if not os.path.isfile(infile):
                # At this point we should only be seeing existing .wav files:
                raise IOError(f"File {infile} does not exist, but should.")
            
            try:
                if file_family.file_type == AudioType.WAV:
                    actions = ['spectro', 'cleanspectro']
                elif file_family.file_type == AudioType.LABEL:
                    actions = ['labelmask']
                    if copy_label_files:
                        try:
                            full_label_path = file_family.fullpath(AudioType.LABEL)
                            shutil.copy(full_label_path, outdir)
                        except Exception as e:
                            self.log.err(f"Could not copy label file {full_label_path} to {outdir}: {repr(e)}")
                            continue
                else:
                    continue
                
                self.log.info(f"Submitting {actions} request on '{infile}' to spectrogrammer...")
                Spectrogrammer(infile,
                               actions,
                               outdir=outdir,
                               normalize=normalize,
                               low_freq=low_freq,
                               high_freq=high_freq,
                               threshold_db=threshold_db,
                               spectrogram_freq_cap=spectrogram_freq_cap
                               )
                self.log.info(f"Spectrogrammer finished; done with {infile}...")

            except Exception as e:
                self.log.err(f"Processing failed for '{infile}: {repr(e)}")
                continue

            files_done += 1
            if limit is not None and (files_done >= limit):
                self.log.info(f"Completed {files_done}, completing the limit of {limit}")
                break
            if files_done > 0 and (files_done % 10) == 0:
                if limit is not None:
                    self.log.info(f"\nBatch gated {files_done} of {limit} wav files.")
                else:
                    self.log.info(f"\nBatch gated {files_done} of {len(my_file_families)} wav files.")


    #------------------------------------
    # select_my_infiles
    #-------------------

    def select_my_infiles(self, file_families, num_workers, this_worker):
        '''
        Used when gnu parallel starts multiple copies of this
        file. Each resulting worker must select a batch of
        infiles. This selection is accomplished via the 
        num_workers and this_worker quantities. The passed-in
        file_families comprise all infiles. Each worker 
        partitions theses into num_worker batches. Worker with
        this_worker == 0 works on the first batch. Worker 1
        on the second, and so on. If the number of infiles is
        not divisible by num_workers, the last worker process
        the left-overs in addition to their regular share.
        
        @param file_families: file info about each infile
        @type file_families: [FileFamily]
        @param num_workers: total number of workers deployed
        @type num_workers: int
        @param this_worker: rank of this worker in the list of workers
        @type this_worker: int
        '''
        
        batch_size = math.floor(len(file_families) / num_workers)
        my_share_start = this_worker * batch_size
        my_share_end   = my_share_start + batch_size
        
        # Do I need to take leftovers?
        if this_worker == num_workers - 1:
            # Yes:
            left_overs = len(file_families) % num_workers
            my_share_end += left_overs
        
        my_families = file_families[my_share_start:my_share_end]
        # Get the .wav files to the front of the 
        # to-do list so that spectrograms are created
        # before 
         
        return my_families

    #------------------------------------
    # get_files_todo_info 
    #-------------------

    def get_files_todo_info(self, infiles):
        '''
        Input: a possibly mixed list of .wav, .txt, files,
        and directories. Returns a list of FileFamily instances

        Where each FileFamily instance is guaranteed to have both, 
        a .wav and .txt entry. While existence of the .wav file
        is verified, the .txt file may not exist.
        
        @param infiles: list of files/directories
        @type infiles: [str]
        @return: list of file information FileFamily instances
        @rtype: [FileFamily]
        '''
        # Collect .wav audio, and .txt label files.
        # Receives instaces of FileFamily. For each
        # family is is guaranteed the .wav version (foo.wav)
        # exists. None of the others, such as the label file
        # variant (foo.txt) need exist:
        file_todos = self.collect_file_families(infiles)
        
        # Remove entries that only have a label file:
        # Strategy: search the list of file families from end to front,
        # so removal does not change positions of yet-to-check
        # entries:
        indices_backw = range(len(file_todos)-1, -1, -1)
        for i in indices_backw:
            if not os.path.exists(file_todos[i].fullpath(AudioType.WAV)):
                del file_todos[i]
        
        return file_todos

    #------------------------------------
    # collect_file_families 
    #-------------------

    def collect_file_families(self, files_or_dirs, file_family_list=None):
        '''
        Recursive.
        Takes a possibly mixed list of files and directories.
        Returns a list of FileFamily instances. Workhorse
        for get_files_todo_info.
            
        ******???Note that either the 'wav' or the 'txt' entry
        ******???may be missing if they were not in the file tree.
        
        Walks down any directory.
        
        @param files_or_dirs: list of files and subdirectories
        @type files_or_dirs: [str]
        @param file_family_list: dict of info about each recursively found file
            only .wav and .txt files are included
        @type file_family_list: {str : str}
        '''
        if file_family_list is None:
            file_family_list = []
            
        for file_or_dir in files_or_dirs:
            if os.path.isdir(file_or_dir):
                new_paths = [os.path.join(file_or_dir, file_name) for file_name in os.listdir(file_or_dir)]
                # Ohhh! Recursion!:
                file_family_list = self.collect_file_families(new_paths, file_family_list)
                continue
            
            # A file (not a directory)
            filename = file_or_dir
            # A file name; is it a .wav file that does not exist?
            if  filename.endswith('.wav') and not os.path.exists(filename):
                self.log.warn(f"Audio file '{filename}' does not exist; skipping it.")
                continue 
            
            # Make a family instance:
            try:
                file_family = FileFamily(file_or_dir)
            except ValueError:
                self.log.warn(f"Unrecognized filename convention: {file_or_dir}; skipping")
                continue

            # Only keep audio and label files:    
            if file_family.file_type not in [AudioType.WAV, AudioType.LABEL]:
                continue

            file_family_list.append(file_family)    

        return(file_family_list)

# ---------------- Main -------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Create noise gated wav files from input wav files."
                                     )

    parser.add_argument('-l', '--logfile',
                        help='fully qualified log file name to which info and error messages \n' +\
                             'are directed. Default: stdout.',
                        default=None);
    parser.add_argument('-o', '--outdir',
                        help='directory for outfiles; default /tmp',
                        default='/tmp');
    parser.add_argument('-n', '--no_normalize',
                        action='store_true',
                        default=False,
                        help='do not normalize wave form to fill int16 range',
                        );
    parser.add_argument('-t', '--threshold_db',
                        type=int,
                        default=-40,
                        help='dB off peak voltage below which signal is zeroed; default -40',
                        );
    parser.add_argument('-m', '--low_freq',
                        type=int,
                        default=10,
                        help='low end of front end bandpass filter; default 10Hz'
                        );
    parser.add_argument('-i', '--high_freq',
                        type=int,
                        default=50,
                        help='high end of front end bandpass filter; default 50Hz'
                        );
    parser.add_argument('-s', '--freq_cap',
                        type=int,
                        default=150,
                        help='highest frequencies to keep in the spectrograms; default 150Hz'
                        );
    parser.add_argument('-x', '--limit',
                        type=int,
                        default=None,
                        help='maximum number of wav files to process; default: all'
                        );
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
                        help='Repeatable: .wav input files and directories')

    args = parser.parse_args();
    
    if args.threshold_db > 0:
        print("Threshold dB value should be less than 0.")
        sys.exit(1)
        
    WavMaker(args.infiles,
             outdir=args.outdir,
             # If no_normalize is True, don't normalize:
             normalize=not args.no_normalize,
             low_freq=args.low_freq,
             high_freq=args.high_freq,
             threshold_db=args.threshold_db,
             spectrogram_freq_cap=args.freq_cap,
             limit=args.limit,
             num_workers=args.num_workers,
             this_worker=args.this_worker,
             logfile=args.logfile
             )