#!/usr/bin/env python
'''
Created on Jul 28, 2020

@author: paepcke
'''
import argparse
from datetime import datetime
import math
import os
import sqlite3
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from spectrogram_dataset import SpectrogramDataset
from DSP.dsp_utils import AudioType, FileFamily
from elephant_utils.logging_service import LoggingService

class SpectrogramChopper(object):
    '''
    Initiate chopping of 24-hr spectrograms.
    This class may be instantiated multiple
    times to distribute the work of chopping.
    When this file is executed an one instance
    is thereby created, the total number of
    workers is passed in, and a 'rank' among
    these workers is assigned to this particular
    instance. 
    
    Each instance selects a subset of all spectrograms
    to chop such that no other worker will work
    on that chosen subset. This decentralized 
    partitioning is modeled on pytorch's 
    DistributedDataParallel.
    
    The best method for invoking this file multiple
    times is to use launch_chopping.sh script.
    '''

    #------------------------------------
    # Constructor
    #-------------------

    def __init__(self,
                 infiles,
                 sqlite_db_path=None,
                 recurse=False,
                 snippet_outdir=None,
                 num_workers=0,
                 this_worker=0,
                 logfile=None
                 ):
        '''
        Constructor
        '''
            
        if type(infiles) != list:
            infiles = [infiles]
            
        self.log = LoggingService(logfile=logfile,
                                  msg_identifier=f"chop_spectrograms#{this_worker}")
        
        # Make sure the full path to the 
        # outdir exists:
        if not os.path.exists(snippet_outdir):
            os.makedirs(snippet_outdir)
            
        # Same with sqlite_db_path:
        if sqlite_db_path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H_%M_%S")
            sqlite_db_path = os.path.join(os.path.dirname(__file__),
                                          f"spectrogram_snippets_{timestamp}.sqlite"
                                          )
        else:
            if os.path.exists(sqlite_db_path):
                # Ensure the specified file is 
                # an sqlite db:
                try:
                    conn = sqlite3.connect(sqlite_db_path)
                    # All good:
                    conn.close()
                except Exception as e:
                    self.log.err(f"Could not open specified Sqlite db '{sqlite_db_path}': {repr(e)}")
                    sys.exit(1)

            else:
                # Path to a not-yet existing sqlite file was given.
                # If that path exists, use it; else
                # ensure that the intermediate dirs
                # exist, and a fresh sqlite db will
                # be created later in the specified
                # dir, with the specified name:

                os.makedirs(os.path.dirname(sqlite_db_path))

        # Get a list of FileFamily instances. The
        # list includes all recursively found files:
        file_families = self.get_files_todo_info(infiles)

        # If this instance is one of several gnu parallel
        # workers, find which of the files this worker
        # is supposed to do:
        
        if num_workers > 0:
            my_file_families = self.select_my_infiles(file_families, num_workers, this_worker)
        else:
            my_file_families = file_families

        my_spectro_files = [family for family in my_file_families 
                                 if family.file_type == AudioType.SPECTRO]
        num_spectro_files = len(my_spectro_files)

        self.log.info(f"Todo: {num_spectro_files} 24-hr spectrogram files")

        self.dataset = SpectrogramDataset(
                         dirs_of_spect_files=infiles,
                         sqlite_db_path=sqlite_db_path,
                         recurse=recurse,
                         snippet_outdir=snippet_outdir,
                         )

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
        Input: a possibly mixed list of .pickle, .txt, files,
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
            if not os.path.exists(file_todos[i].fullpath(AudioType.SPECTRO)):
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
            if  filename.endswith('.pickle') and not os.path.exists(filename):
                self.log.warn(f"Spectrogram '{filename}' does not exist; skipping it.")
                continue 
            
            # Make a family instance:
            try:
                file_family = FileFamily(file_or_dir)
            except ValueError:
                self.log.warn(f"Unrecognized filename convention: {file_or_dir}; skipping")
                continue

            # Only keep audio and label files:    
            if file_family.file_type not in [AudioType.SPECTRO, AudioType.LABEL]:
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
                        help='Repeatable: spectrogram input files and directories')

    args = parser.parse_args();
    
    SpectrogramChopper(args.infiles,
        snippet_outdir=args.outdir,
        limit=args.limit,
        num_workers=args.num_workers,
        this_worker=args.this_worker,
        logfile=args.logfile)
    
