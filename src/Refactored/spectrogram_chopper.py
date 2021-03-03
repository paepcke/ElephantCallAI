#!/usr/bin/env python
'''
Created on Feb. 11, 2020

@author: Jonathan Gomes Selman
'''
import argparse
import math
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from spectrogram_dataset import SpectrogramDataset
from DSP.dsp_utils import AudioType, FileFamily
from elephant_utils.logging_service import LoggingService


"""
    What I want for this class!
    I want to simply pass in a list of files that need to be chopped 
    and this should do the job for me!!! Should we try to multi-process this?
    Let us just do the straight forward non-multi process for now
"""

class SpectrogramChopper(object):
    '''
    Given a list of spectrograms, chop the spectrograms into 
    the specified window size! For now avoid doing any 
    parallelization to just get this done functionality wise
    '''

    #------------------------------------
    # Constructor
    #-------------------

    def __init__(self,
                 infiles,
                 snippet_outdir,
                 logfile=None,
                 test_snippet_width=-1,
                 ):
        '''
        
        @param infiles: list of spectrogram files, where the label files
        are expected to also be in the same directory
        @type infiles: {str|[str]}
        @param snippet_outdir: where snippets are to be placed.
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
            
        if type(infiles) != list:
            infiles = [infiles]
            
        # Again we may just ditch this
        self.log = LoggingService(logfile=logfile,
                                  msg_identifier=f"chop_spectrograms#{this_worker}")
        
        # Make sure the full path to the 
        # outdir exists:
        if snippet_outdir is None:
            raise ValueError("Snippet destination directory must be provided; was None")
        # If snippet_outdir's path does not exist
        # yet, create all dirs along the path:
        if not os.path.exists(snippet_outdir):
            os.makedirs(snippet_outdir)
            
        
        """
            We want to step through the files provided and chop chop chop
        """
        for spect in infiles
            # Create the file family so that we can get the spectrogram
            # and the label file
            file_family = FileFamily(spect)

            self.chop_spectrogram(spect, file_family.fullpath(AudioType.MASK) )


    #------------------------------------
    # sqlite_name_by_worker 
    #-------------------
    
    @classmethod
    def sqlite_name_by_worker(cls, worker_rank):
        '''
        Given a worker rank, create a unique 
        file name that will be the sqlite db file
        used by the respective worker. 
        
        Method is class level so outsiders can 
        find the sqlite files
        
        
        @param worker_rank: the rank of a worker
            in the list of all workers that chop
        @type worker_rank: int
        @return: file name (w/o dirs) of sqlite file that
            will only be used by given worker.
        @rtype str
        '''
        
        sqlite_name = f"snippet_db_{worker_rank}.sqlite"
        return sqlite_name

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
        not divisible by num_workers, the last worker processes
        the left-overs in addition to their regular share.
        
        @param file_families: file info about each infile
        @type file_families: [FileFamily]
        @param num_workers: total number of workers deployed
        @type num_workers: int
        @param this_worker: rank of this worker in the list of workers
        @type this_worker: int
        @return the file family instances for this worker to process
        @rtype [FileFamily]
        '''
        
        # Note: in case of only one file to do, with more
        # than one worker, the last worker will pick
        # up the file in the 'Do I need to take leftovers'
        # below:
        batch_size = math.floor(len(file_families) / num_workers)
        my_share_start = this_worker * batch_size
        my_share_end   = my_share_start + batch_size
        
        # Do I need to take leftovers?
        if this_worker == num_workers - 1:
            # Yes:
            left_overs = len(file_families) % num_workers
            my_share_end += left_overs
        
        my_families = file_families[my_share_start:my_share_end]

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
    
