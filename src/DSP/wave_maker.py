#!/usr/bin/env python
'''
Created on May 4, 2020

Batch-creates noise gated elephant files.
Results are placed into a command line specified
destination dir. On request, associated text files
are copied to that destination as well. 

Input paths may be a mix of .wav files, and directories.
All .wav files in the subdirs are recursively included.

@author: paepcke
'''
import argparse
import os
import shutil
import sys
import time

from scipy.io import wavfile
import torch
import torchaudio

from amplitude_gating import AmplitudeGater
from elephant_utils.logging_service import LoggingService


sys.path.append(os.path.join(os.path.dirname(__file__), '.'))


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
                 threshold_db=-46,
                 copy_label_files=True,
                 low_freq=10,
                 high_freq=50,
                 spectrogram_freq_cap=150, # Hz
                 spectrogram_dest=None,
                 limit=None,
                 logfile=None
                 ):
        '''
        Constructor
        '''
        # Make sure the full path to the 
        # outdir exists:
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        self.log = LoggingService(logfile=logfile)

        
        files_info_todo = self.get_files_todo_info(infiles)
        num_wav_files = len(files_info_todo)
        
        # Show todo info:
        if copy_label_files:
            # We are to copy txt files where available:
            num_lab = len([True for info_dict in files_info_todo \
                                if 'txt' in info_dict.keys()])
            self.log.info(f"Todo: {num_wav_files} .wav files; copy {num_lab} label files.")
            if num_wav_files > num_lab:
                self.log.warn(f"Missing label files for {num_wav_files - num_lab} files.")
        else:
            self.log.info(f"Todo: {num_wav_files} .wav files")

        files_done = 0
        #*************
        transformer = torchaudio.transforms.Spectrogram(n_fft=4096) 
        #*************
        for file_info in files_info_todo:
            
            # Reconstruct the full .wav file path
            infile = os.path.join(file_info['dir'], file_info['wav'])
            if not os.path.isfile(infile):
                # At this point we should only be seeing existing .wav files:
                self.log.warn(f"File {infile} does not exist or is a directory.")
                continue

            #************
            start_time = time.time()
            sig = (framerate, sig) = wavfile.read(infile)
            spect_t = transformer(torch.Tensor(sig))
            print(f"{time.time()-start_time} secs")
            continue
            #************
            self.log.info(f"Processing {infile}...")
            try:
                gater = AmplitudeGater(infile,
                                       amplitude_cutoff=threshold_db,
                                       low_freq=low_freq,
                                       high_freq=high_freq,
                                       spectrogram_freq_cap=spectrogram_freq_cap,
                                       spectrogram_dest=spectrogram_dest,
                                       outdir=outdir
                                       )
            except Exception as e:
                self.log.err(f"Processing failed for '{infile}: {repr(e)}")
                continue
            perc_zeroed = gater.percent_zeroed
            self.log.info(f"Done processing {os.path.basename(infile)}; removed {round(perc_zeroed)} percent")
            if copy_label_files:
                try:
                    full_label_path = os.path.join(file_info['outdir'],
                                                   file_info['txt'])
                    shutil.copy(full_label_path, outdir)
                except KeyError:
                    # Case of .wav without a .txt
                    # We already warned above:
                    pass
            files_done += 1
            if limit is not None and (files_done >= limit):
                self.log.info(f"Completed {files_done}, completing the limit of {limit}")
                break
            if files_done > 0 and (files_done % 10) == 0:
                if limit is not None:
                    self.log.info(f"\nBatch gated {files_done} of {limit} wav files.")
                else:
                    self.log.info(f"\nBatch gated {files_done} of {len(files_info_todo)} wav files.")

    #------------------------------------
    # get_files_todo_info 
    #-------------------

    def get_files_todo_info(self, infiles):
        '''
        Input: a possibly mixed list of .wav, .txt, files,
        and directories. Returns a list of dicts:

            o id:  the location and date of a .wav file 
                    parsed from its filename
            o dir: the directory of the file
            o 'wav': the basename of the .wav file
            o 'txt': the basename of the corresponding .txt label file
        
        Where each dict is guaranteed to have both, a .wav and .txt
        entry.
        
        @param infiles: list of files/directories
        @type infiles: [str]
        @return: list of dicts with file info
        @rtype: [{str : str}]
        '''
        # Collect file info dicts:
        file_info_todo = list(self.build_data_pairs(infiles).values())
        # Remove entries that only have a label file:
        # Strategy: search the list of dicts from end to front,
        # so removal does not change positions of yet-to-check
        # entries:
        indices_backw = range(len(file_info_todo)-1, -1, -1)
        for i in indices_backw:
            try:
                file_info_todo[i]['wav']
            except KeyError:
                del file_info_todo[i]
        
        return file_info_todo

    #------------------------------------
    # build_data_pairs 
    #-------------------

    def build_data_pairs(self, files_or_dirs, data_pairs=None):
        '''
        Takes a possibly mixed list of files and directories.
        Returns a dict with keys:
        
            o id:  the location and date of a .wav file 
                    parsed from its filename
            o dir: the directory of the file
            o 'wav': the basename of the .wav file
            o 'txt': the basename of the corresponding .txt label file
            
        Note that either the 'wav' or the 'txt' entry
        may be missing if they were not in the file tree.
        
        Walks down any directory.
        
        @param files_or_dirs: list of files and subdirectories
        @type files_or_dirs: [str]
        @param data_pairs: dict of info about each recursively found file
            only .wav and .txt files are included
        @type data_pairs: {str : str}
        '''
        if data_pairs is None:
            data_pairs = {}
            
        for file_or_dir in files_or_dirs:
            if os.path.isdir(file_or_dir):
                new_paths = [os.path.join(file_or_dir, file_name) for file_name in os.listdir(file_or_dir)]
                # Ohhh! Recursion!:
                data_pairs = self.build_data_pairs(new_paths, data_pairs)
                continue
            # Strip off the location and time tags
            tags = os.path.basename(file_or_dir).split('_')
            
            # Guard against file names without an underscore:
            if len(tags) < 2:
                self.log.err(f"Unrecognized filename convention: {file_or_dir}; skipping")
                continue
            
            data_id = tags[0] + '_' + tags[1]
            (_file_stem, file_type) = file_or_dir.split('.')
    
            if (file_type not in ['wav', 'txt']):
                continue
    
            # Insert the file name into the dictionary
            # with the file type tag for a given id
            if not data_id in data_pairs:
                data_pairs[data_id] = {}
                data_pairs[data_id]['id'] = data_id
                data_pairs[data_id]['dir'] = os.path.dirname(file_or_dir)
    
            data_pairs[data_id][file_type] = os.path.basename(file_or_dir)
        return(data_pairs)

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
    parser.add_argument('-t', '--threshold',
                        type=int,
                        default=-46,
                        help='dB off peak voltage below which signal is zeroed; default -46',
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
    parser.add_argument('-s', '--spectrofreqcap',
                        type=int,
                        default=150,
                        help='highest frequencies to keep in the spectrograms; default 150Hz'
                        );
    parser.add_argument('-d', '--spectrodest',
                        help='fully qualified log file name to dir where spectrograms will be placed. Default: /tmp',
                        default='/tmp');
    parser.add_argument('-x', '--limit',
                        type=int,
                        default=None,
                        help='maximum number of wav files to process; default: all'
                        );
    parser.add_argument('infiles',
                        nargs='+',
                        help='Repeatable: .wav input files')

    args = parser.parse_args();
    
    if args.threshold > 0:
        print("Threshold dB value should be less than 0.")
        sys.exit(1)

    WavMaker(args.infiles,
             outdir=args.outdir,
             threshold_db=args.threshold,
             spectrogram_freq_cap=args.spectrofreqcap,
             spectrogram_dest=args.spectrodest,
             limit=args.limit,
             logfile=args.logfile
             )