#!/usr/bin/env python
'''
Created on March 4, 2020

@author: Jonathan Gomes Selman
'''

""" 

This file is going to be a general data processing utils file.
We will have two classes, one that is responsible for gathering
the files to be passed to the spectrogrammer and the second
class will deal with spectrogram chopping

"""

import argparse
import os
import sys

from spectrogrammer import Spectrogrammer
from spectrogram_chopper import SpectrogramChopper
from data_utils import FileFamily
from data_utils import AudioType

class Spectrogram_Generator(object):
    """
        Responsible for gathering and properly passing the files
        that we want to be processed by the 'Spectrogrammer' class.
        Allows for several different methods of input data:
            1) A directory from which we want to process all of the files
            of a given type

            2) A text file containing newline seperated names of the 
            files that we want to process
    """

    def __init__(self, data, actions, outdir, spect_parameters, data_path=None):
        '''
        @param data: Either a data directory with the '.wav'/'.txt' files 
            or a file with the '.wav'/'.txt' files that we want to process
        @type data: str
        @param actions: the tasks to accomplish: 
            {spectro|melspectro|labelmask|copyraven}
            NOTE: copy raven is used simply to copy over the raven gt label .txt
            file if we are moving the spectrogram to a new location
        @type actions: [str] 
        @param outdir: The directory where the processed files are written
        @type outdir str
        @param spect_parameters: A dictionary containing the spectrogram parameters
            needed for Spectrogrammer!
        @type spect_parameters: dict
        @param data_path: Only used if data is a text file! This specifies the
            path to where the processed spect files exist
        @type data_path: str
        '''
        super(Spectrogram_Generator, self).__init__()
        
        # Get a list of the data files that we want to process. 
        # These will be passed on to the spectrogrammer class
        if os.path.isfile(data):
            # Make sure data_path is not None
            if data_path is None:
                raise ValueError("Since we are reading from a file of spects to process, data_path cannot be None")

            data_file_paths = self.extract_from_file(data, data_path)
        else:
            data_file_paths = self.extract_from_dir(data)

        # Process the files
        Spectrogrammer(data_file_paths,
                   actions,
                   outdir=outdir,
                   normalize=spect_parameters['normalize'],
                   to_db=spect_parameters['to_db'],
                   min_freq=spect_parameters['min_freq'],
                   max_freq=spect_parameters['max_freq'],
                   framerate=spect_parameters['framerate'],
                   nfft=spect_parameters['nfft'],
                   pad_to=spect_parameters['pad_to'],
                   hop=spect_parameters['hop']
                   )



    def extract_from_file(self, data_file, data_path):
        """
            Given a data file, extract a list of data files to be processed.
            For simplicity sake, assume that the file
            contains exactly the files we want to process, rather than
            just the general spect ids. 

            NOTE: The spectrogram files in the data_file are ONLY file 
            names. Thus, when we extract them we add the 'data_path'
            so that they can be read!

            @return list of .wav/.txt files
        """
        with open(data_file, "r") as f:
            lines = f.readlines()

        data_paths = [os.path.join(data_path, line) for line in lines]
        return data_paths

    def extract_from_dir(self, data_dir):
        """
            Given a data directory, extract a list of the data files to be processed.

            NOTE: Make sure we extract the full file paths so that they can be read!

            @return list of .wav/.txt files
        """
        data_paths = []
        for(dirpath, dirnames, filenames) in os.walk(data_dir):
            for file in filenames:
                # Make sure that we don't get garbage files
                file_type = file.split('.')[1]
                if (file_type not in ['wav', 'txt']):
                    continue

                data_paths.append(os.path.join(dirpath, file))

        return data_paths

"""
    We now want to make a class that handles the spectrogram chopping! This comes after the 
    spect generation!!!
"""
class LaunchChopper(object):
    """
        Responsible for launching the chopping of the generated spectrograms. The 
        purpose of this class is to collect the files that need to be chopped 
        and then chop them!

        Expects input data either from a directory containing the processed spect
        and mask files or a file containing a list of the spect names to be chopped. 

        The output file is to be likely a Train or Test dir!
    """
    def __init__(self, data, output_dir, window_size=256, data_path=None):
        '''
        @param data: Either a data directory with the processed spectrogram files and
            corresponding mask files or a file with the '_spectro.npy' files that we 
            want to chop
        @type data: str
        @param output_dir: The directory where the chopped files are written. 
        @type output_dir str
        @param window_size: Size to make the chops!
        @type window_size: int
        @param data_path: Only used if data is a text file! This specifies the
            path to where the processed spect files exist
        @type data_path: str
        '''
        super(LaunchChopper, self).__init__()
        
        # Get a list of the (spect, mask) tuples that we want to chop! 
        # These will be passed on to the spectrogrammer_chopper class
        if os.path.isfile(data):
            # Make sure data_path is not None
            if data_path is None:
                raise ValueError("Since we are reading from a file of spects to process, data_path cannot be None")

            spect_file_pairs = self.extract_from_file(data, data_path)
        else:
            spect_file_pairs = self.extract_from_dir(data)

        for spect_file, mask_file in spect_file_pairs:
            SpectrogramChopper(spect_file,
                                mask_file,
                                output_dir,
                                window_size=window_size,
                                )


    def extract_from_file(self, data_file, data_path):
        """
            Given a data file, extract a list of (spect, mask) file
            tuples that are to be chopped. We assume that the file
            contains new line seperated '_spectro.npy' file names

            NOTE: The spectrogram files in the data_file are ONLY file 
            names. Thus, when we extract them we add the 'data_path'
            so that they can be read!

            @return list of ('_spectro.npy', '_label_mask.npy') tuples
        """
        with open(data_file, "r") as f:
            lines = f.readlines()

        # Add the complete path to each file name so that we can
        # properly load them into the CHOPPER
        data_paths = [os.path.join(data_path, line) for line in lines]

        spect_file_pairs = []
        for path in data_paths:
            # Use the file family to easily get the names of the different
            # file types!
            file_family = FileFamily(path)

            # Append tuple with (spect_file, label_mask_file)
            spect_file_pairs.append((file_family.fullpath(AudioType.SPECTRO), file_family.fullpath(AudioType.MASK)))

        return spect_file_pairs

    def extract_from_dir(self, data_dir):
        """
            Given a data directory, extract a list of (spect, mask) file
            tuples that are to be chopped.

            NOTE: Make sure we extract the full file paths so that they can be read!

            @return list of ('_spectro.npy', '_label_mask.npy') tuples
        """
        spect_file_pairs = []
        for(dirpath, dirnames, filenames) in os.walk(data_dir):
            for file in filenames:
                # Remember to add the directory!
                full_file = os.path.join(dirpath, file)
                # Apply the file family collection on the spectro files
                if file.endswith('_spectro.npy'):
                    file_family = FileFamily(full_file)

                    # Append tuple with (spect_file, label_mask_file)
                    spect_file_pairs.append((full_file, file_family.fullpath(AudioType.MASK)))

        return spect_file_pairs

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Spectrogram creation"
                                     )

    # Parameters for Spectrogram_Generator
    parser.add_argument('--process_spects', 
                        action='store_true', 
                        help='Flag indicating that we want to process the raw data') 

    parser.add_argument('--raw_data', 
                        type=str, 
                        help='Either a folder or file containing the raw data files we want to process') 

    parser.add_argument('--actions',
                        nargs='+',
                        choices=['spectro', 'melspectro','labelmask', 'copyraven'],
                        help="Which tasks to accomplish (repeatable)"
                        )

    parser.add_argument('--spect_outdir', 
                        default=None, 
                        help='Output directory for any processed spect files'
                        )

    parser.add_argument('--raw_data_path', 
                        default=None, 
                        help='If "raw_data" is a file, then we need this to specify the data path to append to these files'
                        )

    # Spect parameters
    parser.add_argument('--nfft', 
                        type=int, 
                        default=Spectrogrammer.NFFT, 
                        help='Window size used for creating spectrograms') 

    parser.add_argument('--hop', 
                        type=int, 
                        default=Spectrogrammer.HOP_LENGTH, 
                        help='Hop size used for creating spectrograms')

    parser.add_argument('--window', 
                        type=int, 
                        default=256, 
                        help='Deterimes the window size in frames ' +
                             'of the resulting spectrogram') 
                        # Default corresponds to 21s

    parser.add_argument('--pad', 
                        dest='pad_to', 
                        type=int, 
                        default=4096, 
                        help='Deterimes the padded window size that we ' +
                             'want to give a particular grid spacing (i.e. 1.95hz')

    parser.add_argument('-f', '--framerate',
                        type=int,
                        default=8000,
                        help='Framerate at which original .wav file was recorded; def: 8000')


    parser.add_argument('--max_freq', 
                    type=int, 
                    default=Spectrogrammer.MAX_FREQ,
                    help='Max frequency of leading bandpass filter')

    parser.add_argument('--min_freq', 
                    type=int, 
                    default=Spectrogrammer.MIN_FREQ,
                    help='Min frequency of leading bandpass filter')


    parser.add_argument('-n', '--normalize', 
                        default=False,
                        action='store_true', 
                        help='do not normalize to fill 16-bit -32K to +32K dynamic range'
                        )

    parser.add_argument('--to_db', 
                        action='store_true', 
                        default=False, 
                        help='Convert to log scale')


    # LaunchChopper arguments
    parser.add_argument('--chop_spects', 
                        action='store_true', 
                        help='Flag indicating that we want to chop the spectrograms!') 

    parser.add_argument('--spect_data', 
                        type=str, 
                        help='Either a folder or file containing the processed spects we want to chop. Note,' \
                        + 'if we run "process_spects" also we dont need to specify this!') 

    parser.add_argument('--chopped_outdir', 
                        default=None, 
                        help='Output directory for the chopped spect files. Usually either Train/Test'
                        )

    parser.add_argument('--spect_data_path', 
                        default=None, 
                        help='If "spect_data" is a file, then we need this to specify the data path to append to these files'
                        )


    args = parser.parse_args();

    # Do the logic of processing and chopping the data
    if args.process_spects:
        # First collect the spectrogram parameters
        spect_parameters = {'normalize': args.normalize,
                            'to_db': args.to_db,
                            'min_freq': args.min_freq,
                            'max_freq': args.max_freq,
                            'framerate': args.framerate,
                            'nfft': args.nfft,
                            'pad_to': args.pad_to,
                            'hop': args.hop
                            }

        Spectrogram_Generator(args.raw_data, args.actions, args.spect_outdir, spect_parameters, data_path=args.raw_data_path)

    if args.chop_spects:
        # Chop chop chop
        # If we have just processed the data then just use spect_outdir as input!
        # Remember we need to add in the subdirectory created!??
        if args.process_spects:
            LaunchChopper(args.spect_outdir, args.chopped_outdir, window_size=args.window, data_path=args.spect_data_path)
        else:
            LaunchChopper(args.spect_data, args.chopped_outdir, window_size=args.window, data_path=args.spect_data_path)
        

        