#!/usr/bin/env python
'''
Created on Feb, 2021

@author: Jonathan
'''
import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path

# Try it with our ml specgram function but see if fails
from scipy.io import wavfile
from scipy.signal.spectral import stft

from scipy.signal import spectrogram
from matplotlib import mlab as ml


import torch
#from torchaudio import transforms
#import torchaudio

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# See which of these we actually need!
from data_utils import FileFamily
from data_utils import DATAUtils
from data_utils import AudioType


# We can do better logging later!
# See if this works!
from logging_service import LoggingService
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#from plotting.plotter import Plotter

class Spectrogrammer(object):
    '''
    Create and manipulate spectrograms from a given folder of spectrogram
    files and (potentially) .txt label filies. With a possible set of input
    actions can generate spectrograms from .wav files, generate corresponding
    spectrogram 0/1 labelings, generate spectrogram time-slicings, and
    copy over gt_labeling
    '''

    # Default specifications
    NFFT = 4096 # was 3208 # We want a frequency resolution of 1.95 Hz
    # Used by the elephant people but not us! If NFFT is less than 
    # Pad_to does something special!
    PAD_TO = NFFT
    
    OVERLAP = 1/2    
    HOP_LENGTH = 800 # Want second resolution of 0.1 seconds
    # Maximum frequency retained in spectrograms 
    MAX_FREQ = 150 # This we can honestly consider decreasing. But let us leave it for now!
    MIN_FREQ = 0
    # Primarily taken from reading in a .wav file!
    DEFAULT_FRAMERATE = 8000
    # We should not be using these!!


    #------------------------------------
    # Constructor 
    #-------------------

    """
        Active thoughts: It would be nice to have functionality for dealing with a whole folder,
        a list of files, etc. However, this should not be the job of the spectrogramer. I like
        the idea of being passed a list of the files that we want to do stuff too! The question is
        how should this be formatted?? This is my thinking!! We want to allow several options
            - Create just spectrograms
            - create just labelmasks
            - create spectrograms and label masks
            - copy just the raven file.
        For this need flexability in accepting a collection of .wav and .txt files. Here is what 
        we shall do! Let us a list of generic files like Andreas does and then do as we want!
    """

    def __init__(self, 
                 infiles,
                 actions,
                 outdir=None,       # if None Output to same dir as files are located
                 normalize=False,   # We may want to consider using this!
                 to_db=False,
                 min_freq=0,        # Hz 
                 max_freq=150,      # Hz
                 nfft=4096,
                 pad_to=4096,
                 hop=800,
                 framerate=8000,
                 logfile=None,
                 ):
        '''
        @param infiles: List of files to process (containing potentially .wav and .txt)
        @type infiles: [str]
        @param actions: the tasks to accomplish: 
            {spectro|melspectro|labelmask|copyraven}
            NOTE: copy raven is used simply to copy over the raven gt label .txt
            file if we are moving the spectrogram to a new location
        @type actions: [str] 
        @param outdir: if provided, everything that is created is written
            to this directory. If None, is written to the directory of the file
            on which computed was based. This is a good default!
        @type outdir {None | str}
        @param normalize: whether or not to normalize the signal to be within 16 bits
        @type normalize: bool
        @param to_db: whether to convert to db scale (log scale essentially)
        @type to_db: bool
        @param framerate: framerate of the recording. Normally 
            obtained from the wav file itself.
        @type framerate: int
        @param min_freq: min frequency in the processed spectrogram
        @type min_freq: int
        @param max_freq: max frequence in the processed spectrogram
        @type max_freq: int
        @param nfft: window width
        @type nfft: int,
        @param logfile: destination for log. Default: display
        @type logfile: {None|str}
        '''

        # Set up class variables related to spectrogram generation
        self.nfft = nfft
        self.pad_to = pad_to
        self.hop = hop
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.framerate = framerate

        # Output directory
        self.outdir = outdir
        
        # We should figure out exactly what the shmuck is going on here
        if logfile is None:
            self.log = LoggingService()
        else:
            self.log = LoggingService(logfile,
                                      msg_identifier="spectrogrammer")
        
        
        if type(infiles) != list:
            infiles = [infiles]
        

        # Step through the input files and process them depending on
        # their file signature and actions specified as args.
        for infile in infiles:
            # Super basic file checking
            if not os.path.exists(infile):
                print(f"File {infile} does not exist.")
                continue

            # Get a dict with the file_root and names related to 
            # the infile in our file naming scheme:
            # Note this is useful for associating 
            # .wav and .txt files
            file_family = FileFamily(infile)

            # Output the files to the same path as input
            # Note this allows self.outdir to change for each file
            if outdir is None:
                self.outdir = file_family.path

            # Make sure the outdir exists!!
            if not os.path.exists(outdir):
                os.mkdir(outdir)

            # Process wav file if spectro / melspectro in actions
            if infile.endswith('.wav') and ('spectro' in actions or 'melspectro' in actions):
                # Process the wave file
                try: 
                    self.log.info(f"Reading wav file {infile}...")
                    (_, samples) = wavfile.read(infile)                    
                    self.log.info(f"Done reading wav file {infile}.")
                except Exception as e:
                    self.log.warn(f"Cannot process .wav file: {repr(e)}")
                    # We should continue onto the next one!! 
                    # this we have seen with currupted .wav files
                    continue

                if 'spectro' in actions:
                    try:
                        spect, times = self.make_spectrogram(samples)
                    except Exception as e: # This likely will not happen
                        print(f"Cannot create spectrogram for {infile}: {repr(e)}")

                    # Save the spectrogram
                    spectro_outfile = os.path.join(self.outdir, file_family.spectro)
                    np.save(spectro_outfile, spect)
                    # Save the time mask
                    times_outfile = os.path.join(self.outdir, file_family.time_labels)
                    np.save(times_outfile, times)
                else: # melspectro
                    print ("TODO") 
            
            # Process label files
            elif infile.endswith('.txt') and 'labelmask' in actions or 'copyraven' in actions:
                # Generate the 0/1 spectrogram labels
                if 'labelmask' in actions:
                    # If we have a label file there should be a corresponding wav file
                    # in the same folder!
                    wav_file = file_family.fullpath(AudioType.WAV) 
                    if not os.path.exists(wav_file):
                        print(f"File {wav_file} does not exist so we cannot generate label mask")
                        continue

                    label_mask = self.create_label_mask_from_raven_table(wav_file, infile)
                    if label_mask is None:
                        print(f"Issue generating {infile} due to error in .wav file")
                        continue

                    np.save(os.path.join(self.outdir, file_family.mask), label_mask)

                if 'copyraven' in actions:
                    print ("TODO")


    #------------------------------------
    # make_spectrogram
    #-------------------

    def make_spectrogram(self, raw_audio, chunk_size=1000):
        '''
        Given data, compute a spectrogram. To avoid slow memory
        issues build the spectrogram in a stream-like fashion
        where we build it in chunk sizes of 1000 spectrogram frames

        Assumptions:
            o self.framerate contains the data framerate
            o self.nfft contains the window size
            o self.hop contains the hop size for fft
            o self.pad_to contains the zero-padding that 
            we add if self.pad_to > self.nfft
             
        Returns a two-tuple: an array of segment times
            and a 2D spectrogram array

        @param raw_audio: the time/amplitude data
        @type raw_audio: np.array([float])
        @param chunk_size: controls the incremental size used to 
        build up the spectrogram
        @type chunk_size: int
        @return: (spectrogram, time slices array) 
        @rtype: (np.array([float][time x freq]), np.array([float]))
        
        '''
        
        # The time_labels will be in seconds. But
        # they will be fractions of a second, like
        #   0.256, ... 1440
        self.log.info("Creating spectrogram...")
        
        # Compute the number of raw audio frames
        # needed to generate chunk_size spec frames
        len_chunk = (chunk_size - 1) * self.hop + self.nfft

        final_spec = None
        slice_times = None
        start_chunk = 0
        # Generate 1000 spect frames at a time, being careful to follow the correct indexing at the boarders to 
        # "simulate" the full fft. Namely, if we use indeces (raw_start, raw_end) to get the raw_audio frames needed
        # to generate the spectrogram chunk, remember the next chunk does not start at raw_end but actually start 
        # and (raw_end - NFFT) + hop. **THIS IS KEY** to propertly simulate the full spectrogram creation process
        iteration = 0
        print ("Approx number of chunks:", int(raw_audio.shape[0] / len_chunk))
        while start_chunk + len_chunk < raw_audio.shape[0]:
            if (iteration % 100 == 0):
                print ("Chunk number " + str(iteration))
            [spectrum, freqs, t] = ml.specgram(raw_audio[start_chunk: start_chunk + len_chunk], 
                    NFFT=self.nfft, Fs=self.framerate, noverlap=(self.nfft - self.hop), 
                    window=ml.window_hanning, pad_to=self.pad_to)

            # Cutout uneeded high frequencies!
            spectrum = spectrum[(freqs <= self.max_freq)]

            if start_chunk == 0:
                final_spec = spectrum
                slice_times = t
            else:
                final_spec = np.concatenate((final_spec, spectrum), axis=1)
                # Shift t to be 0 started than Offset the new times 
                # by the last frame's time + the time gap between frames (= hop / fr)
                t = t - t[0] + slice_times[-1] + (self.hop / self.framerate)
                slice_times = np.concatenate((slice_times, t))

            # Remember that we want to start as if we are doing one continuous sliding window
            start_chunk += len_chunk - self.nfft + self.hop 
            iteration += 1

        # Do one final chunk for whatever remains at the end
        [spectrum, freqs, t] = ml.specgram(raw_audio[start_chunk: ], 
                NFFT=self.nfft, Fs=self.framerate, noverlap=(self.nfft - self.hop), 
                window=ml.window_hanning, pad_to=self.pad_to)
        # Cutout the high frequencies that are not of interest
        spectrum = spectrum[(freqs <= self.max_freq)]
        final_spec = np.concatenate((final_spec, spectrum), axis=1)
        # Update the times: 
        t = t - t[0] + slice_times[-1] + (self.hop / self.framerate)
        slice_times = np.concatenate((slice_times, t))


        # check the shape of this
        self.log.info("Done creating spectrogram.")
     
        # This we may actually want to do! Log transform!!!
        # Transformer for magnitude to power dB:
        #amp_to_dB_transformer = torchaudio.transforms.AmplitudeToDB()
        #freq_time_dB_tensor = amp_to_dB_transformer(torch.Tensor(freq_time))

        # Note transpose the spectrogram to be of shape - (time, freq)
        return final_spec.T, slice_times

    #------------------------------------
    # make_mel_spectrogram
    #-------------------
    # NEED TO UPDATE THIS STUFF!!!!!
    def make_mel_spectrogram(self, sig_t, framerate):
        
        # Get tensor (128 x num_samples), where 128
        # is the default number of mel bands. Can 
        # change in call to MelSpectrogram:
        mel_spec_t = transforms.MelSpectrogram(
            sample_rate=framerate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length
            )(sig_t)
            
        # Turn energy values to db of max energy
        # in the spectrogram:
        mel_spec_db_t = transforms.AmplitudeToDB()(mel_spec_t)
        
        (num_mel_bands, _num_timebins) = mel_spec_t.shape
        
        # Number of columns in the spectrogram:
        num_time_label_choices = DSPUtils.compute_timeticks(framerate, 
                                                            mel_spec_db_t
                                                            )
        # Enumeration of the mel bands to use as y-axis labels: 
        freq_labels = np.array(range(num_mel_bands))
        
        return(freq_labels, num_time_label_choices, mel_spec_db_t)


    #------------------------------------
    # create_label_mask_from_raven_table
    #-------------------
    
    def create_label_mask_from_raven_table(self,
                                           wav_file,
                                           label_file):
        '''
        Given a .wav recording, plus a manually created 
        selection table as produced by the Raven program,  
        create a mask file with 1s where the spectrogram 
        time bins would match labels, and 0s elsewhere.
        
        Label files are of the form:
        
            Col1    ...    Begin Time (s)    End Time (s)    ... Coln
             foo             6.326            4.653              bar
                    ...

        @param wav_file: tthe name of a .wav recording file
        @type wav_file_or_sig: {str}
        @param label_file: label file as produced with Raven
        @type label_file: {str}
        '''
            
        # The x-axis time labels that a spectrogram would have:
        time_tick_secs_labels = DATAUtils.time_ticks_from_wav(wav_file,
                                                             hop_length=self.hop,
                                                             nfft=self.nfft
                                                             )
        # Check if the wav file failed to read
        if time_tick_secs_labels is None:
            return None
                                     
        # Start with an all-zero label mask:
        label_mask = np.zeros(len(time_tick_secs_labels), dtype=int)
        
        try:
            fd = open(label_file, 'r')
            reader = csv.DictReader(fd, delimiter='\t')
            for (start_bin_idx, end_bin_idx) in self._get_label_indices(reader, 
                                                                        time_tick_secs_labels,
                                                                        label_file):
            
                # Fill the mask with 1s in the just-computed range:
                label_mask[start_bin_idx:end_bin_idx] = 1
                
        finally:
            # Only close an fd that we may have
            # opened in this method. Fds passed
            # in remain open for caller to close:
            fd.close()
            
        return label_mask


            
    #------------------------------------
    # _get_label_indices 
    #-------------------

    def _get_label_indices(self, 
                           reader,
                           label_times,
                           label_txt_file):

        if type(label_times) != np.ndarray:
            label_times = np.array(label_times)
        
        file_offset_key = 'File Offset (s)'
        begin_time_key = 'Begin Time (s)'
        end_time_key   = 'End Time (s)'

        # Get each el call time range spec in the labels:
        for label_dict in reader:
            try:
                #start_time = float(row['File Offset (s)'])
                begin_time = float(label_dict[file_offset_key])
                call_length = float(label_dict[end_time_key]) - float(label_dict[begin_time_key])
                end_time = begin_time + call_length
            except KeyError:
                raise IOError(f"Raven label file {label_txt_file} does not contain one "
                              f"or both of keys '{begin_time_key}', {end_time_key}'")
            
            if end_time < begin_time:
                self.log.err(f"Bad label: end label less than begin label: {end_time} < {begin_time}")
                continue
            
            if begin_time > label_times[-1]:
                self.log.err(f"Bad label: begin label after end of recording: {begin_time} > {label_times[-1]}")
                continue
            if end_time < label_times[0]:
                self.log.err(f"Bad label: end label before start of recording: {end_time} < {label_times[0]}")
                continue
            

            # To deal very loosely with noise around the boundaries
            # let us be on the stricter end of setting the bounds.
            # Find the lower and upper time boarders that do not 
            # include the start/end times and then shrink these 
            # boarders in by 1.
            
            # Get all of the indeces that have time less than the start time
            pre_begin_indices = np.nonzero(label_times < begin_time)[0] 
            if len(pre_begin_indices) == 0:
                start_bin_idx = 0
            else:
                # Make the bounds a bit tighter by adding one to the last
                # index with the time < begin_time
                start_bin_idx = pre_begin_indices[-1] + 1
            
            # Similarly with end time:
            post_end_indices = np.nonzero(label_times > end_time)[0]
            if len(post_end_indices) == 0:
                # Label end time is beyond recording. Just 
                # go up to the end:
                end_bin_idx = len(label_times)
            else:
                # Similar, make bounds a bit tighter 
                end_bin_idx = post_end_indices[0] - 1

            yield (start_bin_idx, end_bin_idx)


# ---------------------------- Main ---------------------
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Spectrogram creation/manipulation"
                                     )

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
                    help='Min frequency of leading bandpass filter')

    parser.add_argument('-s', '--start', 
                        type=int, 
                        default=0, 
                        help='Seconds into recording when to start spectrogram')

    parser.add_argument('-e', '--end', 
                        type=int, 
                        default=None, 
                        help='Seconds into recording when to stop spectrogram; default: All')

    parser.add_argument('-n', '--no_normalize', 
                        default=False,
                        action='store_true', 
                        help='do not normalize to fill 16-bit -32K to +32K dynamic range'
                        )

    parser.add_argument('-o', '--outdir', 
                        default=None, 
                        help='Outfile for any created files'
                        )

    parser.add_argument('--labelfiles',
                        nargs='+',
                        help="Input .txt/.npy file(s)"
                        )

    parser.add_argument('--actions',
                        nargs='+',
                        choices=['spectro', 'melspectro','labelmask', 'copyraven'],
                        help="Which tasks to accomplish (repeatable)"
                        )

    parser.add_argument('infiles',
                        nargs='+',
                        help="Input .wav/.pickle file(s)"
                        )

    args = parser.parse_args();
    # Should get the infiles from a given folder!

    Spectrogrammer(args.infiles,
                   args.actions,
                   outdir=args.outdir,
                   # If no_normalize is True, don't normalize:
                   normalize=not args.no_normalize,
                   framerate=args.framerate,
                   nfft=args.nfft
                   )
    



