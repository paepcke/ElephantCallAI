#!/usr/bin/env python
'''
Created on Feb 23, 2020

@author: paepcke
'''
import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path

from scipy.io import wavfile
from scipy.signal.spectral import stft, istft
import torch
from torchaudio import transforms
import torchaudio

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dsp_utils import AudioType
from dsp_utils import DSPUtils
from dsp_utils import FileFamily
from amplitude_gating import AmplitudeGater
from elephant_utils.logging_service import LoggingService
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotting.plotter import Plotter

class Spectrogrammer(object):
    '''
    Create and manipulate spectrograms. Includes starting
    from a .wav file, or an already prepared .npy file. Allows
    spectrograms to be prepared from segments of a .wav
    file, as specified by start and end time.
    
    Several frequency filters are available.
    '''
    NFFT = 4096 # was 3208 # We want a frequency resolution of 1.95 Hz
    WIN_LENGTH = NFFT
    OVERLAP = 1/2    
    HOP_LENGTH = int(WIN_LENGTH * OVERLAP)
    # Maximum frequency retained in spectrograms if 
    # they are cleaned:
    FREQ_MAX = 30
    DEFAULT_FRAMERATE=8000

    # Energy below which spectrogram is set to zero:
    ENERGY_SUPPRESSION = -50 # dBFS
    
    #------------------------------------
    # Constructor 
    #-------------------


    def __init__(self, 
                 infiles,
                 actions,
                 outdir=None,
                 start_sec=0,
                 end_sec=None,
                 normalize=True,
                 framerate=None,
                 threshold_db=-30, # dB
                 low_freq=20,      # Hz 
                 high_freq=40,     # Hz
                 spectrogram_freq_cap=None,
                 nfft=None,
                 logfile=None,
                 testing=False
                 ):
        '''
        @param infiles:
        @type infiles:
        @param actions: the tasks to accomplish: 
            {spectro|melspectro|plot|plotexcerpts|labelmask}
        @type actions: [str] 
        @param outdir: if provided, everything that is created is written
            to this directory (spectrograms, label masks, gated versions of 
            wav files. If None, is written to the directory of the file
            on which computed was based.
        @type outdir {None | str}
        @param start_sec: start second in the wav file
        @type start_sec: int
        @param end_sec: end second in the wav file
        @type end_sec: int
        @param normalize: whether or not to normalize the signal
        @type normalize: bool
        @param framerate: framerate of the recording. Normally 
            obtained from the wav file itself.
        @type framerate: int
        @param low_freq: low frequency bound of leading bandpass filter
        @type low_freq: int
        @param high_freq: high frequency bound of leading bandpass filter
        @type high_freq: int
        @param nfft: window width
        @type nfft: int,
        @param logfile: destination for log. Default: display
        @type logfile: {None|str}
        @param testing: if True, only create the instance, and return
        @type testing: bool
        '''

        if logfile is None:
            self.log = LoggingService()
        else:
            self.log = LoggingService(logfile,
                                      msg_identifier="spectrogrammer")
        
        if testing:
            # Leave all calling of methods to the unittest:
            return 
        
        if type(infiles) != list:
            infiles = [infiles]
        
        # Depending on what caller wants us to do,
        # different arguments must be passed. Make
        # all those checks to avoid caller waiting a long
        # time for processing to be done only to fail
        # at the end:

        # Prerequisites:

        if not self._ensure_prerequisites(infiles,
                                          actions,
                                          framerate,
                                          threshold_db,
                                          low_freq,
                                          high_freq,
                                          nfft,
                                          outdir):
            return

        # Go through each infile, which could we label
        # masks, spectrogram dataframes, or label csv files.
        # Do appropriate work, and operate on the result
        # accoring to spectified 'actions':
        
        # Compute data structures:
        
        # Need to create all spectrograms before creating
        # masks or cleaning spectrograms. So sort infiles 
        # to have .wav files first:
        
        infiles = sorted(infiles,
                         key=lambda x: x.endswith('.wav'),
                         reverse=True)

        for infile in infiles:
            if not os.path.exists(infile):
                print(f"File {infile} does not exist.")
                continue

            spect = None
            clean_spect = None
            spectro_outfile = None
            label_mask = None

            # Get a dict with the file_root and 
            # names related to the infile in our
            # file naming scheme:
            file_family = FileFamily(infile)

            if outdir is None:
                outdir = file_family.path
            
            
            if infile.endswith('.wav'):
                
                try:
                    self.log.info(f"Reading wav file {infile}...")
                    (self.framerate, samples) = wavfile.read(infile)
                    self.log.info(f"Processing wav file {infile}...")
                    # Get as a dict:
                    #    the gated samples:                           'gated_samples',
                    #    the percent of the wav file that was zeroed: 'perc_zeroed'
                    #    path to the gated wav file on disk:          'result_file'
                    #    path to untreated excerpt file if requested: 'excerpt_file'
                    
                    process_result_dict = \
                        self.process_wav_file(samples,
                                              file_family, 
                                              start_sec=start_sec, 
                                              end_sec=end_sec, 
                                              normalize=normalize,
                                              threshold_db=threshold_db,
                                              low_freq=low_freq,
                                              high_freq=high_freq,
                                              outdir=outdir
                                              )
                    self.log.info(f"Done processing wav file {infile}.")
                except Exception as e:
                    print(f"Cannot process .wav file: {repr(e)}")
                    return

                if 'spectro' in actions:
                    try:
                        # Create spectrogram, receiving a dataframe
                        # with magnitudes in dB:
                        spect = \
                            self.make_spectrogram(process_result_dict['gated_samples'])
                    except Exception as e:
                        print(f"Cannot create spectrogram for {infile}: {repr(e)}")
                        return
                    # Save the spectrogram:
                    spectro_outfile = os.path.join(outdir,file_family.spectro)
                    DSPUtils.save_spectrogram(spect, spectro_outfile)

            elif infile.endswith('.txt'):
                # Get label mask with 1s at time periods with an
                # elephant call.
                wav_file = file_family.fullpath(AudioType.WAV),
                label_mask = self.create_label_mask_from_raven_table(wav_file,
                                                                     infile,
                                                                     self.hope_length
                                                                     )
                DSPUtils.save_label_mask(label_mask, 
                                         os.path.join(outdir,file_family.mask))
                
            elif infile.endswith('.pickle'):
                # Infile is a .pickle spectrogram file:
                self.log.info(f"Loading spectrogram file {infile}...")
                try:
                    spect = DSPUtils.load_spectrogram(infile)
                except Exception as e:
                    print(f"Could not read spectrogram {infile}: {repr(e)}")
                    return
                self.log.info(f"Done loading spectrogram file {infile}.")
                
                maybe_mask_file = file_family.fullpath(AudioType.MASK)
                if os.path.exists(maybe_mask_file):
                    self.log.info("Reading label mask...")
                    label_mask = DSPUtils.load_label_mask(maybe_mask_file)
                    self.log.info("Done reading label mask.")
                else:
                    self.log.info("No corresponding label mask file (that's fine).")

            # Perform requested actions:
            
            clean_spect = spect
            if 'cleanspectro' in actions:
                self.log.info(f"Cleaning spectrogram...")
                clean_spect = self.process_spectrogram(spect, spectrogram_freq_cap=spectrogram_freq_cap)
                self.log.info(f"Done cleaning spectrogram.")
                
            if 'plotexcerpts' in actions and clean_spect is not None:
                # The default for the following call is to plot 
                # regularly spaced excerpts of a given spectrogram:
                self.log.info("Plotting spectrogram excerpts...")
                self.plotter.plot_spectrogram_from_magnitudes(
                    clean_spect,
                    title=f"{os.path.basename(infile)}"
                    )
                self.log.info("Done plotting spectrogram excerpts.")
            if 'plot' in actions and clean_spect is not None:
                self.log.info("Plotting entire spectrogram ...")
                # Plot the whole 24 hour spectro in on chart:
                self.plotter.plot_spectrogram_from_magnitudes(
                    clean_spect,
                    time_intervals_to_cover=[], # Plot whole spectrogram
                    title=f"{os.path.basename(infile)}"
                    )
                self.log.info("Done plotting entire spectrogram")

            # Plot spectro pieces with labeled calls below?
            if 'plothits' in actions and clean_spect is not None:
                relevant_label_file = file_family.fullpath(AudioType.LABEL)
                if not os.path.exists(relevant_label_file):
                    self.log.err(f"The plothits option requires label_file {relevant_label_file} to exist; it doen't. Skipping {infile}")
                    continue
                label_intervals = self.create_label_intervals_from_raven_table(
                    relevant_label_file,
                    clean_spect)
                label_mask      = self.create_label_mask_from_raven_table(
                    relevant_label_file,
                    clean_spect)
                self.log.info("Plotting spectrogram plus truths ...")
                self.plotter.visualize_predictions(
                    label_intervals,
                    clean_spect,
                    label_mask=label_mask,
                    title=f"{os.path.basename(infile)}: True Positives", 
                    filters=None
                    )
                self.log.info("Done plotting spectrogram plus truths ...")

    #------------------------------------
    # _ensure_prerequisites 
    #-------------------
    
    def _ensure_prerequisites(self,
                              infiles,
                              actions,
                              framerate,
                              threshold_db,
                              low_freq,
                              high_freq,
                              nfft,
                              outdir
                              ):
        # Prerequisites:

        if outdir is not None and not os.path.exists(outdir):
                os.makedirs(outdir)

        if 'labelmask' in actions:
            # Need true framerate and spectrogram bin size
            # to compute time ranges in spectrogram:
            if nfft is None:
                self.log.warn(f"Assuming default time bin nfft of {self.NFFT}!\n" 
                              "If this is wrong, label allignment will be wrong")
            if framerate is None:
                self.log.warn(f"Assuming default framerate of {self.DEFAULT_FRAMERATE}!\n"
                              "If this is wrong, label allignment will be wrong")
            
            # A .txt file must either be given in the list of infiles,
            # or must be in the directory of one of the infiles:    
            if not any(Path(filename).parent.joinpath(Path(filename).stem + '.txt') 
                       for filename in infiles) and \
                not any(filename.endswith('.txt') for filename in infiles):
                self.log.err("For creating a label mask, must have a Raven selection file")
                return False
                
        if 'spectro' in actions or 'melspectro' in actions:
            if not any(filename.endswith('.wav') for filename in infiles):
                self.log.err("For creating a spectrogram, a .wav file must be provided")
                return False
            
            if framerate is not None:
                self.log.warn(f"Framerate was provided, but will be ignore: using framerate from .wav file.")

            if threshold_db is None or low_freq is None or high_freq is None:
                self.log.err("This module always does bandwidth prefilter, and noise gating\n"
                             "In future turning those off may be added. For now,\n"
                             "set low_freq to 0 and high_freq to something high\n",
                             "and set threshold_db to something like -100 to achieve"
                             "the same goal"
                             )
                return False
                
        if 'plot' in actions or 'plotexcerpts' in actions or 'plothits' in actions: 
            # We either need a .pickle file that must be
            # a spectrogram, or we need a .wav file that will
            # be turned into a spectrogram to be plotted
            if not any(filename.endswith('.pickle') or filename.endswith('.wav') 
                       for filename in infiles):
                self.log.err("To plot something, there must be either a .pickle spectrogram file\n"
                             "or a .wav file"
                             )
                return False
            self.plotter = Plotter()

        if framerate is None:
            self.framerate = self.DEFAULT_FRAMERATE
            
        if nfft is None:
            self.log.info(f"Assuming NFFT == {Spectrogrammer.NFFT}")
            nfft = Spectrogrammer.NFFT
        
        if type(infiles) != list:
            infiles = [infiles]

        return True


    #------------------------------------
    # process_spectrogram 
    #-------------------
    
    def process_spectrogram(self, 
                            spectro,
                            energy_suppression=None,
                            spectrogram_freq_cap=None):
        '''
        Given a spectrogram, return a new spectrogram with
        the following modifications:
        
           o The magnitudes at energy_suppression dB
             of maximum and below are zeroed
           o Frequencies above spectrogram_freq_cap are
             removed
              
        
        @param spectro: spectrogram on which to operate
        @type spectro: pd.DataFrame
        @param energy_suppression: magnitudes in dB below max energy
            in the spectrogram are zeroed
        @type energy_suppression: int
        @param spectrogram_freq_cap: frequency above which all
            rows in the spectrogram are removed
        @type spectrogram_freq_cap: int
        @return: a new spectrogram.
        @rtype: pd.DataFrame 
        '''
        
        new_spectro = spectro.copy()
        # Chop off high freqs:
        if spectrogram_freq_cap is not None:
            trimmed_spectro = new_spectro.iloc[spectro.index <= spectrogram_freq_cap]
        else:
            trimmed_spectro = new_spectro

        if energy_suppression is None:
            energy_suppression = Spectrogrammer.ENERGY_SUPPRESSION
        mags_only = trimmed_spectro.to_numpy()
        max_energy = np.amax(mags_only)
        low_energy_thres = max_energy * 10**(energy_suppression/20)
        spect_low_energy_indices = np.nonzero(mags_only <= low_energy_thres)
        mags_only[spect_low_energy_indices] = 0.0
        
        processed_spectro = pd.DataFrame(mags_only,
                                         index=trimmed_spectro.index,
                                         columns=trimmed_spectro.columns)
        return processed_spectro

    #------------------------------------
    # make_spectrogram
    #-------------------

    def make_spectrogram(self, data):
        '''
        Given data, compute a spectrogram.

        Assumptions:
            o self.framerate contains the data framerate
    
        o The (Hanning) window overlap used (Hanning === Hann === Half-cosine)
        o Length of each FFT segment: 4096 (2**12)
        o Number of points to overlap with each window
             slide: 1/2 the segments size: 2048
        o Amount of zero-padding at the end of each segment:
             the length of the segment again, i.e. doubling the window
             so that the conversion to positive-only frequencies makes
             the correct lengths
             
        Returns a three-tuple: an array of sample frequencies,
            An array of segment times. And a 2D array of the SFTP:
            frequencies x segment times
        
        @param data: the time/amplitude data
        @type data: np.array([float])
        @return: spectrogram dataframe with index being frequencies,
            and columns being time labels.
        @rtype: pd.DataFrame
        
        '''
        
        # The time_labels will be in seconds. But
        # they will be fractions of a second, like
        #   0, 0.256, ... 1440
        self.log.info("Creating spectrogram...")
        (freq_labels, time_labels, complex_freq_by_time) = \
            stft(data, 
                 self.framerate, 
                 nperseg=self.NFFT
                 #nperseg=int(self.framerate)
                 )
            
        self.log.info("Done creating spectrogram.")

        freq_time = np.absolute(complex_freq_by_time)
        
        # Transformer for magnitude to power dB:
        amp_to_dB_transformer = torchaudio.transforms.AmplitudeToDB()
        freq_time_dB_tensor = amp_to_dB_transformer(torch.Tensor(freq_time))

        spectrogram = pd.DataFrame(freq_time_dB_tensor.numpy(),
                                   index=freq_labels,
                                   columns=time_labels
                                   )
        return spectrogram

    #------------------------------------
    # make_mel_spectrogram
    #-------------------
    
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
    # make_inverse_spectrogram 
    #-------------------
    
    def make_inverse_spectrogram(self, spectrogram):
    
        self.log.info("Inverting spectrogram to time domain...")
        (_time, voltage) = istft(spectrogram, 
                                 self.framerate, 
                                 nperseg=self.NFFT
                                 #nperseg=int(self.framerate)
                                 )
        self.log.info("Done inverting spectrogram to time domain.")

        # Return just the voltages:
        return voltage.astype(np.int16)

    #------------------------------------
    # filter_spectrogram
    #-------------------
    
    def filter_spectrogram(self, 
                           freq_labels, 
                           freq_time, 
                           freq_bands):
        '''
        Given a spectrogram, return a new spectrogram
        with only frequencies within given bands retained.
        
        freq_time is a matrix whose rows each contain energy
        contributions by one frequency over time. 
        
        The freq_labels is an np.array with the frequency of
        each row. I.e. the y-axis labels.
        
        freq_bands is an array of frequency intervals. The
        following would only retain rows for frequencies 
              0 <= f < 5,  
             10 <= f < 20,
         and  f >= 40:
        
           [(None, 5), (10,20), (40,None)]
           
        So: note that these extracts are logical OR.
            Contributions from each of these three
            intervals will be present, even though the 
            (10,20) would squeeze out the last pair,
            due to its upper bound of 20.
            
        Note: Rows will be removed from the spectrogram. Its
              width will not change. But if the spectrogram
              were to be turned into a wav file, that file 
              would be shorter than the original.
         
        @param freq_labels: array of frequencies highest first
        @type freq_labels: np.array[float]
        @param freq_time: 2d array of energy contributions
        @type freq_time: np.array(rows x cols) {float || int || complex}
        @param freq_bands: bands of frequencies to retain.
        @type freq_bands: [({float | int})]
        @return revised spectrogram, and correspondingly reduced
            frequency labels
        @rtype: (np_array(1), np_array(n,m))
        '''
        # Prepare a new spectrogram matrix with
        # the same num of cols as the one passed
        # in, but no rows:
        
        (_num_rows, num_cols) = freq_time.shape
        new_freq_time    = np.empty((0,num_cols))
        
        # Same for the list of frequencies:
        new_freq_labels  = np.empty((0,))
        
        for (min_freq, out_freq) in freq_bands:
            if min_freq is None and out_freq is None:
                # Degenerate case: keep all:
                continue
            if min_freq is None:
                min_freq = 0
            if out_freq is None:
                # No upper bound, so make a ceiling
                # higher than maximum frequency:
                out_freq = np.amax(freq_labels) + 1.
                
            # Get the indices of the frequency array 
            # where the frequency is within this interval.
            # The np.where returns a tuple, therefore [0]

            filter_indices =  np.where(np.logical_and(freq_labels >= min_freq,
                                                      freq_labels < out_freq
                                                      ))[0]
        
            # Keep only rows (axis=0) that contain the energies for
            # included frequencies:
            new_freq_time = np.vstack(
                (new_freq_time, np.take(freq_time, filter_indices, axis=0))
                )

            # Also take away the row labels that where excluded:
            new_freq_labels = np.hstack(
                (new_freq_labels, np.take(freq_labels, filter_indices))
                )

        return (new_freq_labels, new_freq_time)

    #------------------------------------
    # process_wav_file
    #-------------------

    def process_wav_file(self,
                         infile_or_samples,
                         file_family,
                         start_sec=None, 
                         end_sec=None,
                         threshold_db=-40, # dB
                         low_freq=10,      # Hz 
                         high_freq=50,     # Hz
                         normalize=False,
                         outdir=None,
                         keep_excerpt=False,
                         ):
        '''
        Takes either a wav file path, or the audio samples
        from a wav file. Subjects the signal to a bandpass
        filter plus a noise filter. Optionally only works
        on an excerpt of the file.
        
        
        @param infile_or_samples: either audio samples, or a wav file path
        @type infile_or_samples: {str | np.array}
        @param start_sec: if non-none, the first audio second to include
            If None, 0 is assumed.
        @type start_sec: {None|int}
        @param end_sec: one second behond where the excerpt should stop
            If None, to the end is assumed
        @type end_sec: {None|int}
        @param threshold_db: dB of RMS of audio signal below which 
            signal is zeroed out.
        @type threshold_db: int
        @param low_freq: low frequency of the bandpass filter
        @type low_freq: int
        @param high_freq: high frequency of the bandpass filter
        @type high_freq: int
        @param normalize: whether or not to normalize the audio
        @type normalize: bool
        @param outdir: directory where to write the gated outfile.
            the name will be returned (see results)
        @type outdir: str
        @param keep_excerpt: if excerpt was requested, whether or
            not to keep the excerpt in the file system after processing.
        @type keep_excerpt: bool
        @return: dict with:

             the gated samples:                           'gated_samples',
             the percent of the wav file that was zeroed: 'perc_zeroed'
             path to the gated wav file on disk:          'result_file'
             path to untreated excerpt file if requested: 'excerpt_file'

          The latter only if (a) at least one of start_sec or end_sec
          where not None, and (b) the keep_excerpt option was True          
        @rtype: {str : {str|{str|float|np.array}}
         
        '''

        excerpted = False
        
        if (start_sec or end_sec) is not None:
            excerpted = True
            
            # If given an infile path, we need the samples themselves:
            if type(infile_or_samples) == str:
                (self.framerate, samples) = wavfile.read(infile_or_samples)
            else:
                samples = infile_or_samples

            
            # Pull out the excerpt, and write it to a temp file:
            start_sample_indx = self.framerate * start_sec
            end_sample_indx   = self.framerate * end_sec

            excerpt_infile = tempfile.NamedTemporaryFile(prefix='excerpt_',
                                                         suffix='.wav'
                                                         )
            
            wavfile.write(excerpt_infile, 
                          self.framerate,
                          samples[start_sample_indx:end_sample_indx]
                          )
        elif type(infile_or_samples) == str:
            excerpt_infile = infile_or_samples
        else:
            # Were given samples, which we didn't need to 
            # excerpt. AmplitudeGater does want an infile,
            # so we temporarily write one out, containing
            # all the samples
            excerpt_infile = tempfile.NamedTemporaryFile(prefix='full_file_',
                                                         suffix='.wav'
                                                         )
            
            wavfile.write(excerpt_infile, 
                          self.framerate,
                          infile_or_samples  # These are samples
                          )

        try:
            try:
                # Because we pass no spectrogram destination,
                # none is created in AmplitudeGater; just the
                # gated .wav file:
                gated_file_dest = os.path.join(outdir, file_family.gated_wav)
                gater = AmplitudeGater(excerpt_infile.name,
                                       amplitude_cutoff=threshold_db,
                                       low_freq=low_freq,
                                       high_freq=high_freq,
                                       outdir=outdir,
                                       outfile=gated_file_dest,
                                       normalize=normalize
                                       )
            except Exception as e:
                self.log.err(f"Processing failed for '{excerpt_infile}: {repr(e)}")
                return None
            
            perc_zeroed = gater.percent_zeroed

            return {'gated_samples': gater.gated_samples,
                    'perc_zeroed'  : perc_zeroed,
                    'result_file'  : gater.gated_outfile,
                    'excerpt_file' : excerpt_infile if (start_sec or end_sec) is not None
                                                    and keep_excerpt
                                                    else None
                    }
        finally:
            if excerpted and not keep_excerpt:
                os.remove(excerpt_infile)

    #------------------------------------
    # create_label_mask_from_raven_table
    #-------------------
    
    def create_label_mask_from_raven_table(self,
                                           wav_file_or_sig,
                                           label_txt_file,
                                           framerate,
                                           hop_length):
        '''
        Given a .wav recording, plus a manually created 
        selection table as produced by the Raven program, 
        plus two parameters used to create spectrograms, 
        create a mask file with 1s where the spectrogram 
        time bins would match labels, and 0s elsewhere.
        
        Label files are of the form:
        
            Col1    ...    Begin Time (s)    End Time (s)    ... Coln
             foo             6.326            4.653              bar
                    ...

        @param wav_file_or_sig: either the name of a .wav
            recording file, or an array of .wav signals
        @type wav_file_or_sig: {str|[float]}
        @param label_txt_file: either a path to a label file,
            or an open fd to such a file:
        @type label_txt_file: {str|file-like}
        @param label_file: label file as produced with Raven
        @type label_file: str
        '''
        # The x-axis time labels that a spectrogram
        # would have:
        time_tick_secs_labels = DSPUtils.time_ticks_from_wav(wav_file_or_sig,
                                                             framerate, 
                                                             hop_length
                                                             )
                                     
        # Start with an all-zero label mask:
        label_mask = np.zeros(len(time_tick_secs_labels), dtype=int)
        
        try:
            if type(label_txt_file) == str:
                fd = open(label_txt_file, 'r')
            else:
                fd = label_txt_file 
            reader = csv.DictReader(fd, delimiter='\t')
            for (start_bin_idx, end_bin_idx) in self._get_label_indices(reader, 
                                                                        time_tick_secs_labels,
                                                                        label_txt_file):
            
                # Fill the mask with 1s in the just-computed range:
                label_mask[start_bin_idx:end_bin_idx] = 1
                
        finally:
            # Only close an fd that we may have
            # opened in this method. Fds passed
            # in remain open for caller to close:
            if type(label_txt_file) == str:
                fd.close()
            
        return label_mask

    #------------------------------------
    # create_label_intervals_from_raven_table
    #-------------------
    
    def create_label_intervals_from_raven_table(self, 
                                                label_txt_file,
                                                spectrogram_or_spect_file):
        '''
        Given a manually created selection table as produced
        by the Raven program, plus a spectrogram, create a
        list of time intervals ele calls occurred.
        
        Label files are of the form:
        
            Col1    ...    Begin Time (s)    End Time (s)    ... Coln
             foo             6.326            4.653              bar
                    ...

        @param label_txt_file: either a path to a label file,
            or an open fd to such a file:
        @type label_txt_file: {str|file-like}
        @param spectrogram_or_spect_file: either an in-memory spectrogram DataFrame
            or a path to a pickled DataFrame
        @type spectrogram_or_spect_file: {str|pandas.DataFrame}
        '''
        # Get the spectro from file, if needed:
        spect = self._read_spectrogram_if_needed(spectrogram_or_spect_file)
        label_intervals = []
        # The x-axis time labels:
        label_times    = spect.columns.astype(float)
        
        try:
            if type(label_txt_file) == str:
                fd = open(label_txt_file, 'r')
            else:
                fd = label_txt_file 
            reader = csv.DictReader(fd, delimiter='\t')
            for (start_bin_idx, end_bin_idx) in self._get_label_indices(reader, 
                                                                        label_times,
                                                                        label_txt_file):
            
                # Next time interval:
                time_interval = pd.Interval(left=label_times[start_bin_idx], 
                                            right=label_times[end_bin_idx]
                                            )
                
                label_intervals.append(time_interval)
                
        finally:
            # Only close an fd that we may have
            # opened in this method. Fds passed
            # in remain open for caller to close:
            if type(label_txt_file) == str:
                fd.close()
            
        return label_intervals

    #------------------------------------
    # _read_spectrogram_if_needed
    #-------------------
    
    def _read_spectrogram_if_needed(self, spectrogram_or_spect_file):
        '''
        Given either a spectrogram dataframe, or a previously
        saved spectrogram, return the spectrogram
        
        @param spectrogram_or_spect_file:
        @type spectrogram_or_spect_file:
        '''
        if type(spectrogram_or_spect_file) == str:
            # Spectrogram is in a file; get it:
            try:
                spect = DSPUtils.load_spectrogram(spectrogram_or_spect_file)
            except Exception as e:
                self.log.err(f"While reading spectrogram file {spectrogram_or_spect_file}: {repr(e)}")
                raise IOError from e
        else:
            if type(spectrogram_or_spect_file) != pd.DataFrame:
                raise ValueError(f"Argument must be path to pickled spectrogram, or a DataFrame; was {type(spectrogram_or_spect_file)}")
            spect = spectrogram_or_spect_file
        return spect

            
    #------------------------------------
    # _get_label_indices 
    #-------------------

    def _get_label_indices(self, 
                           reader,
                           label_times,
                           label_txt_file
                           ):

        
        begin_time_key = 'Begin Time (s)'
        end_time_key   = 'End Time (s)'

        # Get each el call time range spec in the labels:
        for label_dict in reader:
            try:
                begin_time = float(label_dict[begin_time_key])
                end_time   = float(label_dict[end_time_key])
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
            
            # Want the time bin that's just below
            # the start of the label's
            # start time. Method nonzero returns a
            # 1 el tuple for 1D arrays; therefore the [0] 
            
            pre_begin_indices = np.nonzero(label_times <= begin_time)[0]
            # Is label start time beyond end of recording?
            if len(pre_begin_indices) == 0:
                start_bin_idx = 0
            else:
                # The last match is closest in the spectro
                # time line to the start time of the label:
                start_bin_idx = pre_begin_indices[-1]
            
            # Similarly with end time:
            post_end_indices = np.nonzero(label_times > end_time)[0]
            if len(post_end_indices) == 0:
                # Label end time is beyond recording. Just 
                # go up to the end:
                end_bin_idx = len(label_times)
            else:
                end_bin_idx = post_end_indices[0]
            yield (start_bin_idx, end_bin_idx)

    #------------------------------------
    # normalize
    #-------------------
    
    def normalize(self, samples):
        '''
        Make audio occupy the maximum dynamic range
        of int16: -2**15 to 2**15 - 1 (-32768 to 32767)
        Formula to compute new Intensity of each sample:
        
           I = ((I-Min) * (newMax - newMin)/Max-Min)) + newMin
           
        where Min is minimum value of current samples,
              Max  is maximum value of current samples,
              newMax is 32767
              newMin is -32768
        
        @param samples: samples from .wav file
        @type samples: np.narray('int16')
        @result: a new np array with normalized values
        @rtype: np.narray('int16')
        '''
        new_max = 2**15 - 1   # 32767
        new_min = -2**15      #-32768
        min_val = np.amin(samples)
        max_val = np.amax(samples)

        # self.log.info("Begin normalization...")

        normed_samples = ((samples - min_val) * (new_max - new_min)/(max_val - min_val)) + new_min
        # Convert back from float to int16:
        normed_samples = normed_samples.astype('int16')
        
        #self.log.info("Done normalization.")
        return normed_samples

    #------------------------------------
    # plot_simple
    #-------------------    
    
    def plot_simple(self, spectrum, time):
        pass
    
    #------------------------------------
    # plot
    #------------------- 
    
    def plot(self, 
            times, 
            frequencies,
            spectrum, 
            title='My Title'
            ):
#         new_features = 10*np.log10(spectrum)
#         min_dbfs = new_features.flatten().mean()
#         max_dbfs = new_features.flatten().mean()
#         min_dbfs = np.maximum(new_features.flatten().min(),min_dbfs-2*new_features.flatten().std())
#         max_dbfs = np.minimum(new_features.flatten().max(),max_dbfs+6*new_features.flatten().std())

        fig = plt.figure()
        ax  = fig.add_subplot(1,1,1)
        ax.pcolormesh(times, frequencies, spectrum)
        ax.imshow(#new_features,
                  spectrum, 
                  #cmap="magma_r", 
                  cmap="jet", 
                  #vmin=min_dbfs, 
                  #vmax=max_dbfs, 
                  #interpolation='none', 
                  origin="lower", 
                  #***aspect="auto"
                  )
        ax.set_title(title)
        ax.set_xlabel('Time')
        ax.set_ylabel('Frequency')
        
        #     # Make the plot appear in a specified location on the screen
        #     mngr = plt.get_current_fig_manager()
        #     geom = mngr.window.geometry()  
        #     mngr.window.wm_geometry("+400+150")
        
    #------------------------------------
    # make_time_freq_seqs 
    #-------------------
    
    def make_time_freq_seqs(self, max_freq, spect):
        
        # Num rows is num of frequency bands.
        # Num cols is number of time ticks:
        (num_freqs, num_times) = spect.shape
        # Ex: if max freq is 150Hz, and the number of
        # freq ticks on the y axis is 77, then each
        # tick is worth 150Hz/77 = 1.95Hz

        freq_band = max_freq / num_freqs
        freq_scale = list(np.arange(0,max_freq,freq_band))
        time_scale = list(np.arange(num_times))
        return(freq_scale, time_scale)

                        
    #------------------------------------
    # get_label_filename 
    #-------------------
    
    def get_label_filename(self, spect_numpy_filename):
        '''
        Given the file name of a numpy spectrogram 
        file of the forms:
           nn03a_20180817_neg-features_10.npy
           nn03a_20180817_features_10.npy
           
        create the corresponding numpy label mask file
        name:
           nn03a_20180817_label_10.npy
           
        
           
        @param spect_numpy_filename:
        @type spect_numpy_filename:
        '''
        # Check extension:
        (_fullname, ext) = os.path.splitext(spect_numpy_filename)
        if ext != '.npy':
            raise ValueError("File needs to be a .npy file.")
        
        # Maybe a dir is included, maybe not:
        dirname  = os.path.dirname(spect_numpy_filename)
        filename = os.path.basename(spect_numpy_filename)

        try:
            (loc_code, date, _file_content_type, id_num_plus_rest) = filename.split('_')
        except ValueError:
            raise ValueError(f"File name {spect_numpy_filename} does not have exactly four components.")
        label_filename = f"{loc_code}_{date}_labels_{id_num_plus_rest}"
        full_new_name = os.path.join(dirname, label_filename)
        return full_new_name


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

    parser.add_argument('--freq_cap', 
                    type=int, 
                    default=Spectrogrammer.FREQ_MAX, 
                    help='Determines the maximum frequency band in spectrogram')

    parser.add_argument('--max_freq', 
                    type=int, 
                    default=40,
                    help='Min frequency of leading bandpass filter')

    parser.add_argument('--min_freq', 
                    type=int, 
                    default=20,
                    help='Min frequency of leading bandpass filter')

    parser.add_argument('--threshold_db', 
                    type=int, 
                    default=-30,
                    help='Decibels relative to max signal below which signal is set to zero')

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
                        choices=['spectro', 'melspectro','cleanspectro','plot',
                                 'plotexcerpts','plothits', 'labelmask'],
                        help="Which tasks to accomplish (repeatable)"
                        )

    parser.add_argument('infiles',
                        nargs='+',
                        help="Input .wav/.pickle file(s)"
                        )

    args = parser.parse_args();
    Spectrogrammer(args.infiles,
                   args.actions,
                   outdir=args.outdir,
                   low_freq=args.min_freq,
                   high_freq=args.max_freq,
                   start_sec=args.start,
                   end_sec=args.end,
                   # If no_normalize is True, don't normalize:
                   normalize=not args.no_normalize,
                   framerate=args.framerate,
                   spectrogram_freq_cap=args.freq_cap,
                   nfft=args.nfft
                   )
    # Keep charts up till user kills the windows:
    Plotter.block_till_figs_dismissed()
