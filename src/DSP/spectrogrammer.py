'''
Created on Feb 23, 2020

@author: paepcke
'''
import argparse
import os
import sys
import tempfile

from scipy.io import wavfile
from scipy.signal.spectral import stft, istft
import torch
import torchaudio
from torchaudio import transforms

from amplitude_gating import AmplitudeGater
from elephant_utils.logging_service import LoggingService
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotting.plotter import Plotter
from wave_maker import WavMaker
from deprecated_code.process_rawdata import FREQ_MAX


sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.dirname(__file__))




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
    FREQ_MAX = 150.

    # Energy below which spectrogram is set to zero:
    ENERGY_SUPPRESSION = -50 # dBFS
    
    #------------------------------------
    # Constructor 
    #-------------------


    def __init__(self, 
                 infiles,
                 clean_spectrogram=False,
                 label_files=None,
                 start_sec=0,
                 end_sec=None,
                 normalize=False,
                 framerate=8000,
                 threshold_db=-40, # dB
                 low_freq=10,      # Hz 
                 high_freq=50,     # Hz
                 spectrogram_freq_cap=FREQ_MAX,
                 plot=True,
                 model_input_plot=False,
                 nfft=None,
                 ):
        '''
        
        
        @param infiles:
        @type infiles:
        @param clean_spectrogram: whether or not to clean the spectrogram
        @type clean_spectrogram: bool
        @param max_f: max frequency to keep in the spectrogram(s)
        @type max_f: float
        @param label_files: text files with manually produced labels
        @type label_files: str
        @param start_sec: start second in the wav file
        @type start_sec: int
        @param end_sec: end second in the wav file
        @type end_sec: int
        @param normalize: whether or not to normalize the signal
        @type normalize: bool
        @param framerate: framerate of the recording. Normally 
            obtained from the wav file itself.
        @type framerate: int
        @param plot: whether or not to plot the spectrogram
        @type plot: bool
        @param model_input_plot: whether or not to plot the model
            as seen by the classifier.
        @type model_input_plot: bool
        @param nfft: window width
        @type nfft: int
        '''
        '''
        Constructor
        '''
        
        self.log = LoggingService()

        self.framerate = framerate
        if nfft is None:
            self.log.info(f"Assuming NFFT == {Spectrogrammer.NFFT}")
            nfft = Spectrogrammer.NFFT
            
        self.plotter = Plotter()

        if type(infiles) != list:
            infiles = [infiles]
        for infile in infiles:
            if not os.path.exists(infile):
                print(f"File {infile} does not exist.")
                continue
    
            if infile.endswith('.wav'):
                try:
                    self.log.info(f"Reading wav file {infile}...")
                    (self.framerate, samples) = wavfile.read(infile)
                    self.log.info(f"Processing wav file {infile}...")
                    process_result_dict = \
                        self.process_wav_file(samples, 
                                              start_sec=start_sec, 
                                              end_sec=end_sec, 
                                              normalize=normalize,
                                              threshold_db=threshold_db,
                                              low_freq=low_freq,
                                              high_freq=high_freq,
                                              spectrogram_freq_cap=spectrogram_freq_cap                                              
                                              )
                except Exception as e:
                    print(f"Cannot process .wav file: {repr(e)}")
                    sys.exit(1)

                try:
                    # Create spectrogram:
                    (freq_labels, time_labels, freq_time_dB) = \
                        self.make_spectrogram(process_result_dict['gated_samples'])
                except Exception as e:
                    print(f"Cannot create spectrogram for {infile}: {repr(e)}")
                    sys.exit(1)
                spect = freq_time_dB
            else:
                # Infile is a .pickle spectrogram file:
                label_mask = None
                self.log.info(f"Loading spectrogram file {infile}...")
                try:
                    spect = pd.read_pickle(infile)
                except Exception as e:
                    print(f"Could not read spectrogram {infile}: {repr(e)}")
                    sys.exit(1)
                freq_labels = spect.index
                time_labels = spect.columns
                self.log.info(f"Done loading spectrogram file {infile}.")
                
                if label_files is not None:
                    # One or more .npy label masks was provided. See
                    # any of them match the .npy infile by name pattern:
                    label_file = self.get_label_filename(infile)
                    if label_file in label_files:
                        self.log.info("Reading label mask...")
                        label_mask = np.load(label_file)

            if clean_spectrogram:
                self.log.info(f"Cleaning spectrogram...")
                clean_spect = self.process_spectrogram(spect)
                self.log.info(f"Done cleaning spectrogram.")
            else:
                clean_spect = spect
            if plot:
                self.plotter.plot_spectrogram_excerpts(
                    [(0,len(time_labels)-1)], 
                    clean_spect, 
                    time_labels, 
                    freq_labels, 
                    plot_grid_width=1, 
                    plot_grid_height=1, 
                    title=f"{os.path.basename(infile)}")
                
            if model_input_plot:
                if label_files is None:
                    self.log.err("The model_input_plot option requires label_files to be specified.")
                self.plot_spectrogram_with_labels_truths(clean_spect, 
                                                 labels=label_mask,
                                                 title=os.path.basename(infile), 
                                                 vert_lines=None, 
                                                 filters=None)

#     #------------------------------------
#     # process_wav_file 
#     #-------------------
#     
#     def process_wav_file(self,
#                          samples,
#                          start_sec,
#                          end_sec,
#                          normalize,
#                          plot=True
#                          ):
# 
#         # Total time in seconds:
#         audio_secs = math.floor(samples.size / self.framerate)
#         
#         # Seek in to start time, and cut all beyond stop time:
#         if end_sec is None or end_sec > audio_secs:
#             end_sec = audio_secs
#         start_sample = self.framerate * start_sec
#         stop_sample  = self.framerate * end_sec
#         
#         if normalize:
#             samples_to_use = self.normalize(samples[start_sample : stop_sample])
#         else:
#             samples_to_use = samples
#             
#         (freq_labels, time_labels, freq_time) = \
#             self.make_spectrogram(samples_to_use)
#             
#         if plot:
#             (_spectrum,_freqs,_t_bins,_im) = \
#                 self.plot_spectrogram_from_magnitudes(samples_to_use, 
#                                       self.framerate,
#                                       start_sec,
#                                       end_sec
#                                       )
#         return (freq_labels, time_labels, freq_time)

    #------------------------------------
    # process_spectrogram 
    #-------------------
    
    def process_spectrogram(self, freq_time):
        
        max_energy = np.amax(freq_time)
        db_cutoff  = Spectrogrammer.ENERGY_SUPPRESSION # dB
        low_energy_thres = max_energy * 10**(db_cutoff/20)
        spect_low_energy_indices = np.nonzero(freq_time <= low_energy_thres)
        freq_time[spect_low_energy_indices] = 0.0

        #delete all but row <= 60?
        #take the log
        return freq_time

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
        @return: (frequency_labels, time_labels, spectrogram_matrix)
        @rtype: (np.array, np.array, np.array)
        
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
        
        return (freq_labels, time_labels, freq_time_dB_tensor.numpy())

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
        num_time_label_choices = self.compute_timeticks(framerate, 
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
                         start_sec=None, 
                         end_sec=None,
                         threshold_db=-40, # dB
                         low_freq=10,      # Hz 
                         high_freq=50,     # Hz
                         spectrogram_freq_cap=150,
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
                gater = AmplitudeGater(excerpt_infile.name,
                                       amplitude_cutoff=threshold_db,
                                       low_freq=low_freq,
                                       high_freq=high_freq,
                                       spectrogram_freq_cap=spectrogram_freq_cap,
                                       outdir=outdir,
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
    # compute_timeticks 
    #-------------------
    
    def compute_timeticks(self, framerate, spectrogram_t):
        '''
        Given a spectrogram, compute the x-axis
        time ticks in seconds, minutes, and hours.
        Return all three in a dict:
           {'seconds' : num-of-ticks,
            'minutes' : num-of-ticks,
            'hours'   : num-of-ticks
            }
            
        Recipient can pick 
        
        @param framerate:
        @type framerate:
        @param spectrogram_t:
        @type spectrogram_t:
        '''
        
        # Number of columns in the spectrogram:
        estimate_every_samples = self.n_fft - self.overlap * self.n_fft
        estimate_every_secs    = estimate_every_samples / framerate
        one_sec_every_estimates= 1/estimate_every_secs
        (_num_freqs, num_estimates) = spectrogram_t.shape
        num_second_ticks = num_estimates / one_sec_every_estimates
        num_minute_ticks = num_second_ticks / 60
        num_hour_ticks   = num_second_ticks / 3600
        num_time_ticks = {'seconds': int(num_second_ticks),
                          'minutes': int(num_minute_ticks),
                          'hours'  : int(num_hour_ticks)
                          }
        
        #time_labels       = np.array(range(num_timeticks))
        return num_time_ticks
                        
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

    parser.add_argument('--pad', dest='pad_to', type=int, default=4096, 
                        help='Deterimes the padded window size that we ' +
                             'want to give a particular grid spacing (i.e. 1.95hz')

    parser.add_argument('-f', '--framerate',
                        type=int,
                        default=8000,
                        help='Framerate at which original .wav file was recorded; def: 8000')

    parser.add_argument('--max_f', 
                    dest='max_freq', 
                    type=int, 
                    default=Spectrogrammer.FREQ_MAX, 
                    help='Deterimes the maximum frequency band in spectrogram')

parser.add_argument('-s', '--start', 
                        type=int, 
                        default=0, 
                        help='Seconds into recording when to start spectrogram')

    parser.add_argument('-e', '--end', 
                        type=int, 
                        default=None, 
                        help='Seconds into recording when to stop spectrogram; default: All')

    parser.add_argument('-n', '--normalize', 
                        default=False,
                        action='store_true', 
                        help='Normalize to fill 16-bit -32K to +32K dynamic range'
                        )

    parser.add_argument('-c', '--clean', 
                        default=False,
                        action='store_true', 
                        help='Clean spectrogram'
                        )
    
    parser.add_argument('-p', '--plot', 
                        default=False,
                        action='store_true', 
                        help='Plot charts as appropriate'
                        )
    
    parser.add_argument('-m', '--modelplot', 
                        default=False,
                        action='store_true', 
                        help='Plot spectrogram as classifier will see it.'
                        )

    parser.add_argument('-o', '--outfile', 
                        default=None, 
                        help='Outfile for cleaned spectrogram; only needed with -c'
                        )

    parser.add_argument('--labelfiles',
                        nargs='+',
                        help="Input .txt/.npy file(s)"
                        )
    

    parser.add_argument('infiles',
                        nargs='+',
                        help="Input .wav/.pickle file(s)"
                        )

    args = parser.parse_args();
    Spectrogrammer(args.infiles,
                   args.clean,
                   args.max_freq,
                   start_sec=args.start,
                   end_sec=args.end,
                   normalize=args.normalize,
                   framerate=args.framerate,
                   plot=args.plot,
                   model_input_plot=args.modelplot,
                   label_mask_files=args.labelfiles,
                   spectrogram_freq_cap=args.max_freq,
                   nfft=args.nfft
                   )
    # Keep charts up till user kills the windows:
    Plotter.block_till_figs_dismissed()
