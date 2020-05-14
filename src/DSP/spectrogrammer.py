'''
Created on Feb 23, 2020

@author: paepcke
'''
import argparse
import math
import os
import sys

from scipy.io import wavfile
from scipy.signal.spectral import stft, istft

import matplotlib.gridspec as grd
import matplotlib.pyplot as plt
import numpy as np
from elephant_utils.logging_service import LoggingService

sys.path.append(os.path.dirname(__file__))
from plotting.plotter import Plotter


class Spectrogrammer(object):
    '''
    Manipulate spectrograms. Includes starting
    from a .wav file, or an already prepared .npy file. Allows
    spectrograms to be prepared from segments of a .wav
    file, as specified by start and end time.
    
    Several frequency filters are available.
    '''
    NFFT = 4096 # was 3208 # We want a frequency resolution of 1.95 Hz
    HOP_LENGTH = 800,
    FREQ_MAX = 150.
    OVERLAP = 50
    # Energy below which spectrogram is set to zero:
    ENERGY_SUPPRESSION = -50 # dBFS
    
    #------------------------------------
    # Constructor 
    #-------------------


    def __init__(self, 
                 infiles,
                 clean_spectrogram,
                 max_f,
                 start_sec=0,
                 end_sec=None,
                 normalize=False,
                 framerate=8000,
                 plot=True,
                 nfft=None,
                 ):
        '''
        Constructor
        '''
        
        self.log = LoggingService()

        self.framerate = framerate
        if nfft is None:
            self.log.info(f"Assuming NFFT == {Spectrogrammer.NFFT}")
            nfft = Spectrogrammer.NFFT
            
        self.plotter = Plotter(framerate)

        for infile in infiles:
            if not os.path.exists(infile):
                print(f"File {infile} does not exist.")
                continue
    
            if infile.endswith('.wav'):
                try:
                    self.log.info(f"Reding wav file {infile}...")
                    (self.framerate, samples) = wavfile.read(infile)
                    self.log.info(f"Processing wav file {infile}...")
                    (freq_labels, time_labels, spect) = \
                        self.process_wav_file(samples, 
                                              self.framerate, 
                                              start_sec, 
                                              end_sec, 
                                              normalize,
                                              plot)
                        
                except Exception as e:
                    print(f"Cannot read .wav file: {repr(e)}")
                    sys.exit(1)
            else:
                # Infile is a .npy spectrogram file:
                self.log.info(f"Loading spectrogram file {infile}...")
                spect = np.load(infile)
                self.log.info(f"Done loading spectrogram file {infile}.")
                self.log.info("Computing freq and time ticks...")
                (freq_labels, time_labels) = self.make_time_freq_seqs(max_f, spect)
                
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

    #------------------------------------
    # process_wav_file 
    #-------------------
    
    def process_wav_file(self,
                         samples,
                         start_sec,
                         end_sec,
                         normalize,
                         plot=True
                         ):

        # Total time in seconds:
        audio_secs = math.floor(samples.size / self.framerate)
        
        # Seek in to start time, and cut all beyond stop time:
        if end_sec is None or end_sec > audio_secs:
            end_sec = audio_secs
        start_sample = self.framerate * start_sec
        stop_sample  = self.framerate * end_sec
        
        if normalize:
            samples_to_use = self.normalize(samples[start_sample : stop_sample])
        else:
            samples_to_use = samples
            
        (freq_labels, time_labels, freq_time) = \
            self.make_spectrogram(samples_to_use)
            
        if plot:
            (_spectrum,_freqs,_t_bins,_im) = \
                self.plot_spectrogram_from_audio(samples_to_use, 
                                      self.framerate,
                                      start_sec,
                                      end_sec
                                      )
        return (freq_labels, time_labels, freq_time)

    #------------------------------------
    # process_spectrogram 
    #-------------------
    
    def process_spectrogram(self, spect):
        
        max_energy = np.amax(spect)
        db_cutoff  = Spectrogrammer.ENERGY_SUPPRESSION # dB
        low_energy_thres = max_energy * 10**(db_cutoff/20)
        spect_low_energy_indices = np.nonzero(spect <= low_energy_thres)
        spect[spect_low_energy_indices] = 0.0

        #delete all but row <= 60?
        #take the log
        return spect

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
        
        self.log.info("Creating spectrogram...")
        (freq_labels, time_labels, complex_freq_by_time) = \
            stft(data, 
                 self.framerate, 
                 nperseg=self.FFT_WIDTH
                 #nperseg=int(self.framerate)
                 )
            
        self.log.info("Done creating spectrogram.")
        
        freq_time = np.absolute(complex_freq_by_time)
        return (freq_labels, time_labels, freq_time)

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
    # plot_spectrogram_from_audio 
    #-------------------
        
    def plot_spectrogram_from_audio(self, raw_audio, samplerate, start_sec, end_sec, plot):

        (spectrum, freqs, t_bins, im) = plt.specgram(raw_audio, 
                                                     Fs=samplerate,
                                                     cmap='jet'
                                            		 )
        if plot:
            t = np.arange(start_sec, end_sec, 1/samplerate)
            _fig = plt.Figure()
            grid_spec = grd.GridSpec(nrows=2,
                                     ncols=1
                                     ) 
            ax_audio = plt.subplot(grid_spec[0])
            plt.xlabel('Time')
            plt.ylabel('Audio Units')
            ax_audio.plot(t, raw_audio)
            
            plt.show()
        return (spectrum,freqs,t_bins,im)

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

    parser.add_argument('--max_f', 
                        dest='max_freq', 
                        type=int, 
                        default=Spectrogrammer.FREQ_MAX, 
                        help='Deterimes the maximum frequency band in spectrogram')
    
    parser.add_argument('--pad', dest='pad_to', type=int, default=4096, 
                        help='Deterimes the padded window size that we ' +
                             'want to give a particular grid spacing (i.e. 1.95hz')

    parser.add_argument('-f', '--framerate',
                        type=int,
                        default=8000,
                        help='Framerate at which original .wav file was recorded; def: 8000')

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


    parser.add_argument('-o', '--outfile', 
                        default=None, 
                        help='Outfile for cleaned spectrogram; only needed with -c'
                        )

    parser.add_argument('infiles',
                        nargs='+',
                        help="Input .wav/.npy file(s)"
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
                   nfft=args.nfft
                   )
    # Keep charts up till user kills the windows:
    Plotter.block_till_figs_dismissed()
