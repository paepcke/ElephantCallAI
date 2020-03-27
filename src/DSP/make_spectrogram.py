'''
Created on Feb 23, 2020

@author: paepcke
'''
import argparse
import math
import os
import sys

from scipy.io import wavfile

import matplotlib.pyplot as plt
import matplotlib.gridspec as grd
import numpy as np


class Spectrogrammer(object):
    '''
    classdocs
    '''
    NFFT = 4096 # was 3208 # We want a frequency resolution of 1.95 Hz
    HOP_LENGTH = 256
    FREQ_MAX = 100.
    OVERLAP = 50
    
    #------------------------------------
    # Constructor 
    #-------------------


    def __init__(self, 
                 wav_file_path, 
                 start_sec=0,
                 end_sec=None,
                 normalize=False
                 ):
        '''
        Constructor
        '''
        try:
            (framerate, samples) = wavfile.read(wav_file_path)
        except Exception as e:
            print(f"Cannot read .wav file: {repr(e)}")
            sys.exit(1)
    
        # Total time in seconds:
        audio_secs = math.floor(samples.size / framerate)
        
        # Seek in to start time, and cut all beyond stop time:
        if end_sec is None or end_sec > audio_secs:
            end_sec = audio_secs
        start_sample = framerate * start_sec
        stop_sample  = framerate * end_sec
        
        if normalize:
            samples_to_use = self.normalize(samples[start_sample : stop_sample])
        else:
            samples_to_use = samples
        self.plot_spectrogram(samples_to_use, 
                              framerate,
                              start_sec,
                              end_sec
                              )
        
    #------------------------------------
    # plot_spectrogram 
    #-------------------
        
    def plot_spectrogram(self, raw_audio, samplerate, start_sec, end_sec):

        # Extract the spectogram
        t = np.arange(start_sec, end_sec, 1/samplerate)
        _fig = plt.Figure()
        grid_spec = grd.GridSpec(nrows=2,
                                 ncols=1
                                 ) 
        ax_audio = plt.subplot(grid_spec[0])
        plt.xlabel('Time')
        plt.ylabel('Audio Units')
        ax_audio.plot(t, raw_audio)
        
        #ax_spectrum = plt.subplot(grid_spec[1])
        (spectrum, freqs, t_bins, im) = plt.specgram(raw_audio, 
                                                      Fs=samplerate,
                                                      cmap='jet'
                                            		  )
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
        new_features = 10*np.log10(spectrum)
        min_dbfs = new_features.flatten().mean()
        max_dbfs = new_features.flatten().mean()
        min_dbfs = np.maximum(new_features.flatten().min(),min_dbfs-2*new_features.flatten().std())
        max_dbfs = np.minimum(new_features.flatten().max(),max_dbfs+6*new_features.flatten().std())

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
        
        
# ---------------------------- Main ---------------------
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Quick Spectrogram from .wav"
                                     )

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

    parser.add_argument('wavefile',
                        help="Input .wav file"
                        )

    args = parser.parse_args();
    Spectrogrammer(args.wavefile,
                   args.start,
                   args.end,
                   args.normalize
                   )
    
