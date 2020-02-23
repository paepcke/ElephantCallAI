'''
Created on Feb 23, 2020

@author: paepcke
'''
import argparse
import os
import sys
import wave

from matplotlib import mlab as ml

import matplotlib.pyplot as plt
import numpy as np

class Spectrogrammer(object):
    '''
    classdocs
    '''

    #------------------------------------
    # Constructor 
    #-------------------


    def __init__(self, wav_file_path):
        '''
        Constructor
        '''
        try:
            wave_obj = self.wave_fd(wav_file_path)
        except Exception as e:
            print(f"Cannot read .wav file: {repr(e)}")
            sys.exit(1)
    
        samples_npa = self.read(wave_obj)
        framerate   = wave_obj.getframerate()
        (spectrum, freqs, times) = self.make_spectrogram(samples_npa, framerate)
        
        self.plot(times, 
                  freqs,
                  spectrum,
                  title=f"Spectrogram for {os.path.basename(wav_file_path)}" 
                  )
        
    #------------------------------------
    # make_spectrogram 
    #-------------------
        
    def make_spectrogram(self, 
                         raw_audio, 
                         samplerate,
                         start=0,    # secs
                         stop=10,    # secs
                         NFFT=3208,
                         hop=641
                         ):

        start_sample = samplerate * start
        stop_sample  = samplerate * stop
        # Extract the spectogram
        [spectrum, freqs, t] = ml.specgram(raw_audio[start_sample: stop_sample], 
                                           NFFT=NFFT, 
                                           Fs=samplerate, 
                                           noverlap=(NFFT - hop), 
                                           window=ml.window_hanning)

        return (spectrum,freqs,t)

    #------------------------------------
    # read
    #-------------------    

    def read(self, wave_read_obj):
        '''
        Given a wave_read instance, return
        a numpy array of 16bit int samples
        of the entire file. Must fit in memory!
        
        @param wave_read_obj: result of having opened
            a file using wave.open()
        @type wave_read_obj: wave_read instance
        @return: numpy array of samples
        @rtype: narray(dtype=int16)
        '''
        
        num_frames          = wave_read_obj.getnframes()
        byte_arr            = wave_read_obj.readframes(num_frames)
        samples_readonly    = np.frombuffer(byte_arr, np.uint16)
        samples = np.copy(samples_readonly)
        return samples

    #------------------------------------
    # wave_fd
    #-------------------    
        
    def wave_fd(self, file_path):
        return wave.open(file_path, 'rb')

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
        ax.imshow(new_features, 
                  cmap="magma_r", 
                  vmin=min_dbfs, 
                  vmax=max_dbfs, 
                  interpolation='none', 
                  origin="lower", 
                  aspect="auto"
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

#     parser.add_argument('-NFFT', 
#                         type=int, 
#                         default=3208, 
#                         help='Window size used for creating spectrograms')
# 
#     parser.add_argument('-hop', 
#                         type=int, 
#                         default=641, 
#                         help='Hop size used for creating spectrograms')
# 
#     parser.add_argument('-w', '--window', 
#                         type=int, 
#                         default=10, 
#                         help='Determines the window size in seconds of the resulting spectrogram')

    parser.add_argument('--wavefile',
                        help="Input .wav file"
                        )

    args = parser.parse_args();
    Spectrogrammer(args.wavefile)
    
