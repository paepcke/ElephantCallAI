'''
Created on Feb 18, 2020

@author: paepcke
'''

import matplotlib.pyplot as plt
import numpy as np
import wave

#import scypi
#from scipy import signal

class Direction(enumerate):
    UP = 0
    DOWN = 1
    
class AmplitudeGater(object):
    '''
    classdocs
    '''

    #------------------------------------
    # Constructor 
    #-------------------    


    def __init__(self, wav_file_path, freq=50):
        '''
        Constructor
        '''
#         (time, amplitude) = self.make_sinewave(freq)
# 
#         self.plot(time, 
#                   amplitude,
#                   'Sine Wave',
#                   f"Number of 1/{freq} seconds",
#                   'Amplitude'
#                   )
        self.framerate = 8000
        (time, amplitude) = self.make_envelope(attack_length=50, 
                                               direction=Direction.DOWN)
        #****self.plot(time,amplitude, 'Envelope Up', 'time', 'amplitude')
        
        #release = self.mirror_curve(amplitude)
        self.plot(time,amplitude, 'Envelope Up', 'time (msec)', 'amplitude')
        return
        
        wave_obj = self.wave_fd(wav_file_path)
        
        num_frames = wave_obj.getnframes()
        sample_width = wave_obj.getsampwidth()
        self.framerate = wave_obj.getframerate()
        print(f"Framerate: {self.framerate}")
        print(f"Frames: {num_frames}")
        print(f"Sample width: {sample_width}")
        
        samples = self.read(wave_obj)
        normed_samples = self.normalize(samples)
        
        gated  = self.amplitude_gate(normed_samples, -20)
        print(gated)
        
        
    #------------------------------------
    # amplitude_gate
    #-------------------    
        
    def amplitude_gate(self, sample_npa, threshold_db):
        
        # Compute the threshold below which we
        # set amplitude to 0. It's -20dB of max
        # value. Note that for a normalized array
        # that max val == 1.0
        
        max_voltage = np.max(sample_npa)
        Vthresh = 10**(threshold_db/20 + 20*np.log(max_voltage))

        sample_npa[sample_npa < Vthresh] = 0
        signal_mask = np.where(sample_npa > 0, 1, 0)
        (attack,release) = self.create_envelope(50, self.framerate)
        print('foo')

    #------------------------------------
    # normalize
    #-------------------
    
    def normalize(self, samples):
        largest_val = np.max(samples)
        normed_samples = samples/largest_val
        return normed_samples    
    
    #------------------------------------
    # samples_from_framerate
    #-------------------
    
    def samples_from_framerate(self, time_msec, framerate):
        return round(time_msec * framerate / 1000.)
     
    #------------------------------------
    # make_envelope
    #-------------------
    
    def make_envelope(self, 
                      attack_length=50, 
                      target_amplitude=None, 
                      direction=Direction.UP):
        '''
        Return time/amplitude arrays that exponentially
        approach either 1 or 0, depending on direction. 
        Options are up from zero towards 1:
        
	1	          -
		       -
		    -
		  -
	0	 -
		 
		Or:
		
	1			   -                  
				      -
				        -
				         -
	0			          -
				   
		             
        Assumptions: 
              1. self.framerate contains framerate, so
                 that time per sample can be computed.
              2. the given amplitudes are normalized: between 0 and 1.
        
        Algorithm from https://www.music.mcgill.ca/~gary/307/week1/envelopes.html:
        
            y[n] = ay[n-1] + (1 -a) * target_amplitude
        
            where $a = e^{(-T/\tau)}$, T is the sample period, 
            and $\tau$ is a user provided time constant. 
        
        Tau was experimentally set to 0.001. A is set to 1.
        
        NOTE: if you will generate both up and down curves, just 
              call this method once in the UP direction, and call
              self.mirror_curve() with the result to get the other
              portion.
        
        @param attack_length: number of msecs to approach 1 from 0,
             or 0 from 1.
        @type attack_length: int
        @param target_amplitude: asymptode that amplitude should approach.
            Default: 1 if curve is Direction.UP, else 0
        @type target_amplitude: float
        @param direction: where the exponential curve should be headed.
            If 1, curve slopes up from 0 to 1.
        @type direction:
        '''
        #framerate = 8000
        
        if target_amplitude is None:
            target_amplitude = 1
        T = 1./self.framerate
        
        samples_per_msec = round(self.framerate/1000.)
        #********curve_width = attack_length * samples_per_msec
        curve_width = attack_length

        # Constant A controls gentleness: 0.5==>steep, 1==>gentle
        A = 1
        time = np.arange(1,curve_width + 1)

        tau = 0.001
        c = np.e**(-T/tau)
        #*****amplitude = np.zeros([curve_width])
        amplitude = np.zeros([curve_width])
        for n in time[:-1]:
            amplitude[n] = A*c * amplitude[n-1] + (1 - c) * target_amplitude

        # If direction is downwards, mirror it around
        # the vertical symmetry axis:
        
        if direction == Direction.DOWN:
            amplitude = self.mirror_curve(amplitude)
            
        return (time, amplitude)

    #------------------------------------
    # make_sinewave
    #-------------------    
        
    def make_sinewave(self, freq):
        time = np.arange(0,freq,0.1)
        amplitude = np.sin(time)
        return (time, amplitude)
    
    #------------------------------------
    # mirror_curve
    #-------------------    
    
    def mirror_curve(self, amplitudes):
        res = np.zeros(amplitudes.size)
        for indx in np.arange(amplitudes.size) + 1:
            res[indx - 1] = amplitudes[-indx]
        return res
    
    #------------------------------------
    # db_from_sample
    #-------------------    
    
    def db_from_sample(self, sample):
        return 20 * np.log10(sample)
    
    #------------------------------------
    # get_max_db
    #------------------

    def get_max_db(self, npa):
        
        max_val = npa.amax()
        max_db  = 20 * np.log10(max_val)
        return max_db

    #------------------------------------
    # read
    #-------------------    

    def read(self, wave_read_obj):
        '''
        Given an wave_read instance, return
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
    
    def plot(self, x_arr, y_arr, title='My Title', xlabel='X-axis', ylabel='Y-axis'):
        fig = plt.figure()
        ax  = fig.add_subplot(1,1,1)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.plot(x_arr,y_arr)
        
        
# --------------------------- Main -----------------------

if __name__ == '__main__':
    
    AmplitudeGater('/Users/paepcke/tmp/nn01c_20180311_000000.wav', 
                   freq=100)
    print('Done')