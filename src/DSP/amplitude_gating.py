#!/usr/bin/env python
'''
Created on Feb 18, 2020

@author: paepcke

Given a .wav file, read it, and normalize it
so max voltage is 1. Then set all voltages that 
are less than -20dB of the maximum voltage to zero.
However, to avoid the 0->non-zero transitions to
be impulse responses that will generate energy in many
frequencies, create exponential attack and release envolopes
for each transition between zero and non-zero, and non-zero
to zero, respectively.

Data structures for efficiently tracking bursts of non-zero voltages: 

   sample_npa       : a numpy array of voltages from the .wav file,
                      after normalization
                      All small amplitudes have been set to zero
    signal_index    : an array of pointers into the sample_npa. At
                      the pointed-to places in sample_npa the voltages
                      are non-zero.
    pt_into_index   : pointer into the signal_index
    
Example:
     sample_npa   : array([1,0,0,0,4,5,6,0,10])
     signal_index : array([0,4,5,6,8])
     pt_into_index: 4 points to the 8 in signal_index, and thus refers to
                    the 10V in the sample_npa    
'''

import argparse
import datetime
import math
import os
import sys

from scipy.io import wavfile
from scipy.signal import butter, lfilter 
from scipy.signal.filter_design import freqz

from elephant_utils.logging_service import LoggingService
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
class AmplitudeGater(object):
    '''
    classdocs
    '''
    #------------------------------------
    # Constructor 
    #-------------------    


    def __init__(self,
                 wav_file_path,
                 outfile=None,
                 amplitude_cutoff=-5,   # dB of peak
                 cutoff_freq=100,
                 normalize=True,
                 plot_result=False,
                 logfile=None,
                 framerate=None,  # Only used for testing.
                 testing=False
                 ):
        '''

        @param wav_file_path: path to .wav file to be gated
            Can leave at None, if testing is True
        @type wav_file_path: str
        
        @param outfile: where gated, normalized .wav will be written.
        @type outfile: str
        
        @param amplitude_cutoff: dB attenuation from maximum
            amplitude below which voltage is set to zero
        @type amplitude_cutoff: int
        
        @param framerate: normally extracted from the .wav file.
            Can be set here for testing. Samples/sec
        @type framerate: int

        @param logfile: file where to write logs; Default: stdout
        @type logfile: str

        @param logging_period: number of seconds between reporting
            envelope placement progress.
        @type logging_period: int
        
        @param plot_result: whether or not to create a plot of the 
            new .wav signals.
        @type plot_result: False
        
        @param testing: whether or not unittests are being run. If
            true, __init__() does not initiate any action, allowing
            the unittests to call individual methods.
        @type testing: bool
        '''

        # Make sure the outfile can be opened for writing,
        # before going into lengthy computations:
        
        #*******************
        #amplitude_cutoff = -25
        #*******************
        if outfile is not None:
            try:
                with open(outfile, 'wb') as _fd:
                    pass
            except Exception as e:
                print(f"Outfile cannot be access for writing; doing nothing: {repr(e)}")
                sys.exit(1)

        self.plot_result = plot_result
                
        AmplitudeGater.log = LoggingService(logfile=logfile)
        
        # For testing; usually framerate is read from .wav file:
        self.framerate = framerate

        if not testing:
            try:
                self.log.info("Reading .wav file...")        
                (self.framerate, samples) = wavfile.read(wav_file_path)
                self.log.info("Done reading .wav file.")        
            except Exception as e:
                print(f"Cannot read .wav file: {repr(e)}")
                sys.exit(1)
        
        if testing:
            self.recording_length_hhmmss = "<unknown>"
        else:
            num_samples = samples.size
            recording_length_secs = num_samples / self.framerate
            self.recording_length_hhmmss = str(datetime.timedelta(seconds = recording_length_secs))

        self.samples_per_msec = round(self.framerate/1000.)
        
        if testing:
            return

        samples_float = samples.astype(float)
        # Normalize:
        if normalize:
            normed_samples = self.normalize(samples_float)
        else:
            normed_samples = samples_float.copy()
         
        # Noise gate: Chop off anything with amplitude above amplitude_cutoff:
        gated_samples  = self.amplitude_gate(normed_samples, 
                                             amplitude_cutoff, 
                                             cutoff_freq=cutoff_freq)
 
        # Result back to int16:
        gated_samples = gated_samples.astype(np.int16)
               
        if outfile is not None and not testing:
            # Write out the result:
            wavfile.write(outfile, self.framerate, gated_samples)
        
        if self.plot_result:
            # Find a series of 100 array elements where at least
            # the first is not zero. Just to show an interesting
            # area, not a flat line. The nonzero() function returns
            # a *tuple* of indices where arr is not zero. Therefore
            # the two [0][0] to get the first non-zero:

            start_indx = self.find_busy_array_section(gated_samples)
            end_indx   = start_indx + 100
            
            self.log.info(f"Plotting a 100 long series of result from {start_indx}...")
            self.plot(np.arange(start_indx, end_indx),
                      gated_samples[start_indx:end_indx],
                      title=f"Amplitude-Gated {os.path.basename(wav_file_path)}",
                      xlabel='Sample Index', 
                      ylabel='Voltage'
                      )
        
        print('Done')
        
        
    #------------------------------------
    # amplitude_gate
    #-------------------    
        
    def amplitude_gate(self, sample_npa, threshold_db, order=1, cutoff_freq=100):

        # Don't want to open the gate *during* a burst.
        # So make a low-pass filter to only roughly envelops

        self.log.info("Taking abs val of values...")
        samples_abs = np.abs(sample_npa)
        self.log.info("Done taking abs val of values.")
        
        self.log.info("Applying low pass filter...")
        envelope = self.butter_lowpass_filter(samples_abs, cutoff_freq, order)
        self.log.info("Done applying low pass filter.")        

        #**********************
#         self.over_plot(samples_abs[1000:1100], 'ABS(samples)')
#         self.over_plot(envelope[1000:1100], f"Env Order {order}")
#         
#         order = 3
#         envOrd3 = self.butter_lowpass_filter(samples_abs, cutoff_freq, order)
#         self.over_plot(envOrd3[1000:1100], 'Order 3')
# 
#         order = 1
#         envOrd1 = self.butter_lowpass_filter(samples_abs, cutoff_freq, order)
#         self.over_plot(envOrd1[1000:1100], 'Order 1')
        
        #**********************

        # Compute the threshold below which we
        # set amplitude to 0. It's threshold_db of max
        # value. Note that for a normalized array
        # that max val == 1.0

        max_voltage = np.amax(envelope)
        self.log.info(f"Max voltage: {max_voltage}")
        
        # Compute threshold_db of max voltage:
        Vthresh = max_voltage * 10**(threshold_db/20)
        self.log.info(f"Cutoff threshold amplitude: {Vthresh}")

        # Zero out all amplitudes below threshold:
        self.log.info("Zeroing sub-threshold values...")

        sample_npa[abs(sample_npa) < Vthresh] = 0

        return sample_npa

    #------------------------------------
    # butter_lowpass_filter
    #-------------------    

    def butter_lowpass_filter(self, data, cutoff, order=5):
        b, a = self.get_butter_lowpass_parms(cutoff, order=order)
        y = lfilter(b, a, data)
        
        if self.plot_result:
            self.plot_frequency_response(b, a, cutoff, order)
        
        return y

    #------------------------------------
    # get_butter_lowpass_parms
    #-------------------    

    def get_butter_lowpass_parms(self, cutoff, order=5):
        nyq = 0.5 * self.framerate
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    #------------------------------------
    # suppress_small_voltages
    #-------------------
    
    def suppress_small_voltages(self, volt_vec, thresh_volt, padding_secs):
        '''
        Given an array of numbers, set all elements smaller than
        thres_volt to zero. But: leave padding array elements before
        and after each block of new zeroes alone.
        
        Return the resulting array of same length as volt_vec.
        
        Strategy: 
           o say volt_vec == array([1, 2, 3, 4, 5, 6, 7, 8, 0, 1])
                 thres_volt == 5
                 padding == 2    # samples

           o to not worry about out of bound index, add padding zeros
             to the voltage vector:
             
                 padded_volt_vec = np.hstack((volt_vec, np.zeros(2).astype(int))) 
                     ==> array([1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 0, 0])
                     
           o create an array of indexes that need to be set to 0,
             because voltages at their location exceed thres_volt.
             The np.nonzero returns a one-tuple, therefore the [0]
             
                 indexes_to_zero = np.nonzero(a>5)[0]
                     ==> (array([5, 6, 7]),)

           o we need to respect the padding ahead of the elements 
             to zero. So add padding samples to each index:
             
                 indexes_to_zero = indexes_to_zero + 2

           o 
                 
        @param volt_vec:
        @type volt_vec:
        @param thresh_volt:
        @type thresh_volt:
        @param padding_secs:
        @type padding_secs:
        '''
        
        padding = self.samples_from_secs(padding_secs)

        # Get a mask with True where we will zero out the voltage:
        volt_mask = volt_vec < thresh_volt
        
        pt_next_mask_pos = 0
        while True:
            (volt_mask, pt_next_mask_pos) = self.narrow_mask_segment(volt_mask,
                                                                    pt_next_mask_pos, 
                                                                    padding
                                                                    )
            if pt_next_mask_pos is None:
                # Got a finished mask with padding.
                break

        # Do the zeroing
        volt_vec_zeroed = ma.masked_array(volt_vec,
                                          volt_mask
                                          ).filled(0)
        return volt_vec_zeroed
    
    #------------------------------------
    # narrow_mask_segment
    #------------------- 
    
    def narrow_mask_segment(self, mask, ptr_into_mask, padding):
        
        # Erroneous args or end of mask:
        mask_len = mask.size
        if ptr_into_mask >= mask_len:
            # None ptr to indicate end:
            return (mask, None)
        
        zeros_start_ptr = ptr_into_mask
        
        # Find next Truth value in mask, i.e.
        # the start of a zeroing sequence
        while zeros_start_ptr < mask_len and not mask[zeros_start_ptr]:
            zeros_start_ptr += 1
            
        # Pointing to the first True (first zeroing index)
        # after a series of False, or end of mask:
        if zeros_start_ptr >= mask_len:
            return (mask, None)
        
        # Find end of the zeroing sequence (i.e. True vals in mask):
        zeros_end_ptr = zeros_start_ptr
        while zeros_end_ptr < mask_len and mask[zeros_end_ptr]:
            zeros_end_ptr += 1

        # Is the zeroing sequence long enough to accommodate
        # padding to its left?
        zeros_len = zeros_end_ptr - zeros_start_ptr
        
        if zeros_len < padding:
            # Just don't zero at all for this seq:
            mask[zeros_start_ptr:zeros_end_ptr] = False
        else:
            # Don't zero padding samples:
            mask[zeros_start_ptr : min(zeros_start_ptr + padding, mask_len)] = False    
        
        # New start of zeroing seq: in steady state
        # it's just the start pt moved right by the amount
        # of padding. But the burst of zeroing was too narrow,
        # 
        zeros_start_ptr = min(zeros_start_ptr + padding,
                              zeros_end_ptr
                              )
         
        # Same at the end: Stop zeroing a bit earlier than
        # where the last below-threshold element sits:
        
        zeros_len = zeros_end_ptr - zeros_start_ptr
        if zeros_len <= padding:
            # Just don't do any zeroing:
            mask[zeros_start_ptr : zeros_end_ptr] = False
        else:
            # Just stop zeroing a bit earlier
            mask[zeros_end_ptr - padding : zeros_end_ptr] = False    
            zeros_end_ptr = zeros_end_ptr - padding

        return (mask, zeros_end_ptr)

    #------------------------------------
    # normalize
    #-------------------
    
    def normalize(self, samples):
        '''
        Make audio occupy the maximum dynamic range
        of int16: -2**15 to 2**15 - 1 (-32768 to 32767)
        
        Formula to compute new Intensity of each sample:

           I = ((I-Min) * (newMax - newMin)/Max-Min)) + newMin

        @param samples: samples from .wav file
        @type samples: np.narray('int16')
        @result: a new np array with normalized values
        @rtype: np.narray('int16')
        '''
        new_min = -2**15       # + 10  # Leave a little bit of room with min val of -32768
        new_max = 2**15        # - 10   # same for max:
        min_val = np.amin(samples)
        max_val = np.amax(samples)
        
        self.log.info("Begin normalization...")
        
        normed_samples = ((samples - min_val) * (new_max - new_min)/(max_val - min_val)) + new_min
        
        # Or, using scikit-learn:
        #   normed_samples = preprocessing.minmax_scale(samples, feature_range=[new_min, new_max])

        self.log.info("Done normalization.")
        return normed_samples    

    #------------------------------------
    # make_sinewave
    #-------------------    
        
    def make_sinewave(self, freq):
        time = np.arange(0,freq,0.1)
        amplitude = np.sin(time)
        return (time, amplitude)
    
    #------------------------------------
    # db_from_sample
    #-------------------    
    
    def db_from_sample(self, sample):
        return 20 * np.log10(sample)
    
    #------------------------------------
    # samples_from_msecs
    #-------------------
    
    def samples_from_msecs(self, msecs):
        
        return msecs * self.samples_per_msec
    
    #------------------------------------
    # samples_from_secs
    #-------------------
    
    def samples_from_secs(self, secs):
        '''
        Possibly fractional seconds turned into
        samples. Fractional seconds are rounded up.
         
        @param secs: number of seconds to convert
        @type secs: {int | float}
        @return: number of corresponding samples
        @rtype: int
        '''
        
        return math.ceil(secs * self.framerate)
    
    #------------------------------------
    # msecs_from_samples 
    #-------------------
    
    def msecs_from_samples(self, num_samples):
        
        return num_samples * self.samples_per_msec
    
    
    #------------------------------------
    # get_max_db
    #------------------

    def get_max_db(self, npa):
        
        max_val = npa.amax()
        max_db  = 20 * np.log10(max_val)
        return max_db

    #------------------------------------
    # export_snippet
    #-------------------    

    def export_snippet(self, samples, start_sample, end_sample, filename, to_int16=True):
        '''
        Write part of the samples to a two-col CSV.
        
        @param samples: sample array
        @type samples: np.array
        @param start_sample: index of first sample to export
        @type start_sample: int
        @param end_sample: index of sample after the last one exported
        @type end_sample: int
        @param filename: output file name
        @type filename: str
        @param to_int16: whether or not to convert samples to 16 bit signed int before writing
        @type to_int16: bool
        '''
        
        snippet = samples[start_sample : end_sample]
        if to_int16:
            snippet = snippet.astype(np.int16)
        with open(filename, 'w') as fd:
            for (indx, val) in enumerate(snippet):
                fd.write(f"{indx},{val}\n")
        
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
    
    #------------------------------------
    # plot_frequency_response
    #-------------------
    
    def plot_frequency_response(self, b, a, cutoff, order):
        '''
        b,a come from call to get_butter_lowpass_parms()
        
        @param b:
        @type b:
        @param a:
        @type a:
        @param cutoff: frequency (Hz) at which filter is supposed to cut off
        @type cutoff: int
        @param order: order of the filter: 1st, 2nd, etc.; i.e. number of filter elements.
        @type order: int
        '''
    
        w, h = freqz(b, a, worN=8000)
        plt.subplot(1, 1, 1)
        plt.plot(0.5 * self.framerate * w/np.pi, np.abs(h), 'b')
        plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
        plt.axvline(cutoff, color='k')
        plt.xlim(0, 0.5*self.framerate)
        plt.title(f"Lowpass Filter Frequency Response (order: {order}, cutoff: {cutoff} Hz")
        plt.xlabel('Frequency [Hz]')
        plt.grid()

    #------------------------------------
    # over_plot
    #-------------------    
    
    def over_plot(self, y_arr, legend_entry, title="Filters", xlabel="Time", ylabel='Voltage'):

        try:
            self.ax
        except AttributeError:
            fig = plt.figure()
            self.ax  = fig.add_subplot(1,1,1)
            self.ax.set_title(title)
            self.ax.set_xlabel(xlabel)
            self.ax.set_ylabel(ylabel)
            
        self.ax.plot(np.arange(y_arr.size),
                     y_arr,
                     label=legend_entry
                     )
        self.ax.legend()

    #------------------------------------
    # find_busy_array_section
    #-------------------    

    def find_busy_array_section(self, arr):
        
        non_zeros = np.nonzero(arr)[0]
        for indx_to_non_zero in non_zeros:
            if arr[indx_to_non_zero] > 0 and\
                arr[indx_to_non_zero + 1] > 0 and\
                arr[indx_to_non_zero + 2] > 0:
                return indx_to_non_zero
            
        # Nothing found, return start of array:
        return 0

        
# --------------------------- Burst -----------------------

class Burst(object):

    def __init__(self):
        '''
        
        @param start:
        @type start:
        @param stop:
        @type stop:
        @param attack_start:
        @type attack_start:
        @param release_start:
        @type release_start:
        @param: averaged_value:
        @type: averaged_value:
        @param averaged_value:
        @type averaged_value:
        '''
        self._start           = None
        self._stop            = None
        self._padding_start    = None
        self._release_start   = None
        self._averaging_start = None
        self._averaging_stop  = None
        self.signal_index_pt  = None

    @property
    def start(self):
        return self._start
    @start.setter
    def start(self, val):
        self._start = val

    @property
    def stop(self):
        return self._stop
    @stop.setter
    def stop(self, val):
        self._stop = val
        
    @property
    def attack_start(self):
        return self._padding_start
    
    @attack_start.setter
    def attack_start(self, val):
        self._padding_start = val

    @property
    def release_start(self):
        return self._release_start
    @release_start.setter
    def release_start(self, val):
        self._release_start = val

    @property
    def averaging_start(self):
        return self._averaging_start
    @averaging_start.setter
    def averaging_start(self, val):
        self._averaging_start = val


    @property
    def averaging_stop(self):
        return self._averaging_stop
    @averaging_stop.setter
    def averaging_stop(self, val):
        self._averaging_stop = val

    @property
    def signal_index_pt(self):
        return self._signal_index_pt
    @signal_index_pt.setter
    def signal_index_pt(self, val):
        self._signal_index_pt = val


# --------------------------- Main -----------------------

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Apply amplitude filter to a given .wav file"
                                     )

    parser.add_argument('-l', '--logfile',
                        help='fully qualified log file name to which info and error messages \n' +\
                             'are directed. Default: stdout.',
                        dest='logfile',
                        default=None);

    parser.add_argument('-c', '--cutoff',
                        help='dB attenuation from max amplitude below which signal \n' +\
                            'is set to zero; default: -20dB',
                        type=int,
                        default='-20'
                        )
    
    parser.add_argument('-f', '--filter',
                        help='highest frequency to keep.',
                        type=int,
                        default=10);
    
    parser.add_argument('-p', '--padding',
                        help='seconds to keep before/after events; default: 5',
                        type=int,
                        default=5);
               
    parser.add_argument('-r', '--raw',
                        action='store_true',
                        default=False,
                        help="Set to prevent amplitudes to be  normalized to range from -32k to 32k; default is to normalize"
                        )
                        
    parser.add_argument('-t', '--plot',
                        action='store_true',
                        default=False,
                        help="Whether or not to plot result."
                        )

    parser.add_argument('wavefile',
                        help="Input .wav file"
                        )
    parser.add_argument('outfile',
                        help="Path to where result .wav file will be written; if None, nothing other than plotting is output",
                        default=None
                        )
    
    args = parser.parse_args();

    cutoff = args.cutoff
    if cutoff >= 0:
        print(f"Amplitude cutoff must be negative, not {cutoff}")
        sys.exit(1)

    # AmplitudeGater('/Users/paepcke/tmp/nn01c_20180311_000000.wav',
    #                plot_result=True)
    AmplitudeGater(args.wavefile,
                   args.outfile,
                   amplitude_cutoff=cutoff,
                   filter=args.filter,
                   padding=args.padding,
                   normalize=not args.raw,
                   plot_result=args.plot,
                   logfile=args.logfile,
                   )

    sys.exit(0)
