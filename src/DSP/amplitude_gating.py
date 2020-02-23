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
import time
import wave

from elephant_utils.logging_service import LoggingService
import matplotlib.pyplot as plt
import numpy as np


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


class Direction(enumerate):
    UP = 0
    DOWN = 1
    
class AmplitudeGater(object):
    '''
    classdocs
    '''

    # Duratin of attack and release envelopes
    ATTACK_RELEASE_MSECS = 50 # milliseconds
    
    # Number of samples in attack and release envelopes:
    ATTACK_RELEASE_SAMPLES = None # Filled in after framerate is known

    #------------------------------------
    # Constructor 
    #-------------------    


    def __init__(self, 
                 wav_file_path,
                 outfile=None,
                 amplitude_cutoff=-20,  # dB
                 attack_release_duration_msecs=None, # default: ATTACK_RELEASE_MSECS 
                 framerate=None,
                 logfile=None,
                 logging_period=10, # seconds
                 plot_result=False,
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
        
        @param attack_release_duration_msecs: duration of the 
            exponential attack or release envelope that is placed
            before and after each signal burst (signals above threshold)
        @type attack_release_duration_msecs: int
        
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
        
        if outfile is not None:
            try:
                with open(outfile, 'wb') as _fd:
                    pass
            except Exception as e:
                print(f"Outfile cannot be access for writing; doing nothing: {repr(e)}")
                sys.exit(1)
                
        AmplitudeGater.log = LoggingService(logfile=logfile)
        self.logging_period = logging_period
        
        # Mostly for testing; usually framerate is
        # read from .wav file:
        self.framerate = framerate

        # Customize width of attack and release?
        if attack_release_duration_msecs is not None:
            AmplitudeGater.ATTACK_RELEASE_MSECS = attack_release_duration_msecs
            
        if not testing:
            try:
                wave_obj = self.wave_fd(wav_file_path)
            except Exception as e:
                print(f"Cannot read .wav file: {repr(e)}")
                sys.exit(1)
        
        # For .wav files the sample width is 2,
        # i.e. 16 bit unsigned voltage readings:
        # sample_width = wave_obj.getsampwidth()

        if testing:
            self.recording_length_hhmmss = "<unknown>"
        else:
            self.framerate = wave_obj.getframerate()
            num_samples = wave_obj.getnframes()
            recording_length_secs = round(self.framerate / num_samples)
            self.recording_length_hhmmss = str(datetime.timedelta(seconds = recording_length_secs))

            
        self.samples_per_msec = round(self.framerate/1000.)
        AmplitudeGater.ATTACK_RELEASE_SAMPLES = self.ATTACK_RELEASE_MSECS * self.samples_per_msec
        
        #  print(f"Framerate: {self.framerate}")
        #  print(f"Frames: {num_frames}")
        #  print(f"Sample width: {sample_width}")
        
        if testing:
            return

        self.log.info("Reading .wav file...")        
        samples = self.read(wave_obj)
        self.log.info("Done reading .wav file.")        
        wave_obj.close()
        
        normed_samples = self.normalize(samples)
        gated_samples  = self.amplitude_gate(normed_samples, amplitude_cutoff)
        
        if outfile is not None and not testing:
            # Write out the result:
            self.write_wav(gated_samples, outfile)
        
        if plot_result:
            self.plot(np.arange(gated_samples.size),
                      gated_samples, 
                      title=f"Amplitude-Gated {os.path.basename(wav_file_path)}",
                      xlabel='Sample Index', 
                      ylabel='Voltage'
                      )
        
        print('Done')
        
        
    #------------------------------------
    # amplitude_gate
    #-------------------    
        
    def amplitude_gate(self, sample_npa, threshold_db):
        
        # Compute the threshold below which we
        # set amplitude to 0. It's threshold_db of max
        # value. Note that for a normalized array
        # that max val == 1.0
        
        max_voltage = np.max(sample_npa)
        self.log.info(f"Max voltage: {max_voltage}")
        
        # Compute threshold_db of max voltage:
        Vthresh = 10**(threshold_db/20 + 20*np.log(max_voltage))
        self.log.info(f"Cutoff threshold amplitude: {Vthresh}")

        # Zero out all amplitudes below threshold:
        self.log.info("Zeroing sub-threshold values...")
        sample_npa[sample_npa < Vthresh] = 0
        self.log.info("Done zeroing sub-threshold values.")
        
        # Get indexes of all non-null samples:
     
        # Two choices: indexes of where signial is >0:
        #  a : array([1, 0, 0, 4, 5, 6, 5, 4, 0, 0, 0])
        #  >>> a.nonzero()
        #  ==>  (array([0, 3, 4, 5, 6, 7]),)
        # Or: 1/0 where signal is >0/0:
        # >>> np.where(a>0, 1, 0)
        # ==> array([1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0])

        # The [0] is b/c .nonzero() returns a two tuple
        # with the result in slot 0, and nothing beyond:
        self.log.info("Building index to non-zero values...")
        self.signal_index = sample_npa.nonzero()[0]
        self.log.info("Done building index to non-zero values.")

        # Not yet a 'previous signal burst' object:
        prev_burst = None

        # Will log progress every x seconds:        
        next_report_due = self.logging_period
        
        self.log.info("Begin envelope creation...")
        
        while True:
            burst = self.next_burst(prev_burst)
            if burst is None:
                self.log.info("Last burst processed.")
                break
            # If attack_start is not None, we have enough
            # zero samples before burst to do a ramp-up attack
            # envelope:
            if burst.attack_start is None:
                sample_npa = self.average_the_gap(burst, sample_npa)
            else:
                sample_npa = self.place_attack(burst, sample_npa)
            
            # Is there room for a release?
            # If sig indx is None, we reached the
            # end of the signal_index. So there is
            # definitely no room for a release:
        
            if burst.signal_index_pt is not None:
                num_zeros_to_next_burst = self.signal_index[burst.signal_index_pt] - burst.stop
                if num_zeros_to_next_burst >= self.ATTACK_RELEASE_SAMPLES:
                    # Yes, there is room:
                    sample_npa = self.place_release(burst, sample_npa)

            prev_burst = burst
            # Time to report progress?
            curr_time =  math.floor(time.time())
            if  curr_time > next_report_due:
                secs_into_recording = round(burst.start / self.framerate)
                secs_into_hhmmss = str(datetime.timedelta(seconds = secs_into_recording))
                self.log.info(f"Secs into recording: {secs_into_hhmmss} of {self.recording_length_hhmmss}")
                next_report_due = curr_time + self.logging_period

        return sample_npa   
    #------------------------------------
    # place_attack 
    #-------------------

    def place_attack(self, burst, sample_npa):
        '''
        Modify sample_npe to have an attack envelope
        starting at burst.attack_start. Assumes that
        caller has ensured that the attack width number
        of samples before burst.start may be clobbered with
        the envelope samples.
        
        @param burst:
        @type burst:
        @param sample_npa:
        @type sample_npa:
        @return: modified sample_npa
        @rtype: np.array([float])
        '''
        # Target of attack is the first non-zero voltage:
        envelope_asymptode = sample_npa[burst.start]
        envelope = self.make_envelope(attack_length=self.ATTACK_RELEASE_MSECS, 
                                      target_amplitude=envelope_asymptode, 
                                      direction=Direction.UP)
        sample_npa[burst.attack_start: burst.attack_start + self.ATTACK_RELEASE_SAMPLES] = envelope
        return sample_npa
    
    #------------------------------------
    # place_release 
    #-------------------

    def place_release(self, burst, sample_npa):
        '''
        Modify sample_npa to have a release envelope
        starting at burst.stop. Assumes that
        caller has ensured that the release width number
        of samples after burst.stop may be clobbered with
        the envelope samples.
        
        @param burst:
        @type burst:
        @param sample_npa:
        @type sample_npa:
        @return: modified sample_npa
        @rtype: np.array([float])
        '''
        envelope = self.make_envelope(attack_length=self.ATTACK_RELEASE_MSECS,
                                      target_amplitude=0.01, #******??? 
                                      direction=Direction.UP)
        # Turn the attack envelope into a symmetric release:
        envelope = self.mirror_curve(envelope)
        sample_npa[burst.stop : burst.stop + self.ATTACK_RELEASE_SAMPLES] = envelope
        return sample_npa

    #------------------------------------
    # average_the_gap 
    #-------------------

    def average_the_gap(self, burst, sample_npa):
        '''
        If two bursts are closer together than the width
        of an attack/release envelope, we bridge the two
        bursts by a constant voltage: the average of the 
        first burst's end, and the second burst's start.
        
        @param burst:
        @type burst:
        @return: modified sample_npa
        @rtype: np.array([float])
        '''
        
        end_voltage_burst1   = sample_npa[burst.averaging_start]
        start_voltage_burst2 = sample_npa[burst.averaging_stop]
        avg_voltage = round((end_voltage_burst1 + start_voltage_burst2) / 2.)
        # The '1 +' below starts filling the average
        # into the first sample *after* the last non-zero
        # sample:
        sample_npa[1 + burst.averaging_start : burst.averaging_stop] = avg_voltage
        return sample_npa

    #------------------------------------
    # next_burst 
    #-------------------

    def next_burst(self, curr_burst):
        '''
        Given a burst instance, find the next burst
        of non-zeros. Burst instances contain a pointer
        (signal_index_pt) into the index array entry.
        I.e. the signal_index contains at this pointer
        the first non-zero entry in the samples. We
        start from there. If no more bursts are left,
        returns None.
        
        @param curr_burst: the previous burst
        @type curr_burst: Burst
        @return a new burst instance, which contains all
            information about the next burst.
        @rtype { Burst | None }
        '''
        
        # If the index pt of the passed-in burst
        # is None, we reached the end of the index
        # array, and there won't be another burst:
        if curr_burst is not None and curr_burst.signal_index_pt is None:
            return None
        
        next_burst = Burst()
        if curr_burst is None:
            # First burst. Enough zeros before to
            # allow for a full attack?
            next_burst.start = self.signal_index[0]   
            
            if self.signal_index[0] >= self.ATTACK_RELEASE_SAMPLES:
                # Yes, there is room:
                next_burst.attack_start = self.signal_index[0] - self.ATTACK_RELEASE_SAMPLES
            else:
                next_burst.averaging_start = 0
                next_burst.averaging_stop  = self.signal_index[0]
            pt_into_index = 0
            # Get sample_npa for first 0 after burst,
            # and a pt into signal_index where the next
            # burst is pointed to:
            (burst_end_indx, pt_into_index) = self._find_burst_end(pt_into_index)

            next_burst.stop            = burst_end_indx
            next_burst.signal_index_pt = pt_into_index
            return next_burst
        
        # *Not* the first-burst corner case:
        (burst_end_indx, pt_into_index) = self._find_burst_end(curr_burst.signal_index_pt)        
        next_burst.start = self.signal_index[curr_burst.signal_index_pt]
        next_burst.stop  = burst_end_indx
        next_burst.signal_index_pt = pt_into_index
        
        # First position in sample_npa of non-zero voltage
        # after previous burst:  
        prev_burst_stop  = curr_burst.stop 
        zeros_before_next_burst = 1 + next_burst.start - prev_burst_stop
        if zeros_before_next_burst >= self.ATTACK_RELEASE_SAMPLES:
            # There is room for an attack envelope:
            next_burst.attack_start = next_burst.start - self.ATTACK_RELEASE_SAMPLES
        else:
            # Not enough room for an attack envelope.
            # We will average between the last non-zero
            # sample of the previous burst, and the first
            # sample of the current one:
            next_burst.averaging_start = prev_burst_stop -1 
            next_burst.averaging_stop  = next_burst.start

        return next_burst
        
    #------------------------------------
    # _find_burst_end 
    #-------------------

    def _find_burst_end(self, pt_into_index):
        '''
        Given a pt into the signal_index, find the 
        end of the respective burst, returning the
        index of sample_npa's first 0 after the burst.
        
        Ex.: signal_index  = [array([0,4,5,6,8])]
             pt_into_index = 1
             Goal: find the first non-zero index into sample_npa.
             That would be one after the end of the burst: element 6
             in the sample_npa. Because there is a break in 
             the signal_index sequence.
             
             The method would return (7,4). I.e. the first 0-volt
             cell in sample_npa, and a pt into the signal_index,
             where the next burst starts.
        
        @param pt_into_index:
        @type pt_into_index:
        @return the index into the sample_npa of the first 0V after
             the burst of non-zeroes. Also returns the pt into the 
             signal_index that holds the start of the first burst.
             If no more burst available, returns None
        @rtype {(int, int) | None} 
        '''
        
        # Is there no more burst left?
        if pt_into_index is None:
            return None
        
        prev_indx_pt = pt_into_index
        # Examine successive index entries, and
        # find the end of a series. That end indicates
        # the end of a burst:
        for indx_pt in range(pt_into_index+1, len(self.signal_index)):
            if self.signal_index[indx_pt] == 1 + self.signal_index[prev_indx_pt]:
                prev_indx_pt = indx_pt
                continue
            
            # Reached a gap. The returned pointer
            # into the signal_index will point to where
            # the next burst-start is noted. If we reached
            # the end of the signal_index array, we set the
            # next-pt to None:
            return (1 + self.signal_index[prev_indx_pt], 
                    indx_pt if indx_pt > prev_indx_pt else None)
        
        # If we land here, we reached the end of signal_index array.
        # The sample_npa ends with the start of a burst.
        # There is no 'pointer to next burst' in the index:
        return(1 + self.signal_index[prev_indx_pt], None)
               

    #------------------------------------
    # normalize
    #-------------------
    
    def normalize(self, samples):
        self.log.info("Begin normalization...")
        largest_val = np.max(samples)
        normed_samples = samples/largest_val
        self.log.info("Done normalization.")
        return normed_samples    
    
    #------------------------------------
    # make_envelope
    #-------------------
    
    def make_envelope(self, 
                      attack_length=None, 
                      target_amplitude=None, 
                      direction=Direction.UP):
        '''
        Return amplitude array that exponentially
        approaches either target_amplitude or 0, if
        direction is Direction.DOWN.

        
	Target        -
		       -
		    -
		  -
	0	 -
		 
		Or:
		
				   -                  
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
        
        @param attack_length: number of msecs to approach an 
             asymptode: the voltage of the burst being 'approached'
             by the attack, of 0 on the tail end.
             Default: AmplitudeGater.ATTACK_RELEASE_MSECS 
        @type attack_length: int
        @param target_amplitude: asymptode that amplitude should approach.
            Default: 1 if curve is Direction.UP, else 0
        @type target_amplitude: float
        @param direction: where the exponential curve should be headed.
            If 1, curve slopes up from 0 to 1.
        @type direction: Direction
        @return: array of voltage amplitudes. 
            Length: AmplitudeGater.ATTACK_RELEASE_SAMPLES
        @type: np.array([float])
        '''
        if target_amplitude is None:
            target_amplitude = 1
            
        if attack_length is None:
            attack_length = self.ATTACK_RELEASE_MSECS
            
        T = 1./self.framerate
        
        curve_width = self.samples_from_msecs(attack_length)

        # Constant A controls gentleness: 0.5==>steep, 1==>gentle
        A = 1
        time = np.arange(1,curve_width + 1)

        tau = 0.001
        c = np.e**(-T/tau)
        amplitude = np.zeros([curve_width])
        for n in time[:-1]:
            amplitude[n] = A*c * amplitude[n-1] + (1 - c) * target_amplitude

        # If direction is downwards, mirror it around
        # the vertical symmetry axis:
        
        if direction == Direction.DOWN:
            amplitude = self.mirror_curve(amplitude)
            
        return amplitude

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
    # samples_from_msecs
    #-------------------
    
    def samples_from_msecs(self, msecs):
        
        return msecs * self.samples_per_msec
    
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
    # write_wav 
    #-------------------

    def write_wav(self, 
                  sample_npa, 
                  outfile_path,
                  sample_width=2,
                  framerate=None,
                  num_channels=1,
                  compress_type="NONE",
                  compress_name='not compressed'
                  ):
        
        if framerate is None:
            framerate = self.framerate
            
        
        with wave.open(outfile_path, 'wb') as wav_obj:
            wav_obj.setparams((num_channels,
                               sample_width,
                               framerate,
                               0,            # set by writeframes() below
                               compress_type,
                               compress_name
                               )
                              )
            wav_obj.writeframes(sample_npa)
        
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
        self._attack_start    = None
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
        return self._attack_start
    
    @attack_start.setter
    def attack_start(self, val):
        self._attack_start = val

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

    parser.add_argument('-t', '--logperiod',
                        help='Seconds between reporting progress during envelope phase.',
                        dest='logperiod',
                        type=int,
                        default=10);
                        
    parser.add_argument('-c', '--cutoff',
                        help='dB attenuation from max amplitude below which signal \n' +\
                            'is set to zero; default: -20dB',
                        dest='cutoff',
                        type=int,
                        default='-20'
                        )
    parser.add_argument('-d', '--duration',
                        help='Number of msecs to take for attack and release envelopes. Default: 50msec',
                        dest='duration',
                        type=int,
                        default='50'
                        )
    parser.add_argument('-p', '--plot',
                        action='store_true',
                        default=False,
                        help="Whether or not to plot result."
                        )
    parser.add_argument('wavefile',
                        help="Input .wav file"
                        )
    parser.add_argument('outfile',
                        help="Path to where result .wav file will be written."
                        )
    
    args = parser.parse_args();

    cutoff = args.cutoff
    if cutoff >= 0:
        print(f"Amplitude cutoff must be negative, not {cutoff}")
        sys.exit(1)
        
    duration = args.duration
    if duration <= 0:
        print(f"Duration must be msecs for attack and release envolope duration, not {duration}")
        sys.exit(1)
        


    # AmplitudeGater('/Users/paepcke/tmp/nn01c_20180311_000000.wav',
    #                plot_result=True)
    AmplitudeGater(args.wavefile,
                   args.outfile,
                   amplitude_cutoff=cutoff,
                   attack_release_duration_msecs=duration,
                   logfile=args.logfile,
                   logging_period=args.logperiod,
                   plot_result=args.plot)

    sys.exit(0)