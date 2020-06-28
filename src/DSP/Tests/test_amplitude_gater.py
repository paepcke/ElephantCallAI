'''
Created on Feb 21, 2020

@author: paepcke
'''

import sys, os
from tempfile import NamedTemporaryFile
import unittest
import wave

from scipy.io import wavfile

from DSP.amplitude_gating import AmplitudeGater
import numpy as np


sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))



TEST_ALL = True
#TEST_ALL = False

class Test(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.setUp(self)

        self.samples_for_voltage_suppression = np.array([1,12,13,14,5,4,3,20])
        
        self.samples_normal_immediate_start = np.array([1,0,0,10,11,0,0,0,0,20])
        self.samples_normal_immediate_start_sig_index = np.array([0,3,4,9])

        self.samples_for_attack_and_release = np.array([0,0,0,0,1,0.8,0,0,0,0,0,0,0,0,0.4])
        self.samples_for_attack_and_release_sig_index = np.array([4,14])
        
        # Compress everything down so we only need small vectors
        # like the ones above: we'll take 2msecs for the attack/release: 
        # Attacks and releases take 2msecs, rather than the default 50msec:
        AmplitudeGater.ATTACK_RELEASE_MSECS = 2
        self.gater = AmplitudeGater(None, # No .wav file 
                                    spectrogram_freq_cap=0,
                                    testing=True,
                                    framerate=2 # frames per second
                                    )
        
    #------------------------------------
    # test_suppress_none
    #-------------------
    
    @unittest.skipIf(not TEST_ALL, "Temporarily skipping")
    def test_suppress_none(self):
        new_volts = self.gater.suppress_small_voltages(self.samples_for_voltage_suppression, 
                                                       5,   # volt_thres
                                                       1)   # 1 sec === 2 samples padding
        self.assertEqual(new_volts.all(), 
                         self.samples_for_voltage_suppression.all(),
                         )

    #------------------------------------
    # test_suppress_start_array_fit
    #-------------------
    
    @unittest.skipIf(not TEST_ALL, "Temporarily skipping")
    def test_suppress_mid_array(self):

        volts = np.array([ 1,2,3,14,5,4,3,1,3,2,2,20])
        # With padding of 1 sample (0.5 sec) should come out as:
        # [ 1,0,3,14,5,4,0,0,0,0,2,20]
        new_volts = self.gater.suppress_small_voltages(volts,
                                                       5,   # volt_thres
                                                       0.5) # 0.5 sec === 1 samples padding

        self.assertEqual(new_volts.all(), 
                         np.array([ 1,0,3,14,5,4,0,0,0,0,2,20]).all()
                         )

    #------------------------------------
    # test_lead_pad_exact_fit_at_end
    #-------------------    

    @unittest.skipIf(not TEST_ALL, "Temporarily skipping")
    def test_lead_pad_exact_fit_at_end(self):
        
        volts = np.array([ 1,12,13,14,5,4,3,1,3,2,2,1])
        # With padding of 3 samples, should come out as:
        # we should get [ 1,12,13,14,5,4,3,1,0,2,2,1])
        new_volts = self.gater.suppress_small_voltages(volts,
                                                       5,   # volt_thres
                                                       1.5)   # 1.5 sec === 3 samples padding

        self.assertEqual(new_volts.all(), 
                         np.array([1,12,13,14,5,4,3,1,0,2,2,1]).all()
                         )
    #------------------------------------
    # test_not_fit_at_start
    #-------------------    

    @unittest.skipIf(not TEST_ALL, "Temporarily skipping")
    def test_not_fit_at_start(self):
        
        volts = np.array([ 1,2,13,14])
        # With padding of 3 samples, should come out as:
        # we should get [ 1,2,13, 14]
        new_volts = self.gater.suppress_small_voltages(volts,
                                                       5,   # volt_thres
                                                       1)   # 1 sec === 2 samples padding

        self.assertEqual(new_volts.all(), 
                         np.array([ 1,2,13, 14]).all()
                         )
    
    #------------------------------------
    # test_no_padding
    #-------------------    
    
    unittest.skipIf(not TEST_ALL, "Temporarily skipping")
    def test_no_padding(self):

        volts = np.array([ 1,2,13,14])
        # With padding of 3 samples, should come out as:
        # we should get [ 0,0,13,14]
        new_volts = self.gater.suppress_small_voltages(volts,
                                                       5,   # volt_thres
                                                       0)   # 0 sec === 0 samples padding

        self.assertEqual(new_volts.all(), 
                         np.array([ 0,0,13,14]).all()
                         )

    #------------------------------------
    # test_filter_spectrogram_simple
    #-------------------
    
    @unittest.skipIf(not TEST_ALL, "Temporarily skipping")
    def test_filter_spectrogram_simple(self):
        
        freq_labels = np.array([10,20,30,40])
        # Make:
        #    array([[ 0,  1,  2],
        #           [ 3,  4,  5],
        #           [ 6,  7,  8],
        #           [ 9, 10, 11]])

        freq_time   = self.make_matrix()
        # Retain only [[3,4,5,]]:
        # Keep frequency rows greater than 20, less than 30:
        freq_bands   = [(20, 30)]
        
        (new_freq_labels, new_freq_time) = self.gater.filter_spectrogram(freq_labels, 
                                                                         freq_time, 
                                                                         freq_bands)
        self.assertEqual(new_freq_labels, np.array([20]))
        self.assertEqual(new_freq_time.all(), np.array([[3,4,5]]).all())
       
    #------------------------------------
    # test_filter_spectrogram_open_low_freq
    #-------------------
    
    @unittest.skipIf(not TEST_ALL, "Temporarily skipping")
    def test_filter_spectrogram_open_low_freq(self):
        
        freq_labels = np.array([10,20,30,40])
        # Make:
        #    array([[ 0,  1,  2],
        #           [ 3,  4,  5],
        #           [ 6,  7,  8],
        #           [ 9, 10, 11]])

        freq_time   = self.make_matrix()
        # Retain only [[0,1,2], 
        #              [3,4,5]]:
        # Keep rows for frequencies under 30Hz:
        freq_bands   = [(None, 30)]
        
        (new_freq_labels, new_freq_time) = self.gater.filter_spectrogram(freq_labels, 
                                                                         freq_time, 
                                                                         freq_bands)
        self.assertEqual(new_freq_labels.all(), np.array([10,20]).all())
        self.assertEqual(new_freq_time.all(), np.array([[0,1,2],
                                                        [3,4,5]]).all())

    #------------------------------------
    # test_filter_spectrogram_two_simples
    #-------------------
    
    @unittest.skipIf(not TEST_ALL, "Temporarily skipping")
    def test_filter_spectrogram_two_simples(self):
        
        freq_labels = np.array([10,20,30,40])
        # Make:
        #    array([[ 0,  1,  2],
        #           [ 3,  4,  5],
        #           [ 6,  7,  8],
        #           [ 9, 10, 11]])

        freq_time   = self.make_matrix()
        # Retain only rows 1, 2, and 3:
        #             [[3,4,5],
        #              [6,7,8],
        #              [9,10,11]]:

        # The first constraint chokes off
        # what the second would allow:
        freq_bands   = [(11, 30), (31, 50)]
        
        (new_freq_labels, new_freq_time) = self.gater.filter_spectrogram(freq_labels, 
                                                                         freq_time, 
                                                                         freq_bands)
        self.assertEqual(new_freq_labels.all(), np.array([20,40]).all())
        self.assertEqual(new_freq_time.all(), np.array([[3,4,5],
                                                        [9,10,11]
                                                        ]).all())

       
    #------------------------------------
    # test_wav_read_write
    #-------------------

    @unittest.skipIf(not TEST_ALL, "Temporarily skipping")
    def test_wav_read_write(self):
        
        test_sound_path = os.path.join(os.path.dirname(__file__), 'testsound.wav')
        
        (framerate, sound_data) = wavfile.read(test_sound_path)
        
        # Get a tmp file and write the data back out:
        tmp_file_obj = NamedTemporaryFile(mode='w+b',
                                          prefix='test_amp_gate',
                                          suffix='.wav')
        tmp_file_obj.close()
        tmp_file_nm = tmp_file_obj.name
        
        wavfile.write(tmp_file_nm, framerate, sound_data) 
        
        # Original file and just-written file equal?
        
        orig_file_len = os.stat(test_sound_path).st_size
        tmp_file_len  = os.stat(tmp_file_nm).st_size
        
        # NOTE: the original file is 105886 bytes long.
        #       method wavfile.write() writes this back
        #       as 105884 bytes. The difference is that
        #       the original file pads the 'Bits per sample'
        #       information in 1 byte, rather than 2.
        #       Both files play the same, and have the same
        #       length data section of the wav file:

        try:
            self.assertEqual(tmp_file_len, orig_file_len - 2)
            
            # Check the metadata:
            
            tmp_wave_read_obj = wave.open(tmp_file_nm)
            test_sound_wave_read_obj = wave.open(test_sound_path)
            
            # Get all metadata in one tuple:
            
            (num_channels,
             sample_width,
             framerate,
             num_frames,
             compress_type,
             compress_name
             ) = test_sound_wave_read_obj.getparams()

            tmp_metadata = tmp_wave_read_obj.getparams()
            self.assertTupleEqual(tmp_metadata,
                                  (num_channels,
                                   sample_width,
                                   framerate,
                                   num_frames,
                                   compress_type,
                                   compress_name
                                   )
                                  )

        finally:
            os.remove(tmp_file_obj.name)

# --------------------------- Utils ---------------
 
    #------------------------------------
    # make_matrix
    #-------------------
    
    def make_matrix(self):
        '''
        Return:
            array([[ 0,  1,  2],
                   [ 3,  4,  5],
                   [ 6,  7,  8],
                   [ 9, 10, 11]])
        
        '''
        return np.array(np.arange(12)).reshape(4,3)

 
# --------------------------- Main ---------------
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()