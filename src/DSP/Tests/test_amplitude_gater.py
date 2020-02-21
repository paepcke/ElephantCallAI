'''
Created on Feb 21, 2020

@author: paepcke
'''

import sys,os
sys.path.append(os.path.dirname(__file__))
import unittest
import numpy as np

from DSP.amplitude_gating import AmplitudeGater

class Test(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.setUp(self)
        self.samples_normal_immediate_start = np.array([1,0,0,10,11,0,0,0,0,20])
        self.samples_normal_immediate_start_sig_index = np.array([0,3,4,9])
        # Compress everything down so we only need small vectors
        # like the ones above: we'll take 2msecs for the attack/release: 
        AmplitudeGater.ATTACK_RELEASE_MSECS = 2
        self.gater = AmplitudeGater(None, # No .wav file 
                                    testing=True,
                                    framerate=2000 # frames per second
                                    )

    def testNextBurst(self):
        samples_npa = self.samples_normal_immediate_start
        signal_index = self.samples_normal_immediate_start_sig_index
        self.gater.samples_na   = samples_npa
        self.gater.signal_index = signal_index

        burst1 = self.gater.next_burst(None)
        self.assertEqual(burst1.start, 0)
        self.assertEqual(burst1.stop, 1)
        self.assertEqual(burst1.signal_index_pt, 1)
        
        burst2 = self.gater.next_burst(burst1)
        self.assertEqual(burst2.start, 3)
        self.assertEqual(burst2.stop, 5)
        self.assertEqual(burst2.averaging_start, 0)
        self.assertEqual(burst2.averaging_stop, 3)
        self.assertEqual(burst2.signal_index_pt, 3)
        
        burst3 = self.gater.next_burst(burst2)
        self.assertEqual(burst3.start, 9)
        self.assertEqual(burst3.stop, 10)
        self.assertEqual(burst3.attack_start, 5)
        self.assertIsNone(burst3.signal_index_pt)
        
        print(burst1, burst2, burst3)
        


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()