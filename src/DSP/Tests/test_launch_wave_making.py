'''
Created on Jul 31, 2020

@author: paepcke
'''
import os
import subprocess
import tempfile
import unittest


class Test(unittest.TestCase):

    #------------------------------------
    # setUp 
    #-------------------

    def setUp(self):
        self.curr_dir = os.path.dirname(__file__)
        self.test_sound_crow = os.path.join(self.curr_dir, 'crow_test_sound.wav')
        self.test_sound_nightingale = os.path.join(self.curr_dir, 'nightingale_test_sound.wav')
        self.wav_file_list = [self.test_sound_crow, self.test_sound_nightingale]

    #------------------------------------
    # tearDown
    #-------------------


    def tearDown(self):
        pass

    #------------------------------------
    # testLaunching
    #-------------------

    def testLaunching(self):

                
        # Temporary dest dir:
        with tempfile.TemporaryDirectory(prefix="Tmpdir",
                                         dir=self.curr_dir,
                                         ) as dir_name:
            cmd_arr = ['bash',
                       f'{self.curr_dir}/../launch_wave_making.sh',
                       '--outdir',
                       dir_name,
                       self.wav_file_list[0],
                       self.wav_file_list[1],
                       ]
            completed_proc = subprocess.call(cmd_arr,
                                             #env=os.environ,
                                            #shell=True,
                                            #capture_output=True
                                            )
        print(completed_proc)



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()