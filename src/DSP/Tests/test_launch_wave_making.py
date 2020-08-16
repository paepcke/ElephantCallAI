'''
Created on Jul 31, 2020

@author: paepcke
'''
import os
import subprocess
import sys
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
        self.crow_res_file = 'crow_test_sound_gated.wav'
        self.nightingale_res_file = 'nightingale_test_sound_gated.wav'

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

#        dir_name = tempfile.TemporaryDirectory(prefix="Tmpdir",  dir=self.curr_dir).name
#        dir_name = '/tmp'
            script_path = os.path.abspath(os.path.join(self.curr_dir,
                                                       '../launch_wave_making.sh'))
            cmd = f"{script_path} --outdir {dir_name} {self.wav_file_list[0]} {self.wav_file_list[1]}"
    
            #************
            #print(f"Cmd: {cmd}")
            #************
    
            completed_proc = subprocess.run(cmd, shell=True)
    
            # Should report that it launched 2 processes:
            crow_file_path = os.path.join(dir_name, self.crow_res_file)
            nightingale_file_path = os.path.join(dir_name, self.nightingale_res_file)
    
            self.assertTrue(os.path.exists(os.path.join(dir_name, 
                                                        self.crow_res_file)),
                            f"Non existent: {crow_file_path}")
            self.assertTrue(os.path.exists(os.path.join(dir_name, self.nightingale_res_file)),
                            f"Non existent: {nightingale_file_path}")
                                                        



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()