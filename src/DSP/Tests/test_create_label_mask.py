'''
Created on Jul 8, 2020

@author: paepcke
'''
import io
import os, sys
import unittest

from testfixtures import LogCapture

import numpy as np
import pandas as pd

# Only for testing that error messages are 
# logged when appropriate:
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from spectrogrammer import Spectrogrammer

TEST_ALL = True
#TEST_ALL = False

class Test(unittest.TestCase):

    #------------------------------------
    # setUp
    #-------------------

    def setUp(self):
        
        # Build a make-believe spectrogram:
        
        #        0.5  1.0  1.5  2.0  2.5  3.0
        # freq1    1    2    3    4    5    6
        # freq2    7    8    9   10   11   12
        # freq3   13   14   15   16   17   18

        self.spectro = pd.DataFrame.from_dict(
            {'freq1': [1,2,3,4,5,6],
             'freq2': [7,8,9,10,11,12],
             'freq3': [13,14,15,16,17,18]
             },
            columns=['0.5', '1.0', '1.5', '2.0', '2.5', '3.0'],
            orient='index'
            )
        
        # Make an empty spectrogrammer: No infile, no actions:
        self.spectrogrammer = Spectrogrammer('foo', [], testing=True)
        self.spectrogrammer.DEFAULT_FRAMERATE = 2
        
        # Col header row for an in-memory string label 'file'
        col_headers = "col1\tBegin Time (s)\tcol2\tEnd Time (s)\tcol3\n"
        self.label_file_fd = io.StringIO(col_headers)
        self.label_file_fd.write(col_headers)
        
        curr_dir = os.path.dirname(__file__)
        
    #------------------------------------
    # tearDown
    #-------------------

    def tearDown(self):
        try:
            self.label_file_fd.close()
        except Exception:
            pass
        try:
            del self.spectrogrammer
        except Exception:
            pass

    #------------------------------------
    # testLabelsOnTimeBinBoundaries
    #-------------------

    @unittest.skipIf(not TEST_ALL, "Temporarily skipping")
    def testLabelsOnTimeBinBoundaries(self):

        # Create in-memory label CSV 'file'
        # 0.5sec to 1.0sec
        
        self.label_file_fd.write("foo1\t1024.0\tfoo2\t2048.0\tfoo3\n")
        self.label_file_fd.seek(0)
        
        # Create a fake .wav signal:
        wav_signal = range(6*2048)

        label_mask = self.spectrogrammer.create_label_mask_from_raven_table(
                wav_signal, self.label_file_fd, framerate=2
                )        

        # Times are: [0.0, 1024.0, 2048.0, 3072.0, 4096.0, 5120.0, 6144.0]
        expected = np.array([0,1,1,0,0,0,0])
        self.assertTrue((expected == label_mask).all())
        
    #------------------------------------
    # testLabelsOffBoundaries
    #-------------------

    @unittest.skipIf(not TEST_ALL, "Temporarily skipping")
    def testLabelsOffBoundaries(self):
        # Create in-memory label CSV 'file'
        # for label 1000sec to 3000sec
        
        self.label_file_fd.write("foo1\t1024.0\tfoo2\t3072.0\tfoo3\n")
        self.label_file_fd.seek(0)
        
        # Create a fake .wav signal:
        wav_signal = range(6*2048)
        
        label_mask = self.spectrogrammer.create_label_mask_from_raven_table(
                wav_signal, self.label_file_fd, framerate=2
                )
        # Times are [0.0, 1024.0, 2048.0, 3072.0, 4096.0, 5120.0, 6144.0]
        expected = np.array([0,1,1,1,0,0,0])
        self.assertTrue((expected == label_mask).all())

    #------------------------------------
    # testLabelStartBeforeRecording
    #-------------------

    @unittest.skipIf(not TEST_ALL, "Temporarily skipping")
    def testLabelStartBeforeRecording(self):
        # Create in-memory label CSV 'file'
        # for label 2048sec to 4096sec
        
        self.label_file_fd.write("foo1\t2048\tfoo2\t4096\tfoo3\n")
        self.label_file_fd.seek(0)
        
        # Create a fake .wav signal:
        wav_signal = range(1,6*2048)

        label_mask = self.spectrogrammer.create_label_mask_from_raven_table(
                wav_signal, self.label_file_fd, framerate=2
                )

        # Times are [0.0, 1024.0, 2048.0, 3072.0, 4096.0, 5120.0, 6144.0]
        expected = np.array([0,0,1,1,1,0,0])
        self.assertTrue((expected == label_mask).all())

    #------------------------------------
    # testLabelEndBeyondRecording
    #-------------------

    @unittest.skipIf(not TEST_ALL, "Temporarily skipping")
    def testLabelEndBeyondRecording(self):
        # Create in-memory label CSV 'file'
        # for label 10,000sec to 12,000sec
        
        self.label_file_fd.write("foo1\t10000\tfoo2\t12000\tfoo3\n")
        self.label_file_fd.seek(0)
        
        # Create a fake .wav signal:
        wav_signal = range(6*2048)
        
        label_mask = self.spectrogrammer.create_label_mask_from_raven_table(
                wav_signal, self.label_file_fd, framerate=2
                )
        # Times are [0.0, 1024.0, 2048.0, 3072.0, 4096.0, 5120.0, 6144.0]
        expected = np.array([0,0,0,0,0,0,0])
        self.assertTrue((expected == label_mask).all())

    #------------------------------------
    # testBeginBeforeEndTime
    #-------------------
    
    @unittest.skipIf(not TEST_ALL, "Temporarily skipping")
    def testBeginBeforeEndTime(self):
        # Create in-memory label CSV 'file'
        # for label 0.2sec to 1.0sec
        
        self.label_file_fd.write("foo1\t2.5\tfoo2\t0.5\tfoo3\n")
        self.label_file_fd.seek(0)
        
        # Create a fake .wav signal:
        wav_signal = range(6*2048)
        
        with LogCapture('logging_service') as log_content:
            label_mask = self.spectrogrammer.create_label_mask_from_raven_table(
                    wav_signal, self.label_file_fd, framerate=2
                    )
        # Ensure we got an error log msg about begin time
        # later than end time:
        log_content.check(
            ('logging_service', 'ERROR', "Bad label: end label less than begin label: 0.5 < 2.5"),
            )
        
        # Times are ['0.5', '1.0', '1.5', '2.0', '2.5', '3.0']
        expected = np.array([0,0,0,0,0,0,0])
        self.assertTrue((expected == label_mask).all())

    #------------------------------------
    # testNoLabelInRange
    #-------------------

    @unittest.skipIf(not TEST_ALL, "Temporarily skipping")
    def testNoLabelInRange(self):
        # Create in-memory label CSV 'file'
        # for label 0.2sec to 1.0sec
        
        # Start is after end of recording:
        self.label_file_fd.write("foo1\t5.5\tfoo2\t3.0\tfoo3\n")
        self.label_file_fd.seek(0)
        
        # Create a fake .wav signal:
        wav_signal = range(6*2048)
        
        with LogCapture('logging_service') as log_content:
            _label_mask = self.spectrogrammer.create_label_mask_from_raven_table(
                    wav_signal, self.label_file_fd, framerate=2
                    )

        # Ensure we got an error log msg about begin time
        # later than end time:
        log_content.check(
            ('logging_service', 'ERROR',
             'Bad label: end label less than begin label: 3.0 < 5.5'
            ))
        
        # End before start of recording:
        self.label_file_fd.write("foo1\t0.1\tfoo2\t0.2\tfoo3\n")
        self.label_file_fd.seek(0)
        with LogCapture('logging_service') as log_content:
            label_mask = self.spectrogrammer.create_label_mask_from_raven_table(
                    wav_signal, self.label_file_fd, framerate=2
                    )

        # Ensure we got an error log msg about begin time
        # later than end time:
        log_content.check(
            ('logging_service', 'ERROR', "Bad label: end label less than begin label: 3.0 < 5.5")
            )
        # Times are [0.0, 1024.0, 2048.0, 3072.0, 4096.0, 5120.0, 6144.0]
        expected = np.array([1,0,0,0,0,0,0])
        self.assertTrue((expected == label_mask).all())

# -------------------------- Main -------------------
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testLabelMaskFromSpectrogram']
    unittest.main()