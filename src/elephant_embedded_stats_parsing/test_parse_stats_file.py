'''
Created on Jun 22, 2021

@author: paepcke
'''
import os
from pathlib import Path
import unittest

from elephant_embedded_stats_parsing.parse_stats_files import EleStatsParser


TEST_ALL = True
#TEST_ALL = False

class TestParseStatsFile(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cur_dir  = os.path.dirname(__file__)
        cls.data_dir = os.path.join(cls.cur_dir, 'test_data')
        cls.resnet101_dir = os.path.join(cls.data_dir, 'float16-resnet-101-batchsize-64')
        cls.mobilenetV2   = os.path.join(cls.data_dir, 'float16-mobilenetV2-batchsize-16')
        
        cls.all_stats_truth = os.path.join(cls.data_dir, 'all_stats_truth.csv')
        
    def setUp(self):
        self.parser = EleStatsParser(unittesting=True)

    def tearDown(self):
        # Where the output file was written:
        out_path = self.parser.out_fd.name
        # Close the output file:
        self.parser.out_fd.close()
        # Remove the output file:
        os.remove(out_path)

# ------------------------ Tests -------------

    #------------------------------------
    # test_parse_dir_name 
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_parse_dir_name(self):
        dir_name = Path(self.resnet101_dir).stem
        res_dict = self.parser.parse_dir_name(dir_name)
        truth    = {'model_name' : 'resnet101',
                    'word_width' : 16,
                    'batch_size' : 64}
        self.assertDictEqual(res_dict, truth)

    #------------------------------------
    # test_add_to_results 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_add_to_results(self):
        '''
        Call sig:
		       def add_to_results(self, 
		                          stats_file_path, 
		                          model_name, 
		                          word_width, 
		                          batch_size, 
		                          csv_writer)
        '''
        stats_path = os.path.join(self.resnet101_dir, 'stats.txt')
        self.parser.add_to_results(stats_path,
                                   'resnet101',
                                   16, # word width
                                   64, # batch size
                                   self.parser.csv_writer
                                   )
        # Do the 2nd model:
        stats_path = os.path.join(self.mobilenetV2, 'stats.txt')
        self.parser.add_to_results(stats_path,
                                   'mobilenetV2',
                                   16, # word width
                                   16, # batch size
                                   self.parser.csv_writer
                                   )
        self.parser.out_fd.flush()

        # Read the out csv file back, and compare:
        out_path = self.parser.out_fd.name
        with open(out_path, 'r') as res_fd:
            res = res_fd.read()
        with open(self.all_stats_truth, 'r') as truth_fd:
            truth = truth_fd.read()
        
        self.assertEqual(res, truth)


# -------------------- Main ------------------
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()