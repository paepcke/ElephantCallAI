'''
Created on Mar 27, 2020

@author: paepcke
'''
import csv
import os, sys
import statistics
import tempfile
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from precision_recall_from_wav import PerformanceResult

class TestPrecisionRecall(unittest.TestCase):

    #------------------------------------
    # setUpClass
    #-------------------

    @classmethod
    def setUpClass(cls):
        super(TestPrecisionRecall, cls).setUpClass()
        
        # A dict with all properties that PerformanceResult
        # instances hold, initialized to artificial values
        # 0,1,2,...:
        cls.articifial_prop_values = {}
        val = 0
        for prop_name in PerformanceResult.props:
            cls.articifial_prop_values[prop_name] = val 
            val += 1
            
        # The overlap_percentages needs to be a list
        # of floats:
        cls.articifial_prop_values['overlap_percentages'] = [1.0,2.0,3.0]

        # Add mean of overlap percentages, as will be
        # done when PerformanceResult is instantiated
        # later:
        cls.articifial_prop_values['mean_overlaps'] = \
            int(statistics.mean(cls.articifial_prop_values['overlap_percentages']))
        
    #------------------------------------
    # setUp
    #-------------------

    def setUp(self):
        self.perf_res = PerformanceResult(self.articifial_prop_values)

    #------------------------------------
    # tearDown
    #-------------------

    def tearDown(self):
        pass

    #------------------------------------
    # test_to_csv
    #-------------------

    def test_to_csv(self):
        tmp_file_fd = tempfile.NamedTemporaryFile(mode='w', delete=False)
        tmp_file_name = tmp_file_fd.name + '.tsv'
        tmp_file_fd.close()
        try:
            self.perf_res.to_tsv(include_col_header=True, outfile=tmp_file_name, append=True)
            with open(tmp_file_name, 'r') as tmp_file_fd:
                reader = csv.reader(tmp_file_fd, delimiter='\t')
                col_header = next(reader)
                self.assertListEqual(col_header, list(PerformanceResult.props.keys()))
                res_line = next(reader)
                quoted_perf_nums = [str(num) for num in self.articifial_prop_values.values()]
                self.assertListEqual(res_line, quoted_perf_nums)
        finally:
            try:
                os.remove(tmp_file_name)
            except Exception:
                pass
            


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()