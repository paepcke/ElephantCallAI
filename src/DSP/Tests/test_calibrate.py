'''
Created on Apr 5, 2020

@author: paepcke
'''
import csv
import os
import statistics
import tempfile
import unittest
from collections import OrderedDict

import numpy as np

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from calibrate_preprocessing import Experiment
from precision_recall_from_wav import PerformanceResult, PrecRecComputer


TEST_ALL = True
#TEST_ALL = False

class TestCalibrate(unittest.TestCase):

    # A dict with all properties that PerformanceResult
    # instances hold, initialized to artificial values
    # 0,1,2,...:

    artificial_prop_values = {}
    
    #------------------------------------
    # setUpClass
    #-------------------

    @classmethod
    def setUpClass(cls):
        super(TestCalibrate, cls).setUpClass()

        # Create a dict of artificial results that would
        # be created by the PrecRecComputer. This dict will
        # be used for creating PerformanceResult instances
        # for testing:
        
        val = 0
        # The props class variable of Experiment is
        # a dict {prop-name : <required-class-of-prop>}:
        for prop_name in Experiment.props:
            try:
                # All properties need to be floats or ints.
                # But experiment_result must be a PerformanceResult
                # instance,  and will cause an error here. We take
                # care of that after the loop:
                cls.artificial_prop_values[prop_name] = Experiment.props[prop_name](val)
            except TypeError:
                cls.artificial_prop_values[prop_name] = None
            val += 1

    #------------------------------------
    # setUp
    #-------------------

    def setUp(self):
        # Create a fresh PerformanceResult instance for each test 
        # to use.
        self.perf_res = self.create_perf_result()
        self.experiment = Experiment(self.artificial_prop_values)
        self.experiment['experiment_res'] = self.perf_res
        
        # Create a non-acting PrecRecComputer instance:
        self.prec_rec_computer = PrecRecComputer(None,  # No wavfile
                                                 None,  # No labelfile
                                                 testing=True)

        
        # Create a temp file for each test to use:
        tmp_file_fd = tempfile.NamedTemporaryFile(mode='w', delete=False)
        self.tmp_file_name = tmp_file_fd.name + '.tsv'
        tmp_file_fd.close()

    #------------------------------------
    # tearDown
    #-------------------

    def tearDown(self):
        try:
            # If temp file still exists, get rid of it:
            os.remove(self.tmp_file_name)
        except Exception:
            pass

    #------------------------------------
    # test_to_tsv
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_to_tsv(self):
        # Save an Experiment instance  to a .tsv file:
        
        self.experiment.to_tsv(include_col_header=True, outfile=self.tmp_file_name, append=True)
        with open(self.tmp_file_name, 'r') as tmp_file_fd:
            reader = csv.reader(tmp_file_fd, delimiter='\t')
            col_header = next(reader)
            col_header_set = set(col_header)
            self.assertSetEqual(col_header_set, set(Experiment.props.keys()))
            res_line = next(reader)
            res_dict = {}
            for (col_num, val) in enumerate(res_line):
                key = col_header[col_num]
                required_type = Experiment.props[key]
                if isinstance(required_type, list):
                    # The list will be single-element, with
                    # value being the type of elements in the
                    # array:
                    required_type_in_arr = required_type[0]
                    res_arr = [required_type_in_arr(el) for el in eval(val)]
                    res_dict[key] = res_arr
                # Can't get EQ to work for required_type/PerformanceResult,
                # so use following hack:
                elif issubclass(required_type, OrderedDict):
                    perf_res = self.create_perf_result()
                    res_dict[key] = perf_res                    
                else:
                    res_dict[key] = required_type(val)
            # From what we picked out of the .tsv file,
            # create an Experiment instance:
            retrieved_experiment = Experiment(res_dict)
            self.assertTrue(self.performance_results_equality(retrieved_experiment, self.experiment))

    #------------------------------------
    # test_from_tsv
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_from_tsv(self):
        # Make a new test result instance from a results
        # .tsv file:
        
        # Write our known Experiment instance to tsv:
        self.experiment.to_tsv(include_col_header=True, outfile=self.tmp_file_name, append=True)
        # Make a new instance, and read that tsv file:
        experiment_inst_list = Experiment.instances_from_tsv(self.tmp_file_name, first_line_is_col_header=True)

        restored_experiment = experiment_inst_list[0]
        for (prop_name, prop_value) in self.experiment.items():
            if prop_name == 'experiment_res':
                # The PerformanceResult instance:
                self.performance_results_equality(restored_experiment['experiment_res'],
                                                  self.perf_res
                                                  )
            self.assertEqual(prop_value, restored_experiment[prop_name])

# --------------------- Utils ---------------

    #------------------------------------
    # create_perf_res
    #-------------------

    def create_perf_result(self):
        '''
        Creates a simple PerformanceResult instance, where
        all but one value is, successively 1,2,3, etc.
        '''
        artificial_prop_values = {}
        val = 0
        # The props class variable of PerformanceResult is
        # a dict {prop-name : <required-class-of-prop>}:
        for prop_name in PerformanceResult.props:
            try:
                # All properties need to be floats or ints.
                # But overlap_percentages must be a list,
                # and will cause an error here. We take
                # care of that after the loop:
                artificial_prop_values[prop_name] = PerformanceResult.props[prop_name](val)
            except TypeError:
                continue
            val += 1
            
        # The overlap_percentages needs to be a list
        # of floats:
        artificial_prop_values['overlap_percentages'] = [1.0,2.0,3.0]

        # Add mean of overlap percentages, as will be
        # done when PerformanceResult is instantiated
        # later:
        artificial_prop_values['mean_overlaps'] = \
            statistics.mean(artificial_prop_values['overlap_percentages'])

        perf_res_inst = PerformanceResult(artificial_prop_values)
        return perf_res_inst
    
    #------------------------------------
    # performance_results_equality
    #-------------------
    
    def performance_results_equality(self, perf_res_left, perf_res_right):
        errors = []
        for (key,val) in perf_res_left.items():
            if isinstance(val, np.ndarray):
                # Values are np.arrays; simple Python
                # equality won't work. Also, for floats,
                # array_equal also fails over slight diffs.
                # Rather than using isclose(), just round
                # to 2 decimals for both:
                if not np.array_equal(np.round(val,2), 
                                      np.round(perf_res_right[key], 2)
                                      ):
                    errors.append(key)
                continue
            if val != perf_res_right[key]:
                errors.append(key)
                continue
        if len(errors) == 0:
            return True
        for problem_key in errors:
            print(f"{problem_key}: Left: {perf_res_left[problem_key]}; Right: {perf_res_right[problem_key]}")
        return False


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()