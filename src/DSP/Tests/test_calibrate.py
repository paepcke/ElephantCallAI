'''
Created on Apr 5, 2020

@author: paepcke
'''

import csv
import os
import tempfile
import unittest

import numpy as np

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dsp_utils import SignalTreatmentDescriptor
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
    overlap_percentages    = np.array([1.0,2.0,3.0])
    min_required_overlap = 10
    
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
                # All properties except experiment_result and signal_treatment
                # need to be floats or ints. These will cause a TypeError
                # here. We take care of these later.
                target_type = Experiment.props[prop_name]
                if prop_name == 'experiment_res':
                    raise TypeError()
                cls.artificial_prop_values[prop_name] = target_type(val)
            except TypeError:
                if target_type == SignalTreatmentDescriptor:
                    cls.artificial_prop_values[prop_name] = SignalTreatmentDescriptor(-30,300,10)
                else:
                    cls.artificial_prop_values[prop_name] = None
            val += 1
        cls.artificial_prop_values['experiment_res'] = None
        # Standardize on min_required_overlap == 10:
        cls.artificial_prop_values['min_required_overlap'] = 10
        
    #------------------------------------
    # setUp
    #-------------------

    def setUp(self):
        # Create a fresh PerformanceResult instance for each test 
        # to use.
        self.perf_res = self.create_perf_result()
        self.artificial_prop_values['experiment_res'] = self.perf_res
        self.experiment = Experiment(self.artificial_prop_values)
        
        # Dict of all props and their types, down to everything expanded:
        # Must happen after the experiment was instantiated at least
        # once, b/c that's when the flat props are computed:
        all_props = Experiment.props.copy()
        all_props.update(PerformanceResult.props)
        all_props.update({
                            'overlaps_summary_min' : float,
                            'overlaps_summary_max' : float,
                            'overlaps_summary_mean' : float,
                            'overlaps_summary_med' : float,
                            'overlaps_summary_sd' : float,
                            'overlaps_summary_num_ge_min' : int,
                            'overlaps_summary_perc_ge_min' : float
                        })
        # The following two props aren't in a fully
        # expanded set; they are expanded there:
        del(all_props['experiment_res'])
        del(all_props['overlaps_summary'])
        
        self.all_prop_types = all_props

        # Create a non-acting PrecRecComputer instance:
        self.prec_rec_computer = PrecRecComputer(self.experiment['signal_treatment'],
                                                 None,  # No wavfile
                                                 None,  # No labelfile
                                                 testing=True)

        
        # Create a temp file for each test to use:
        tmp_file_fd = tempfile.NamedTemporaryFile(mode='w', delete=False)
        self.tmp_file_name = tmp_file_fd.name + '.pickle'
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
        # Some tests use tmp_file_name with .pickle removed,
        # and .tsv added instead:
        (root, _ext) = os.path.splitext(self.tmp_file_name)
        try:
            os.remove(root + '.tsv')
        except Exception:
            pass

    #------------------------------------
    # test_to_save
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_save_load(self):
        # Save an Experiment instance  to a pickle file:
        
        self.experiment.save(outfile=self.tmp_file_name, append=False)
        retrieved_exp = next(self.experiment.load(self.tmp_file_name))
        self.assertTrue(self.experiments_equality(retrieved_exp, self.experiment))
        
        # Save a second experiment to the same file (using the same instance):
        self.experiment.save(outfile=self.tmp_file_name, append=True)
        
        retrieved_experiments = [exp for exp in Experiment.load(self.tmp_file_name)]
        self.assertEqual(len(retrieved_experiments), 2)

        self.assertEqual(retrieved_experiments[0], self.experiment)
        self.assertEqual(retrieved_experiments[1], self.experiment)
        
    #------------------------------------
    # test_to_flat_tsv
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_to_flat_tsv(self):
        # Save an Experiment instance  to a .tsv file:
        
        # Want tmp file to end not in .pickle, but in .tsv:
        (root, _ext) = os.path.splitext(self.tmp_file_name)
        tmp_file_name = root + '.tsv'
        self.experiment.to_flat_tsv(include_col_header=True, 
                                    outfile=tmp_file_name, 
                                    append=True)
        with open(tmp_file_name, 'r') as tmp_file_fd:
            reader = csv.reader(tmp_file_fd, delimiter='\t')
            col_header = next(reader)
            col_header_set = set(col_header)

            props_set = set(self.all_prop_types.keys())
            self.assertSetEqual(col_header_set, props_set)
            
            res_line = next(reader)
            for (col_num, val) in enumerate(res_line):
                key = col_header[col_num]
                # Find the required type in the top level
                # experiment or in its PerformanceRes:
                required_type = self.all_prop_types[key]
                
                if isinstance(required_type, list):
                    # The list will be single-element, with
                    # value being the type of elements in the
                    # array:
                    required_type_in_arr = required_type[0]
                    res_arr = [required_type_in_arr(el) for el in eval(val)]
                    self.assertEqual(res_arr, self.experiment[key])
                # Can't get EQ to work for required_type/PerformanceResult,
                # so use following hack:
                elif required_type == PerformanceResult:
                    self.assertEqual(val, self.experiment['experiment_res'])

    #------------------------------------
    # test_from_tsv
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_from_tsv(self):
        # Make a new test result instance from a results
        # .tsv file:
        
        # Write our known Experiment instance to tsv:
        self.experiment.save(outfile=self.tmp_file_name, append=True)
        # Make a new instance, and read that tsv file:
        experiment_inst_list = Experiment.instances_from_saved(self.tmp_file_name)

        restored_experiment = experiment_inst_list[0]
        for (prop_name, prop_value) in self.experiment.items():
            if prop_name == 'experiment_res':
                # The PerformanceResult instance:
                self.assertTrue(self.performance_results_equality(restored_experiment['experiment_res'],
                                                                  self.perf_res
                                                  ))
            elif prop_name == 'signal_treatment':
                # The SignalTreatmentDescriptor instance:
                self.assertTrue(prop_value.__eq__(restored_experiment[prop_name]))
            else:
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
                # and signal_treatment must be a SignalTreatmentDescriptor.
                # Both will cause an error here. We take
                # care of that after the loop:
                artificial_prop_values[prop_name] = PerformanceResult.props[prop_name](val)
            except TypeError:
                continue
            val += 1
        
        # Adjust the min_required_overlap to be 10:
        artificial_prop_values['min_required_overlap'] = 10
        
#         # The overlap_percentages needs to be a list
#         # of floats:
#         artificial_prop_values['overlap_percentages'] = [1.0,2.0,3.0]
# 
#         # Add mean of overlap percentages, as will be
#         # done when PerformanceResult is instantiated
#         # later:
#         artificial_prop_values['mean_overlaps'] = \
#             statistics.mean(artificial_prop_values['overlap_percentages'])

        # signal_treatment must be a SignalTreatmentDescriptor:
        artificial_prop_values['signal_treatment'] = SignalTreatmentDescriptor(-30,300,10)

        perf_res_inst = PerformanceResult(artificial_prop_values,
                                          overlap_percentages=self.overlap_percentages
                                          )
        return perf_res_inst

    #------------------------------------
    # performance_results_equality
    #-------------------
    
    def experiments_equality(self, exp_left, exp_right):
        errors = []
        for (key,val) in exp_left.items():
            if isinstance(val, np.ndarray):
                # Values are np.arrays; simple Python
                # __eq__ won't work. Also, for floats,
                # array_equal also fails over slight diffs.
                # Rather than using isclose(), just round
                # to 2 decimals for both:
                if not np.array_equal(np.round(val,2), 
                                      np.round(exp_right[key], 2)
                                      ):
                    errors.append(key)
                continue
            elif isinstance(val, PerformanceResult):
                if not self.performance_results_equality(val, exp_right[key]):
                    errors.append(key)
                continue
            elif isinstance(val, SignalTreatmentDescriptor):
                if not val.__eq__(exp_right[key]):
                    errors.append(key)
                continue
            if val != exp_right[key]:
                errors.append(key)
                continue
        if len(errors) == 0:
            return True
        for problem_key in errors:
            print(f"{problem_key}: Left: {exp_left[problem_key]}; Right: {exp_right[problem_key]}")
        return False

    
    #------------------------------------
    # performance_results_equality
    #-------------------
    
    def performance_results_equality(self, perf_res_left, perf_res_right):
        errors = []
        for (key,val) in perf_res_left.items():
            if isinstance(val, np.ndarray):
                # Values are np.arrays; simple Python
                # __eq__ won't work. Also, for floats,
                # array_equal also fails over slight diffs.
                # Rather than using isclose(), just round
                # to 2 decimals for both:
                if not np.array_equal(np.round(val,2), 
                                      np.round(perf_res_right[key], 2)
                                      ):
                    errors.append(key)
                continue
            elif isinstance(val, PerformanceResult):
                if not val.__eq__(perf_res_right[key]):
                    errors.append(key)
                continue
            elif isinstance(val, SignalTreatmentDescriptor):
                if not val.__eq__(perf_res_right[key]):
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