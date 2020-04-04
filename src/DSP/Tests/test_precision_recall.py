'''
Created on Mar 27, 2020

@author: paepcke
'''
import csv
import os, sys
import statistics
import tempfile
import unittest

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from precision_recall_from_wav import PerformanceResult, PrecRecComputer

TEST_ALL = True
# ***TEST_ALL = False

class TestPrecisionRecall(unittest.TestCase):
    
    # A dict with all properties that PerformanceResult
    # instances hold, initialized to artificial values
    # 0,1,2,...:

    artificial_prop_values = {}
    framerate = 4000
    
    #------------------------------------
    # setUpClass
    #-------------------

    @classmethod
    def setUpClass(cls):
        super(TestPrecisionRecall, cls).setUpClass()

        # Create a dict of artificial results that would
        # be created by the PrecRecComputer. This dict will
        # be used for creating PerformanceResult instances
        # for testing:
        
        val = 0
        # The props class variable of PerformanceResult is
        # a dict {prop-name : <required-class-of-prop>}:
        for prop_name in PerformanceResult.props:
            try:
                # All properties need to be floats or ints.
                # But overlap_percentages must be a list,
                # and will cause an error here. We take
                # care of that after the loop:
                cls.artificial_prop_values[prop_name] = PerformanceResult.props[prop_name](val)
            except TypeError:
                continue
            val += 1
            
        # The overlap_percentages needs to be a list
        # of floats:
        cls.artificial_prop_values['overlap_percentages'] = [1.0,2.0,3.0]

        # Add mean of overlap percentages, as will be
        # done when PerformanceResult is instantiated
        # later:
        cls.artificial_prop_values['mean_overlaps'] = \
            statistics.mean(cls.artificial_prop_values['overlap_percentages'])
            

    #------------------------------------
    # setUp
    #-------------------

    def setUp(self):
        # Create a fresh PerformanceResult instance for each test 
        # to use.
        self.perf_res = PerformanceResult(self.artificial_prop_values)
        
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
        # Save a test result instance  to a .tsv file:
        
        self.perf_res.to_tsv(include_col_header=True, outfile=self.tmp_file_name, append=True)
        with open(self.tmp_file_name, 'r') as tmp_file_fd:
            reader = csv.reader(tmp_file_fd, delimiter='\t')
            col_header = next(reader)
            col_header_set = set(col_header)
            self.assertSetEqual(col_header_set, set(PerformanceResult.props.keys()))
            res_line = next(reader)
            res_dict = {}
            for (col_num, val) in enumerate(res_line):
                key = col_header[col_num]
                required_type = PerformanceResult.props[key]
                if isinstance(required_type, list):
                    # The list will be single-element, with
                    # value being the type of elements in the
                    # array:
                    required_type_in_arr = required_type[0]
                    res_arr = [required_type_in_arr(el) for el in eval(val)]
                    res_dict[key] = res_arr                    
                else:
                    res_dict[key] = required_type(val)
            # From what we picked out of the .tsv file,
            # create a PerformanceResult instance:
            retrieved_perf_res = PerformanceResult(res_dict)
            self.assertTrue(self.performance_results_equality(retrieved_perf_res, self.perf_res))

    #------------------------------------
    # test_from_tsv
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_from_tsv(self):
        # Make a new test result instance from a results
        # .tsv file:
        
        # Write our known PerformanceResult instance to tsv:
        self.perf_res.to_tsv(include_col_header=True, outfile=self.tmp_file_name, append=True)
        # Make a new instance, and read that tsv file:
        perf_inst_list = PerformanceResult.instances_from_tsv(self.tmp_file_name, first_line_is_col_header=True)

        restored_perf_res = perf_inst_list[0]
        for (prop_name, prop_value) in self.perf_res.items():
            self.assertEqual(prop_value, restored_perf_res[prop_name])

    #------------------------------------
    # test_compute_overlap_percentages
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_compute_overlap_percentages(self):
        
        audio_burst_indices = \
        np.array([[ 5, 10],      # +     covers exactly                  0
                  [40, 46],      # +     covers beyond borders           2
                  [47, 49],      # -     
                  [51, 66],      # +     lower within borders            3
                  [70, 75],      # +     upper within borders            5
                  [77, 80],      # +     use ele burst twice             5
                  [90, 95]       # -     more in aud than in labels      
                  ])
        
        elephant_burst_indices = \
        np.array([[ 5, 10],      #                                       0
                 [10, 11],       # abutting intervals                    -
                 [42, 45],       #                                       1
                 [50, 65],       #                                       3
                 [55, 70],       # overlapping intervals                 3
                 [74, 80]        #                                       5
                 ])
        
        (percent_overlaps, matches_aud) = self.prec_rec_computer.compute_overlap_percentage(audio_burst_indices, 
                                                                                            elephant_burst_indices)
        res_perc_overlaps = np.array([100., 100., 93.33333333, 73.33333333, 16.66666667, 50.])
        self.assertTrue(np.array_equal(np.round(percent_overlaps, 1), 
                                       np.round(res_perc_overlaps, 1)))
        res_aud_matches = np.array([[ 5, 10],
                                    [40, 46],
                                    [51, 66],
                                    [51, 66],
                                    [70, 75],
                                    [77, 80]
                                    ])
        self.assertTrue(np.array_equal(matches_aud, res_aud_matches))

    #------------------------------------
    # test_compute_performance
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_compute_performance(self):
        voltages = np.array([0,0,1,2,3,0,4,0,0,0,0,5,6,7,0,0])
        label_start_stops = np.array([[ 1, 3],  # incl on both sides
                                      [11,13],  # incl on both sides
                                      ])
        # The start/stop seconds (1,4,11,14) will at some
        # point be turned into sample indices by multiplying 
        # by the framerate. To keep the numbers small, pre-divide
        # by framerate to get 1,4,11,14 as the actual elephant
        # samples:
        label_start_stops = label_start_stops / self.framerate
        label_file_name = self.create_label_file(label_start_stops)
        perf_res = self.prec_rec_computer.compute_performance(voltages, label_file_name, 10)
        
        res_dict = {'recall_events'         : 1.0,
                    'precision_events'      : 0.6666666666666666,
                    'f1score_events'        : 0.8,
                    'recall_samples'        : 0.8333333333333334,
                    'precision_samples'     : 0.7142857142857143,
                    'f1score_samples'       : 0.7692307692307692,
                    'overlap_percentages'   : np.array([66.66666667, 100.]),
                    'true_pos_samples'      : 5,
                    'false_pos_samples'     : 2,
                    'true_neg_samples'      : 8,
                    'false_neg_samples'     : 1,
                    'true_pos_events'       : 2,
                    'false_pos_events'      : 1,
                    'true_neg_events'       : 4,
                    'false_neg_events'      : 0,
                    'mean_overlaps'         : 83.33333333333334
                    }
        true_res = PerformanceResult(res_dict)
        self.assertTrue(self.performance_results_equality(perf_res, true_res))

    #------------------------------------
    # test_label_file_reader
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_label_file_reader(self):
        
        # Label file will have burst start/stops
        # in seconds: 
        label_start_stops = np.array([[1.0,2.0],    # start/stop in seconds
                                      [3.0,4.0],
                                      [5.0,6.0],
                                      [7.0,8.0]])

        label_file_name = self.create_label_file(label_start_stops)
        
        # We will receive indices to samples that 
        # correspond to the above seconds: 
        label_start_stops *= self.framerate
        label_start_stops = label_start_stops.astype(int)
        # Add 1 to the end samples, b/c we work
        # with intervals open at the high end:
        label_start_stops[:,1] += 1
        
        (elephant_burst_indices, el_mask) = self.prec_rec_computer.label_file_reader(label_file_name)
        self.assertTrue(np.array_equal(elephant_burst_indices, label_start_stops))
        sanctified_el_mask = self.create_mask(label_start_stops)
        self.assertTrue(np.array_equal(el_mask, sanctified_el_mask))



# --------------------  Utils --------------------

    #------------------------------------
    # create_mask
    #-------------------
    
    def create_mask(self, start_stop_index_arr):
        '''
        Given an array of start/stop index pairs,
        return a mask that reflects the pairs.

        @param start_stop_index_arr: nx2 array of burst start/stop 
            indices into an imagined audio sample stream:  
        @type start_stop_index_arr: [[int, int]]
        @return array of 1s/0s
        @rtype [int]
        '''
        # Mask will be as long as the end index of
        # the last burst:
        mask = np.zeros(start_stop_index_arr[-1,-1] + 1, dtype=int)
        for (start,stop) in start_stop_index_arr:
            mask[start:stop] = 1
        
        return mask
        
    #------------------------------------
    # create_label_file
    #-------------------
    
    def create_label_file(self, start_stop_arr):
        '''
        Given a 2D array in which each row is a 
        2-column burst start-stop pair of ints:
        
        1. Create a file that looks like a Raven-exported 
           label file.
        2. Compute a 1/0 mask from the matrix
        3. Return a pair: (label_file_name, mask)
         
        @param start_stop_arr: nx2 array of burst start/stop 
            indices into an imagined audio sample stream:  
        @type start_stop_arr: [[int, int]]
        @return: label_file_name
        @rtype: str
        '''
        try:
            label_file_name = os.path.join(os.path.dirname(__file__),
                                           'fake_label_file'
                                           )
            os.remove(label_file_name)
        except Exception:
            pass
    
        with open(label_file_name, 'w') as fd:
            fd.write("Selection	View	Channel	Begin Time (s)	End Time (s)	Low Freq (Hz)	High Freq (Hz)	Begin Path	File Offset (s)	Begin File	site	hour	file date	date(raven)	Tag 1	Tag 2	notes	analyst")
            fd.write('\n')

            # Create one line of 'label' for each
            # row in the given start-stop array:
            generic_label_entry = "1	Spectrogram 1	1	1.293	6.464999999991385	19.7	43.3	L:\ELP\Projects\GabonBais\gb_sounds\ceb1_oct2011\CEB1_20111107_000000.wav	1.293	CEB1_20111107_000000.wav	CEB1	0	7-Nov-2011	7-Nov-2011				al"
            # Turn the string into an array of strings;
            generic_label_entry_arr = [str(entry) for entry in generic_label_entry.split('\t')]
            
            for (start, stop) in start_stop_arr:
                generic_label_entry_arr[4] = str(start)
                generic_label_entry_arr[5] = str(stop)
                fd.write('\t'.join(generic_label_entry_arr) + '\n')

        return label_file_name

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