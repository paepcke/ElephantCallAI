'''
Created on Apr 6, 2020

@author: paepcke
'''
# This file is just a support service for testing

import os
import statistics
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from calibrate_preprocessing import Experiment
from precision_recall_from_wav import PerformanceResult

class ExperimentObjCreator(object):
    '''
    Support for testing. Create a toy instance
    of Experiment, with values of results being
    1,2,3,4,...
    '''

    artificial_prop_values = {}
    
    #------------------------------------
    # make_experiment_instance
    #-------------------
    
    @classmethod
    def make_experiment_instance(cls):

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
            
            
        # Create a fresh PerformanceResult instance 
        perf_res = cls.create_perf_result()
        experiment = Experiment(cls.artificial_prop_values)
        experiment['experiment_res'] = perf_res
        return experiment
        
    #------------------------------------
    # create_perf_res
    #-------------------

    @classmethod
    def create_perf_result(cls):
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

        
# ----------------------------- Main ----------------

if __name__ == '__main__':
    creator = ExperimentObjCreator()
    exp_inst = creator.make_experiment_instance()
    #print(exp_inst)
    exp_inst.print()
        