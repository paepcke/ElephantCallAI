'''
Created on Apr 6, 2020

@author: paepcke
'''
import os
import re
import sys

from calibrate_preprocessing import Experiment
from dsp_utils import DSPUtils, PrecRecFileTypes


class ExperimentExplorer(object):
    
    class NoExperimentFile(Exception):
        pass
    
    class NotAnExperimentFile(Exception):
        pass
    
    #------------------------------------
    # Constructor
    #-------------------

    def __init__(self, experiment_pointer):
        
        self.experiment_pointers = []
        try:
            # If a dir is given, find all experiment files 
            # in it. If arg is a file, ensure that it is an
            # experiment file. If no file found in given dir, 
            # or given file is not an experiment file, print
            # msg and exit:
             
            if os.path.isdir(experiment_pointer):
                exp_dir = os.path.dirname(experiment_pointer)
                # Find all experiment result files 
                # the dir:
                files_in_dir = os.listdir(exp_dir)
                # Piece of file name that indicates the file
                # is a saved experiment:
                experiment_indicator = PrecRecFileTypes.EXPERIMENT.value
                for file in files_in_dir:
                    if re.match(experiment_indicator, file):
                        self.experiment_pointers.append(os.path.join(exp_dir,file))
                if len(self.experiment_pointers) == 0:
                    raise self.NoExperimentFile()
            else:
                # arg is path to individual experiment file:
                if not os.path.exists(experiment_pointer):
                    raise self.NoExperimentFile
                if not self.is_experiment_file(experiment_pointer):
                    raise self.NotAnExperimentFile()
                self.experiment_pointers.append(experiment_pointer)
        except self.NoExperimentFile:
            print(f"No experiment file found in directory {experiment_pointer}")
            sys.exit(1)
        except self.NotAnExperimentFile:
            print(f"File {experiment_pointer} is not an experiment result file.")
            sys.exit(1)

        self.materialize_experiments()
        
    #------------------------------------
    # materialize_experiments
    #-------------------
    
    def materialize_experiments(self):
        self.experiments = []
        for exp_path in self.experiment_pointers:
            experiment = Experiment.instance_from_tsv(exp_path)
            self.experiments.append(experiment)

    #------------------------------------
    # print
    #-------------------

    def print(self, experiment_file=None):
        for experiment in self.experiments:
            experiment.print(experiment_file)
            
        
# ------------------ Main ---------------
if __name__ == '__main__':
    pass