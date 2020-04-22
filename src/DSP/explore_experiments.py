'''
Created on Apr 6, 2020

@author: paepcke
'''
import os
import re
import sys

from calibrate_preprocessing import Experiment
from dsp_utils import DSPUtils, PrecRecFileTypes

from prettytable import PrettyTable
from DSP.precision_recall_from_wav import PerformanceResult

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
            # If a dir is given, find all pickled experiment files 
            # in it. If arg is a file, ensure that it is an
            # experiment file. If no file found in given dir, 
            # or given file is not an experiment file, print
            # msg and exit:
             
            if os.path.isdir(experiment_pointer):
                # Find all experiment result files 
                # the dir:
                files_in_dir = os.listdir(experiment_pointer)
                # Piece of file name that indicates the file
                # is a saved experiment:
                experiment_indicator = PrecRecFileTypes.PICKLE.value
                re_pat = re.compile(f".*{experiment_indicator}.*")
                for file in files_in_dir:
                    if re.match(re_pat, file):
                        self.experiment_pointers.append(os.path.join(experiment_pointer,file))
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

        experiments = self.materialize_experiments()
        
        #html_tables = self.make_html_tables(experiments)
        txt_tables = self.make_txt_tables(experiments)
        for tbl in txt_tables:
            print(tbl)
        print("Done")
        
    #------------------------------------
    # make_html_tables
    #-------------------

    def make_html_tables(self, experiments):
        
        if not isinstance(experiments, list):
            experiments = [experiments]
        
    #------------------------------------
    # make_txt_tables
    #-------------------

    def make_txt_tables(self, experiments):
        
        txt_tables = []
        for experiment in experiments:
            tbl = PrettyTable()
            tbl.field_names = ['Measurement', 'Value']
            tbl.align["Measurement"] = "l"
            tbl.align["Value"] = "l"
            for (prop_name, prop_type) in experiment.props.items():
                if prop_type in [int, str, float]:
                    tbl.add_row([prop_name, experiment[prop_name]])
            # Get the PerformanceResult:
            perf_res = experiment['experiment_res']
            for (prop_name, prop_val) in perf_res.items():
                tbl.add_row([prop_name, prop_val])
            txt_tables.append(tbl)
            #overlap_stats = experiment['overlaps_summary']
        return txt_tables
    
    #------------------------------------
    # print
    #-------------------

    def print(self, experiment_file=None):
        for experiment in self.experiments:
            experiment.print(experiment_file)
            
    #------------------------------------
    # materialize_experiments
    #-------------------
    
    def materialize_experiments(self):
        experiments = []
        for exp_path in self.experiment_pointers:
            # Each .tsv may contain rows for multiple 
            # experiments; the the first only for now:
            experiment = Experiment.instances_from_tsv(exp_path)[0]
            experiments.append(experiment)
        return experiments


    #------------------------------------
    # is_experiment_file
    #-------------------
    
    def is_experiment_file(self, file_path):
        pass


# ------------------ Main ---------------
if __name__ == '__main__':
    exp_explorer = ExperimentExplorer('/tmp')
