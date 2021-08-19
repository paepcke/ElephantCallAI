#!/usr/bin/env python3

import argparse
from collections import defaultdict
import os
from pathlib import Path
import re
from typing import List, Dict

import numpy as np
import pandas as pd


TEGRASTATS_LINE_REGEX = "RAM\s(\d+\/\d+)MB\s\(.*\)\sSWAP\s(\d+\/\d+)MB\s\(.*\)\sCPU\s(\[.*\]).*?CPU@(\d+(?:\.\d+)?)C\s.*?GPU@(\d+(?:\.\d+)?)C.*?POM_5V_IN\s(\d+\/\d+)\sPOM_5V_GPU\s(\d+\/\d+)\sPOM_5V_CPU\s(\d+\/\d+)"

"""
The following is an example line from a tegrastats output log:

RAM 1738/3956MB (lfb 172x4MB) SWAP 0/1978MB (cached 0MB) CPU [7%@1224,0%@1224,3%@1224,7%@1224] EMC_FREQ 0% GR3D_FREQ 0% PLL@29C CPU@31C PMIC@100C GPU@31.5C AO@37C thermal@31.25C POM_5V_IN 1997/2089 POM_5V_GPU 40/57 POM_5V_CPU 203/289
"""

class JetsonLogParser:

    def __init__(self, 
                 log_file_list,
                 output_dir,
                 logging_rate
                 ):
        
        # Sample period in msecs:
        self.logging_period = 1/logging_rate

        df_active = None
        df_idle   = None
        for log_fname in log_file_list:
            try:
                metrics = self.fill_one_metrics_dict(log_fname)
                self.num_measurements = len(metrics["cpu_temp"])
            except Exception as e:
                print(f"Could not extract measures from {log_fname}: {repr(e)}")
            # Create or append to a growing dataframe,
            # either tare (idle processing), or active):

            exp_cond = self.experiment_conds_from_fname(log_fname)
            if exp_cond['model_name'] == 'tare':
                df_idle = self.add_measures_to_output(exp_cond,
                                                 metrics, 
                                                 df_idle
                                                 )
            else:
                df_active = self.add_measures_to_output(exp_cond,
                                                        metrics, 
                                                        df_active
                                                        )

        # Output consolidated active and/or tare measurments
        if df_active is not None:
            df_out_fpath = os.path.join(output_dir, 'power_measures.csv')
            df_active.to_csv(df_out_fpath, index=False)
            print(f"Active-power measurements are in {df_out_fpath}")
            
        if df_idle is not None:
            df_out_fpath = os.path.join(output_dir, 'tare_power_measures.csv')
            df_idle.to_csv(df_out_fpath, index=False)
            print(f"Tare-power measurements are in {df_out_fpath}")

    def experiment_conds_from_fname(self, log_path):

        log_fname = Path(log_path).stem
        if log_fname.startswith('tare'):
            cond = {'model_name' : 'tare'}
            if log_fname.find('not-listening') > 1:
                cond['activity'] = 'not-listening'
            else:
                cond['activity'] = 'constant-prediction'
            return cond

        # Must be a log from a real run:
        # Get word width, model name and batch size:
        try:
            return self.parse_logfile_name(log_fname)
        except ValueError:
            raise ValueError(f"Log file {log_fname} is not in standard file name form")

    
    def init_metrics_dict(self):
        metrics = defaultdict(lambda: [])
        return metrics
    
    
    def update_metrics_dict(self, metrics: Dict[str, List[float]], line: str):
        match = re.match(TEGRASTATS_LINE_REGEX, line)
    
        # ram and swap are fractions
        ram_str = match.group(1)
        swap_str = match.group(2)
    
        # cpu util is formatted like: [7%@1224,0%@1224,3%@1224,7%@1224]
        #cpu_util_str = match.group(3)
    
        # temp is in C
        cpu_temp_str = match.group(4)
        gpu_temp_str = match.group(5)
    
        # power draw is in mW, formatted like "X/Y" where X is current reading and Y is average reading (we throw Y away)
        powerdraw_str = match.group(6)
        gpupower_str = match.group(7)
        cpupower_str = match.group(8)
    
        # TODO: use more of these metrics?
    
        ram_util = self.parse_fraction(ram_str)
        swap_util = self.parse_fraction(swap_str)
    
        cpu_temp = float(cpu_temp_str)
        gpu_temp = float(gpu_temp_str)
    
        total_power = self.parse_cur_avg_reading(powerdraw_str)
        gpu_power = self.parse_cur_avg_reading(gpupower_str)
        cpu_power = self.parse_cur_avg_reading(cpupower_str)
    
        metrics["ram_util"].append(ram_util)
        metrics["swap_util"].append(swap_util)
        metrics["cpu_temp"].append(cpu_temp)
        metrics["gpu_temp"].append(gpu_temp)
        metrics["total_power"].append(total_power)
        metrics["gpu_power"].append(gpu_power)
        metrics["cpu_power"].append(cpu_power)
    
    
    def parse_fraction(self, frac_str: str):
        """
        Parses a fraction string into a floating point number
    
        :param frac_str: a string like "2/3"
        :return:
        """
        parts = frac_str.split("/")
        numerator, denominator = parts[0], parts[1]
        return float(numerator)/float(denominator)
    
    
    def parse_cur_avg_reading(self, reading: str):
        parts = reading.split("/")
        return float(parts[0])

    def create_overview_txt(self, metrics, savepath: str):
        keys_of_interest = ["total_power", "gpu_power", "cpu_power", "gpu_temp", "cpu_temp", "ram_util", "swap_util"]
    
        lines = []
        for key in keys_of_interest:
            lines.append(self.create_overview_string(metrics, key))
    
        with open(savepath, "w") as outfile:
            outfile.writelines(lines)
    
    
    def create_overview_string(self, metrics, key):
        array = np.array(metrics[key])
        mean = array.mean()
        var = array.var()
        arr_min = array.min()
        arr_max = array.max()
    
        return f"{key} - mean: {mean}, variance: {var}, minimum: {arr_min}, maximum: {arr_max}\n"
    
    
    def add_measures_to_output(self, exp_conds_dict, metrics, out_df):
        '''
        Given a completed metrics dict, create a dataframe.
        If out_df is not None, catenate that df to out_df.
        Return that df.
        
        The log_fname is used to 
        
        Metrics will have keys:
        
          ['ram_util', 'swap_util', 'cpu_temp', 
           'gpu_temp', 'total_power', 'gpu_power', 'cpu_power']
        
        The values for each key will be a list of measured
        result numbers. All lists will have equal length. 
    
        :param log_fname: jetson log file path
        :type str
        :param metrics: dict of measurement results
        :type metrics: {str : [float]}
        :param out_df dataframe to which result is appended,
            or, if None, a df is created
        :return: dataframe of accumulating results
        :rtype: pd.DataFrame
        '''
        
    
        if exp_conds_dict['model_name'] == 'tare':
            fld_names  = ['seconds', 'activity'] + list(metrics.keys())
        else:
            fld_names = ['seconds', 'model_name', 'batch_size', 'word_width'] + list(metrics.keys())
    
        sample_times = [i*self.logging_period for i in range(self.num_measurements)]
        df = pd.DataFrame(metrics, columns=fld_names)
        df.loc[:,'seconds'] = sample_times
        
        # Fill model name, batch size and word width
        # with the constants taken from the log file name:
        df.loc[:,'model_name'] = [exp_conds_dict['model_name']]*self.num_measurements
        
        if exp_conds_dict['model_name'] == 'tare':
            df.loc[:,'activity'] = [exp_conds_dict['activity']]*self.num_measurements
        else:
            df.loc[:,'batch_size'] = [exp_conds_dict['batch_size']]*self.num_measurements
            df.loc[:,'word_width'] = [exp_conds_dict['word_width']]*self.num_measurements
    
        if out_df is None:
            out_df = df
        else:
            out_df = pd.concat([out_df, df], axis=0)
        return  out_df
    
    def fill_one_metrics_dict(self, log_file_name):
        metrics = self.init_metrics_dict()
        with open(log_file_name, "r") as logfile:
            for line in logfile:
                # Some log files contain a line of
                # all zeroes at the end, just ignore:
                if ord(line[0]) == 0:
                    continue
                self.update_metrics_dict(metrics, line)
        return metrics
    
    def parse_logfile_name(self, dirname):
        '''
        Given a dir name like float16-mobilenetV2-batchsize-16,
        or float32-resnet-101-batchsize-16, or float32-trained-model-batchsize-64,
        or tare-not-listening, pull out word width, model name, and
        batch size. Return them as a tuple.
        
        :param dir_name:
        :type dir_name:
        :return: dict with keys word_width, model_name, batch_size
        :rtype: {str : int, str : str, str : int}
        '''
    
        try:
            word_width = int(dirname[5:7])
        except Exception as e:
            msg = f"{repr(e)}; dirname: {dirname}"
            e.args = (msg,)
            raise 
        rest = dirname[8:]
        # Now have one of:
        #    resnet-101
        #    trained-model
        #    mobilenetV2
        if rest.startswith('resnet-101'):
            model_name = 'resnet101'
        elif rest.startswith('resnet-50'):
            model_name = 'resnet-50'
        elif rest.startswith('mobilenetV2'):
            model_name = 'mobilenetV2'
        elif rest.startswith('trained-model'):
            model_name = 'resnet18'
        else:
            raise ValueError(f"Cannot find model name in '{rest}'")
    
        fragments  = rest.split('-')
        batch_size = int(fragments[-1])
        
        return {'word_width' : word_width, 
                'model_name' : model_name, 
                'batch_size' : batch_size
                }
        
# --------------------- Main -------------

if __name__ == "__main__":

    def get_args():
        '''
        Augments args with new args.log_file_list. 
        If tegrastats-log-file is a file, the value
        of log_file_list is that file placed inside 
        a list.
        
        If tegrastats-log-file is a directory, the value
        of log_file_list will be a list of all .txt file
        names in the directory.
        
        Additionally, if neither --csv nor --charts are provided,
        will set --charts to true.
        '''
        parser = argparse.ArgumentParser()
    
        parser.add_argument('--logs-loc', type=str, required=True, help="path to a tegrastats log file or dir of logs")
        parser.add_argument('--output-dir', type=str, required=True, help="path to a directory to store output in")
        parser.add_argument('--logging_rate', 
                            type=int, 
                            required=False,
                            default=5, 
                            help="sampling frequency of tegrastats data in Hz; default 5Hz")
    
        args = parser.parse_args()
            
        # Check whether log file info is a dir or .txt file:
        if os.path.isfile(args.logs_loc):
            args.log_file_list = [args.logs_loc]
        else:
            # Find all .txt files:
            args.log_file_list = [os.path.join(args.logs_loc, log_fname)
                                  for log_fname
                                  in os.listdir(args.logs_loc)
                                  if log_fname.endswith('.txt')
                                  ]
        return args

    args = get_args()
    JetsonLogParser(args.log_file_list,
                    args.output_dir,
                    args.logging_rate
                    )
