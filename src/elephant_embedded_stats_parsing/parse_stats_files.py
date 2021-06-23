#!/usr/bin/env python3
'''
Created on Jun 22, 2021

@author: paepcke
'''
import csv
import os
from pathlib import Path


class EleStatsParser(object):
    '''
    Finds all files called stats.txt under the SEARCH_ROOT
    directory. Creates in this script's dir a file all_stats.csv
    with a spreadsheet for all values of power consumption.
    
    Prediction accuracy is NOT covered.
    '''
    SEARCH_ROOT = '/Users/paepcke/Project/Wildlife/Elephants/Papers/embedding/Data/ElephantsEmbedded/data-for-paper-v1/power-analysis'

    def __init__(self, unittesting=False):
        '''
        Constructor
        '''
        cur_dir = os.path.dirname(__file__)
        outfile = os.path.join(cur_dir, 'all_stats.csv')
        if os.path.exists(outfile):
            print(f"Overwriting existing all_stats.csv file")
        
        col_header = [
            'ModelType','WordWidth','BatchSize',
            'TotalPowerMean','TotalPowerVar','TotalPowerMin','TotalPowerMax',
            'GPUPowerMean','GPUPowerVar','GPUPowerMin','GPUPowerMax',
            'CPUPowerMean','CPUPowerVar','CPUPowerMin','CPUPowerMax',
            'GPUTempMean','GPUTempVar','GPUTempMin','GPUTempMax',
            'CPUTempMean','CPUTempVar','CPUTempMin','CPUTempMax',
            'RAMUtilMean','RAMUtilVar','RAMUtilMin','RAMUtilMax']

        if unittesting:
            # Unittests must close!
            self.out_fd = open(outfile, 'w')
            self.csv_writer = csv.writer(self.out_fd)
            self.csv_writer.writerow(col_header)
            return
        
        with open(outfile, 'w') as out_fd:
            csv_writer = csv.writer(out_fd)
            csv_writer.writerow(col_header)

            for root, _dirs, files in os.walk(self.SEARCH_ROOT):
                for file in files:
                    dir_name = Path(root).stem
                    if file == 'stats.txt' and dir_name.startswith('float'):
                        # Directory name holds experiment conditions:
                        hyperparams_dict = self.parse_dir_name(dir_name)
                    else:
                        continue
                    self.add_to_results(os.path.join(root, file),
                                        hyperparams_dict['model_name'],
                                        hyperparams_dict['word_width'],
                                        hyperparams_dict['batch_size'],
                                        csv_writer)
            print(f"Results are in {out_fd.name}")

    #------------------------------------
    # parse_dir_name
    #-------------------
    
    def parse_dir_name(self, dirname):
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

        word_width = int(dirname[5:7])
        rest = dirname[8:]
        # Now have one of:
        #    resnet-101
        #    trained-model
        #    mobilenetV2
        if rest.startswith('resnet-101'):
            model_name = 'resnet101'
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

    #------------------------------------
    # add_to_results 
    #-------------------
    
    def add_to_results(self, 
                       stats_file_path, 
                       model_name, 
                       word_width, 
                       batch_size, 
                       csv_writer):
        '''
        
        Finds file 'stats.txt' in the given directory, and 
        parses it. Adds one line to the csv writer.
        
        Example stats file:
            total_power - mean: 1647.2224350361626, variance: 15028.646071667012, minimum: 1556.0, maximum: 3951.0
            gpu_power - mean: 50.873845802006805, variance: 5064.043460207662, minimum: 40.0, maximum: 1854.0
            cpu_power - mean: 171.8763614533604, variance: 2189.5561665843106, minimum: 122.0, maximum: 933.0
            gpu_temp - mean: 27.40270432520511, variance: 0.12991635947616126, minimum: 26.0, maximum: 29.0
            cpu_temp - mean: 27.61487664732283, variance: 0.07990389619316765, minimum: 26.5, maximum: 29.0
            ram_util - mean: 0.8102191759222557, variance: 0.0038489174379519138, minimum: 0.32088799192734613, maximum: 0.8377901109989909
            swap_util - mean: 0.4107095213947914, variance: 0.00014229665385782252, minimum: 0.35317860746720486, maximum: 0.41422805247225025
         
         
        Assumption: a column name header line has already been written
            to the csv file:
            
              ModelType, WordWidth, BatchSize, 
              TotalPowerMean,TotalPowerVar,TotalPowerMin,TotalPowerMax,
              GPUPowerMean,GPUPowerVar,GPUPowerMin,GPUPowerMax, 
              CPUPowerMean,CPUPowerVar,CPUPowerMin,CPUPowerMax 
              GPUTempMean,GPUTempVar,GPUTempMin,GPUTempMax,
              CPUTempMean,CPUTempVar,CPUTempMin,CPUTempMax, 
              RAMUtilMean,RAMUtilVar,RAMUtilMin,RAMUtilMax
              
        the swap_util input line is ignored. 

        :param stats_file_path: directory containing the stats file
        :type stats_file_path: str
        :param model_name: one of 'resnet18', 'resnet101', and 'mobilenetV2'
        :type model_name: str
        :param word_width: 16 or 32 processor word width
        :type word_width: int
        :param batch_size: batch size for inference
        :type batch_size: int
        :param csv_writer: open CSV writer to which a line is added
        :type csv_writer: csv.Writer
        '''
        
        csv_line = [model_name, word_width, batch_size]
        with open(stats_file_path, 'r') as in_fd:
            # Ensure that the sequence of lines is consistent:
            line = in_fd.readline().strip()
            
            total_power_dict = self.verify_stats_line(line, 'total_power')
            csv_line.extend(list(total_power_dict.values()))
            line = in_fd.readline().strip()
            
            gpu_power_dict   = self.verify_stats_line(line, 'gpu_power')
            csv_line.extend(list(gpu_power_dict.values()))
            line = in_fd.readline().strip()
            
            cpu_power_dict   = self.verify_stats_line(line, 'cpu_power')
            csv_line.extend(list(cpu_power_dict.values()))
            line = in_fd.readline().strip()
            
            gpu_temp_dict    = self.verify_stats_line(line, 'gpu_temp')
            csv_line.extend(list(gpu_temp_dict.values()))
            line = in_fd.readline().strip()
            
            cpu_temp_dict    = self.verify_stats_line(line, 'cpu_temp')
            csv_line.extend(list(cpu_temp_dict.values()))
            line = in_fd.readline().strip()
            
            ram_util_dict    = self.verify_stats_line(line, 'ram_util')
            csv_line.extend(list(ram_util_dict.values()))
            
        csv_writer.writerow(csv_line)

    #------------------------------------
    # verify_stats_line 
    #-------------------
    
    def verify_stats_line(self, line, line_start):
        '''
        Take one line like:
        
           gpu_temp - mean: 27.40270432520511, variance: 0.12991635947616126, minimum: 26.0, maximum: 29.0
           
        and the line starter, in the above example 'gpu_temp'.
        Ensures that the line starts with "<line_start> - ".
        Then creates dict from the statistic/value pairs. Returns
        that dict. 
               
        :param line: line from stats.txt file
        :type line: str
        :param line_start: header of the line
        :type line_start: str
        :return: dict {'mean' : float,
                       'variance': float,
                       'minimum': float,
                       'maximim': float
                       }
        :rtype: {str : float}
        :raise ValueError if line does not start with line_start
        '''
        
        full_starter = f"{line_start} - "
        if not line.startswith(full_starter):
            raise ValueError(f"Unexpected line: {line}; expecting {line_start}")
        # Get "mean: xxx, variance: ..., minimum: ... maximum: ..."
        data_txt = line[len(full_starter):]
        # Now have:
        #    'mean: 1796.3218939620015, variance: 106223.79810515656, ...'
        # As intermediate step, get:
        #    ['mean: 1796.3218939620015', ' variance: 106223.79810515656', ...]
        stat_name_val_stringlets = data_txt.split(',')
        
        # Finally, get:
        #   {'mean': 1796.3218939620015, 'variance': 106223.79810515656, ...}
        # by splitting each 'stringlet' on colon:

        data = {statistic.strip() : float(val.strip())
                for statistic, val
                in map(lambda stringlet: stringlet.split(':'), 
                       stat_name_val_stringlets)
                }
        
        return data
        

# ------------------------ Main ------------
if __name__ == '__main__':

    EleStatsParser()