#!/usr/bin/env python3

import argparse
from collections import defaultdict
import os
from pathlib import Path
import re
from typing import List, Dict

from matplotlib import pyplot as plt

import numpy as np
import pandas as pd


TEGRASTATS_LINE_REGEX = "RAM\s(\d+\/\d+)MB\s\(.*\)\sSWAP\s(\d+\/\d+)MB\s\(.*\)\sCPU\s(\[.*\]).*?CPU@(\d+(?:\.\d+)?)C\s.*?GPU@(\d+(?:\.\d+)?)C.*?POM_5V_IN\s(\d+\/\d+)\sPOM_5V_GPU\s(\d+\/\d+)\sPOM_5V_CPU\s(\d+\/\d+)"

"""
The following is an example line from a tegrastats output log:

RAM 1738/3956MB (lfb 172x4MB) SWAP 0/1978MB (cached 0MB) CPU [7%@1224,0%@1224,3%@1224,7%@1224] EMC_FREQ 0% GR3D_FREQ 0% PLL@29C CPU@31C PMIC@100C GPU@31.5C AO@37C thermal@31.25C POM_5V_IN 1997/2089 POM_5V_GPU 40/57 POM_5V_CPU 203/289
"""

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
    parser.add_argument('--csv', action='store_true',required=False, help="whether or not to output csv files")
    parser.add_argument('--charts', action='store_true',required=False, help="whether or not to output graphs as png files")
    parser.add_argument('--sample-rate-hz', 
                        type=int, 
                        required=True,
                        default=None, 
                        help="sampling frequency of tegrastats data in Hz")

    args = parser.parse_args()
    # If no action specified, assume charts:
    if not (args.csv or args.charts):
        args.charts = True
        
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

def init_metrics_dict():
    metrics = defaultdict(lambda: [])
    return metrics


def update_metrics_dict(metrics: Dict[str, List[float]], line: str):
    match = re.match(TEGRASTATS_LINE_REGEX, line)

    # ram and swap are fractions
    ram_str = match.group(1)
    swap_str = match.group(2)

    # cpu util is formatted like: [7%@1224,0%@1224,3%@1224,7%@1224]
    cpu_util_str = match.group(3)

    # temp is in C
    cpu_temp_str = match.group(4)
    gpu_temp_str = match.group(5)

    # power draw is in mW, formatted like "X/Y" where X is current reading and Y is average reading (we throw Y away)
    powerdraw_str = match.group(6)
    gpupower_str = match.group(7)
    cpupower_str = match.group(8)

    # TODO: use more of these metrics?

    ram_util = parse_fraction(ram_str)
    swap_util = parse_fraction(swap_str)

    cpu_temp = float(cpu_temp_str)
    gpu_temp = float(gpu_temp_str)

    total_power = parse_cur_avg_reading(powerdraw_str)
    gpu_power = parse_cur_avg_reading(gpupower_str)
    cpu_power = parse_cur_avg_reading(cpupower_str)

    metrics["ram_util"].append(ram_util)
    metrics["swap_util"].append(swap_util)
    metrics["cpu_temp"].append(cpu_temp)
    metrics["gpu_temp"].append(gpu_temp)
    metrics["total_power"].append(total_power)
    metrics["gpu_power"].append(gpu_power)
    metrics["cpu_power"].append(cpu_power)


def parse_fraction(frac_str: str):
    """
    Parses a fraction string into a floating point number

    :param frac_str: a string like "2/3"
    :return:
    """
    parts = frac_str.split("/")
    numerator, denominator = parts[0], parts[1]
    return float(numerator)/float(denominator)


def parse_cur_avg_reading(reading: str):
    parts = reading.split("/")
    return float(parts[0])


def create_plots(args, metrics):
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    #**********
    # Should be number of msecs: 1/args.tegrastats_sample_freq_hz/
    # and have x-axis be msecs, rather than secs?
    sample_period_s = args.tegrastats_sample_freq_hz/1000
    #**********

    # create the x axis labels
    sample_times = [i*sample_period_s for i in range(len(metrics["cpu_temp"]))]

    # plot CPU and GPU temp
    plt.figure()
    plt.plot(sample_times, metrics["cpu_temp"], label="CPU temp", color="blue")
    plt.plot(sample_times, metrics["gpu_temp"], label="GPU temp", color="red")
    plt.legend()
    plt.title("Processor Temperature During Operation")
    plt.xlabel("Time (s)")
    plt.ylabel("Component Temperature (C)")
    plt.savefig(f"{args.output_dir}/component_temp.png")

    # plot RAM/SWAP utilization
    plt.figure()
    plt.plot(sample_times, metrics["ram_util"], label="RAM utilization", color="orange")
    plt.plot(sample_times, metrics["swap_util"], label="Swap utilization", color="cyan")
    plt.legend()
    plt.title("Memory Utilization During Operation")
    plt.xlabel("Time (s)")
    plt.ylabel("Fraction of memory utilized")
    plt.savefig(f"{args.output_dir}/memory_utilization.png")

    # plot power separately and combined for convenience
    plot_individual_power(sample_times, metrics["cpu_power"], "CPU Power Draw During Operation", f"{args.output_dir}/cpu_power.png")
    plot_individual_power(sample_times, metrics["gpu_power"], "GPU Power Draw During Operation", f"{args.output_dir}/gpu_power.png")
    plot_individual_power(sample_times, metrics["total_power"], "Total Power Draw During Operation", f"{args.output_dir}/total_power.png")

    # plot power curves combined
    plt.figure()
    plt.plot(sample_times, metrics["cpu_power"], label="CPU power", color="blue")
    plt.plot(sample_times, metrics["gpu_power"], label="GPU power", color="red")
    plt.plot(sample_times, metrics["total_power"], label="Total power", color="purple")
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Power Consumption (mW)")
    plt.title("Power Consumption During Operation")
    plt.savefig(f"{args.output_dir}/power_draw.png")

    # create overview txt file
    create_overview_txt(metrics, f"{args.output_dir}/stats.txt")


def create_overview_txt(metrics, savepath: str):
    keys_of_interest = ["total_power", "gpu_power", "cpu_power", "gpu_temp", "cpu_temp", "ram_util", "swap_util"]

    lines = []
    for key in keys_of_interest:
        lines.append(create_overview_string(metrics, key))

    with open(savepath, "w") as outfile:
        outfile.writelines(lines)


def create_overview_string(metrics, key):
    array = np.array(metrics[key])
    mean = array.mean()
    var = array.var()
    arr_min = array.min()
    arr_max = array.max()

    return f"{key} - mean: {mean}, variance: {var}, minimum: {arr_min}, maximum: {arr_max}\n"


def plot_individual_power(x, y, title, savepath: str):
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Power Consumption (mW)")
    plt.savefig(savepath)

def output_csv_files(log_fname, metrics, out_df, sr_hz):
    '''
    Given a completed metrics dict, create an
    informative filename, and output as CSV to
    args.output_dir.
    
    Metrics will have keys:
    
      ['ram_util', 'swap_util', 'cpu_temp', 
       'gpu_temp', 'total_power', 'gpu_power', 'cpu_power']
    
    The values for each key will be a list of measured
    result numbers. All lists will have equal length. 

    :param args: command line args
    :type args: {str : Any}
    :param metrics: dict of measurement results
    :type metrics: {str : [float]}
    :return: full path to csv output file
    :rtype: str
    '''
    
    # Get word width, model name and batch size:
    try:
        log_experiment_id = Path(log_fname).stem
        exp_conds_dict = parse_logfile_name(log_experiment_id)
    except ValueError:
        print(f"Log file {log_experiment_id} is not in standard file name form")
        # Do nothing:
        return out_df

    fld_names = ['msecs', 'model_name', 'batch_size', 'word_width'] + list(metrics.keys())

    # Sample period in msecs:
    sample_period_ms = 1000 * 1/sr_hz
    num_measures = len(metrics["cpu_temp"])
    sample_times = [i*sample_period_ms for i in range(num_measures)]
    df = pd.DataFrame(metrics, columns=fld_names)
    df.loc[:,'msecs'] = sample_times
    
    # Fill model name, batch size and word width
    # with the constants taken from the log file name:
    df.loc[:,'model_name'] = [exp_conds_dict['model_name']]*num_measures
    df.loc[:,'batch_size'] = [exp_conds_dict['batch_size']]*num_measures
    df.loc[:,'word_width'] = [exp_conds_dict['word_width']]*num_measures

    if out_df is None:
        out_df = df
    else:
        out_df = pd.concat([out_df, df], axis=0)
    return  out_df

def fill_one_metrics_dict(log_file_name):
    metrics = init_metrics_dict()
    with open(log_file_name, "r") as logfile:
        for line in logfile:
            update_metrics_dict(metrics, line)
    return metrics

def parse_logfile_name(dirname):
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

def main():
    args = get_args()

    df = None
    for log_fname in args.log_file_list:
        try:
            metrics = fill_one_metrics_dict(log_fname)
        except Exception as e:
            print(f"Could not extract measures from {log_fname}: {repr(e)}")
        if args.csv:
            # Create or append to a growing dataframe:
            df = output_csv_files(log_fname, 
                                  metrics, 
                                  df, 
                                  args.sample_rate_hz)
        # Needs updating to do charts for all log files
        #if args.charts:
        #    create_plots(args, metrics)
    if df is not None:
        df_out_fpath = os.path.join(args.output_dir, 'power_measures.csv')
        df.to_csv(df_out_fpath, index=False)
        print(f"Power measurements are in {df_out_fpath}")

if __name__ == "__main__":
    main()
