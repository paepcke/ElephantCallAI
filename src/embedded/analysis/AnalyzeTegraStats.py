import argparse
from collections import defaultdict
from typing import List, Dict
import re
import os
from matplotlib import pyplot as plt
import numpy as np


TEGRASTATS_LINE_REGEX = "RAM\s(\d+\/\d+)MB\s\(.*\)\sSWAP\s(\d+\/\d+)MB\s\(.*\)\sCPU\s(\[.*\]).*?CPU@(\d+(?:\.\d+)?)C\s.*?GPU@(\d+(?:\.\d+)?)C.*?POM_5V_IN\s(\d+\/\d+)\sPOM_5V_GPU\s(\d+\/\d+)\sPOM_5V_CPU\s(\d+\/\d+)"

"""
The following is an example line from a tegrastats output log:

RAM 1738/3956MB (lfb 172x4MB) SWAP 0/1978MB (cached 0MB) CPU [7%@1224,0%@1224,3%@1224,7%@1224] EMC_FREQ 0% GR3D_FREQ 0% PLL@29C CPU@31C PMIC@100C GPU@31.5C AO@37C thermal@31.25C POM_5V_IN 1997/2089 POM_5V_GPU 40/57 POM_5V_CPU 203/289
"""


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--tegrastats-log-file', type=str, required=True, help="path to a tegrastats log file")
    parser.add_argument('--tegrastats-sample-freq-ms', type=int, required=True, help="sampling frequency of tegrastats data")
    parser.add_argument('--output-dir', type=str, required=True, help="path to a directory to store output in")

    return parser.parse_args()


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

    sample_period_s = args.tegrastats_sample_freq_ms/1000

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


def main():
    args = get_args()

    metrics = init_metrics_dict()
    with open(args.tegrastats_log_file, "r") as logfile:
        for line in logfile:
            update_metrics_dict(metrics, line)

    create_plots(args, metrics)


if __name__ == "__main__":
    main()
