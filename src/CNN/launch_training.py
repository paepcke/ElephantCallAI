#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals

from argparse import ArgumentParser, REMAINDER
import argparse
import json
from json.decoder import JSONDecodeError
import os
import sys
import re
import socket
import subprocess
import signal

import GPUtil


r"""
Based on torch.distributed.launch, with additions by Andreas Paepcke
(https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py)
Many of the comments are the originals. For more relevant documentation,
see git@github.com:paepcke/bert_train_parallel.git

o Adds ability to read world configuration from JSON file.

---------------------------------------------
`torch.distributed.launch` is a module that spawns up multiple distributed
training processes on each of the training nodes.

The utility can be used for single-node distributed training, in which one or
more processes per node will be spawned. The utility can be used for either
CPU training or GPU training. If the utility is used for GPU training,
each distributed process will be operating on a single GPU. This can achieve
well-improved single-node training performance. It can also be used in
multi-node distributed training, by spawning up multiple processes on each node
for well-improved multi-node distributed training performance as well.
This will especially be benefitial for systems with multiple Infiniband
interfaces that have direct-GPU support, since all of them can be utilized for
aggregated communication bandwidth.

In both cases of single-node distributed training or multi-node distributed
training, this utility will launch the given number of processes per node
(``--nproc_per_node``). If used for GPU training, this number needs to be less
or equal to the number of GPUs on the current system (``nproc_per_node``),
and each process will be operating on a single GPU from *GPU 0 to
GPU (nproc_per_node - 1)*.

**How to use this module:**

1. Single-Node multi-process distributed training

::

    >>> python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE
               YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3 and all other
               arguments of your training script)

2. Multi-Node multi-process distributed training: (e.g. two nodes)


Node 1: *(IP: 192.168.1.1, and has a free port: 1234)*

::

    >>> python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE
               --nnodes=2 --node_rank=0 --master_addr="192.168.1.1"
               --master_port=1234 YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3
               and all other arguments of your training script)

Node 2:

::

    >>> python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE
               --nnodes=2 --node_rank=1 --master_addr="192.168.1.1"
               --master_port=1234 YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3
               and all other arguments of your training script)

3. To look up what optional arguments this module offers:

::

    >>> python -m torch.distributed.launch --help


**Important Notices:**

1. This utility and multi-process distributed (single-node or
multi-node) GPU training currently only achieves the best performance using
the NCCL distributed backend. Thus NCCL backend is the recommended backend to
use for GPU training.

2. In your training program, you must parse the command-line argument:
``--local_rank=LOCAL_PROCESS_RANK``, which will be provided by this module.
If your training program uses GPUs, you should ensure that your code only
runs on the GPU device of LOCAL_PROCESS_RANK. This can be done by:

Parsing the local_rank argument

::

    >>> import argparse
    >>> parser = argparse.ArgumentParser()
    >>> parser.add_argument("--local_rank", type=int)
    >>> args = parser.parse_args()

Set your device to local rank using either

::

    >>> torch.cuda.set_device(arg.local_rank)  # before your code runs

or

::

    >>> with torch.cuda.device(arg.local_rank):
    >>>    # your code to run

3. In your training program, you are supposed to call the following function
at the beginning to start the distributed backend. You need to make sure that
the init_method uses ``env://``, which is the only supported ``init_method``
by this module.

::

    torch.distributed.init_process_group(backend='YOUR BACKEND')

4. In your training program, you can either use regular distributed functions
or use :func:`torch.nn.parallel.DistributedDataParallel` module. If your
training program uses GPUs for training and you would like to use
:func:`torch.nn.parallel.DistributedDataParallel` module,
here is how to configure it.

::

    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[arg.local_rank],
                                                      output_device=arg.local_rank)

Please ensure that ``device_ids`` argument is set to be the only GPU device id
that your code will be operating on. This is generally the local rank of the
process. In other words, the ``device_ids`` needs to be ``[args.local_rank]``,
and ``output_device`` needs to be ``args.local_rank`` in order to use this
utility

5. Another way to pass ``local_rank`` to the subprocesses via environment variable
``LOCAL_RANK``. This behavior is enabled when you launch the script with
``--use_env=True``. You must adjust the subprocess example above to replace
``args.local_rank`` with ``os.environ['LOCAL_RANK']``; the launcher
will not pass ``--local_rank`` when you specify this flag.

.. warning::

    ``local_rank`` is NOT globally unique: it is only unique per process
    on a machine.  Thus, don't use it to decide if you should, e.g.,
    write to a networked filesystem.  See
    https://github.com/pytorch/pytorch/issues/12042 for an example of
    how things can go wrong if you don't do this correctly.

"""

num_gpus_here = len(GPUtil.getGPUs())

#------------------------------------
# parse_world_layout_config 
#-------------------

def parse_world_layout_config(other_gpu_config_file):
    '''
    Parse JSON config file that describes how many
    GPUs different machines have. Expect any entries
    like:
    
       {"foo.bar.com" : 4,
        "127.0.0.1"   : 5,
        "localhost"   : 3,
        "172.12.145.1 : 6
       }

    Ensures there is an entry for 
    "localhost"
    
    @param other_gpu_config_file:
    @type other_gpu_config_file:
    '''
    try:
        with open(other_gpu_config_file, 'r') as config_fd:
            config_dict = json.load(config_fd)
    except FileNotFoundError:
        print(f"Could not find or open config file {other_gpu_config_file}")
        sys.exit(1)
    except JSONDecodeError as e:
        print(f"Bad JSON in config file {other_gpu_config_file}: {repr(e)}")
        sys.exit(1)

    # Ensure that this machine's node entry is
    # 'localhost'. It is legal in the config file
    # to use 'localhost', '127.0.0.1', or the hostname's FQDN as
    # key in the config file for local host.
    # Get name of this machine. 
    my_hostname = socket.gethostname().split('.')[0]
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("google.com",80))
    
    # Get something like ('172.24.75.114', 44572)  
    my_global_ip_addr = s.getsockname()[0] 

    # Find whether some version of this host
    # is referenced in the dict; if it's anything
    # other than 'localhost', replace the entry
    # with 'lcoalhost' as key:
    for node_name_or_addr in config_dict.copy().keys():
        if node_name_or_addr   == '127.0.0.1' or\
            node_name_or_addr.split('.')[0] == my_hostname or\
            node_name_or_addr  == my_global_ip_addr:
            
            config_dict['localhost'] = config_dict[node_name_or_addr]
            del config_dict[node_name_or_addr]

    if 'localhost' not in config_dict.keys():
        config_dict['localhost'] = len(GPUtil.getGPUs())

    # Values should all be ints: number of GPUs:
    for (node, num_gpus) in config_dict.items():
        if type(num_gpus) != int:
            print(f"Number of GPUs at node {node} in config file {other_gpu_config_file} should be an int.")
            sys.exit(1)

    return config_dict

#------------------------------------
# parse_args 
#-------------------
class BlankLinesHelpFormatter (argparse.HelpFormatter):
    
    def _split_lines(self, text, width):
        if text.find('\n') == -1:
            lines = super()._split_lines(text, width)
        else:
            lines = text.split('\n')
        split_lines = []
        for line in lines:
            while True:
                try:
                    nl_pos = line.index('\n')
                    one_line = line[:nl_pos]
                    split_lines.append(one_line)
                    line = line[nl_pos+1:]
                except ValueError:
                    # No more NLs:
                    split_lines.append(line)
                    break
        return split_lines
    
def parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    
    curr_dir = os.path.dirname(__file__)
    training_script = os.path.join(curr_dir, 'spectrogram_train_parallel.py')
    
    # Get the help string from spectrogram_train_parallel.py:
    proc = subprocess.run([training_script, '-h'], capture_output=True)
    # Decode needed b/c proc.stdout is byte string:
    script_help = proc.stdout.decode('utf8')
    
    parser = ArgumentParser(formatter_class=BlankLinesHelpFormatter,
                            description="PyTorch distributed training launch "
                                        "helper utility that will spawn up "
                                        "multiple distributed processes")

    # Optional arguments for the launch helper

    parser.add_argument("--node_rank", 
                        type=int, 
                        default=0,
                        help=("this machine's index into the number of machines (0 is the master); "
                              "default: 0"
                              )
                       )
    parser.add_argument("--other_gpus",
                        default=0,
                        help=("either: path to GPU global-config file, or total\n"
                              "number of GPUs used on other nodes; default: 0"
                              )
                        )
    parser.add_argument("--here_gpus", 
                        type=int,
                        help=f"number of GPUs to use on this node; default is all: {num_gpus_here}",
                        default=num_gpus_here
                        )
    parser.add_argument("--master_addr", default="127.0.0.1", type=str,
                        help="Master node (rank 0)'s address, should be either "
                             "the IP address or the hostname of node 0, for "
                             "single node multi-proc training, the "
                             "--master_addr can simply be 127.0.0.1")
    parser.add_argument("--master_port", default=29500, type=int,
                        help="Master node (rank 0)'s free port that needs to "
                             "be used for communication during distributed "
                             "training")
    parser.add_argument("-m", "--module", default=False, action="store_true",
                        help="Changes each process to interpret the launch script "
                             "as a python module, executing with the same behavior as"
                             "'python -m'.")
    parser.add_argument("--no_python", default=False, action="store_true",
                        help="Do not prepend the training script with \"python\" - just exec "
                             "it directly. Useful when the script is not a Python script, "
                             "or has a #! at the top."
                        )
    parser.add_argument("-q", "--quiet", 
                        action='store_true',
                        help=f"do not print status and other info messages",
                        default=False
                        )
    # Allow training script to run everywhere
    # to be defaulted even though it is positional
    # (the '?'). 
    parser.add_argument("training_script", type=str,
                        nargs=1,
                        default=training_script,
                        help=f"Default {os.path.basename(training_script)}: Add training script arguments after the above: \n"
                             f"\n{script_help}"
                        )

    # Rest of args are for the training program:
    parser.add_argument('training_script_args', nargs=REMAINDER)
    args = parser.parse_args()
    if type(args.training_script) == list:
            args.training_script = args.training_script[0]
    return args

#------------------------------------
# main
#-------------------

def main():
    args = parse_args()
    
    #*********
    print("CLI Arguments:")
    for key in vars(args):
        print(f"{key}: {getattr(args, key)}")
    #*********

    # world size is number of processes, which is
    # equal to number of GPUs used on all machines with
    # lower node rank than this one, plus this node's GPUs.
    # If args.other_gpus is an int, we assume that 
    # all machines have the same number of GPUs. 
    # Else the arg is the path to a config file
    # that lays out which node is to use how many 
    # of its GPUs.
    
    other_gpus = args.other_gpus
    here_gpus  = args.here_gpus
    node_rank  = args.node_rank
    
    world_layout = {}
    try:
        other_gpus   = int(other_gpus)
    except:
        # other_gpus is path to world layout
        # config file.
        pass
    
    if type(other_gpus) == int:
        # Every node (machine) has same number of args:
        # Account for all nodes that came before this one.
        # We could just lump those together as:
        #    other_gpus * node_rank
        # but enumerating them matches what would be
        # in a config file if it was used:
        if node_rank == 0:
            # Master node must know true sum of GPUs:
            world_layout['others'] = other_gpus
        else:
            # All other nodes care only about GPUs
            # in nodes with lower ranks than themselves:
            for prev_node_rank in range(node_rank):
                world_layout[prev_node_rank] = other_gpus
        world_layout['localhost'] = here_gpus
    else:
        # Unequal number of GPUs used in various
        # machines. Parse the config file, filling
        # in world_layout with each prior node's 
        # number of GPUs.
        world_layout = parse_world_layout_config(other_gpus)

    # Handle special case: no GPUs anywere, and
    # we are on node 0: in that case start a single
    # copy of the training script. If it is written
    # properly, it will detect the absence of a GPU,
    # and use the CPU. This happens during debugging
    # on a laptop:
    
    dist_world_size = sum(world_layout.values())
    if dist_world_size == 0 and int(node_rank) == 0:
        world_layout['localhost'] += 1
        dist_world_size += 1
    
    # If trying to launch on a node without GPUs,
    # when GPUs are available elsewhere, refuse to
    # start the script:
    if world_layout['localhost'] == 0:
        print("This machine does not have any GPU; training script not started.")
        sys.exit(1)

    # If host name was provided instead of an
    # IP address, resolve that:
    if re.search('^[\d.]*$', args.master_addr) is None:
        # Got not an IP address, but something
        # with letters:
        master_addr = socket.gethostbyname(args.master_addr)
    else:
        master_addr = args.master_addr 

    # set PyTorch distributed related environmental variables
    current_env = os.environ.copy()
    current_env["MASTER_ADDR"] = master_addr
    current_env["MASTER_PORT"] = str(args.master_port)
    current_env["WORLD_SIZE"] = str(dist_world_size)
    
    processes = []

    if 'OMP_NUM_THREADS' not in os.environ and args.here_gpus > 1:
        current_env["OMP_NUM_THREADS"] = str(1)
        if not args.quiet:
            print("*****************************************\n"
                  "Setting OMP_NUM_THREADS environment variable for each process "
                  "to be {} in default, to avoid your system being overloaded, "
                  "please further tune the variable for optimal performance in "
                  "your application as needed. \n"
                  "*****************************************".format(current_env["OMP_NUM_THREADS"]))

    # All GPUs in lower ranked nodes (i.e. exclusive
    # the ones in this node:
    other_gpus =  sum([num_gpus for (node_name, num_gpus) 
                       in world_layout.items() 
                       if node_name != 'localhost']) 

    # Compute a unique number for each GPU within
    # the group of nodes (machines). Starting with
    # the master node, whose numbers are 0,1,...<ngpus_here>:

    current_env['NODE_RANK'] = str(args.node_rank)

    for local_rank in range(0, world_layout['localhost']):

        dist_rank = other_gpus * args.node_rank + local_rank

        current_env["RANK"] = str(dist_rank)
        current_env["LOCAL_RANK"] = str(local_rank)

        # Spawn the process:
        with_python = not args.no_python
        cmd = []
        if with_python:
            cmd = [sys.executable, "-u"]
            if args.module:
                cmd.append("-m")
        else:
            if args.module:
                raise ValueError("Don't use both the '--no_python' flag and the '--module' flag at the same time.")

        cmd.append(args.training_script)

        # To the args for the train script processes,
        # add --started_from_launch to let each of
        # them know:
        args_for_train_scripts = ['--started_from_launch'] + \
                                 args.training_script_args
        cmd.extend(args_for_train_scripts)

        # Copy stdin, and give the copy to the subprocess.
        # This enables the subprocess to ask user whether
        # to save training state in case of a cnt-C:
        newstdin = os.fdopen(os.dup(sys.stdin.fileno()))
        process = subprocess.Popen(cmd, stdin=newstdin, env=current_env)
        processes.append(process)
    
    if not args.quiet:
        print(f"Node {args.node_rank} launch.py: Num processes launched: {len(processes)}")
        if node_rank == 0:
            print(f"Awaiting {sum(world_layout.values())} processes to finish...")
        else:
            print(f"Awaiting {world_layout['localhost']} processes to finish...")
    
    # Let subprocesses deal with cnt-C (keyboard interrupt):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    for process in processes:
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(returncode=process.returncode,
                                                cmd=cmd)

if __name__ == "__main__":
    main()
