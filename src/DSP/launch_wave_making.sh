#!/usr/bin/env bash

# Uses gnu parallel to run multiple copies of
# wave_maker.py with a fixed number of parameters.
# Change those in the 'parallel' cmd below.
#
# Each worker will run one wave_maker.py copy,
# which is uniquely identified by a worker_rank
# integer. Each worker selects a unique set of
# input .wav files. Each such file is put through
# audio preprocessing (see amplitude_gating.py).
# The resulting gated .wav file is turned into a
# 24-hr spectrogram, which is deposited in the
# destination directory.

#*********
#echo "Infiles at start: $@ "
#*********


if [ "$(uname)" == "Darwin" ]
then
    if [[ ! -e /usr/local/Cellar/gnu-getopt/1.1.6/bin/getopt ]]
    then
        echo "Getopt program on Mac is incompatible. Do: 'brew install gnu-getopt' and try again."
        exit 1
    fi
    getopt="/usr/local/Cellar/gnu-getopt/1.1.6/bin/getopt"
else
    getopt=$(which getopt)
fi

read -r -d '' USAGE <<EOF
Usage: $(basename $0) [-j --jobs][-o --outdir] <spectrogram/label files and/or dirs>
    -h: This message
    -j: Number of wave file noise gating jobs to run simultaneouly;
        default: number of cores
    -o: Output directory for gated files.
        default: same dir as original .wav file
\n
EOF

OPTS=$(${getopt} --options hj:o: --long help,jobs:,outdir: -- "$@")

if [ $? != 0 ]
then
    # Use printf instead of echo to get the
    # newlines right:
    printf "Failed parsing options;\n${USAGE}"
    exit 1
fi

eval set -- $OPTS

# Number of copies gnu parallel should run.
# Default: as many as there # are cores:

NUM_WORKERS=$(getconf _NPROCESSORS_ONLN)

OUTDIR=''

while true; do
  case "$1" in
      -h | --help ) SAVED_IFS=$IFS; IFS=; printf $USAGE; IFS=${SAVED_IFS} ; exit ;;
      -j | --jobs ) NUM_WORKERS=$2; shift; shift ;;
      -o | --outdir ) OUTDIR=$2; shift; shift ;;
      -- ) shift ;  break ;;
       * ) echo "Could not assign options." ; exit 1 ;;
  esac
done

# The rest of the args are input .wav files and  dirs:
infiles=$@

#*********
#echo "Infiles: '$infiles'"
#*********

# Don't spawn more workers than
# there are input files:

if [[ $# -lt $NUM_WORKERS ]]
then
    NUM_WORKERS=$#
fi

# Must have at least one in-file/dir:
if [[ -z $infiles ]]
then
    echo "Must provide at least one infile or in-directory."
    printf $USAGE
    exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

#********
# echo "Number of jobs: $NUM_WORKERS"
# echo "Destination:    $OUTDIR"
# echo "Infiles         $infiles"
# echo "Script dir      $SCRIPT_DIR"
# echo "Exiting intentionally"
# exit
#********

# Our worker ranks are 0...(NUM_WORKERS - 1):
let MAX_WORKER_RANK="$NUM_WORKERS - 1"

# Generate a sequence of worker rank integers.
# This results in :0 1 2 ... The -s changes
# separator from \n to space:
WORKER_RANKS=$(seq -s ' ' 0 $MAX_WORKER_RANK)
               
echo "Starting $NUM_WORKERS copies of wave_maker.py"

# For testing, use the following as the first
# line of the command, commented the line below it:

# The --bar creates a progress bar in the terminal
# for each worker. The --joblog enables check of
# exit status, as well as use of parallel's --resume
# option:

#cmd="/usr/local/bin/parallel --link --bar echo "
cmd="time /usr/local/bin/parallel --bar --joblog /tmp/parallel_wave_making.log $SCRIPT_DIR/wave_maker.py "

if [[ ! -z $OUTDIR ]]
then
    cmd="$cmd --outdir $OUTDIR"
fi
cmd="$cmd --threshold_db=-30 --low_freq=20 --high_freq=40 --freq_cap=30 "
cmd="$cmd --num_workers=$NUM_WORKERS --this_worker ::: $WORKER_RANKS ::: $infiles"

#**********
#echo "Cmd: $cmd"
#exit
#**********

# Execute:
$cmd
