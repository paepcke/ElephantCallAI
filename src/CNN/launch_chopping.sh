#!/usr/bin/env bash

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
Usage: $(basename $0) [-j --jobs][-d --destination] <spectrogram/label files and/or dirs>
    -h: This message
    -j: Number of spectrogram chopping jobs to run simultaneouly; default: number of cores
    -d: Destination directory for chopped spectrograms; default: with corresponding 24-hr spectrogram
EOF

OPTS=$(${getopt} --options hj:d: --long help,jobs:,destination: -- "$@")

if [ $? != 0 ]
then 
   echo "Failed parsing options;\n${USAGE}"
   exit 1
fi

eval set -- $OPTS

# Number of copies gnu parallel should run.
# Default: as many as there # are cores:

NUM_WORKERS=$(getconf _NPROCESSORS_ONLN)

DEST_DIR=''

while true; do
  case "$1" in
      -h | --help ) SAVED_IFS=$IFS; IFS=; echo $USAGE; IFS=${SAVED_IFS} ; exit ;;
      -j | --jobs ) NUM_WORKERS=$2; shift; shift ;;
      -d | --destination ) DEST_DIR=2; shift; shift ;;
      -- ) shift ;  break ;;
       * ) echo "Could not assign options." ; exit 1 ;;
  esac
done

# The rest of the args are input files and  dirs:
infiles=$@

# Must have at least one in-file/dir:
if [[ -z $infiles ]]
then
    echo "Must provide at least one infile or in-directory."
    echo $USAGE
    exit 1
fi

# Our worker ranks are 0...(NUM_WORKERS - 1):
let MAX_WORKER_RANK="$NUM_WORKERS - 1"

# Generate a sequence of worker rank integers.
# This results in :0\n1\n2\n...:
WORKER_RANKS=$(seq 0 $MAX_WORKER_RANK)
# Replace the \n by spaces:
WORKER_RANKS="${WORKER_RANKS//\\n/ }"
               
echo "Starting $NUM_WORKERS copies of chop_spectrograms.py"

# For testing, use the following as the first
# line of the command, commented the line below it:
#cmd="time parallel echo --outdir $DEST_DIR "
cmd="time parallel ./chop_spectrograms.py --outdir $DEST_DIR "
cmd="$cmd --threshold_db=-30 --low_freq=20 --high_freq=40 --freq_cap=30 "
cmd="$cmd --num_workers=$NUM_WORKERS --this_worker ::: $WORKER_RANKS ::: $infiles"

$cmd


