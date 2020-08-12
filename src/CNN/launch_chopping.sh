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
Usage: $(basename $0) [-j --jobs][-o --outdir] <spectrogram/label files and/or dirs>
    -h: This message
    -j: Number of spectrogram chopping jobs to run simultaneouly;
        default: number of cores
    -o: Destination directory for chopped spectrograms and sqlite db; 
        default: with corresponding 24-hr spectrogram
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

# The rest of the args are input files and  dirs:
infiles=$@

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
               
echo "Starting $NUM_WORKERS copies of chop_spectrograms.py"

# For testing, use the following as the first
# line of the command, commented the line below it:

# The --bar creates a progress bar in the terminal
# for each worker. The --joblog enables check of
# exit status, as well as use of parallel's --resume
# option:

#*cmd="time parallel --bar --joblog /tmp/chopping.log echo  "
cmd="time parallel --bar --joblog /tmp/chopping.log ${SCRIPT_DIR}/chop_spectrograms.py "
if [[ ! -z $OUTDIR ]]
then
    cmd="$cmd --outdir $OUTDIR"
fi
cmd="$cmd --num_workers=$NUM_WORKERS --this_worker ::: $WORKER_RANKS ::: $infiles"

# Invoke all the copies of chop_spectrograms.py:
$cmd

# When we get here, all chopping
# will be finished. Each worker will have
# created an sqlite file with info from their
# share of the work. Must combine those into
# one result sqlite db:

sqlite_files=""
for worker_rank in $WORKER_RANKS
do
    sqlite_files="$sqlite_files $OUTDIR/snippet_db_${worker_rank}.sqlite"
done

#***********
#echo "Worker ranks: '${WORKER_RANKS}'s"
#echo "sqlite_files: $sqlite_files"
#***********

# Add the destination sqlite:
sqlite_files="$sqlite_files $OUTDIR/snippet_db.sqlite"

#**************
#echo "Sqlite db files: $sqlite_files"
#echo "Exiting prematurely on purpose"
#exit
#**************

$SCRIPT_DIR/../elephant_utils/sqlite_db_merger.py $sqlite_files
