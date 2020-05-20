#!/bin/bash

# Takes:
#
#   o a destination dir
#   o a list of directories or .wav or .txt files.
#
# Collects all .wav and .txt files below each of the
# directories, and runs the wave_maker.py on it.

# Uses gnu parallel to call wave_maker.py for each. 
# Result is a set of gated wav files at the destination.
# Filenames there will have "_gated" added before the
# extension portion.
#
# Logs to /tmp/wav_gating.log

USAGE="Usage: "`basename $0`" wavDestDir srcFileOrDict1 srcFileOrDict2 ..."

if [ $# -lt 2 ]
then
    echo $USAGE
    exit 1
fi

LOGFILE=/tmp/wav_gating.log
echo "Logging to $LOGFILE"

# Determine whether this is a Mac. Some bash
# commands are not available there:
if [[ $(echo $OSTYPE | cut -c 1-6) == 'darwin' ]]
then
    PLATFORM='macos'
    BASH_VERSION=$(echo $(bash --version) | sed -n 's/[^0-9]*version \([0-9]\).*/\1/p')
    if [[ $BASH_VERSION < 4 ]]
    then
        echo "On MacOS Bash version must be 4.0 or higher."
        exit 1
    fi
else
    PLATFORM='other'
fi    

destDir=$1
#echo $destDir
if [ ! -d $destDir ]
then
    echo "First argument must be a directory; $USAGE"
    exit 1
fi
shift
#echo ${@}

thisScriptDir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cmdfile=$thisScriptDir/../src/DSP/wave_maker.py
logfile=$LOGFILE

# BSD bash seems not to pass PATH down into
# subshells; so hard-code parallel's location
# for the MacOS case:

if [ -e $logfile ]
then
    rm $logfile
fi

# Recursively collect all files with a .txt or .wav
# extension. In contrast to the -regex alternative,
# this -o method works on Mac and Linux:

FILES_OR_DIRS=$(find "$@" -type f -name  "*.txt" -o -name "*.wav" -print)

echo "Wav gating starting: `date`" > $logfile
if [[ $PLATFORM == 'macos' ]]
then   
    time /usr/local/bin/parallel --gnu --progress $cmdfile -l $logfile -o $destDir ::: "${FILES_OR_DIRS}";
else
    time parallel --gnu --progress $cmdfile -l $logfile -o $destDir ::: "${FILES_OR_DIRS}";
fi    
echo "Wav gating done: `date`" >> /tmp/transformOnly.txt

