#!/bin/bash

# Takes:
#
#   o a destination dir
#   o a list of .wav files
#
# Uses gnu parallel to call wave_maker.py for each. 
# Result is a set of gated wav files at the destination.
# Filenames there will have "_gated" added before the
# extension portion.

USAGE="Usage: "`basename $0`" wavDestDir srcWavFile1 srcWavFile2 ..."

if [ $# -lt 2 ]
then
    echo $USAGE
    exit 1
fi

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
logfile=/tmp/wav_gating.log

# BSD bash seems not to pass PATH down into
# subshells; so hard-code parallel's location
# for the MacOS case:

if [ -e $logfile ]
then
    rm $logfile
fi

echo "Wav gating starting: `date`" > $logfile
if [[ $PLATFORM == 'macos' ]]
then   
    time /usr/local/bin/parallel --gnu --progress $cmdfile -l $logfile -o $destDir ::: ${@};
else
    time parallel --gnu --progress $cmdfile -l $logfile -o $destDir ::: ${@};
fi    
echo "Wav gating done: `date`" >> /tmp/transformOnly.txt

