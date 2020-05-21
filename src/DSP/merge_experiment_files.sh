#!/usr/bin/env bash

# Merge multiple .csv or .tsv files. Assumes
# the files have a header column. The -n flag
# indicates that no header column is present in
# any of the files.
#
# Output to stdout
#
# Strategy: output the header of the first argument,
# then the bodies of all (incl. the first)

USAGE="Usage: $(basename $0) experimentResultFile1 [experimentResultFile2 [...]]"

if [[ $# < 1 ]]
then
    echo "Not enough args: $USAGE"  >/dev/stderr
    exit 1
fi

# ---------------------------- Command Line Args -------------

# By default: assume each file has a col header:
HAS_COL_HEADER=1

while getopts ":nh" opt; do
  case ${opt} in
    n )
        HAS_COL_HEADER=0 ;;
    h )
        echo $USAGE
        exit
  esac
done
shift $((OPTIND -1))

# --------------------------- The Work ----------------

# If no column headers, just cat the files to stdout:
if [[ $HAS_COL_HEADER == 0 ]]
then
    cat $@
    exit 0
fi

# Grab the column header of the first file:
# NOTE: the quotes around $1 and $COL_HEAD
#       are needed for .tsv files, which
#       linux otherwise takes as arg sequences.
COL_HEAD=$(head -1 "$1")
if [ -z "$COL_HEAD" ]
then
    # First file was empty; error:
    echo "No column header in first file: $1" >/dev/stderr
    exit 1
fi

# First output line: column header, which is common
# with all the other given experiment files:
echo $COL_HEAD

# In a loop: output all but the first line of
# each file:

for FILE in $@
do
    if [[ ! -e $FILE ]]
    then
        # Don't pollute the output with the error msg;
        # ensure it goes to stderr:
        echo "File $FILE does not exist; skipping." >/dev/stderr
        continue
    fi
    # Write all but the first line:
    tail -n +2 "$FILE"
done
