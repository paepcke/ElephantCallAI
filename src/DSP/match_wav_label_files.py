#!/usr/bin/env python
'''
Created on May 25, 2020

@author: paepcke
'''
import argparse
import os
import sys

class WavLablesMatcher(object):
    '''
    Takes path to dir with potentially lots of
    files. Finds matching pairs of .wav and .txt
    files. Writes the full path of each pair without
    an extension to stdout.
    
    If --wav is set, paths are output with .wav extension.
    If --txt is set, output paths with .txt extension.

    Illegal to specify both options.

    '''

    #------------------------------------
    # Constructor 
    #-------------------

    def __init__(self, 
                 dirpath,
                 wav=False,
                 txt=False,
                 dirs=True):
        '''
        Constructor
        '''
        all_files = os.listdir(dirpath)
        wav_txt_files = filter(lambda file: file.endswith('.wav') or file.endswith('.txt'),
                               all_files)
        # Build a dir with None values. Change
        # each value to the basename when a pair
        # is found:
        match_dir = {}
        for file in wav_txt_files:
            try:
                (file_no_ext, _ext) = os.path.splitext(file)
                # If success, we found the first of this
                # pair earlier:
                match_dir[file_no_ext]
                match_dir[file_no_ext] = os.path.basename(file_no_ext)
            except KeyError:
                # Found first of a pair:
                match_dir[file_no_ext] = None
                
        for file_no_extension in match_dir.values():
            if file_no_extension is None:
                # Orphan: either no .wav or no .txt file:
                continue
            if dirs:
                # Caller wants full path output:
                file_no_extension = os.path.join(dirpath, file_no_extension)
            if wav:
                sys.stdout.write(f"{file_no_extension}.wav\n")
            elif txt:
                sys.stdout.write(f"{file_no_extension}.txt\n")
            else:
                sys.stdout.write(f"{file_no_extension}\n")
            

# ------------------------------ Main ---------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Matches .wav and .txt files in a dir."
                                     )
    parser.add_argument('-w', '--wav',
                        help='output file paths with .wav extension; default: no extension.',
                        action='store_true')
    parser.add_argument('-t', '--txt',
                        help='output file paths with .txt extension; default: no extension.',
                        action='store_true')
    parser.add_argument('-d', '--dirs',
                        help='output full paths; default: only output file name.',
                        action='store_true')
    parser.add_argument('indir',
                        help='directory where to look for .wav and .txt files')

    args = parser.parse_args();

    if args.wav and args.txt:
        print("Not allowed to specify both the wav and txt options")
        sys.exit(1)
        
    if not os.path.isdir(args.indir):
        print("Given directory either does not exist, or is not a directory.")
        sys.exit(1)

    WavLablesMatcher(args.indir,
                     wav=args.wav,
                     txt=args.txt,
                     dirs=args.dirs
                     )