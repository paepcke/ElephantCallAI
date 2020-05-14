#!/usr/bin/env python
'''
Created on May 4, 2020

Takes full paths to one or more Elephant .wav
files. Generates noise gated .wav files. 

Fixed parameters:
  o Uses 1% overlap.


@author: paepcke
'''
import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from amplitude_gating import AmplitudeGater
from elephant_utils.logging_service import LoggingService


class WavMaker(object):
    '''
    classdocs
    '''

    #------------------------------------
    # Constructor
    #-------------------

    def __init__(self, 
                 infiles,
                 outdir='/tmp/',
                 threshold_db=-40,
                 cutoff_freq=10,
                 logfile=None
                 ):
        '''
        Constructor
        '''
        self.log = LoggingService(logfile=logfile)
        files_done = 0
        for infiles in infiles:
            if not os.path.exists(infiles):
                self.log.warn(f"File {infiles} does not exist.")
                continue
            # Get '/tmp/foo' and '.txt' from /tmp/foo.txt:
            (dir_plus_basename, ext) = os.path.splitext(infiles)
            if ext != '.wav':
                self.log.warn(f"File {infiles} apparently not a .wav file.")
                continue
            # Get 'foo' from '/my/directory/foo
            in_basename = os.path.basename(dir_plus_basename)
            # Final out: /my/outdir/foo_gated.wav
            outfile = os.path.join(outdir, f"{in_basename}_gated.wav")
            
            self.log.info(f"Processing {infiles}...")
            try:
                gater = AmplitudeGater(infiles,
                                       outfile=outfile,
                                       amplitude_cutoff=threshold_db,
                                       envelope_cutoff_freq=cutoff_freq,
                                       )
            except Exception as e:
                self.log.err(f"Processing failed for '{infiles}: {repr(e)}")
                continue
            perc_zeroed = gater.percent_zeroed
            self.log.info(f"Done processing {os.path.basename(infiles)}; removed {round(perc_zeroed)} percent")
            files_done += 1
            self.log.info(f"\nBatch gated {files_done} wav files.")
            
# ---------------- Main -------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Create noise gated wav files from input wav files."
                                     )

    parser.add_argument('-l', '--logfile',
                        help='fully qualified log file name to which info and error messages \n' +\
                             'are directed. Default: stdout.',
                        default=None);
    parser.add_argument('-o', '--outdir',
                        help='directory for outfiles; default /tmp',
                        default='/tmp');
    parser.add_argument('-t', '--threshold',
                        type=int,
                        default=-40,
                        help='dB off peak voltage below which signal is zeroed; default -40',
                        );
    parser.add_argument('-f', '--cutoff_freq',
                        type=int,
                        default=10,
                        help='envelope frequency; default 10Hz',
                        );
    parser.add_argument('infiles',
                        nargs='+',
                        help='Repeatable: .wav input files')
    args = parser.parse_args();
    
    if args.threshold > 0:
        print("Threshold dB value should be less than 0.")
        sys.exit(1)

    WavMaker(args.infiles,
             outdir=args.outdir,
             threshold_db=args.threshold,
             cutoff_freq=args.cutoff_freq,
             logfile=args.logfile
             )