'''
Created on Mar 5, 2020

@author: paepcke
'''
from _ast import arg
import argparse
from collections import deque
import os
import sys
import time

from DSP.amplitude_gating import args
from amplitude_gating import AmplitudeGater
from elephant_utils.logging_service import LoggingService
from precision_recall_from_wav import PrecRecComputer


sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))




class PreprocessingCalibration(object):

    #------------------------------------
    # Constructor
    #-------------------    

    def __init__(self,
                 in_wav_file,
                 label_file,
                 thresholds_db, 
                 cutoff_freqs,
                 outfile_dir='/tmp',
                 logfile=None):

        AmplitudeGater.log = LoggingService(logfile=logfile)
        self.log.info("Constructing output file names...")
        outfile_names_deque = self.construct_outfile_names(thresholds_db, cutoff_freqs, outfile_dir)

        
        # Generate a new .wav file from the input file 
        # for each pair of voltage threshold/cutoff-freq pair.
        # Copy the outfile deque, b/c called method will pop
        # from it, and we need it again:
        self.generate_outfiles(in_wav_file, thresholds_db, cutoff_freqs, outfile_names_deque.copy())
        
        # Compute precision/recall and other info for each 
        # of the just created files:
        prec_recall_results = self.generate_prec_recall(outfile_names_deque.copy(),
                                                        label_file)
        print(prec_recall_results) # ******

    #------------------------------------
    # generate_prec_recall
    #-------------------
    
    def generate_prec_recall(self, file_deque, labelfile):
        
        precrec_results = deque()
        try:
            filtered_wavfile = file_deque.popleft()
            precrec_computer = PrecRecComputer(filtered_wavfile, labelfile)
            precrec_results.append(precrec_computer.performance_result)
        except IndexError:
            # No more work to do:
            return precrec_results

    #------------------------------------
    # generate_outfiles
    #-------------------
    
    def generate_outfiles(self, in_wav_file, thresholds_db, cutoff_freqs, outfile_names):
        

        for threshold in thresholds_db:
            for cutoff_freq in cutoff_freqs:
                outfile = outfile_names.popleft()
                self.log.info(f"Generating [{threshold}dB, {cutoff_freq}Hz]...")
                _gater = AmplitudeGater(in_wav_file,
                                        outfile=outfile,
                                        amplitude_cutoff=threshold,
                                        cutoff_freq=cutoff_freq
                                        )
                self.log.info(f"Done generating [{threshold}dB, {cutoff_freq}Hz] into {os.path.basename(outfile)}...")
        self.log.info("Done generating outfiles.")


    #------------------------------------
    # construct_outfile_names
    #-------------------
    
    def construct_outfile_names(self, thresholds_db, cutoff_freqs, outfile_dir):
        
        out_files = deque()
        for threshold in thresholds_db:
            for cutoff_freq in cutoff_freqs:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                out_files.append(os.path.join(outfile_dir),
                                 f"filtered_wav_{threshold}dB_{cutoff_freq}Hz_{timestamp}.wav" 
                                 )
        return out_files


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Compute precision/recall for pairs of voltage cutoff, and lowpass frequencies."
                                     )

    parser.add_argument('-l', '--logfile',
                        help='fully qualified log file name to which info and error messages \n' +\
                             'are directed. Default: stdout.',
                        default=None);
    parser.add_argument('-d', '--outdir',
                        help='fully qualified directory for output wav files.',
                        default='/tmp');                        
    parser.add_argument('threshold_dbs',
                        type=int,
                        nargs='+',
                        default=-20,
                        help='Repeatable: db above which voltages are cut off; default: -20dB FS')
    parser.add_argument('cutoff_freqs',
                        type=int,
                        nargs='+',
                        default=100,
                        help='Repeatable: cutoff frequencies for lowpass filter; default: 100Hz')

    parser.add_argument('wavefile',
                        help='fully qualified log file name to elephant wav file.',
                        default=None);
    parser.add_argument('labelfile',
                        help='fully qualified log file name to corresponding Raven label file.',
                        default=None);

    args = parser.parse_args();
    PreprocessingCalibration(args.wavefile,
                             args.labelfile,
                             args.threshold_dbs,
                             args.cutoff_freqs,
                             outfile_dir=args.outdir,
                             logfile=args.logfile
                             )