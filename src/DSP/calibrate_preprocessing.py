'''
Created on Mar 5, 2020

@author: paepcke
'''
import argparse
from collections import deque
import os
import sys
import time

from amplitude_gating import AmplitudeGater
from amplitude_gating import FrequencyError
from elephant_utils.logging_service import LoggingService
from precision_recall_from_wav import PrecRecComputer

from plotting.plotter import PlotterTasks


sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class PreprocessingCalibration(object):

    #------------------------------------
    # Constructor
    #-------------------    

    def __init__(self,
                 in_wav_file,
                 label_file,
                 overlap_perc,
                 thresholds_db, 
                 cutoff_freqs,
                 outfile_dir='/tmp',
                 spectrogram=None,
                 logfile=None):

        AmplitudeGater.log = LoggingService(logfile=logfile)
        self.log = AmplitudeGater.log
        self.log.info("Constructing output file names...")
        outfile_names_deque = self.construct_outfile_names(thresholds_db, cutoff_freqs, outfile_dir)
        self.log.info(f"Output files will be: {self.list_outfiles_from_deque(outfile_names_deque)}")

        self.spectrogram = spectrogram
                
        # Generate a new .wav file from the input file 
        # for each pair of voltage threshold/cutoff-freq pair.
        # Copy the outfile deque, b/c called method will pop
        # from it, and we need it again:
        
        # Lots of work happens here: all the
        # spectrograms are created, if their
        # plots are requested:

        # To skip generating gated files for
        # testing the precision/recall component,
        # switch the commenting below:
        self.generate_outfiles(in_wav_file, 
                               thresholds_db, 
                               cutoff_freqs, 
                               outfile_names_deque.copy(),
                               spectrogram=self.spectrogram)
        # Option for just one precomputed gated .wav file:
        #outfile_names_deque = deque(['/Users/paepcke/tmp/filtered_wav_-40dB_500Hz_20200321_202339.wav'])
        # Option for two precomputed gated .wav files; 
        #outfile_names_deque = deque(['/tmp/filtered_wav_-20dB_100Hz_20200306_131711.wav',
        #                            '/tmp/filtered_wav_-25dB_100Hz_20200306_131716.wav']
        #                            )
        
        # Compute precision/recall and other info for each 
        # of the just created files:
        prec_recall_results = self.generate_prec_recall(outfile_names_deque.copy(),
                                                        label_file,
                                                        overlap_perc)
        print(prec_recall_results)

    #------------------------------------
    # generate_prec_recall
    #-------------------
    
    def generate_prec_recall(self, file_deque, labelfile, overlap_perc):
        
        precrec_results = deque()
        try:
            filtered_wavfile = file_deque.popleft()
            precrec_computer = PrecRecComputer(filtered_wavfile, labelfile, overlap_perc)
            precrec_results.append(precrec_computer.performance_result)
        except IndexError:
            # No more work to do:
            return precrec_results

    #------------------------------------
    # generate_outfiles
    #-------------------
    
    def generate_outfiles(self, 
                          in_wav_file, 
                          thresholds_db, 
                          cutoff_freqs, 
                          outfile_names,
                          spectrogram=None):
        '''
        Run through each threshold voltage and lowpass filter
        cutoff frequency, and generate a wave file for each.
        Caller provides the output file names.
        
        @param in_wav_file: elephant wave file over which to 
            run each setting to produce cleaned .wav files.
        @type in_wav_file: str
        @param thresholds_db: voltages above which to set wav
            signals to zero. Values are in dB below peak. Thus
            always negative numbers.
        @type thresholds_db: [int]
        @param cutoff_freqs: frequencies for the low-pass filter
            in each experiment.
        @type cutoff_freqs: [int]
        @param outfile_names: list of full path filenames, one
            for each experiment.
        @type outfile_names: [str]
        @param spectrogram: if True, a spectrogram is created,
            and written to outdir, with the same file name as the 
            .wav file, but with an.npy extension. If False, no 
            spectrogram is created.
        @type spectrogram: bool
        '''

        experiments = []
        for threshold in thresholds_db:
            for cutoff_freq in cutoff_freqs:
                outfile = outfile_names.popleft()
                # Create a corresponding file name for spectrograms:
                spectrogram_outfile = os.path.join(os.path.splitext(outfile)[0], '.npy') 
                self.log.info(f"Generating [{threshold}dB, {cutoff_freq}Hz]...")
                try:
                    # Generate an Experiment instance to
                    # which caller can later add precision/recall
                    # results:
                    experiments.append(Experiment(in_wav_file,
                                                  outfile,
                                                  threshold,
                                                  cutoff_freq
                                                  ))
                    _gater = AmplitudeGater(in_wav_file,
                                            outfile=outfile,
                                            amplitude_cutoff=threshold,
                                            spectrogram_freq_cap=cutoff_freq,
                                            spectrogram_outfile=spectrogram_outfile
                                            )
                except FrequencyError as e:
                    self.log.err(f"Bad frequency; skipping it: {repr(e)}")
                    continue
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
                out_files.append(os.path.join(outfile_dir,
                                 f"filtered_wav_{threshold}dB_{cutoff_freq}Hz_{timestamp}.wav") 
                                 )
        return out_files


    #------------------------------------
    # list_outfiles_from_deque
    #-------------------
    
    def list_outfiles_from_deque(self, file_name_deque):
        '''
        Given a deque filled with full pathnames, return a 
        string that lists all files. But: the dir path is only
        listed once. Following lines are indented and only have the
        file names. Ex. output:
            Directory  /tmp:
                foo.txt,
                bar.txt,
        If deque is empty, return "No files in list."
                
        @param file_name_deque: Deque of path names
        @type file_name_deque: Deque[str]
        '''
        if len(file_name_deque) == 0:
            return "No files in list."
        dir_path = os.path.dirname(file_name_deque[0])
        res = f"Directory {dir_path}:"
        for file_name in file_name_deque:
            res += f"\n\t{file_name}"
        return res

# ---------------------------- Experiment Class ---------------

class Experiment(object):
    
    def __init__(self,
                 in_wav_file,
                 outfile,
                 threshold,
                 cutoff_freq
                 ):
        self.__in_wav_file = in_wav_file
        self.__outfile     = outfile
        self.__threshold   = threshold
        self.__cutoff_freq = cutoff_freq
        
        self.__experiment_result = None

    @property
    def get_in_wav_file(self):
        return self.__in_wav_file

    @property
    def get_outfile(self):
        return self.__outfile

    @property
    def get_threshold(self):
        return self.__threshold

    @property
    def get_cutoff_freq(self):
        return self.__cutoff_freq

    @property
    def experiment_result(self):
        return self.__experiment_result
    
    @experiment_result.setter
    def experiment_result(self, val):
        self.__experiment_result = val

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
    parser.add_argument('-o', '--overlap',
                        type=int,
                        help='percentage overlap of prediction with label to count as success; default: 10%',
                        default=10);
    parser.add_argument('-t', '--threshold_dbs',
                        type=int,
                        nargs='+',
                        default=-20,
                        help='Repeatable: db above which voltages are cut off; default: -20dB FS')
    parser.add_argument('-f', '--cutoff_freqs',
                        type=int,
                        nargs='+',
                        default=100,
                        help='Repeatable: cutoff frequencies for lowpass filter; default: 100Hz')
    parser.add_argument('-s', '--spectrogram',
                        default=None,
                        help='If provided, a spectrogram is created and written to outdir as .npy file')
    parser.add_argument('--plot',
                        nargs='+',
                        choices=['gated_wave_excerpt','samples_plus_envelope','spectrogram_excerpts','low_pass_filter'],
                        help="Plots to produce; repeatable; default: no plots"
                        )
    parser.add_argument('wavefile',
                        help='fully qualified path to elephant wav file.',
                        default=None)
    parser.add_argument('labelfile',
                        help='fully qualified path to corresponding Raven label file.',
                        default=None)

    args = parser.parse_args();
    
    # Register the plots to produce:
    for plot_name in args.plot:
        PlotterTasks.add_task(plot_name)

    PreprocessingCalibration(args.wavefile,
                             args.labelfile,
                             args.overlap,
                             args.threshold_dbs,
                             args.cutoff_freqs,
                             outfile_dir=args.outdir,
                             spectrogram=args.spectrogram,
                             logfile=args.logfile
                             )