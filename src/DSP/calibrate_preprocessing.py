#!/usr/bin/env python
'''
Created on Mar 5, 2020

@author: paepcke
'''
from _collections import OrderedDict, deque
import argparse
from collections import deque
import csv
import os
import sys
import time

from precision_recall_from_wav import PerformanceResult
from amplitude_gating import AmplitudeGater
from amplitude_gating import FrequencyError
from dsp_utils import DSPUtils, PrecRecFileTypes
from elephant_utils.logging_service import LoggingService
from plotting.plotter import PlotterTasks
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
                 overlap_perc,
                 thresholds_db, 
                 cutoff_freqs,
                 spectrogram_freq_cap=AmplitudeGater.spectrogram_freq_cap,
                 outfile_dir='/tmp',
                 spectrogram=None,
                 logfile=None):

        AmplitudeGater.log = LoggingService(logfile=logfile)
        self.log = AmplitudeGater.log
        self.log.info("Constructing output file names...")
        outfile_names_deque = self.construct_outfile_names(thresholds_db, cutoff_freqs, outfile_dir)
        names_as_str = self.list_outfiles_from_deque(outfile_names_deque)
        self.log.info(f"Output files will be: {names_as_str}")

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
        experiments = self.generate_outfiles(in_wav_file, 
                                             thresholds_db, 
                                             cutoff_freqs,
                                             outfile_names_deque.copy(),
                                             spectrogram_freq_cap=spectrogram_freq_cap,
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
        
        try:
            outfile_names_deque_copy = outfile_names_deque.copy()
            experiments_copy         = experiments.copy()
            while True:
                gated_outfile_name = outfile_names_deque_copy.popleft()
                prec_recall_res        = prec_recall_results.popleft()
                prec_recall_res_file   = DSPUtils.prec_recall_file_name(gated_outfile_name,
                                                                        PrecRecFileTypes.PREC_REC_RES)
                prec_recall_res.to_tsv(prec_recall_res_file)
                # Get the Experiment instance that holds
                # the parameters of this experiment:
                experiment = experiments_copy.popleft()
                # Add the just computed result:
                experiment['experiment_res']    = prec_recall_res
                experiment['result_file_path']  = prec_recall_res_file
                exp_res_file = DSPUtils.prec_recall_file_name(gated_outfile_name,
                                                              PrecRecFileTypes.EXPERIMENT)
                experiment.to_tsv(include_col_header=True, outfile=exp_res_file)
        except IndexError:
            # Done
            pass

    #------------------------------------
    # generate_prec_recall
    #-------------------
    
    def generate_prec_recall(self, file_deque, labelfile, overlap_perc):
        '''
        For each elephant .wav file in file_deque, compute precision,
        recall, and other measures. Return a deque of PerformanceResult
        instances. The .wav files are normally noise-gated versions
        of one original .wav file. Therfore, only one label file
        exists.
        
        @param file_deque: elephant recordings .wav files
        @type file_deque: Dequeue
        @param labelfile: path to the Raven label file associated
            with all the .wav files.
        @type labelfile: the Raven labels text file.
        @param overlap_perc: percentage requirement for how much
            a detected burst must overlap a labeled burst to count
            as a detection.
        @type overlap_perc: {int | float}
        '''
        
        precrec_results = deque()
        try:
            while True:
                filtered_wavfile = file_deque.popleft()
                self.log.info(f"Compute prec/recall for {filtered_wavfile}...")
                precrec_computer = PrecRecComputer(filtered_wavfile, labelfile, overlap_perc)
                self.log.info(f"Done compute prec/recall for {filtered_wavfile}.")                
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
                          spectrogram_freq_cap=AmplitudeGater.spectrogram_freq_cap,
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
        @return: Dequeu of Experiment instances with all parameters used.
        @rtype: Dequeue(Experiment)
        '''

        experiments = deque()
        for threshold in thresholds_db:
            for cutoff_freq in cutoff_freqs:
                outfile = outfile_names.popleft()
                # Create a corresponding file name for spectrograms:
                spectrogram_outfile = DSPUtils.prec_recall_file_name(outfile,
                                                                     PrecRecFileTypes.SPECTROGRAM)
                self.log.info(f"Generating [{threshold}dB, {cutoff_freq}Hz]...")
                try:
                    # Generate an Experiment instance to
                    # which caller can later add precision/recall
                    # results:
                    experiment = Experiment({'in_wav_file'           : in_wav_file,
                            	    		 'outfile'               : outfile,
                                    		 'spectrogram_outfile'   : spectrogram_outfile,
                                    		 'threshold'             : threshold,
                                    		 'cutoff_freq'           : cutoff_freq,
                                    		 'spectrogram_freq_cap'  : spectrogram_freq_cap,
                                    		 'experiment_res'        : None
                                            })                        
                    experiments.append(experiment)
                    _gater = AmplitudeGater(in_wav_file,
                                            outfile=outfile,
                                            amplitude_cutoff=threshold,
                                            envelope_cutoff_freq=cutoff_freq,
                                            spectrogram_freq_cap=spectrogram_freq_cap,
                                            spectrogram_outfile=spectrogram_outfile
                                            )
                except FrequencyError as e:
                    self.log.err(f"Bad frequency; skipping it: {repr(e)}")
                    continue
                self.log.info(f"Done generating [{threshold}dB, {cutoff_freq}Hz] into {os.path.basename(outfile)}...")
        self.log.info("Done generating outfiles.")
        return experiments

    #------------------------------------
    # construct_outfile_names
    #-------------------
    
    def construct_outfile_names(self, thresholds_db, cutoff_freqs, outfile_dir):
        
        out_files = deque()
        for threshold in thresholds_db:
            for cutoff_freq in cutoff_freqs:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                outfile = os.path.join(outfile_dir,
                                       f"filtered_wav_{threshold}dB_{cutoff_freq}Hz_{timestamp}.wav")
                # Add info that this is a gated wav file:
                outfile = DSPUtils.prec_recall_file_name(outfile, PrecRecFileTypes.GATED_WAV)
                out_files.append(outfile)

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

class Experiment(OrderedDict):

    # Name of all properties stored in instances
    # of this class. Needed when reading from
    # previously saved csv file of a PerformanceResult
    # instance:
    props = {'in_wav_file'           : str,
        	 'outfile'               : str,
             'spectrogram_outfile'   : str,
             'threshold'             : int,
             'cutoff_freq'           : int,
             'spectrogram_freq_cap'  : int,
             'result_file_path'      : str,
             'experiment_res'        : PerformanceResult
            }
    
    #------------------------------------
    # Constructor
    #-------------------
    
    def __init__(self, all_results={}):
        super().__init__(all_results)

    #------------------------------------
    # to_tsv
    #-------------------

    def to_tsv(self, include_col_header=False, outfile=None, append=True):
        if include_col_header:
            # Create column header:
            col_header = '\t'.join([f"{col_name}" for col_name in self.keys()])
        csv_line = '\t'.join([str(res_val) for res_val in self.values()])
        try:
            if outfile is not None:
                if not outfile.endswith('.tsv'):
                    outfile += '.tsv'
                fd = open(outfile, 'a' if append else 'w')
            else:
                fd = sys.stdout
            if include_col_header:
                fd.write(col_header + '\n')
            fd.write(csv_line + '\n')
        finally:
            if outfile:
                fd.close()

    #------------------------------------
    # instances_from_tsv
    #-------------------

    @classmethod
    def instances_from_tsv(cls, infile, first_line_is_col_header=False):
        
        res_obj_list = []
        with open(infile, 'r') as fd:
            reader = csv.reader(fd, delimiter='\t')
            # Get array of one line:
            first_line = next(reader)
            # Check whether first line is header;
            if first_line_is_col_header:
                prop_order = first_line
            else:
                # First line is also first data line: 
                prop_order = Experiment.props.keys()
                res_obj_list.append(cls._make_res_obj(first_line, prop_order))
            try:
                while True:
                    line = next(reader)
                    res_obj_list.append(cls._make_res_obj(line, prop_order))
            except StopIteration:
                # Finished the file.
                pass
        return res_obj_list

    #------------------------------------
    # _make_res_obj
    #-------------------

    @classmethod
    def _make_res_obj(cls, values_arr_strings, prop_order):
    
        res_obj = Experiment()
        for (indx, prop_name) in enumerate(prop_order):
            # To which data type must this string be
            # coerced?:
            dest_type = Experiment.props[prop_name]
            # Does it call for a PerformanceResult?
            # (Can't get EQ dest_type/PerformanceResult
            # to work. So kludge):
            if issubclass(dest_type, OrderedDict):
                # Reconstitute the PerformanceResult
                # instance. We use eval with an empty 
                # namespace to prevent safety issues:
                perf_res_create_str = values_arr_strings[indx]
                # Safe eval: only the PerformanceResult class
                # is available, and no built-ins:
                typed_val = eval(perf_res_create_str,
                                 {"__builtins__":None},
                                 {'PerformanceResult' : PerformanceResult}
                                 )
            else:
                typed_val = dest_type(values_arr_strings[indx])
            res_obj[prop_name] = typed_val
            
        # Materialize the PerformanceResult instance, if
        # we have a path to its .tsv:
        res_obj_tsv_path = res_obj['result_file_path']
        if isinstance(res_obj_tsv_path, str) and\
           os.path.exists(res_obj_tsv_path):
            res_obj['experiment_res'] = \
                PerformanceResult.instances_from_tsv(res_obj_tsv_path, 
                                                     first_line_is_col_header = True) 

        return res_obj 

    #------------------------------------
    # print
    #-------------------
    
    def print(self, outfile=None):
        
        try:
            if outfile is None:
                out_fd = sys.stdout
            else:
                out_fd = open(outfile, 'w')
            
            print('Experiment settings:\n')
            # For nice col printing: find longest 
            # property name:
            col_width = max(len(prop_name) for prop_name in self.keys())
            for prop_name in self.keys():
                # The PerformanceResult instance has its own
                # printing method:
                if prop_name == 'experiment_res':
                    print("\nExperiment result:\n")
                    self[prop_name].print()
                    continue
                print(prop_name.ljust(col_width), self[prop_name])

        finally:
            if out_fd != sys.stdout:
                out_fd.close()

# ------------------------------ Main ------------------------

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
                        help='Repeatable: cutoff frequencies for lowpass envelope filter; default: 100Hz')
    parser.add_argument('-s', '--spectrogram',
                        default=None,
                        action='store_true',
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
    if args.plot is not None:
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