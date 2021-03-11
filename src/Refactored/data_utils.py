'''
Created on Apr 5, 2020

@author: paepcke
'''

from collections import OrderedDict
import csv
from enum import Enum
import os
from pathlib import Path, PosixPath
import re
import sys
import time

from scipy.io import wavfile
import torch
from torchaudio import transforms

from elephant_utils.logging_service import LoggingService
import numpy as np
import pandas as pd
import math


class PrecRecFileTypes(Enum):
    SPECTROGRAM  = '_spectrogram'
    TIME_LABELS  = '_time_labels'
    FREQ_LABELS  = '_freq_labels'
    GATED_WAV    = '_gated'
    PREC_REC_RES = '_prec_rec_res'
    EXPERIMENT   = '_experiment'
    PICKLE       = '_pickle'
    
class AudioType(Enum):
    WAV = 0         # Audio sound wave
    SPECTRO = 1     # 24-hr spectrogram
    LABEL = 3       # Raven label file
    MASK = 4        # Mask Call/No-Call from label file
    TIME = 5       # Time series for spectrogram cols

class DATAUtils(object):
    '''
    classdocs
    '''
    # Try removing this!
    #log = LoggingService()

    #------------------------------------
    # overlap_percentage 
    #-------------------
    
    @classmethod
    def overlap_percentage(cls, interval1, interval2):
        '''
        Given two Pandas Interval instances, return
        percentage overlap as a number between 0 and 1.
        The percentage is number seconds overlap as
        a percentage of interval2
        
        @param interval1: numeric interval
        @type interval1: pdInterval
        @param interval2: numeric interval
        @type interval2: pdInterval
        '''
        num_overlap_seconds = \
             max(0, 
                 min(interval1.right, interval2.right) - \
                 max(interval1.left, interval2.left)
                 )
        percentage = num_overlap_seconds / interval2.length
        return percentage

    #------------------------------------
    # prec_recall_file_name
    #-------------------

    @classmethod
    def prec_recall_file_name(cls, 
                              root_info, 
                              file_type,
                              experiment_id=None):
        '''
        Given root info like one of:
          o filtered_wav_-50dB_10Hz_50Hz_20200404_192128.npy
          o filtered_wav_-50dB_10Hz_50Hz_20200404_192128
          
        return a new name with identifying info after the
        basename:
        
          o filtered_wav_-50dB_10Hz_50Hz_20200404_192128_spectrogram.npy
          o filtered_wav_-50dB_10Hz_50Hz_20200404_192128_time_labels.npy
          o filtered_wav_-50dB_10Hz_50Hz_20200404_192128_freq_labels.npy
          o filtered_wav_-50dB_10Hz_50Hz_20200404_192128_gated.wav
        
        The experiment_id argument is for type PICKLE. Such files
        hold a single experiment object in pickle form:
        
         o filtered_wav_-50dB_10Hz_50Hz_20200404_192128_exp<experiment_id>.pickle
         
        If generated filename already exists, adds a counter before 
        the dot: 
        
          filtered_wav_-50dB_10Hz_50Hz_20200404_192128_exp<experiment_id>_<n>.pickle
        
        @param root_info: example file name
        @type root_info: str
        @param file_type: one of the PrecRecFileTypes
        @type file_type: PrecRecFileTypes
        '''
        (path_no_ext, ext) = os.path.splitext(root_info)
        if file_type in [PrecRecFileTypes.FREQ_LABELS,
                         PrecRecFileTypes.TIME_LABELS]:
            ext = '.npy'
        elif file_type in [PrecRecFileTypes.SPECTROGRAM]:
            ext = '.pickle'
        elif file_type in [PrecRecFileTypes.GATED_WAV]:
            ext = '.wav'
        elif file_type in [PrecRecFileTypes.PREC_REC_RES,
                           PrecRecFileTypes.EXPERIMENT]:
            ext = '.tsv'
        elif file_type == PrecRecFileTypes.PICKLE:
            ext = f"{experiment_id}.pickle"
        
        new_file_path = f"{path_no_ext}{file_type.value}{ext}"
        counter = 0
        
        # Find a file name that does not already exists,
        # guarding against race conditions. The 'x' mode
        # atomically opens a file for writing, but only if 
        # it does not exist. If it does, a FileExistsError
        # is generated:
        
        while True:
            try:
                with open(new_file_path, 'x') as _fd: 
                    return new_file_path
            except FileExistsError:
                counter += 1
                new_file_path = f"{path_no_ext}{file_type.value}_{counter}{ext}"



    #------------------------------------
    # save_spectrogram 
    #-------------------
    
    @classmethod
    def save_spectrogram(cls, 
                         magnitudes_np_or_spectr_df, 
                         spectrogram_dest,
                         freq_labels=None,
                         time_labels=None
                         ):
        '''
        Given magnitudes_np_or_spectr_df magnitudes_np_or_spectr_df, frequency
        and time axes labels, save as a pickled
        dataframe
        
        @param magnitudes_np_or_spectr_df: either a
             2d array of magnitudes, or a DataFrame
             comprising the magnitudes, times and freqs.
        @type magnitudes_np_or_spectr_df: {pd.DataFrame|np.array}
        @param spectrogram_dest: file name to save to
        @type spectrogram_dest: str
        @param freq_labels: array of y-axis labels
        @type freq_labels: np_array
        @param time_labels: array of x-axis labels
        @type time_labels: np_array
        @raise ValueError if parameters are inconsistent.
        '''
        if type(magnitudes_np_or_spectr_df) == pd.DataFrame:
            magnitudes_np_or_spectr_df.to_pickle(spectrogram_dest)
            return
        # Got an np array of magnitudes, not a ready-made
        # df. In that case, freq_labels and time_labels must
        # be available:
        if freq_labels is None or time_labels is None:
            raise ValueError("When magnitudes_np_or_spectr_df is an np array, freq/time labels must be provided.")
        
        # Save the magnitudes_np_or_spectr_df to file.
        # Combine magnitudes_np_or_spectr_df, freq_labels, and time_labels
        # into a DataFrame:
        df = pd.DataFrame(magnitudes_np_or_spectr_df,
                          columns=time_labels,
                          index=freq_labels
                          )
        # Python 3.8's dataframe.to_pickle() writes
        # pickle version 5, which cannot be read by
        # python 3.7's df.read_pickle(). So write using
        # the common protocol:
        
        df.to_pickle(spectrogram_dest, protocol=4)

    #------------------------------------
    # load_spectrogram
    #-------------------
    
    @classmethod
    def load_spectrogram(cls, df_filename):
        '''
        Given the path to a pickled dataframe
        that holds a spectrogram, return the
        dataframe. The df will have the index
        (i.e. row labels) set to the frequencies,
        and the column names to the time labels.
        
        @param df_filename: location of the pickled df
        @type df_filename: str
        @return: spectrogram DataFrame: columns are 
            times in seconds, index are frequency bands
        @rtype pandas.DataFrame
        @raise FileNotFoundError: when pickle file not found
        @raise pickle.UnpicklingError
        '''

        # Safely read the pickled DataFrame
        df = pd.read_pickle(df_filename)
        return df

        #return({'spectrogram' : df.values,
        #        'freq_labels' : df.index,
        #        'time_labels' : df.columns
        #        })

    #------------------------------------
    # spectrogram_to_db 
    #-------------------
    
    @classmethod
    def spectrogram_to_db(cls, spect_magnitude):
        '''
        Takes a numpy spectrogram  of magnitudes.
        Returns a numpy spectrogram containing 
        dB scaled power.
        
        @param spect_magnitude:
        @type spect_magnitude:
        '''
        transformer = transforms.AmplitudeToDB('power')
        spect_tensor = torch.Tensor(spect_magnitude)
        spect_dB_tensor = transformer.forward(spect_tensor)
        spect_dB = spect_dB_tensor.numpy()
        return spect_dB

    #------------------------------------
    # get_spectrogram__from_treatment
    #-------------------
    
    @classmethod
    def get_spectrogram__from_treatment(cls, threshold_db, cutoff_freq, src_dir='/tmp'):
        '''
        Given a list of treatments, construct the file names
        that are created by the calibrate_preprossing.py facility.
        Load them all, and return them.
        
        @param threshold_db: dB below which all values were set to zero
        @type threshold_db: int
        @param cutoff_freq: spectrogram cutoff frequency
        @type cutoff_freq: int
        @param src_dir: directory where all the files 
            are located.
        @type src_dir: src
        @return:  {'spectrogram' : magnitudes,
                   'freq_labels' : y-axis labels,
                   'time_labels' : x-axis labels
                  }

        '''

        files = os.listdir(src_dir)
        spec_pat = re.compile(f'filtered_wav_{str(threshold_db)}dB_{str(cutoff_freq)}Hz.*spectrogram.pickle')
        try:
            spect_file = next(filter(spec_pat.match, files))
            spect_path = os.path.join(src_dir, spect_file)
        except StopIteration:
            raise IOError("Spectrogram file not found.")

        # Safely read the pickled DataFrame
        df = eval(pd.read_pickle(spect_path),
                   {"__builtins__":None},    # No built-ins at all
                   {}                        # No additional func
                   )

        return({'spectrogram' : df.values,
                'freq_labels' : df.index,
                'time_labels' : df.columns
                })
        
    #------------------------------------
    # load_label_time_intervals 
    #-------------------
    
    @classmethod
    def load_label_time_intervals(cls, label_file):
        '''
        Given a .txt file of elephant call labels as 
        written by Raven, return a list of time intervals
        
        @param label_file: label time file in csv format
        @type label_file: str
        @return: list of pd.Interval instances, each holding
            one inclusive start/end time pair of a call
        @rtype: [pd.Interval]
        '''
        label_time_intervals = []
        with open(label_file, 'r') as fd:
            reader = csv.DictReader(fd, delimiter='\t')
            begin_time_key = 'Begin Time (s)'
            end_time_key   = 'End Time (s)'
            # Get each el call time range spec in the labels:
            for label_dict in reader:
                try:
                    begin_time = float(label_dict[begin_time_key])
                    end_time   = float(label_dict[end_time_key])
                except KeyError:
                    raise IOError(f"Raven label file {label_file} does not contain one "
                                  f"or both of keys '{begin_time_key}', {end_time_key}'")
                
                if end_time < begin_time:
                    #cls.log.err(f"Bad label: end label less than begin label: {end_time} < {begin_time}")
                    continue
                label_time_intervals.append(pd.Interval(left=begin_time, right=end_time))
        return label_time_intervals

    #------------------------------------
    # load_label_mask 
    #-------------------

    @classmethod
    def load_label_mask(cls, infile):
        np.load(infile)
    
    #------------------------------------
    # save_label_mask 
    #-------------------

    @classmethod
    def save_label_mask(cls, label_mask, outfile):
        np.save(outfile, label_mask)


    #------------------------------------
    # hrs_mins_secs_from_secs 
    #-------------------

    @classmethod
    def hrs_mins_secs_from_secs(cls, secs):
        return time.strftime("%H:%M:%S", time.gmtime(secs))

    #------------------------------------
    # time_ticks_from_wav 
    #-------------------
    
    @classmethod
    def time_ticks_from_wav(cls, 
                            wav, 
                            hop_length=800, 
                            nfft=4096):
        '''
        Given the file name of a .wav recording, compute the time tick labels
        that would go with a spectrogram. Return an array
        of (fractional) seconds that indicate the time
        at each spectrogram x-timeslice.
        
        To compute, we need to know how the spectrogram
        would be computed. Among all the related parameters
        we need NFFT and hop size. Note, the framerate is retrieved
        from the .wav file. 

        @param wav: file name of a .wav recording
        @type wav_or_sig: {str}
        @param hop_length: distance in samples from start of a window
            to the start of the usually overlapping next window.
        @type hop_length: {int|float}
        @return (time ticks array or None if failed to read .wav)
        '''

        try:
            # Read wave file
            (framerate, sig) = wavfile.read(wav)
        except Exception as e:
            print (f"Cannot process .wav file: {repr(e)}")
            return None

        # Number of voltage readings in the
        # recording: 
        num_wav_samples = len(sig)
        
        # Number of columns in the spectrogram (time bins in the spectrogram)
        num_spectro_ticks = int(math.floor((num_wav_samples - (nfft - hop_length)) / hop_length))
        
        # We are labeling w/r to the center of columns
        start_time_tick = nfft / 2. / framerate
        # Time in fractional seconds corresponding
        # to one spectro column (i.e. time bin):
        time_per_spectro_tick = hop_length / framerate
        
        # Generate the sequence of fractional seconds
        # that would label the middle of each spectrogram column.
        # Cannot use Python range(), b/c it
        # only works with ints:
        
        time_seq = [start_time_tick + tick_idx * time_per_spectro_tick for 
                    tick_idx in range(0, num_spectro_ticks)]

        return time_seq

    #------------------------------------
    # compute_timeticks 
    #-------------------
    
    @classmethod
    def compute_timeticks(cls, framerate, spectrogram):
        '''
        UNTESTED
        
        Given a spectrogram, compute the x-axis
        time ticks in seconds, minutes, and hours.
        Return all three in a dict:
           {'seconds' : num-of-ticks,
            'minutes' : num-of-ticks,
            'hours'   : num-of-ticks
            }
            
        Recipient can pick 
        
        @param framerate:
        @type framerate:
        @param spectrogram: spectrogram
        @type spectrogram: pd.DataFrame
        '''
        
        # Number of columns in the spectrogram:
        estimate_every_samples = spectrogram.columns
        estimate_every_secs    = estimate_every_samples / framerate
        one_sec_every_estimates= 1/estimate_every_secs
        (_num_freqs, num_estimates) = spectrogram.shape
        num_second_ticks = num_estimates / one_sec_every_estimates
        num_minute_ticks = num_second_ticks / 60
        num_hour_ticks   = num_second_ticks / 3600
        num_time_ticks = {'seconds': int(num_second_ticks),
                          'minutes': int(num_minute_ticks),
                          'hours'  : int(num_hour_ticks)
                          }
        
        #time_labels       = np.array(range(num_timeticks))
        return num_time_ticks


    #------------------------------------
    # unix_find 
    #-------------------
    
    @classmethod
    def unix_find(cls, 
                  start_dir, 
                  name_regex, 
                  collected=None,
                  maxdepth=None):
        '''
        Simplified version of Unix 'find' command.
        Given a starting directory, a regular expression,
        and optionally a list of already collected file
        names. Returns a list of all matching file names
        in a recursive walk down all subdirs of the 
        given start_dir.
        
        The regex may be a (usually) raw string, or a
        compiled regex. If a string, it is automatically
        compiled before use.
        
        @param start_dir: directory from which to
            being the file name collection
        @type start_dir: {str|pathlib.PosixPath}
        @param name_regex: regular expression against
            which file names must match to be collected.
        @type name_regex: {str|re.Pattern}
        @param collected: list of file names in the dir
            tree that have already been collected
        @type collected: {None|[str]}
        @param maxdepth: maximum depth to which descent
            will proceed. A value of zero means to 
            stay within start_dir. None: search till
            bottom out.
        @type maxdepth: int
        '''
        
        if type(name_regex) == str:
            name_regex = re.compile(name_regex)
            
        if type(start_dir) == str:
            # Work with Path instances:
            start_dir = Path(start_dir)
        
        # Initial condition:
        if collected is None:
            collected = []
        if maxdepth is None:
            # No limit (in practice):
            maxdepth = sys.maxsize
        if maxdepth < 0:
            # Reach max depth in previous 
            # (recursive) call:
            return
        
        this_dir_iter = start_dir.iterdir()
        for file_name_obj in this_dir_iter:
            # Apply search only to the file name,
            # not the entire path:
            if name_regex.search(file_name_obj.name) is not None:
                # Append the full path as a string:
                collected.append(str(file_name_obj))
            elif Path.is_dir(file_name_obj):
                
                # The recursion:

                # Assignment to collected is not 
                # neccessary, b/c append is in-place.
                # So 'collected' would appear with
                # the additional files automatically.
                # The "-1" implements the maxdepth
                # limit, if imposed by original caller:
                
                collected = cls.unix_find(file_name_obj, 
                                          name_regex, 
                                          collected,
                                          maxdepth - 1)
        return collected

# ---------------------------- Class FileFamily
## SIMPLE FOR NOW WE MAY UPDATE AS WE GO!
class FileFamily(object):
    '''
    Several files are derived from a single recording:
    the 24-hour spectrogram, a label mask, and others. And
    there is the .wav file itself. The filenames have
    standard formats. The core is the file id, such as
    nn05b_20180617_000000. The associated recording is
    nn05b_20180617_000000.wav. A spectrogram of that file
    nn05b_20180617_000000_spectrogram.npy. 
    
    An instance of this class enables easy access to 
    all these associated file names. For the names
    without their path:
    
        o inst.wav
        o inst.spectro
        o inst.label
        o inst.mask
        o inst.time
    
    For full filepaths, use:
    
        inst.fullpath(<AudioType>)
        
    where AudioType is one of 
        AudioType.WAV         # Audio sound wave
                  SPECTRO     # 24-hr spectrogram
                  LABEL       # Raven label file
                  MASK        # Mask Call/No-Call from label file
                  TIME        # Time series mask for spectrogram cols
    '''

    #------------------------------------
    # Constructor 
    #-------------------
    
    def __init__(self, filename):
        # Set up the dict
        self.decode_filename(filename)

    #------------------------------------
    # __str__ 
    #-------------------

    def __str__(self):
        return f"<FileFamily {self.file_root}>"
    
    #------------------------------------
    # fullpath 
    #-------------------
    
    def fullpath(self, filetype):
        '''
        Given a filetype: 
        AudioType.WAV         # Audio sound wave
                  SPECTRO     # 24-hr spectrogram
                  LABEL       # Raven label file
                  MASK        # Mask Call/No-Call from label file
                  TIME        # Time series mask for spectrogram cols

        return the full path to the respective
        file family.

        @param filetype: specifies which file extension is wanted
        @type filetype: AudioType
        @return: full path to the file
        @rtype: str
        '''
        
        if filetype == AudioType.WAV:
            return os.path.join(self.path, self.wav)
        elif filetype == AudioType.SPECTRO:
            return os.path.join(self.path, self.spectro)
        elif filetype == AudioType.LABEL:
            return os.path.join(self.path, self.label)
        elif filetype == AudioType.MASK:
            return os.path.join(self.path, self.mask)
        elif filetype == AudioType.TIME:
            return os.path.join(self.path, self.time_labels)


    #------------------------------------
    # decode_filename 
    #-------------------
    # May update this to include snippets but may not
    def decode_filename(self, filename):
        '''
        Given a file name with or without 
        path that is one of: 
           o .wav file
           o .txt file
           o .npy file
           
        initialize all instance variables in 
        this FileFamily instance to allow
        retrieving file names of other family members.
        return a dict with the other file 
        names:
                
        @param filename: filename to decode. 
        @type filename: str
        @raise ValueError if filename does not conform to 
            elephant file name convention.
        '''

        fpath = Path(filename)

        # Name without path and extension:
        # make sure you have the correct format!!
        self.file_root = fpath.name[:-len(fpath.suffix)]
        if filename.endswith('_label_mask.npy'):
            self.file_root = fpath.name[:-len('_label_mask.npy')]
        elif filename.endswith('_time_mask.npy'):
            self.file_root = fpath.name[:-len('_time_mask.npy')]
        elif filename.endswith('_spectro.npy'):
            self.file_root = fpath.name[:-len('_spectro.npy')]

        # Just the path to the file without filename:
        path = str(fpath.parent)
        # If filename is just a name w/o a
        # path: set path to None:
        if path == '.':
            path = None
        self.path = path

        if fpath.suffix == '.wav':
            self.file_type = AudioType.WAV
        elif fpath.suffix == '.txt':
            self.file_type = AudioType.LABEL
        elif str(fpath).endswith("_label_mask.npy"):
            self.file_type = AudioType.MASK
        elif str(fpath).endswith("_spectro.npy"):
            self.file_type = AudioType.SPECTRO
        elif str(fpath).endswith("_time_mask.npy"):
            self.file_type = AudioType.TIME
        else:
            # Unknown type of file:
            raise ValueError(f"File '{filename} not in family of elephant file name conventions.")

        self.wav     = self.file_root + '.wav'
        self.label   = self.file_root + '.txt'
        self.mask    = f"{self.file_root}_label_mask.npy"
        self.time_labels = f"{self.file_root}_time_mask.npy"
        self.spectro = f"{self.file_root}_spectro.npy"


           
# ----------------------------- Main ----------------

if __name__ == '__main__':
    
    # Just some testing; doesn't normally run as main:
    fname = 'filtered_wav_-50dB_10Hz_50Hz_20200404_192128.npy'
    
    if os.path.exists(fname):
        os.remove(fname)
        
    new_file = DSPUtils.prec_recall_file_name(fname, 
                                               PrecRecFileTypes.FREQ_LABELS)
    
    assert (new_file == f"{os.path.splitext(fname)[0]}_freq_labels.npy"), \
            f"Bad freq_label file: {new_file}"
    os.remove(new_file) 

    new_file = DSPUtils.prec_recall_file_name(fname, 
                                               PrecRecFileTypes.FREQ_LABELS) 
            
    assert (new_file == f"{os.path.splitext(fname)[0]}_freq_labels.npy"), \
            f"Bad freq_label file: {new_file}"
             
    os.remove(new_file)
    
    new_file = DSPUtils.prec_recall_file_name(fname,
                                               PrecRecFileTypes.TIME_LABELS) 

    assert (new_file == f"{os.path.splitext(fname)[0]}_time_labels.npy"),\
            f"Bad freq_label file: {new_file}"
            
    os.remove(new_file) 

    new_file = DSPUtils.prec_recall_file_name(fname,
                                              PrecRecFileTypes.GATED_WAV) 

    assert (new_file == f"{os.path.splitext(fname)[0]}_gated.wav"),\
            f"Bad freq_label file: {new_file}"

    os.remove(new_file)

#     new_file = DSPUtils.prec_recall_file_name('/foo/bar/filtered_wav_-50dB_10Hz_50Hz_20200404_192128.npy', 
#                                                PrecRecFileTypes.GATED_WAV) 
# 
#     assert (new_file == '/foo/bar/filtered_wav_-50dB_10Hz_50Hz_20200404_192128_gated.wav'),\
#             f"Bad freq_label file: {new_file}" 

#    os.remove(new_file)

    print('File name manipulation tests OK')    
                                          
    # Test SignalTreatmentDescriptor:
    # To string:
    
    s1 = "SignalTreatmentDescriptor(-30,10,50,20)"
    d = SignalTreatmentDescriptor(-30, 10, 50, 20)
    assert str(d) == s1

    s2 = "SignalTreatmentDescriptor(-40,20,50,None)"
    d = SignalTreatmentDescriptor(-40, 20, 50)
    assert str(d) == s2

    try:    
        d = SignalTreatmentDescriptor('foo', 10, 50, 10)
    except ValueError:
        pass
    else:
        raise ValueError(f"String 'foo,10,50,10' should have raised an ValueError")
    
    # Test SignalTreatmentDescriptor:
    # From string:

    d = SignalTreatmentDescriptor.from_str(s1)
    assert (str(d) == s1)

    d = SignalTreatmentDescriptor.from_str(s2)
    assert (str(d) == s2)
    
    try:
        d = SignalTreatmentDescriptor.from_str('foo_10Hz_50Hz_30perc')
    except ValueError:
        pass
    else:
        raise AssertionError("Should have ValueError from 'foo_10Hz_50Hz_30perc'")

    # SignalTreatmentDescriptor
    # Adding min overlap after the fact:
    
    d = SignalTreatmentDescriptor(-40, 30, 60)
    assert (str(d) == "SignalTreatmentDescriptor(-40,30,60,None)")
    d.add_overlap(10)
    assert (str(d) == "SignalTreatmentDescriptor(-40,30,60,10)")
    
    # SignalTreatmentDescriptor
    # to_flat_str
    
    s3 = '-30dB_10Hz_50Hz_20perc'
    d = SignalTreatmentDescriptor.from_flat_str(s3)
    assert(d.to_flat_str() == s3)
    
    s4 = '-40dB_10Hz_50Hz_noneperc'
    d = SignalTreatmentDescriptor.from_flat_str(s4)
    assert(d.to_flat_str() == s4)
    
    s5 = '-40dB_10Hz_50Hz_10perc'
    d = SignalTreatmentDescriptor.from_flat_str(s5)
    assert(d.to_flat_str() == s5)
    
    # SignalTreatmentDescriptor
    # Equality
    
    d1 = SignalTreatmentDescriptor(-40,10,50,10)
    d2 = SignalTreatmentDescriptor(-40,10,50,10)
    assert (d1.__eq__(d2))
    
    d1 = SignalTreatmentDescriptor(-40,10,50,None)
    d2 = SignalTreatmentDescriptor(-40,10,50,10)
    assert (not d1.__eq__(d2))
    
    d1 = SignalTreatmentDescriptor(-40,10,50,10)
    d2 = SignalTreatmentDescriptor(-40,10,50,20)
    assert (d1.equality_sig_proc(d2))
    assert (not d1.__eq__(d2))
    
    
    print('SignalTreatmentDescriptor tests OK')
    
    print("Tests done.")
