'''
Created on Apr 5, 2020

@author: paepcke
'''

from collections import OrderedDict
import csv
from enum import Enum
import os
from pathlib import Path
import re
import time

import torch
from torchaudio import transforms

from elephant_utils.logging_service import LoggingService
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class PrecRecFileTypes(Enum):
    SPECTROGRAM  = '_spectrogram'
    TIME_LABELS  = '_time_labels'
    FREQ_LABELS  = '_freq_labels'
    GATED_WAV    = '_gated'
    PREC_REC_RES = '_prec_rec_res'
    EXPERIMENT   = '_experiment'
    PICKLE       = '_pickle'
    
class AudioType(enumerate):
    WAV = 0         # Audio sound wave
    SPECTRO = 1     # 24-hr spectrogram
    SNIPPET = 2     # Snippet of spectrogram 
    LABEL = 3       # Raven label file
    MASK = 4        # Mask Call/No-Call from label file
    IMAGE=5         # PNG of a spectgrogram

class DSPUtils(object):
    '''
    classdocs
    '''
    log = LoggingService()

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
        df.to_pickle(spectrogram_dest)

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
                    cls.log.err(f"Bad label: end label less than begin label: {end_time} < {begin_time}")
                    continue
                label_time_intervals.append(pd.Interval(left=begin_time, right=end_time))
        return label_time_intervals

    #------------------------------------
    # load_label_mask 
    #-------------------

    def load_label_mask(self, infile):
        np.load(infile)
    
    #------------------------------------
    # save_label_mask 
    #-------------------

    def save_label_mask(self, label_mask, outfile):
        np.save(outfile, label_mask)


    #------------------------------------
    # decode_filename 
    #-------------------

    @classmethod
    def decode_filename(cls, filename):
        '''
        Given a file name with or without 
        path that is one of: 
           o .wav file
           o .txt file
           o .png file
           o _spectrogram.pickle
           o _spectrogram_<n>.pickle
        return a dict with the other file 
        names:
        
           {file_type : <AudioType enum>,
            file_root : <no path, stripped ext and [_<n>]_spectrogram
            wav : <.wav file>
            txt : <.txt file>
            png : <.png file>
            mask: <.npy file>
            spectro : <_spectrogram.pickle>
            snippet_id : <the <n> if in given filename, else None>,
            path : <path portion of filename if given, else None>
        
        If a full path is returned, the path up to the filename
        is placed int the path key. All other entries are 
        only the filenames.
        
        @param filename: filename to decode. 
        @type filename: str
        @return: dict of file names derived from
            infile.
        @rtype: {str : str}
        '''
        
        res = {}
        # Will correct later if appropriate:
        res['snippet_id']  = None
        
        fpath = Path(filename)
        
        # Regex to find spectrogram snippet files: 
        # Ex.: nn05b_20180617_000000_5_spectrogram.pickle'
        # A match group will contain: ('nn05b_20180617_000000', '5'). 
        # However the following will not match: 
        #    nn05b_20180617_000000_spectrogram.pickle
        # This last one is a 24-hr, full spectrogram:
        
        snippet_search_pattern = re.compile(r"(.*_[0]+)_([0-9]+)_spectrogram[.]pickle") 

        # Name without path and extension:
        file_root = fpath.name
        # If it's foo_spectrogram.pickle or foo_<n>_spectrogram.pickle: 
        if filename.endswith('_spectrogram.pickle'):
            # Lose the _spectrogram.pickle:
            file_root = file_root[:-len('_spectrogram.pickle')]
            # Get spectro_id if file is a spectrogram snippet:        
            matched_fragments = snippet_search_pattern.search(filename)
            if matched_fragments is not None:
                (_path, snippet_id) = matched_fragments.groups()
                res['snippet_id']  = int(snippet_id)
                # Remove the snippet id from file root:
                # The 1+ is for the leading underscore
                # before the number:
                file_root = file_root[:-(1+len(snippet_id))]
                res['file_type'] = AudioType.SNIPPET # snippet of spectrogram
            else:
                res['file_type'] = AudioType.SPECTRO # 24 hr spectrogram

        res['file_root'] = file_root

        # Just the path to the file without filename:
        path = str(fpath.parent)
        # If filename is just a name w/o a
        # path: set path to None:
        if path == '.':
            path = None
        res['path'] = path

        if fpath.suffix == '.wav':
            res['file_type'] = AudioType.WAV
        elif fpath.suffix == '.txt':
            res['file_type'] = AudioType.LABEL
        elif fpath.suffix == '.npy':
            res['file_type'] = AudioType.Mask
        elif fpath.suffix == '.png':
            res['file_type'] = AudioType.IMAGE

        res['wav']     = file_root + '.wav'
        res['label']   = file_root + '.txt'
        res['mask']    = file_root + '.npy'
        res['png']     = file_root + '.png'
        res['spectro'] = file_root + '_spectrogram.pickle'

            
        return res

    #------------------------------------
    # hrs_mins_secs_from_secs 
    #-------------------

    @classmethod
    def hrs_mins_secs_from_secs(cls, secs):
        return time.strftime("%H:%M:%S", time.gmtime(secs))
    
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

# ---------------------------- Class SignalTreatment ------------
    
class SignalTreatmentDescriptor(object):
    '''
    Hold information on how an original audio signal 
    was transformed to a noise gated file. And how 
    subsequent precision/recall calculations were done
    in different ways.
    
    Instances manage a signal treatment descriptor. These
    are part of both Experiment and PerformanceResult instances.
    They are used to match them reliably. Typically a
    descriptor is first created when a noise-gated signal file
    is created. Later, the minimum required overlap is added.
    It comes into play after the gated file is created.
    
    '''
    
    props = OrderedDict({'threshold_db' : int,
                         'low_freq'  : int,
                         'high_freq'  : int,
                         'min_required_overlap' : int
                         })

    #------------------------------------
    # Constructor 
    #-------------------

    def __init__(self, threshold_db, low_freq, high_freq, min_required_overlap=None):
        '''
        
        @param threshold_db: signal strength below which signal
            is set to 0.
        @type threshold_db: int
        @param low_freq: lowest frequency accepted by front end bandpass filter
        @type low_freq: int
        @param high_freq: highest frequency accepted by front end bandpass filter
        @type high_freq: int
        @param min_required_overlap: percentage of overlap minimally
            required for a burst to count as discovered.
        @type min_required_overlap: float
        '''
        
        try:
            self.threshold_db = int(threshold_db)
        except (ValueError, TypeError):
            raise ValueError("Threshold dB must be int, or str convertible to int")
        
        try:
            self.low_freq = int(low_freq)
        except (ValueError, TypeError):
            raise ValueError("Bandpass freqs must be int, or str convertible to int")

        try:
            self.high_freq = int(high_freq)
        except (ValueError, TypeError):
            raise ValueError("Bandpass freqs must be int, or str convertible to int")
        
        if min_required_overlap in (None, 'none', 'None', 'noneperc'):
            self.min_required_overlap = None
        else:
            try:
                self.min_required_overlap = int(min_required_overlap)
            except (ValueError, TypeError):
                raise ValueError("Minimum required overlap must be int, or str convertible to int")

    #------------------------------------
    # from_str
    #-------------------

    @classmethod
    def from_str(cls, stringified_instance):
        '''
        Reverse of what __str__() produces:
          "SignalTreatmentDescriptor(-30,10,50,20)" ==> an instance
          
        Alternatively, the method will also deal with 
        a flat string: "-20dB_10Hz_5Hz_10perc".
        
        @param cls:
        @type cls:
        @param stringified_instance: Stringified instance: either a
            string that can be evaled, or a flattened string.
        @type stringified_instance: str
        '''
        if not stringified_instance.startswith(cls.__name__):
            return cls.from_flat_str(stringified_instance)
        
        # Safe eval by creating a near-empty environment:
        try:
            inst = eval(stringified_instance,
                        {'__builtins__' : None},
                        {'SignalTreatmentDescriptor' : SignalTreatmentDescriptor}
                        )
        except Exception:
            raise ValueError(f"Expression '{stringified_instance}' does not evaluate to an instance of SignalTreatmentDescriptor")
        return inst

    #------------------------------------
    # from_flat_str
    #-------------------

    @classmethod
    def from_flat_str(cls, str_repr):
        '''
        Assume passed-in string is like "-30dB_10Hz_50Hz_10perc",
        or "-30dB_10Hz_50Hz_Noneperc". Create an instance from
        that info.
        
        @param str_repr:
        @type str_repr:
        '''
        # Already an instance?
        
        (threshold_db_str, 
         low_freq_str,
         high_freq_str, 
         min_overlap_str) = str_repr.split('_')
        
        # Extract the (always negative) threshold dB:
        p = re.compile(r'(^[-0-9]*)')
        err_msg = f"Cannot parse threshold dB from '{str_repr}'"
        try:
            threshold_db = p.search(threshold_db_str).group(1)
        except AttributeError:
            raise ValueError(err_msg)
        else:
            if len(threshold_db) == 0:
                raise ValueError(err_msg)
        
        # Low bandpass freq is int:
        p = re.compile(r'(^[0-9]*)')
        err_msg = f"Cannot parse low bandpass frequency from '{str_repr}'"
        
        try:
            low_freq = p.search(low_freq_str).group(1)
        except AttributeError:
            raise ValueError(err_msg)
        else:
            if len(low_freq) == 0:
                raise ValueError(err_msg)

        # High bandpass freq is int:
        p = re.compile(r'(^[0-9]*)')
        err_msg = f"Cannot parse high bandpass frequency from '{str_repr}'"
        
        try:
            high_freq = p.search(high_freq_str).group(1)
        except AttributeError:
            raise ValueError(err_msg)
        else:
            if len(high_freq) == 0:
                raise ValueError(err_msg)

        # Overlap:
        p = re.compile(r"(^[0-9]*|none)perc")
        err_msg = f"Cannot parse min_required_overlap from '{str_repr}'"
        try:
            min_required_overlap = p.search(min_overlap_str).group(1)
        except AttributeError:
            raise ValueError(err_msg)
        else: 
            if len(min_required_overlap) == 0:
                raise ValueError(err_msg)
            
        return(SignalTreatmentDescriptor(threshold_db,low_freq,high_freq,min_required_overlap))

    #------------------------------------
    # __str__
    #-------------------

    def __str__(self):
        '''
        Produce a string that, if eval is applied, will recreate
        and instance equivalent to this one. Ex:
           SignalTreatmentDescriptor('-40dB_300Hz_10perc')
        '''
        the_str = \
          f"SignalTreatmentDescriptor({self.threshold_db},{self.low_freq},{self.high_freq},{self.min_required_overlap})"
        return the_str

    #------------------------------------
    # __repr__
    #-------------------

    def __repr__(self):
        return f"<{__class__.__name__}: {self.to_flat_str()} at {hex(id(self))}>"
    
    #------------------------------------
    # to_flat_str
    #-------------------

    def to_flat_str(self):
        '''
        
        '''
        descriptor = f"{self.threshold_db}dB_{self.low_freq}Hz_{self.high_freq}Hz"
        if self.min_required_overlap is not None:
            descriptor += f"_{self.min_required_overlap}perc"
        else:
            descriptor += '_noneperc'
        return descriptor


    #------------------------------------
    # add_overlap
    #-------------------

    def add_overlap(self, min_required_overlap):
        '''
        Add the given minimum required overlap to the 
        given signal description.
        
        @param min_required_overlap: minimum percent overlap
        @type min_required_overlap: float
        '''
        self.min_required_overlap = min_required_overlap

    #------------------------------------
    # equality_sig_proc
    #-------------------
    
    def equality_sig_proc(self, other):
        '''
        Return True if at least the signal
        processing is equal with the 'other' 
        instance

        @param other:
        @type other:
        '''
        return self.threshold_db == other.threshold_db and \
            self.low_freq == other.low_freq and \
            self.high_freq == other.high_freq

    #------------------------------------
    # __eq__
    #-------------------
    
    def __eq__(self, other):
        '''
        Return True if all quantities of the 
        passed-in other instance are equal to this
        one.
        
        @param other:
        @type other:
        '''
        return self.equality_sig_proc(other) and \
            self.min_required_overlap == other.min_required_overlap

    def __ne__(self, other):
        if self.threshold_db != other.threshold_db or \
           self.low_freq != other.low_freq or \
           self.high_freq != other.high_freq or \
           self.min_required_overlap != other.min_required_overlap:
            return True
        else:
            return False
           
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
