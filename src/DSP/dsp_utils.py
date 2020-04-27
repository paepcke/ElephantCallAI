'''
Created on Apr 5, 2020

@author: paepcke
'''

from collections import OrderedDict
from enum import Enum
import os
import re

import numpy as np

class PrecRecFileTypes(Enum):
    SPECTROGRAM  = '_spectrogram'
    TIME_LABELS  = '_time_labels'
    FREQ_LABELS  = '_freq_labels'
    GATED_WAV    = '_gated'
    PREC_REC_RES = '_prec_rec_res'
    EXPERIMENT   = '_experiment'
    PICKLE       = '_pickle'
        
class DSPUtils(object):
    '''
    classdocs
    '''

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
          o filtered_wav_-50dB_500Hz_20200404_192128.npy
          o filtered_wav_-50dB_500Hz_20200404_192128
          
        return a new name with identifying info after the
        basename:
        
          o filtered_wav_-50dB_500Hz_20200404_192128_spectrogram.npy
          o filtered_wav_-50dB_500Hz_20200404_192128_time_labels.npy
          o filtered_wav_-50dB_500Hz_20200404_192128_freq_labels.npy
          o filtered_wav_-50dB_500Hz_20200404_192128_gated.wav
        
        The experiment_id argument is for type PICKLE. Such files
        hold a single experiment object in pickle form:
        
         o filtered_wav_-50dB_500Hz_20200404_192128_exp<experiment_id>.pickle
        
        @param root_info: example file name
        @type root_info: str
        @param file_type: one of the PrecRecFileTypes
        @type file_type: PrecRecFileTypes
        '''
        (path_no_ext, ext) = os.path.splitext(root_info)
        if file_type in [PrecRecFileTypes.SPECTROGRAM,
                         PrecRecFileTypes.FREQ_LABELS,
                         PrecRecFileTypes.TIME_LABELS]:
            ext = '.npy'
        elif file_type in [PrecRecFileTypes.GATED_WAV]:
            ext = '.wav'
        elif file_type in [PrecRecFileTypes.PREC_REC_RES,
                           PrecRecFileTypes.EXPERIMENT]:
            ext = '.tsv'
        elif file_type == PrecRecFileTypes.PICKLE:
            ext = f"{experiment_id}.pickle"
            
        new_file_path = f"{path_no_ext}{file_type.value}{ext}"
        return new_file_path

    #------------------------------------
    # get_spectrogram_data
    #-------------------
    
    @classmethod
    def get_spectrogram_data(cls, threshold_db, cutoff_freq):
        #**************
        spectrogram = np.load('/tmp/filtered_wav_-40dB_10Hz_20200423_153323_gated_spectrogram.npy')
        freq_labels = np.load('/tmp/filtered_wav_-40dB_10Hz_20200423_153323_gated_spectrogram_freq_labels.npy')
        time_labels = np.load('/tmp/filtered_wav_-40dB_10Hz_20200423_153323_gated_spectrogram_time_labels.npy')
        return({'spectrogram' : spectrogram,
                'freq_labels' : freq_labels,
                'time_labels' : time_labels
                })

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
                         'cutoff_freq'  : int,
                         'min_required_overlap' : int
                         })

    #------------------------------------
    # Constructor 
    #-------------------

    def __init__(self, threshold_db, cutoff_freq, min_required_overlap=None):
        '''
        
        @param threshold_db: signal strength below which signal
            is set to 0.
        @type threshold_db: int
        @param cutoff_freq: frequency for the envelope creation
        @type cutoff_freq: int
        @param min_required_overlap: percentage of overlap minimally
            required for a burst to count as discovered.
        @type min_required_overlap: float
        '''
        
        try:
            self.threshold_db = int(threshold_db)
        except (ValueError, TypeError):
            raise ValueError("Threshold dB must be int, or str convertible to int")
        
        try:
            self.cutoff_freq = int(cutoff_freq)
        except (ValueError, TypeError):
            raise ValueError("Cutoff freq must be int, or str convertible to int")
        
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
          "SignalTreatmentDescriptor(-30,300,20)" ==> an instance
          
        Alternatively, the method will also deal with 
        a flat string: "-20dB_300Hz_10perc".
        
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
        Assume passed-in string is like "-30dB_300Hz_10perc",
        or "-30dB_300Hz_Noneperc". Create an instance from
        that info.
        
        @param str_repr:
        @type str_repr:
        '''
        # Already an instance?
        
        (threshold_db_str, 
         cutoff_freq_str, 
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
        
        # Cutoff freq is just a number.
        # Usually integer, but allow float:
        p = re.compile(r'(^[0-9.]*)')
        err_msg = f"Cannot parse cutoff frequency from '{str_repr}'"
        
        try:
            cutoff_freq = p.search(cutoff_freq_str).group(1)
        except AttributeError:
            raise ValueError(err_msg)
        else:
            if len(cutoff_freq) == 0:
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
            
        return(SignalTreatmentDescriptor(threshold_db,cutoff_freq,min_required_overlap))

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
          f"SignalTreatmentDescriptor({self.threshold_db},{self.cutoff_freq},{self.min_required_overlap})"
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
        descriptor = f"{self.threshold_db}dB_{self.cutoff_freq}Hz"
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
            self.cutoff_freq == other.cutoff_freq

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
           self.cutoff_freq != other.cutoff_freq or \
           self.min_required_overlap != other.min_required_overlap:
            return True
        else:
            return False
           
# ----------------------------- Main ----------------

if __name__ == '__main__':
    
    # Just some testing; doesn't normally run as main:
    new_file = DSPUtils.prec_recall_file_name('filtered_wav_-50dB_500Hz_20200404_192128.npy', 
                                               PrecRecFileTypes.FREQ_LABELS) 
    assert (new_file == 'filtered_wav_-50dB_500Hz_20200404_192128_freq_labels.npy'),\
            f"Bad freq_label file: {new_file}" 

    new_file = DSPUtils.prec_recall_file_name('filtered_wav_-50dB_500Hz_20200404_192128', 
                                               PrecRecFileTypes.FREQ_LABELS) 
            
    assert (new_file == 'filtered_wav_-50dB_500Hz_20200404_192128_freq_labels.npy'),\
            f"Bad freq_label file: {new_file}" 

    new_file = DSPUtils.prec_recall_file_name('filtered_wav_-50dB_500Hz_20200404_192128.npy', 
                                               PrecRecFileTypes.TIME_LABELS) 

    assert (new_file == 'filtered_wav_-50dB_500Hz_20200404_192128_time_labels.npy'),\
            f"Bad freq_label file: {new_file}" 

    new_file = DSPUtils.prec_recall_file_name('filtered_wav_-50dB_500Hz_20200404_192128.npy', 
                                               PrecRecFileTypes.GATED_WAV) 

    assert (new_file == 'filtered_wav_-50dB_500Hz_20200404_192128_gated.wav'),\
            f"Bad freq_label file: {new_file}" 


    new_file = DSPUtils.prec_recall_file_name('/foo/bar/filtered_wav_-50dB_500Hz_20200404_192128.npy', 
                                               PrecRecFileTypes.GATED_WAV) 

    assert (new_file == '/foo/bar/filtered_wav_-50dB_500Hz_20200404_192128_gated.wav'),\
            f"Bad freq_label file: {new_file}" 


    print('File name manipulation tests OK')    
                                          
    # Test SignalTreatmentDescriptor:
    # To string:
    
    s1 = "SignalTreatmentDescriptor(-30,300,20)"
    d = SignalTreatmentDescriptor(-30, 300, 20)
    assert str(d) == s1

    s2 = "SignalTreatmentDescriptor(-40,400,None)"
    d = SignalTreatmentDescriptor(-40, 400)
    assert str(d) == s2

    try:    
        d = SignalTreatmentDescriptor('foo', 500, 10)
    except ValueError:
        pass
    else:
        raise ValueError(f"String 'foo,500,10' should have raised an ValueError")
    
    # Test SignalTreatmentDescriptor:
    # From string:

    d = SignalTreatmentDescriptor.from_str(s1)
    assert (str(d) == s1)

    d = SignalTreatmentDescriptor.from_str(s2)
    assert (str(d) == s2)
    
    try:
        d = SignalTreatmentDescriptor.from_str('foo_300Hz_30perc')
    except ValueError:
        pass
    else:
        raise AssertionError("Should have ValueError from 'foo_300Hz_30perc'")

    # SignalTreatmentDescriptor
    # Adding min overlap after the fact:
    
    d = SignalTreatmentDescriptor(-40, 400)
    assert (str(d) == "SignalTreatmentDescriptor(-40,400,None)")
    d.add_overlap(10)
    assert (str(d) == "SignalTreatmentDescriptor(-40,400,10)")
    
    # SignalTreatmentDescriptor
    # to_flat_str
    
    s3 = '-30dB_300Hz_20perc'
    d = SignalTreatmentDescriptor.from_flat_str(s3)
    assert(d.to_flat_str() == s3)
    
    s4 = '-40dB_400Hz_noneperc'
    d = SignalTreatmentDescriptor.from_flat_str(s4)
    assert(d.to_flat_str() == s4)
    
    s5 = '-40dB_400Hz_10perc'
    d = SignalTreatmentDescriptor.from_flat_str(s5)
    assert(d.to_flat_str() == s5)
    
    # SignalTreatmentDescriptor
    # Equality
    
    d1 = SignalTreatmentDescriptor(-40,300,10)
    d2 = SignalTreatmentDescriptor(-40,300,10)
    assert (d1.__eq__(d2))
    
    d1 = SignalTreatmentDescriptor(-40,300,None)
    d2 = SignalTreatmentDescriptor(-40,300,10)
    assert (not d1.__eq__(d2))
    
    d1 = SignalTreatmentDescriptor(-40,300,10)
    d2 = SignalTreatmentDescriptor(-40,300,20)
    assert (d1.equality_sig_proc(d2))
    assert (not d1.__eq__(d2))
    
    
    
    
    print('SignalTreatmentDescriptor tests OK')
    
    print("Tests done.")
