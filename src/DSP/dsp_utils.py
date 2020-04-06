'''
Created on Apr 5, 2020

@author: paepcke
'''

from enum import Enum
import os

class PrecRecFileTypes(Enum):
    SPECTROGRAM  = '_spectrogram'
    TIME_LABELS  = '_time_labels'
    FREQ_LABELS  = '_freq_labels'
    GATED_WAV    = '_gated'
    PREC_REC_RES = '_prec_rec_res'
    EXPERIMENT   = '_experiment'
        
class DSPUtils(object):
    '''
    classdocs
    '''

    #------------------------------------
    # prec_recall_file_name
    #-------------------

    @classmethod
    def prec_recall_file_name(cls, root_info, file_type):
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
            
        new_file_path = f"{path_no_ext}{file_type.value}{ext}"
        return new_file_path

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


    print('Tests OK')    
                                          
                                          

                                          