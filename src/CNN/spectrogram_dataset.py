'''
Created on Jul 7, 2020

@author: paepcke
'''

import ast
from collections import deque
import glob
import os
from pathlib import Path
import re
from sqlite3 import OperationalError as DatabaseError
import sqlite3
import sys

from pandas.core.frame import DataFrame
from sklearn.model_selection import KFold
from sklearn.model_selection._split import RepeatedKFold, StratifiedKFold, \
    RepeatedStratifiedKFold
from torch.utils.data import Dataset

from DSP.dsp_utils import AudioType
from DSP.dsp_utils import DSPUtils
from DSP.dsp_utils import FileFamily
from elephant_utils.logging_service import LoggingService
import numpy as np
import pandas as pd


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))




TESTING = False
#TESTING = True

# To change spectrogram snippet width time,
# see class variable in SpectrogramDataset.

class ChopError(Exception):
    '''
    Error during spectrogram chopping:
    '''
    
    def __init__(self, msg):
        super().__init__(msg)

# ------------------------------- Class Frozen Dataset ----------

class FrozenDataset(Dataset):

    SPACE_TO_COMMA_PAT = re.compile(r'([0-9])[\s]+')
    
    #------------------------------------
    # Constructor
    #-------------------

    def __init__(self,
                 log,
                 db,
                 split_id,
                 queue,
                 label_mapping,
                 sample_ids
                 ):
        self.log = log
        self.db = db
        self._split_id = split_id
        self.label_mapping = label_mapping
        self.sample_ids = sample_ids
                
        self.queue = queue
        self.saved_queue = queue.copy()

    #------------------------------------
    # split_id
    #-------------------
    
    def split_id(self):
        try:
            return self._split_id
        except AttributeError:
            return "Not Yet Split"

    #------------------------------------
    # reset
    #-------------------

    def reset(self):
        '''
        Sets the dataset's queue to the beginning.
        '''

        # Replenish the requested queue

        self.queue = self.saved_queue.copy()

# ---------------------- Utilities ---------------

    #------------------------------------
    # to_np_array 
    #-------------------

    def to_np_array(self, array_string):
        '''
        Given a string:
          "[ 124  56  32]"
        return an np_array: np.array([124,56,32]).
        Also works for more reasonable strings like:
          "[1, 2, 5]"
        
        @param array_string: the string to convert
        @type array_string: str
        '''

        # Use the pattern to substitute occurrences of
        # "123   45" with "123,45". The \1 refers to the
        # digit that matched (i.e. the capture group):
        proper_array_str = self.SPACE_TO_COMMA_PAT.sub(r'\1,', array_string)
        # Remove extraneous spaces:
        proper_array_str = re.sub('\s', '', proper_array_str)
        # Turn from a string to array:
        return np.array(ast.literal_eval(proper_array_str))

    #------------------------------------
    # clean_row_res
    #-------------------
    
    def clean_row_res(self, row):
        '''
        Given a row object returned from sqlite, 
        turn tok_ids and attention_mask into real
        np arrays, rather than their original str
        
        @param row:
        @type row:
        '''
        
        # tok_ids are stored as strings:
        row['tok_ids'] = self.to_np_array(row['tok_ids'])
        row['attention_mask'] = self.to_np_array(row['attention_mask'])
        return row

    
    #------------------------------------
    # __next__ 
    #-------------------

    def __next__(self):
        try:
            next_sample_id = self.queue.popleft()
        except IndexError:
            raise StopIteration
        
        res = self.db.execute(f'''
                               SELECT sample_id, tok_ids,attention_mask,label
                                FROM Samples 
                               WHERE sample_id = {next_sample_id}
                             ''')
        row = next(res)
        return self.clean_row_res(dict(row))
    
    #------------------------------------
    # __getitem__ 
    #-------------------

    def __getitem__(self, indx):
        '''
        Return indx'th row from the db.
        The entire queue is always used,
        rather than the remaining queue
        after some popleft() ops. 
        
        @param indx:
        @type indx:
        '''

        ith_sample_id = self.saved_queue[indx]
        res = self.db.execute(f'''
                               SELECT sample_id, tok_ids,attention_mask,label
                                FROM Samples 
                               WHERE sample_id = {ith_sample_id}
                             ''')
        # Return the (only result) row:
        row = next(res)
        return self.clean_row_res(dict(row))
    
    #------------------------------------
    # __iter__ 
    #-------------------
    
    def __iter__(self):
        return self

    #------------------------------------
    # __len__
    #-------------------
    
    def __len__(self):
        '''
        Return length of the current split. Use
        switch_to_split() before calling this
        method to get another split's length.
        The length of the entire queue is returned,
        not just what remains after calls to next()
        '''
        return len(self.saved_queue)

# ------------- SpectrogramDataset ---------

class SpectrogramDataset(FrozenDataset):
    '''
    Torch dataset for iterating over spectrograms.
    Assumption:
        o Given one directory that contains label files
          as those created by Raven.
        o Given any number of directories that only contain 
          spectrograms, or individual spectrogram files
        o Naming convention:
             - The basename of the original .wav files from 
               which one or more of the spectrograms were created
               is called FILE ROOT. Example: all but the .wav
               extension of nn05e_20180504_000000.wav is a
               file root.
             - Label files begin with a file root, and have
               extension .txt
             - Spectrogram file names begin with a 
               file root, and end with _spectrogram.pickle if
               the are 24 hr spectrograms. Else the end with
               _spectrogram_<n>.pickle 
        o The spectrograms are created by Spectrogrammer 
          (see spectrogrammer.py), and are Pandas dataframes.
          The indexes (row labels) are frequencies. The 
          columns names are absolute times in seconds into
          the 24hr recording of which the spectrogram is
          a part.
          
    We create an Sqlite database with metadata about each 
    spectrogram. One spectrogram is one sample. Schema:
        o Table Samples:
                  * sample_id int,
                  * label tinyint,   # Has call or not
                  * num_calls int,   # May have multiple calls in one spectrogram
                  * start_time int,  # Seconds into the original 24hr recording
                  * end_time int,
                  * spectroFilename varchar(255),
                  * recordingFilename varchar(255) # Name of 24hr wav file. 

???Needed?        o Table Files
                  * processed_dir varchar(255)
                  

    This class behaves like an interator and a dict.
    The keys to the dict are sample_id values.
        
    An additional feature is the option for integrated
    train/validation/test splits. Calling split_dataset()
    internally produces input queues that feed three 
    iterators. Callers switch between these iterators via
    the switch_to_split() method. The splits can be reset
    to their beginnings using the reset() method.
    '''

    SNIPPET_WIDTH  = 5 # approximate width of spectrogram snippets.
    LOW_FREQ_BAND  = pd.Interval(left=0, right=21)
    MED_FREQ_BAND  = pd.Interval(left=21, right=41)
    HIGH_FREQ_BAND = pd.Interval(left=41, right=51)

    #------------------------------------
    # Constructor 
    #-------------------

    def __init__(self,
                 dirs_of_spect_files=None,
                 sqlite_db_path=None,
                 recurse=False,
                 snippet_outdir=None,
                 testing=False
                 ):
        #***** if snippet_outdir is None, snippets
        #      go where spectrogram is.
        self.snippet_outdir = snippet_outdir

        # Allow unittests to create an instance and
        # then call methods selectively:
        if testing:
            return
        if dirs_of_spect_files is None and sqlite_db_path is None:
            raise ValueError("Directories and sqlite_db_path args must not both be None")
        
        self.log = LoggingService()
        
        if sqlite_db_path is not None:
            # If a db to that path exists, open it:
            if os.path.exists(sqlite_db_path):
                self.db = sqlite3.connect(sqlite_db_path)
                self.db.row_factory = sqlite3.Row
            else:
                # Path is intended as dest for the new db:
                self.db = self.create_db(sqlite_db_path)
        else:
            # Create a db:
            default_db_name = os.path.join(os.path.dirname(__file__),
                                           'spectrograms.sqlite'
                                           )
            self.db = self.create_db(default_db_name)
        
        # Get already processed dirs. The 'list()' pull all hits from
        # the db at once (like any other iterator)
        processed_dirs = list(self.db.execute('''
                                    SELECT processed_dirs FROM Directories;
                                              '''))
        if dirs_of_spect_files is not None:
            # Process those of the given dirs_of_spect_files that 
            # are not already in the db:
            dirs_or_files_to_do = set(dirs_of_spect_files) - set(processed_dirs)
        else:
            dirs_or_files_to_do = set()

        if len(dirs_or_files_to_do) == 0:
            self.process_spectrograms(dirs_or_files_to_do, recurse=recurse)

        num_samples_row = next(self.db.execute('''SELECT COUNT(*) AS num_samples from Samples'''))
        
        # Total number of samples in the db:
        self.num_samples = num_samples_row['num_samples']

        # Our sample ids go from 0 to n. List of all sample ids:
        self.sample_ids = list(range(self.num_samples))

        # The following only needed in case the 
        # class is used without either the split_dataset()
        # or the kfold*() facilities:
        
        # Make a preliminary train queue with all the
        # sample ids. If split_dataset() is called later,
        # this queue will be replaced:
        self.train_queue = deque(self.sample_ids)
        self.curr_queue  = self.train_queue
        self.saved_queues = {}
        # Again: this saved_queues entry will be
        # replaced upon a split:
        self.saved_queues['train'] = self.train_queue.copy()
        self.num_samples = len(self.train_queue)

    #------------------------------------
    # get_datasplit
    #-------------------

    def get_datasplit(self, split_id):
        if split_id == 'train':
            return  self.train_frozen_dataset
        elif split_id == 'validate':
            return self.validate_frozen_dataset
        elif split_id == 'test':
            return self.test_frozen_dataset
        else:
            raise ValueError("Only train, validate, and test are valid split ids.")

    #------------------------------------
    # switch_to_split 
    #-------------------
    
    def switch_to_split(self, split_id):
        
        if split_id == 'train':
            self.curr_queue = self.train_queue
        elif split_id == 'validate':
            self.curr_queue = self.val_queue
        elif split_id == 'test':
            self.curr_queue = self.test_queue
        else:
            raise ValueError(f"Dataset ID must be one of train/validate/test; was {split_id}")

    #------------------------------------
    # curr_dataset_id 
    #-------------------

    def curr_split_id(self):
        if self.curr_queue == self.train_queue:
            return 'train'
        if self.curr_queue == self.val_queue:
            return 'validate'
        if self.curr_queue == self.test_queue:
            return 'test'
        raise ValueError("Bad curr_queue")
        
    #------------------------------------
    # reset
    #-------------------

    def reset(self, split_id=None):
        '''
        Sets the dataset's queue to the beginning.
        If dataset_id is None, resets the current
        split.
                
        @param split_id:
        @type split_id:
        '''

        # After replenishing the requested
        # queue, check whether that queue was
        # self.curr_queue. If so, change self.curr_queue
        # to point to the new, refilled queue. Else
        # self.curr_queue remains unchanged:
                
        if split_id == 'train':
            old_train = self.train_queue
            self.train_queue = self.saved_queues['train'].copy()
            if self.curr_queue == old_train:
                self.curr_queue = self.train_queue

        elif split_id == 'validate':
            old_val = self.val_queue
            self.val_queue = self.saved_queues['validate'].copy()
            if self.curr_queue == old_val:
                self.curr_queue = self.val_queue
            
        elif split_id == 'test':
            old_test = self.test_queue
            self.test_queue = self.saved_queues['test'].copy()
            if self.curr_queue == old_test:
                self.curr_queue = self.test_queue

        else:
            raise ValueError(f"Dataset ID must be one of train/validate/test; was {split_id}")

    #------------------------------------
    # process_spectrograms 
    #-------------------

    def process_spectrograms(self, dirs_or_files_to_do, recurse=False):
        '''
        Given a list of files and directories, find
        24-hr spectrograms. Find the corresponding label
        files. Chop each spectrogram into self.SNIPPET_WIDTH
        seconds. Place into the Sqlite db the location of each
        spectrogram, its start/end times, the 'parent' 
        24-hr spectrogram, and the filename of the snippet.
        The ****Snippe_id alone is not key!!!!!****
        
        @param dirs_or_files_to_do:
        @type dirs_or_files_to_do:
        @param recurse:
        @type recurse:
        '''

        # Get flat list of files, recursively
        # descending dirs if requested
        files_to_do = self.files_from_dirs(dirs_or_files_to_do, recurse)
        
        # As we encounter 24-hr spectrograms, note
        # them as keys, and set value to the corresponding
        # label file. However, if we find any snippet
        # of that same spectrogram later, we remove the
        # entry: no need to chop up that spectrogram:
        full_spectrograms_queued = {}

        for file in files_to_do:
            file_family = FileFamily(file)
            # Name without path and extention:
            file_root = file_family.file_root
            if file_family.file_type == AudioType.SPECTRO:
                # 24-hour spectrogram: note corresponding label,
                # file:
                full_spectrograms_queued[file_root] = os.path.join(file_family['path'],
                                                       file_family['label'])
                continue
            if file_family.file_type == AudioType.SNIPPET:
                # No need to process the corresponding full 
                # spectrogram, if we encounter it:
                try:
                    del full_spectrograms_queued[file_root]
                except Exception:
                    pass
                continue
            # Ignore all other files:
            continue

        # Chop up any 24-hour spectrograms for which we
        # found no snippet at all:
        
        self.chop_spectograms(full_spectrograms_queued)

    #------------------------------------
    # chop_spectograms 
    #-------------------
    
    def chop_spectograms(self, spectro_dict):
        '''
        Given a dict: {spectrogram_path : label_path},
        chop the spectrogram in snippets, write them
        to self.snippet_outdir, and add them to the 
        db.
        
        Width of each snippet will be close to 
        self.snippet_len, such that there is an
        even number of seconds in each snippet
        
        Assumptions: o self.snippet_outdir holds destination dir
                     o self.snippet_len holds width of snippets
                       in seconds.
        
        @param spectro_dict: 24-hr spectrograms to chop
        @type spectro_dict: {str:str}
        '''
        try:
            for (spectro_file, label_file) in spectro_dict.items():
                # Get dict with file names that are by convention
                # derived from the original .wav file:
                curr_file_family = FileFamily(spectro_file)
                spect_df = pd.read_pickle(spectro_file)
                
                # Compute mean energy in three frequency
                # bands across the whole 24 hr spectrogram:
                freq_energies = self.mean_magnitudes(spect_df)
                # Copy to a new dict to change the key names:
                parent_freq_energies = {'parent_low_freqs_energy' : freq_energies['low_freq_mean'],
                                        'parent_med_freqs_energy' : freq_energies['med_freq_mean'],
                                        'parent_high_freqs_energy': freq_energies['high_freq_mean']
                                        }
                
                # If no outdir for snippets was specified by caller,
                # use the directory of this 24-hr spectrogram:
                if self.snippet_outdir is None:
                    self.chop_one_spectrogram(spect_df, 
                                              label_file,
                                              os.path.dirname(spectro_file),
                                              parent_freq_energies, 
                                              curr_file_family)
                else:
                    self.chop_one_spectrogram(spect_df, 
                                              label_file,
                                              self.snippet_outdir,
                                              parent_freq_energies,
                                              curr_file_family)
        except Exception as e:
            raise ChopError(f"Error trying to close Sqlite db: {repr(e)}") from e

    #------------------------------------
    # mean_magnitudes 
    #-------------------
    
    def mean_magnitudes(self, spectro):
        '''
        Given a spectrogram, compute the mean energy
        across the entire spectro along three frequency bands.
        Return the three means in a dict:
        
            {'low_freq_mags' : <float>,
             'med_freq_mags' : <float>,
             'high_freq_mags': <float>
             }
             
        @param spectro: spectrogram of any width
        @type spectro: pd.DataFrame
        '''
        
        low_freq_slice = spectro[(spectro.index.values >= self.LOW_FREQ_BAND.left) & 
                                 (spectro.index.values < self.LOW_FREQ_BAND.right)]
        low_freq_mean  = low_freq_slice.mean().mean()
        
        med_freq_slice = spectro[(spectro.index.values >= self.MED_FREQ_BAND.left) & 
                                 (spectro.index.values < self.MED_FREQ_BAND.right)]
        med_freq_mean  = med_freq_slice.mean().mean()
        
        high_freq_slice = spectro[(spectro.index.values >= self.HIGH_FREQ_BAND.left) & 
                                 (spectro.index.values < self.HIGH_FREQ_BAND.right)]
        high_freq_mean  = high_freq_slice.mean().mean()
        
        return {'low_freq_mean'  : low_freq_mean,
                'med_freq_mean'  : med_freq_mean,
                'high_freq_mean' : high_freq_mean
                }
        

    #------------------------------------
    # chop_one_spectrogram 
    #-------------------
    
    def chop_one_spectrogram(self, 
                             spect_df, 
                             label_file,
                             snippet_outdir,
                             parent_freq_energies,
                             curr_file_family):
        '''
        Takes one 24-hr spectrogram, and chops it
        into smaller spectrograms of approximately 
        self.SNIPPET_WIDTH seconds. Each snippet
        is written to snippet_outdir, named 
        <24-hr-spectrogram-filename>_<snippet-nbr>_spectrogram.pickle
        Each snippet is a Pandas DataFrame, and can be
        read via pd.read_pickle(filename).
        
        In addition, information on each snippet is
        written to table Samples in Sqlite database
        <24-hr-spectrogram-filename>_snippets.sqlite
        For the db schema, see get_db() header comment.
        Most importantly, each row represents one snippet,
        and contains a sample_id and the full file path
        to the snippet file. It also contains the label
        of that snippet: 1 if the snippet contains an elephant
        call, else 0. The db is used later to 
        generate streams of samples for the classifier. 
        
        @param spect_df: dataframe of a 24-hour spectrogram
        @type spect_df: pd.DataFrame
        @param label_file: full path to Raven true-label file
            that corresponds to the 24-hr spectrogram.
        @type label_file: str
        @param snippet_outdir: directory where snippets will be placed
        @type snippet_outdir: str
        @param parent_freq_energies: mean energies of parent 24-hr
            spectrogram in three frequency bands
        @type parent_freq_energies: {str : float}
        @param curr_file_family: dict with information about the
            various file names associated with the spectrogram.
            See DSPUtils.decode_filename()
        @type curr_file_family: {str : str}
        '''
        
        times = spect_df.columns
        earliest = times[0]
        latest   = times[-1]
        time_per_xtick = times[1]
        # Be conservative in the following by
        # taking the floor. So snippets may be
        # a bit shorter than SNIPPET_WIDTH if
        # the division isn't even:
        xticks_per_snippet = int(np.floor(self.SNIPPET_WIDTH / time_per_xtick))
        time_span = latest - earliest
        # If not an even division, we leave on the table
        # less than self.SNIPPET_WIDTH seconds of samples:
        num_snippets =  int(np.floor(time_span / self.SNIPPET_WIDTH))
        snippet_file_root = curr_file_family.file_root
        # Like /foo/bar/nouabile0012, to which we'll add
        # '_<snippet_id>_spectrogram.pickle below:
        snippet_dest = str(Path(snippet_outdir).joinpath(snippet_file_root)) 
        
        # Walk through spectrogram from its start
        # time to its end time, and pull out 
        # spectrogram snippets:
        
        for snippet_id in range(0,num_snippets):
            # First xtick in this snippet:
            start_xtick = snippet_id * xticks_per_snippet
            # Range of xticks in the snippet:
            snip_xtick_interval = pd.Interval(left=start_xtick, 
                                              right=start_xtick + xticks_per_snippet
                                              )
            # Translated into a range of real time in secs:
            snip_time_interval  = snip_xtick_interval * time_per_xtick
            
            # Get the excerpt:
            snippet = spect_df.iloc[:,snip_xtick_interval.left:snip_xtick_interval.right]
            
          
            label = self.label_for_snippet(snip_xtick_interval, label_file)
            
            # Create a file name for the snippet, but leave
            # out the snippet number, which we will only know
            # once an entry was made to the db:
            snippet_file_name_template = f"{snippet_dest}_???_spectrogram.pickle"
            
            # Get this snippet's mean energy in three frequency
            # bands:
            snippet_freq_energies = self.mean_magnitudes(snippet)
            # Combine parent and snippet energies:
            # Just for clarity:
            freq_energies = parent_freq_energies
            freq_energies.update({'snippet_low_freqs_energy' : snippet_freq_energies['low_freq_mean'],
                                  'snippet_med_freqs_energy' : snippet_freq_energies['med_freq_mean'],
                                  'snippet_high_freqs_energy' : snippet_freq_energies['high_freq_mean']
                                  })
            
            # Get the authoritative snipped id, and the 
            # finalized destination file name:
            (db_snippet_id, snippet_file_name) = self.add_snippet_to_db(
                                                   snippet_file_name_template, 
                                                   label,
                                                   snip_xtick_interval,
                                                   snip_time_interval,
                                                   freq_energies,
                                                   curr_file_family)
            # Fill the snippet id into the file family:
            curr_file_family.snippet_id = db_snippet_id
            
            # Save the snippet to file:
            snippet.to_pickle(snippet_file_name)
            

    #------------------------------------
    # label_for_snippet 
    #-------------------
    
    def label_for_snippet(self, 
                          snippet_interval, 
                          label_file,
                          required_overlap_percentage=None):
        '''
        Given the time interval of a spectrogram snippet
        return True or False for whether the interval overlaps
        with a bona fide elephant cal. The interval is in
        seconds.
        
        The intervals from all label files are cached to obviate
        multiple loading and parsing of the label file into intervals.
        
        @param snippet_interval: begin and end times of a spectrogram snippet
        @type snippet_interval: pdInterval
        @param label_file: path to the Raven label file of the
            full spectrogram
        @type label_file: str
        @param required_overlap_percentage: if provided, the amount of 
            overlap required to count the match as True. I None,
            any degree of overlap is acceptable
        @type required_overlap_percentage: {None | int | float}
        @return: whether or not the interval overlaps a label
            interval in the given label file.
        @rtype: bool
        '''
        
        try:
            label_intervals = self.label_interval_cache[label_file]
        except KeyError:
            label_intervals = DSPUtils.load_label_time_intervals(label_file)
            self.label_interval_cache[label_file] = label_intervals
        except AttributeError:
            # Cache doesn't exist yet:
            label_intervals = DSPUtils.load_label_time_intervals(label_file)
            self.label_interval_cache = {label_file : label_intervals}

        does_overlap = False
        for label_interval in label_intervals:
            does_overlap = label_interval.overlaps(snippet_interval)
            if does_overlap:
                does_overlap = True
                break
        if required_overlap_percentage is None or not does_overlap:
            # Any degree of overlap is fine:
            return does_overlap
        snippet_overlap_percentage = DSPUtils.overlap_percentage(snippet_interval,
                                                                 label_interval)
        # Deal with required_overlap_percentage provided as 
        # number between 0 and 1, as well as between
        # 0 and 100:
        if type(required_overlap_percentage) == float:
            # Our computed snippet_overlap_percentage is in fractions
            # of 1, and so is the given minimal requirement:
            return snippet_overlap_percentage >= required_overlap_percentage
        else:
            # Given snippet_overlap_percentage was an in 0 to 100:
            return int(100 * snippet_overlap_percentage) >= required_overlap_percentage

    #------------------------------------
    # add_snippet_to_db
    #-------------------
    
    def add_snippet_to_db(self, 
                          snippet_file_name_template,
                          label,
                          snippet_xtick_interval,
                          snippet_time_interval,
                          freq_band_energies,
                          curr_file_family):
        '''
        
        Adds a record for a spectrogram snippet into the
        Sqlite db. The record in includes the snippet_id,
        the file where the snippet resides, the recording
        site as derived from the filename in curr_file_family,
        as well as X-axis time slice start/stop index and
        start/stop times of the snippet relative to the start
        of the parent spectrogram.
        
        Trick is that we use Sqlite's automatic ROWID generation
        for snippet ids. And we only know those after inserting
        the record. That id is part of the snippet's future
        file name, which itself has the snippet id in it.
        Since that file name needs to be part of the snippet's
        record, though, we have to do an INSERT, followed
        by an UPDATE. Obtaining the ROWID (i.e. the snippet id)
        after the insert is free. The update of the ultimate
        file name is not. 
        
        @param snippet_file_name_template: partially constructed
            file name where the snippet dataframe will be
            stored on disk: "foo_???_spectrogram.pickle"
            The question marks are replaced in this method
            with the ROWID of the newly created record
        @type snippet_file_name_template: str
        @param label: the snippet's label: 1/0
        @type label: int
        @param snippet_xtick_interval: interval of 
            x slots in spectrogram
        @type snippet_xtick_interval: Pandas Interval
        @param snippet_time_interval: interval of
            true times since start of parent spectrogram
        @type snippet_time_interval: Pandas Interval
        @param freq_band_energies: mean energy in three 
            frequency bands of parent 24hr-spectrogram,
            and of this snippet
        @type freq_band_energies: {str : float}
        @param curr_file_family: info about the snippet's
            file family (see dsp_utils.file_family).
        @type curr_file_family: FileFamily
        '''

        recording_site = curr_file_family.file_root

        insertion = f'''
                    INSERT INTO Samples (recording_site,
                                         label,
                                         start_time_tick,
                                         end_time_tick,
                                         start_time,
                                         end_time,
                                         parent_low_freqs_energy,
                                         parent_med_freqs_energy,
                                         parent_high_freqs_energy,
                                         snippet_low_freqs_energy,
                                         snippet_med_freqs_energy,
                                         snippet_high_freqs_energy
                                         )
                            VALUES ('{recording_site}',
                                    {label},
                                    {snippet_xtick_interval.left},
                                    {snippet_xtick_interval.right},
                                    {snippet_time_interval.left},
                                    {snippet_time_interval.right},
                                    {freq_band_energies['parent_low_freqs_energy']},
                                    {freq_band_energies['parent_med_freqs_energy']},
                                    {freq_band_energies['parent_high_freqs_energy']},
                                    {freq_band_energies['snippet_low_freqs_energy']},
                                    {freq_band_energies['snippet_med_freqs_energy']},
                                    {freq_band_energies['snippet_high_freqs_energy']}
                                    );
                    '''
        try:
            # The Python API to the Sqlite3 db
            # automatically begins a transaction:
            
            cur = self.db.execute(insertion)
            
        except Exception as e:
            self.db.rollback()
            # Raise DatabaseError but with original stacktrace:
            raise DatabaseError(repr(e)) from e
        
        # Get the ROWID that was assigned to the row 
        # we just wrote above:
        db_snippet_id = cur.lastrowid
        
        # Safe to commit the partially filled in
        # snippet record now that we have its
        # ROWID:
        
        self.db.commit()
        
        # Use this ultimate sample id to finalize
        # the file name where the caller will write
        # the spectrogram snippet:
        
        snippet_file_name = snippet_file_name_template.replace('???', str(db_snippet_id))

        # Finally: update the snippet_filename column
        # of the just-written entry:
        
        self.db.execute(f'''UPDATE Samples SET snippet_filename = '{snippet_file_name}'
                           WHERE sample_id = {db_snippet_id}
                           '''
                           )
        
        self.db.commit()
        return (db_snippet_id, snippet_file_name)


    #------------------------------------
    # files_from_dirs 
    #-------------------
    
    def files_from_dirs(self, dirs_and_files, recurse=False):
        '''
        Given a mixed list of files and directories,
        return a list of files. If recurse is True,
        do that.
        
        @param dirs_and_files: mixed list of directories and files 
        @type dirs_and_files: [str]
        @param recurse: whether or not to recurse down the dirs
        @type recurse: bool
        '''

        all_files = []

        for file_or_dir in dirs_and_files:
            if os.path.isfile(file_or_dir):
                all_files.append(file_or_dir)
            else:
                # Make sure the dir as a trailing slash:
                if not file_or_dir.endswith('/'):
                    file_or_dir += '/'
                for filename in glob.iglob(file_or_dir + '**/*', recursive=recurse):
                    all_files.append(filename)

        return all_files

    #------------------------------------
    # __next__ 
    #-------------------

    def __next__(self):
        try:
            next_sample_id = self.curr_queue.popleft()
        except IndexError:
            # We are out of either test samples
            # or validation samples. Get the next
            # fold if there is one, and add the
            # test/validation samples to the test
            # and val queues. The following will
            # throw StopIteration if no folds are
            # left:

            (train_sample_ids, validate_sample_ids) = \
               next(self.folds_iter)
               
            self.train_queue.extend(train_sample_ids)
            self.val_queue.extend(validate_sample_ids)
            
            self.train_labels.extend(self.labels_from_db(self.train_sample_ids))
            self.validate_labels.extend(self.labels_from_db(self.validate_sample_ids))

            # Retry getting a sample:
            try:
                next_sample_id = self.curr_queue.popleft()
            except IndexError:
                # Truly out of folds, though that should
                # have caused an earlier StopIteration.
                raise StopIteration
        
        res = self.db.execute(f'''
                               SELECT sample_id, label, snippet_filename
                                FROM Samples 
                               WHERE sample_id = {next_sample_id}
                             ''')
        row = next(res)
        snippet_df = DSPUtils.load_spectrogram(row['snippet_filename'])
        return {'snippet_df' : snippet_df, 'label' : row['label']}
    
    #------------------------------------
    # __getitem__ 
    #-------------------

    def __getitem__(self, indx):
        '''
        Return indx'th row from the db.
        The entire queue is always used,
        rather than the remaining queue
        after some popleft() ops. 
        
        @param indx:
        @type indx:
        '''

        ith_sample_id = self.saved_queues[self.curr_split_id()][indx]
        res = self.db.execute(f'''
                               SELECT sample_id, label, snippet_filename
                                FROM Samples 
                               WHERE sample_id = {ith_sample_id}
                             ''')
        row = next(res)
        snippet_df = DSPUtils.load_spectrogram(row['snippet_filename'])
        return {'snippet_df' : snippet_df, 'label' : row['label']}

    #------------------------------------
    # __iter__ 
    #-------------------
    
    def __iter__(self):
        return self

    #------------------------------------
    # __len__
    #-------------------
    
    def __len__(self):
        '''
        Return length of the current split. Use
        switch_to_split() before calling this
        method to get another split's length.
        The length of the entire queue is returned,
        not just what remains after calls to next()
        '''
        return len(self.saved_queues[self.curr_split_id()])

    #------------------------------------
    # kfold 
    #-------------------

    def kfold(self, 
               n_splits=5,
               n_repeats=0,
               shuffle=False,
               random_state=None
               ):
        '''
        Uses sklearn's KFold and StratifiedKFold facility. 
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
        See also method kfolds_stratified() for balanced
        folds.
        
        The 'X' in this context are sample ids that are eventually 
        used by the dataloader to retrieve spectrograms. Recall
        that each sample id stands for one spectrogram snippet. The
        'y' vector in the KFold page is the 'label' column in the
        Sqlite db, which is 1 or 0 for each column (i.e. time bin)
        of the spectrogram.
        
        All methods on the sklearn KFold facility
        are available in this class by the same name.
        
        After calling this method, calls to next() will
        return train samples.
        
        
        @param n_splits: number of folds to create 
        @type n_splits: int
        @param n_repeats: number times fold splitting should
            be repeated (n-times k-fold cross validation.
            Set to zero, the method uses sklearn KFold class,
            else it uses the sklearn.RepeatedKFold
        @type n_repeats: int
        @param shuffle: whether or not to shuffle the 
            data before splitting. Once split, the 
            data in the folds are not shuffled 
        @type shuffle: bool
        @param random_state: if shuffle is set to True,
            this argument allows for repeatability over
            multiple runs
        @type random_state: int
        '''
        if n_repeats == 0:
            self.cross_validator = KFold(n_splits=n_splits,
                                         shuffle=shuffle,
                                         random_state=random_state)
        else:
            self.cross_validator = RepeatedKFold(n_splits=n_splits,
                                                 n_repeats=n_repeats,
                                                 random_state=random_state)
            
        # The following retrieves *indices* into 
        # our list of sample_ids. However, since
        # our sample_ids are just numbers from 0 to n,
        # the indices are equivalent to the sample ids
        # themselves
        
        # The split method will return a generator
        # object. Each item in this generator is
        # a 2-tuple: a test set array and a validation
        # set array. There will be n_splits such tuples.
        
        # We grab the first pair:
        
        self.folds_iter = self.cross_validator.split(self.sample_ids) 
        (self.train_sample_ids, self.validate_sample_ids) = \
            next(self.folds_iter)
            
        self.train_queue = deque(self.train_sample_ids)
        self.val_queue   = deque(self.validate_sample_ids)
        self.train_labels = self.labels_from_db(self.train_sample_ids)
        self.validate_labels = self.labels_from_db(self.validate_sample_ids)
        self.switch_to_split('train')

    #------------------------------------
    # kfold_stratified 
    #-------------------

    def kfold_stratified(self, 
               n_splits=5,
               n_repeats=0,
               shuffle=False,
               random_state=None
               ):
        '''
        Uses sklearn's StratifiedKFold and RepeatedStratifiedKFold facility.
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
        See also method kfold() for folding without balancing.
        
        The 'X' in this context are sample ids that are eventually 
        used by the dataloader to retrieve spectrograms. Recall
        that each sample id stands for one spectrogram snippet. The
        'y' vector in the KFold page is the 'label' column in the
        Sqlite db, which is 1 or 0 for each column (i.e. time bin)
        of the spectrogram.
        
        All methods on the sklearn [Repeated]StratifiedKFold facility
        are available in this class by the same name.
        
        After calling this method, calls to next() will
        return train samples. I.e. the current queue is set
        to self.train_queue
        
        
        @param n_splits: number of folds to create 
        @type n_splits: int
        @param n_repeats: number times fold splitting should
            be repeated (n-times k-fold cross validation.
            Set to zero, the method uses sklearn KFold class,
            else it uses the sklearn.RepeatedKFold
        @type n_repeats: int
        @param shuffle: whether or not to shuffle the 
            data before splitting. Once split, the 
            data in the folds are not shuffled 
        @type shuffle: bool
        @param random_state: if shuffle is set to True,
            this argument allows for repeatability over
            multiple runs
        @type random_state: int
        '''
        if n_repeats == 0:
            self.cross_validator = StratifiedKFold(n_splits=n_splits,
                                                   shuffle=shuffle,
                                                   random_state=random_state)
        else:
            self.cross_validator = RepeatedStratifiedKFold(n_splits=n_splits,
                                                           n_repeats=n_repeats,
                                                           random_state=random_state)
            
        # The following retrieves *indices* into 
        # our list of sample_ids. However, since
        # our sample_ids are just numbers from 0 to n,
        # the indices are equivalent to the sample ids
        # themselves
        
        # The split method will return a generator
        # object. Each item in this generator is
        # a 2-tuple: a test set array and a validation
        # set array. There will be n_splits such tuples.
        
        # We grab the first pair:
        
        all_labels = self.labels_from_db(self.sample_ids)
        self.folds_iter = self.cross_validator.split(self.sample_ids, all_labels) 
        (self.train_sample_ids, self.validate_sample_ids) = \
            next(self.folds_iter)
            
        self.train_queue = deque(self.train_sample_ids)
        self.val_queue   = deque(self.validate_sample_ids)
        self.train_labels = self.labels_from_db(self.train_sample_ids)
        self.validate_labels = self.labels_from_db(self.validate_sample_ids)
        self.switch_to_split('train')


    #------------------------------------
    # get_n_splits 
    #-------------------
    
    def get_n_splits(self):
        '''
        Straight pass_through to the sklearn method
        Returns the number of splitting iterations in the cross-validator
        '''
        return self.cross_validator.get_n_splits()

    #------------------------------------
    # split_dataset 
    #-------------------
    
    def split_dataset(self,
                      sample_ids_or_df=None, 
                      train_percent=0.8,
                      val_percent=0.1,
                      test_percent=0.1,
                      save_to_db=True,
                      random_seed=1845):
        '''
        Splits dataset into train, validation, and 
        test sets at the given proportions. One of the
        proportions may be set to None. In that case
        only two splits will be created. Randomly permutes
        samples before splitting
        
        The sample_ids_or_df may be a list of of
        indices into the sqlite db of sample rows from
        the original CSV file, or a dataframe in which 
        each row corresponds to a sample row from the 
        original CSV. If None, uses what this instance
        already knows. If in doubt, let it default.
        
        Creates a deque (a queue) for each split, and
        saves copies of each in a dict (saved_queues).
        Returns a triplet with the queues. 
        
        @param sample_ids_or_df: list of sqlite sample_id, or dataframe
        @type sample_ids_or_df: {list|pandas.dataframe}
        @param train_percent: percentage of samples for training
        @type train_percent: float
        @param val_percent: percentage of samples for validation
        @type val_percent: float
        @param test_percent: percentage of samples for testing
        @type test_percent: float
        @param save_to_db: whether or not to save the indices that
            define each split in the Sqlite db
        @type save_to_db: bool
        @param random_seed: seed for permuting dataset before split
        @type random_seed: int
        '''

        if sample_ids_or_df is None:
            sample_ids_or_df = self.sample_ids
            
        # Deduce third portion, if one of the
        # splits is None:
        if train_percent is None:
            if val_percent is None or test_percent is None:
                raise ValueError("Two of train_percent/val_percent/test_percent must be non-None")
            train_percent = 1-val_percent-test_percent
        elif val_percent is None:
            if train_percent is None or test_percent is None:
                raise ValueError("Two of train_percent/val_percent/test_percent must be non-None")
            val_percent = 1-train_percent-test_percent
        elif test_percent is None:
            if train_percent is None or val_percent is None:
                raise ValueError("Two of train_percent/val_percent/test_percent must be non-None")
            test_percent = 1-train_percent-val_percent
            
        if train_percent+val_percent+test_percent != 1.0:
            raise ValueError("Values for train_percent/val_percent/test_percent must add to 1.0")
            
        np.random.seed(random_seed)
        if type(sample_ids_or_df) == DataFrame:
            sample_indices = list(sample_ids_or_df.index) 
        else:
            sample_indices = sample_ids_or_df
             
        perm = np.random.permutation(sample_indices)
        # Permutations returns a list of arrays:
        #   [[12],[40],...]; turn into simple list of ints:
        num_samples = len(perm)
        
        train_end = int(train_percent * num_samples)
        validate_end = int(val_percent * num_samples) + train_end
        self.train_queue = deque(perm[:train_end])
        self.val_queue = deque(perm[train_end:validate_end])
        self.test_queue = deque(perm[validate_end:])
        
        self.curr_queue = self.train_queue
        
        if save_to_db:
            self.save_queues(self.train_queue, self.val_queue, self.test_queue) 
        
        self.saved_queues = {}
        self.saved_queues['train'] = self.train_queue.copy()
        self.saved_queues['validate'] = self.val_queue.copy()
        self.saved_queues['test'] = self.test_queue.copy()
        
        self.train_frozen_dataset = FrozenDataset(self.log,
                                                  self.db,
                                                  'train',
                                                  self.saved_queues['train'],
                                                  self.label_mapping,
                                                  self.sample_ids
                                                  )
        
        self.validate_frozen_dataset = FrozenDataset(self.log,
                                                     self.db,
                                                     'validate',
                                                     self.saved_queues['validate'],
                                                     self.label_mapping,
                                                     self.sample_ids
                                                     )
        
        self.test_frozen_dataset = FrozenDataset(self.log,
                                                 self.db,
                                                 'test',
                                                 self.saved_queues['test'],
                                                 self.label_mapping,
                                                 self.sample_ids
                                                 )
    #------------------------------------
    # labels_from_db 
    #-------------------
    
    def labels_from_db(self, sample_ids):
        '''
        Given a list of sample ids, return a 
        list of ele yes/no labels:
        
        @param sample_ids: np.array of sample ids whose
            labels are to be retrieved.
        @type sample_ids: np.array
        '''
        
        # Need to turn the sample_ids from
        # ints to strings so that they are
        # usable as a comma-separated list 
        # in the query:
        
        sample_id_list =  ','.join([str(el) for el in list(sample_ids)])
        cmd = f'''SELECT label
                    FROM Samples
                   WHERE sample_id in ({sample_id_list})
                  ORDER BY sample_id;
              '''
        try:
            rows = self.db.execute(cmd)
        except Exception as e:
            raise DatabaseError(f"Could not retrieve labels: {repr(e)}") from e
        labels = [row['label'] for row in rows]
        return labels


    #------------------------------------
    # close
    #-------------------
    
    def close(self):
        try:
            self.db.close()
        except Exception as e:
            raise DatabaseError(f"Could not close sqlite db: {repr(e)}") from e

    #------------------------------------
    # save_queues 
    #-------------------
    
    def save_queues(self, train_queue, val_queue, test_queue):
        '''
        Saving the train, validation, and test queues
        allows post-mortem of a run: the db will contain
        the sequence in which samples were processed.
        
        @param train_queue:
        @type train_queue:
        @param val_queue:
        @type val_queue:
        @param test_queue:
        @type test_queue:
        '''
        
        self.db.execute('DROP TABLE IF EXISTS TrainQueue')
        self.db.execute('CREATE TABLE TrainQueue (sample_id int)')

        self.db.execute('DROP TABLE IF EXISTS ValidateQueue')
        self.db.execute('CREATE TABLE ValidateQueue (sample_id int)')

        self.db.execute('DROP TABLE IF EXISTS TestQueue')
        self.db.execute('CREATE TABLE TestQueue (sample_id int)')
        
        # Turn [2,4,6,...] into tuples: [(2,),(4,),(6,),...]
        train_tuples = [(int(sample_id),) for sample_id in train_queue]
        if len(train_tuples) > 0:
            self.db.executemany("INSERT INTO TrainQueue VALUES(?);", train_tuples)

        val_tuples = [(int(sample_id),) for sample_id in val_queue]
        if len(val_tuples) > 0:
            self.db.executemany("INSERT INTO ValidateQueue VALUES(?);", val_tuples)

        test_tuples = [(int(sample_id),) for sample_id in test_queue]
        if len(test_tuples) > 0:
            self.db.executemany("INSERT INTO TestQueue VALUES(?);", test_tuples)
        
        self.db.commit()

    #------------------------------------
    # save_dict_to_table 
    #-------------------
    
    def save_dict_to_table(self, table_name, the_dict, delete_existing=False):
        '''
        Given a dict, save it to a table in the underlying
        database.
        
        If the table exists, action depends on delete_existing.
        If True, the table is deleted first. Else the dict values
        are added as rows. 
        
        It is the caller's responsibility to ensure that:
        
           - Dict values are db-appropriate data types: int, float, etc.
           - The table name is a legal Sqlite table name  
        
        @param table_name: name of the table
        @type table_name: str
        @param dict: col/value information to store
        @type dict: {str : <any-db-appropriate>}
        '''
        if delete_existing:
            self.db.execute(f'''DROP TABLE IF EXISTS {table_name}''')
            self.db.execute(f'''CREATE TABLE {table_name} ('key_col' varchar(255),
                                                          'val_col' varchar(255));''')
            self.db.commit()

        insert_vals = list(the_dict.items())
        self.db.executemany(f"INSERT INTO {table_name} VALUES(?,?);", insert_vals)
        self.db.commit()

    #------------------------------------
    # yes_no_question 
    #-------------------

    def query_yes_no(self, question, default='yes'):
        '''
        Ask a yes/no question via raw_input() and return their answer.
    
        "question" is a string that is presented to the user.
        "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).
    
        The "answer" return value is True for "yes" or False for "no".
        '''
        valid = {"yes": True, "y": True, "ye": True,
                 "no": False, "n": False}
        if default is None:
            prompt = " [y/n] "
        elif default == "yes":
            prompt = " [Y/n] "
        elif default == "no":
            prompt = " [y/N] "
        else:
            raise ValueError("invalid default answer: '%s'" % default)
    
        while True:
            sys.stdout.write(question + prompt)
            choice = input().lower()
            if default is not None and choice == '':
                return valid[default]
            elif choice in valid:
                return valid[choice]
            else:
                sys.stdout.write("Please respond with 'yes' or 'no' "
                                 "(or 'y' or 'n').\n")        

    #------------------------------------
    # get_db 
    #-------------------
    
    def get_db(self, sqlite_filename):
        '''
        If db exists, open it and return
        the connection instance. Else, 
        create the db, and table Samples 
        
        @param sqlite_filename: file name of Sqlite3 db
        @type sqlite_filename: str
        '''
        
        self.db = sqlite3.connect(sqlite_filename)
        self.db.row_factory = sqlite3.Row
        
        tbl_list = self.db.execute(f'''
            SELECT name
              FROM sqlite_master
             WHERE type='table' AND name='Samples';
             ''').fetchall()
        samples_tbl_exists = len(tbl_list) > 0
        
        if samples_tbl_exists:
            return self.db
        
        self.db.execute('''DROP TABLE IF EXISTS Samples;''')
        self.db.execute('''CREATE TABLE Samples(
                    sample_id INTEGER PRIMARY KEY,
                    recording_site varchar(100),
                    label tinyint,
                    start_time_tick int,
                    end_time_tick int,
                    start_time float,
                    end_time float,
                    parent_low_freqs_energy float,
                    parent_med_freqs_energy float,
                    parent_high_freqs_energy float,
                    snippet_low_freqs_energy float,
                    snippet_med_freqs_energy float,
                    snippet_high_freqs_energy float,
                    snippet_filename varchar(1000)
                    )
                    ''')
        self.db.commit()
        return self.db


