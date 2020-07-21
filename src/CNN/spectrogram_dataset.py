'''
Created on Jul 7, 2020

@author: paepcke
'''

import ast
from collections import deque
import os
import glob
from pathlib import Path
import re
from sqlite3 import OperationalError as DatabaseError
import sqlite3
import sys

import pandas as pd
from pandas.core.frame import DataFrame
from torch.utils.data import Dataset
from DSP.dsp_utils import DSPUtils
from DSP.dsp_utils import AudioType
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__),
                             '..'
                             ))
from elephant_utils.logging_service import LoggingService


TESTING = False
#TESTING = True

# To change spectrogram snippet width time,
# see class variable in SpectrogramDataset.

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

    SNIPPET_WIDTH = 5 # approximate width of spectrogram snippets.

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
        num_samples = num_samples_row['num_samples']
        # Our sample ids go from 0 to n
        self.sample_ids = list(range(num_samples))

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
            file_info = DSPUtils.decode_filename(file)
            # Name without path and extention:
            file_root = file_info['file_root']
            if file_info['file_type'] == AudioType.SPECTRO:
                # 24-hour spectrogram: note corresponding label,
                # file:
                full_spectrograms_queued[file_root] = os.path.join(file_info['path'],
                                                       file_info['label'])
                continue
            if file_info['file_type'] == AudioType.SNIPPET:
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
        for (spectro_file, label_file) in spectro_dict.items():
            # Get dict with file names that are by convention
            # derived from the original .wav file:
            curr_file_family = DSPUtils.decode_filename(spectro_file)
            spect_df = pd.read_pickle(spectro_file)
            
            # Create the name for an Sqlite db file that
            # will hold information about the snippets of
            # this 24-hr spectrogram:
            (spectro_file_root, _pickle_ext) = os.path.splitext(spectro_file)
            db_file = spectro_file_root + '_snippets.sqlite'
            self.db = self.get_db(db_file)
            try:
                # If no outdir for snippets was specified by caller,
                # use the directory of this 24-hr spectrogram:
                if self.snippet_outdir is None:
                    self.chop_one_spectrogram(spect_df, 
                                              label_file,
                                              os.path.dirname(spectro_file), 
                                              curr_file_family)
                else:
                    self.chop_one_spectrogram(spect_df, 
                                              label_file,
                                              self.snippet_outdir,
                                              curr_file_family)
            finally:
                self.db.close()

    #------------------------------------
    # chop_one_spectrogram 
    #-------------------
    
    def chop_one_spectrogram(self, 
                             spect_df, 
                             label_file,
                             snippet_outdir,
                             curr_file_family):
        '''
        Takes one 24-hr spectrogram, and chops it
        into smaller spectrograms of approximately 
        self.SNIPPET_WIDTH seconds. Each snippet
        is written to snippet_outdir, named 
        <24-hr-spectrogram-filename>_<snippet-nbr>.pickle
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
        snippet_file_root = curr_file_family['file_root']
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
            
            # Save the snippet to file:
            snippet_file_name = f"{snippet_dest}_{snippet_id}_spectrogram.pickle"
            snippet.to_pickle(snippet_file_name)
            
            label = self.label_for_snippet(snip_xtick_interval, label_file)
            
            # Fill the snippet id into the file family:
            curr_file_family['snippet_id'] = snippet_id
            
            self.add_snippet_to_db(snippet_file_name, 
                                   label,
                                   snip_xtick_interval,
                                   snip_time_interval,
                                   curr_file_family)

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
                          snippet_file_name,
                          label,
                          snippet_xtick_interval,
                          snippet_time_interval,
                          curr_file_family):

        insertion = f'''
                    INSERT INTO Samples (sample_id,
                                         label,
                                         start_time_tick,
                                         end_time_tick,
                                         start_time,
                                         end_time,
                                         snippet_filename
                                         )
                            VALUES ({curr_file_family['snippet_id']},
                                    {label},
                                    {snippet_xtick_interval.left},
                                    {snippet_xtick_interval.right},
                                    {snippet_time_interval.left},
                                    {snippet_time_interval.right},
                                    '{snippet_file_name}'
                                    );
                    '''
        self.db.execute(insertion)
        self.db.commit()



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

        ith_sample_id = self.saved_queues[self.curr_split_id()][indx]
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
        return len(self.saved_queues[self.curr_split_id()])

    #------------------------------------
    # next_csv_row
    #-------------------
 
    def next_csv_row(self):
        '''
        Returns a dict 'ids', 'label', 'attention_mask'
        '''

        # Still have a row left from a previouse
        # chopping?
        if len(self.queued_samples) > 0:
            return(self.queued_samples.popleft())
        
        # No pending samples from previously
        # found texts longer than sequence_len:
        row = next(self.reader)
        try:
            txt = row[self.text_col_name]
        except KeyError:
            msg = (f"\nCSV file does not have a column named '{self.text_col_name}'\n"
                    "You can invoke bert_train_parallel.py with --text\n"
                    "to specify col name for text, and --label to speciy\n"
                    "name of label column."
                    )
            self.log.err(msg)
            raise ValueError(msg)

        # Tokenize the text of the row (the ad):
        # If the ad is longer than self.SEQUENCE_LEN,
        # then multiple rows are returned.
        # Each returned 'row' is a dict containing
        # just the key self.IDS_COL. Its value is
        # an array of ints: each being an index into
        # the BERT vocab.
        #
        # The ids will already be padded. Get
        #   [{'ids' : [1,2,...]},
        #    {'ids' : [30,64,...]}
        #        ...
        #   ]
        
        # Get list of dicts: {'tokens' : ['[CLS]','foo',...'[SEP]'],
        #                     'ids'    : [2545, 352, ]
        #                    }
        # dicts. Only one if text is <= sequence_len, else 
        # more than one:
        id_dicts = self.text_augmenter.fit_one_row_to_seq_len(txt) 

        # Add label. Same label even if given text was
        # chopped into multiple rows b/c the text exceeded
        # sequence_len:

        try:        
            label = row[self.label_col_name]
        except KeyError:
            msg = f"CSV file does not have col {self.label_col_name}" + '\n' +\
                    "You can invoke bert_train_parallel.py with --label"
            self.log.err(msg)
            raise ValueError(msg)

        try:
            label_encoding = self.label_mapping[label]
        except KeyError:
            # A label in the CSV file that was not
            # anticipated in the caller's label_mapping dict
            self.log.err(f"Unknown label encoding: {label}")
            return
        
        for id_dict in id_dicts:
            id_dict['label'] = label_encoding 

        # Create a mask of 1s for each token followed by 0s for padding
        for ids_dict in id_dicts:
            ids_seq = id_dict[self.IDS_COL_NAME]
            #seq_mask = [float(i>0) for i in seq]
            seq_mask = [int(i>0) for i in ids_seq]
            ids_dict['attention_mask'] = seq_mask

        # We now have a list of dicts, each with three
        # keys: 'ids','label','attention_mask'
        if len(id_dicts) > 1:
            self.queued_samples.extend(id_dicts[1:])
        return id_dicts[0]

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
        return (self.train_queue, self.val_queue, self.test_queue)

    #------------------------------------
    # save_queues 
    #-------------------
    
    def save_queues(self, train_queue, val_queue, test_queue):
        
        self.db.execute('DROP TABLE IF EXISTS TrainQueue')
        self.db.execute('CREATE TABLE TrainQueue (sample_id int)')

        self.db.execute('DROP TABLE IF EXISTS ValidateQueue')
        self.db.execute('CREATE TABLE ValidateQueue (sample_id int)')

        self.db.execute('DROP TABLE IF EXISTS TestQueue')
        self.db.execute('CREATE TABLE TestQueue (sample_id int)')
        
        # Turn [2,4,6,...] into tuples: [(2,),(4,),(6,),...]
        train_tuples = [(int(sample_id),) for sample_id in train_queue]
        self.db.executemany("INSERT INTO TrainQueue VALUES(?);", train_tuples)

        val_tuples = [(int(sample_id),) for sample_id in val_queue]
        self.db.executemany("INSERT INTO ValidateQueue VALUES(?);", val_tuples)

        test_tuples = [(int(sample_id),) for sample_id in test_queue]
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
                    sample_id int primary key,
                    label tinyint,
                    start_time_tick int,
                    end_time_tick int,
                    start_time float,
                    end_time float,
                    snippet_filename varchar(1000)
                    )
                    ''')
        self.db.commit()
        return self.db


