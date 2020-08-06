'''
Created on Jul 7, 2020

@author: paepcke
'''

from collections import deque
import glob
import os
from pathlib import Path
from sqlite3 import OperationalError as DatabaseError
import sqlite3
import sys

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

# ------------- SpectrogramDataset ---------

class SpectrogramDataset(Dataset):
    '''
    Torch dataset for iterating over spectrogram snippets,
    each of which is associated with an integer label.

    The dataset behaves like both an iterator and a sequence.
    That is both methods for interacting with the 
    dataset work, and return dicts with spectrogram-label
    items:
    
        for sample_dict in my_dataset:
            print(sample_dict['spectrogram']
            print(sample_dict['label']
            
    and:
        sample_dict_14 = my_dataset[14]
    
    In addition, the class offers several flavors of
    kfold cross validation:
    
        kfold                : k folds
        repeatedKfold        : k folds repeated n times
        stratifiedKfold      : each fold is guaranteed to
                               be balanced across all label
                               classes
        repated stratified   : stratified kfold repeated n times.
        
    See scikit-learn's kfold facility for documentation. It is
    used under the hood. All information other then the spectrograms
    themselves are kept in the Samples table of an Sqlite3 database.
    The following documentation refers to this db as the "Samples db"
    
    The instantiation of this class accomplishes several tasks,
    as needed. The tasks are derived from the arguments to _init__():
    
    1. If only the location of an already existing Samples db is provided
       its content are used, and no additional work is required.
    2. If no already existing Samples db is specified, but a mixed list
       of directories and files is provided, then the list is assumed
       to contain paths to 24-hr spectrogram files and Raven label files.
       All spectrograms are chopped into SNIPPET_WIDTH second snippets.
       Those snippets are the dataset samples. Using the label file
       that corresponds to each 24-hr spectrogram, an Sqlite db is created.
    3. Both, a Samples db and additional 24-hr spectrograms and their
       label files are specified. In that case the additional spectrograms
       are chopped and added to the Samples db.
    
    Assumptions:
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
               the are 24 hr spectrograms. Else, if they
               are spectrogram snippets choped from a 24-hr
               spectrogram, they end with _spectrogram_<n>.pickle 
        o The spectrograms are created by Spectrogrammer 
          (see spectrogrammer.py), and are Pandas dataframes.
          The indexes (row labels) are frequencies. The 
          column names are absolute times in seconds into
          the 24hr recording of which the spectrogram is
          a part.
          
    We create an Sqlite database with metadata about each 
    spectrogram. One spectrogram is one sample. Schema:
    
        o Table Samples:
        
			CREATE TABLE Samples(
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
			
        
    An additional feature is the option for integrated
    train/validation splits. Calling split_dataset()
    internally produces input queues that feed three 
    iterators. Callers switch between these iterators via
    the switch_to_split() method.
    '''

    SNIPPET_WIDTH  = 30 # approximate width of spectrogram snippets (seconds).
    LOW_FREQ_BAND  = pd.Interval(left=0, right=21)
    MED_FREQ_BAND  = pd.Interval(left=21, right=41)
    HIGH_FREQ_BAND = pd.Interval(left=41, right=51)

    #------------------------------------
    # Constructor 
    #-------------------

    def __init__(self,
                 dirs_or_spect_files=None,
                 sqlite_db_path=None,
                 recurse=False,
                 chop=False,
                 snippet_outdir=None,
                 testing=False,
                 test_db=None,
                 debugging=False
                 ):
        '''
        
        @param dirs_or_spect_files: list of files and/or directories
            where spectrograms reside. These may be 24-hr spectrograms,
            or snippets.
        @type dirs_or_spect_files: {str|[str]}
        @param sqlite_db_path: fully qualified path to the sqlite
            db that holds info of already existing snippets. If None,
            such a db will be created.
        @type sqlite_db_path: str
        @param recurse: whether or not to search for spectrograms
            in subtrees of dirs_or_spect_files
        @type recurse: bool
        @param chop: whether or not to perform any chopping of
            24-hr spectrograms. If all spectrograms in dirs_or_spect_files
            and their subtrees are snippets, set this value to False.
        @type chop: bool
        @param snippet_outdir: if chopping is requested: where to place
            the resulting snippets
        @type snippet_outdir: str
        @param testing: whether caller is a unittest
        @type testing: bool
        @param test_db: in case of testing, a db created by
            the unittest.
        @type test_db: sqlite3.Connection
        @param debugging: set to True if debugging where not all
            sample snippets are available, such as a purely dev machine.
            Will cause only samples from the first dirs_or_spect_files
            to be included.
        @type debugging: bool
        '''

        self.snippet_outdir = snippet_outdir
        
        if type(dirs_or_spect_files) != list:
            dirs_or_spect_files = [dirs_or_spect_files]

        # Allow unittests to create an instance and
        # then call methods selectively:
        if testing:
            if test_db is None:
                raise ValueError("If testing, must provide an Sqlite db instance")
            self.db = test_db
            self.testing = testing
            
            # Indicators that a new fold was just
            # loaded as part of a __next__() call:
            self.new_train_fold_len = None
            self.new_validate_fold_len = None
            
            
        if not testing:
            if dirs_or_spect_files is None and sqlite_db_path is None:
                raise ValueError("Directories and sqlite_db_path args must not both be None")
            
            self.log = LoggingService()
            
            if sqlite_db_path is None:
                sqlite_db_path = os.path.join(os.path.dirname(__file__),
                                              'spectrograms.sqlite'
                                              )

            self.db = SpectrogramDataset.get_db(sqlite_db_path)
            
            if chop:
                # Get already processed dirs. The 'list()' pulls all hits from
                # the db at once (like any other iterator)
                try:
                    processed_dirs = list(self.db.execute('''
                                                SELECT dir_or_file_name FROM DirsAndFiles;
                                                          '''))
                except DatabaseError as e:
                    raise DatabaseError(f"Could not check for already processed work: {repr(e)}") from e
                    
                if dirs_or_spect_files is not None:
                    # Process those of the given dirs_or_spect_files that 
                    # are not already in the db:
                    dirs_or_files_to_do = set(dirs_or_spect_files) - set(processed_dirs)
                else:
                    dirs_or_files_to_do = set()
        
                if len(dirs_or_files_to_do) > 0:
                    # Chop spectrograms:
                    self.process_spectrograms(dirs_or_files_to_do, recurse=recurse)
    
        if debugging:
            # Only keep the samples whose corresponding 
                # pickle files we have in our debug env:
            sample_id_rows = self.db.execute(f'''SELECT sample_id
                                                  FROM Samples
                                                  WHERE snippet_filename LIKE '{dirs_or_spect_files[0]}%'
                                                  ''')
        else:
            sample_id_rows = self.db.execute('''SELECT sample_id
                                                  FROM Samples''')
            
        self.sample_ids  = [row['sample_id'] for row in sample_id_rows]
        self.num_samples = len(self.sample_ids)
        
        # So far, folds*() was not called, so using
        # the entire dataset:
        self.num_folds = 0

        # The following only needed in case the 
        # class is used without either the split_dataset()
        # or the kfold*() facilities:
        
        # Make a preliminary train queue with all the
        # sample ids. If split_dataset() is called later,
        # this queue will be replaced:
        self.train_queue = deque(self.sample_ids)
        self.curr_queue  = self.train_queue

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
            raise ValueError(f"Dataset ID must be 'train' or 'validate'; was {split_id}")

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
                # Looking at a 24hr spectrogram. Map that
                # to its corresponding label file:
                label_file = file_family.fullpath(AudioType.LABEL)
                if not os.path.exists(label_file):
                    self.log.warn(f"No label file for spectrogram {file}; skipping it.")
                    continue
                full_spectrograms_queued[file] = label_file
                continue
            if file_family.file_type == AudioType.SNIPPET:
                # No need to process the corresponding full 
                # spectrogram, if we encounter ever encounter
                # it; we assume that full spectro is taken
                # care of:
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
            mapping full paths to spectrogram to their corresponding
            label files.
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
            raise ChopError(f"Chop err; spectro_file: {spectro_file}, label_file: {label_file}: {repr(e)}") from e

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
        
        # It is possible for some frequency ranges
        # to not be represented in the spectrogram.
        # For those cases we set the corresponding
        # mean to zero:
        
        low_freq_slice = spectro[(spectro.index.values >= self.LOW_FREQ_BAND.left) & 
                                 (spectro.index.values < self.LOW_FREQ_BAND.right)]
        if low_freq_slice.empty:
            low_freq_mean = 0
        else:
            low_freq_mean  = low_freq_slice.mean().mean()
        
        med_freq_slice = spectro[(spectro.index.values >= self.MED_FREQ_BAND.left) & 
                                 (spectro.index.values < self.MED_FREQ_BAND.right)]
        if med_freq_slice.empty:
            med_freq_mean = 0
        else:
            med_freq_mean  = med_freq_slice.mean().mean()
        
        high_freq_slice = spectro[(spectro.index.values >= self.HIGH_FREQ_BAND.left) & 
                                 (spectro.index.values < self.HIGH_FREQ_BAND.right)]
        if high_freq_slice.empty:
            high_freq_mean = 0
        else:
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
        # But: if the time span is smaller than a SNIPPET_WIDTH,
        # we do want that bit:
        num_snippets = max(num_snippets, 1)
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
            
            # Save the snippet to file. Latest protocol
            # is 5, but pd.read_pickle() only works up 
            # to 4 (at least with version <= 1.1.0):
            snippet.to_pickle(snippet_file_name, protocol=4)

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
        '''
        Return the next sample spectrogram and label from the 
        current queue. That current queue is either
        the train set or the validation set. This
        quantity is controlled via the switch_to_split()
        method.
        
        The method serves the iterator protocol.
        
        Support for unittests: when a new fold is loaded
        as part of serving this call, the variables 
        self.new_train_fold_len and self.new_validate_fold_len 
        are set to the length of those respective folds.
        each following call to this method resets those
        values to None.
        
        @return: dictionary {'spectrogram' : <snipped_df>, 
                             'label' : <label
                             }
        @rtype: {str : pd.DataFrame, str : int}
        @raise StopIteration
        
        '''
        # To help unittests: Reset length of current folds
        # to None. This way unittests can check after
        # each call to __next__() whether a new fold
        # was loaded during the call (see setting of these
        # vars in the first IndexError catch below.
        
        self.new_train_fold_len = None
        self.new_validate_fold_len = None
        
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

            # To help unittests: indicate the 
            # lengths of the new folds:
            self.new_train_fold_len = len(train_sample_ids)
            self.new_validate_fold_len = len(validate_sample_ids)

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
        try:
            snippet_fname = row['snippet_filename']
            snippet_df = DSPUtils.load_spectrogram(snippet_fname)
        except Exception as e:
            raise IOError(f"Attempt to load spectro snippet {row['snippet_filename']}: {repr(e)}") from e
        return {'spectrogram' : snippet_df, 'label' : row['label']}

    #------------------------------------
    # __getitem__ 
    #-------------------

    def __getitem__(self, indx):
        '''
        Return indx'th row from the db.
        Due to how the db is structured, 
        indx is the same as the sample_id
        
        @param indx: row to retrieve
        @type indx: int
        @return: dict {'spectrogram' : ..., 'label': ...}, which
            are one spectrogram and it's label.
        @rtype: {pd.dataframe, int}
        '''

        res = self.db.execute(f'''
                               SELECT sample_id, label, snippet_filename
                                FROM Samples 
                               WHERE sample_id = {indx}
                             ''')
        row = next(res)
        snippet_df = DSPUtils.load_spectrogram(row['snippet_filename'])
        return {'spectrogram' : snippet_df, 'label' : row['label']}

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
        Return length of the entire dataset.
        This is different from the number of samples
        in the current queue (test or validate).
        Providing the current queue's length is not
        feasible, because the folds are (a) of slightly
        different sizes, and (b) are created on the
        fly in sklearn's iterator.
        
        @return number of samples in entire dataset
        @rtype int
        '''
        # Consider caching number of rows in 
        # the Samples table. But how to reliably
        # invalidate the cache?
        
        rows = self.db.execute('''SELECT COUNT(*) as num_samples FROM Samples;''')
        return next(rows)['num_samples']

    #------------------------------------
    # kfold 
    #-------------------

    def kfold(self, 
               n_splits=5,
               n_repeats=0,
               shuffle=False,
               random_state=None,
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
            

        
        # The split method will return a generator
        # object. Each item in this generator is
        # a 2-tuple: a test set array and a validation
        # set array. There will be n_splits such tuples.
        
        # We grab the first pair:
        
        all_labels = self.labels_from_db(self.sample_ids)
        self.folds_iter = self.cross_validator.split(self.sample_ids, all_labels) 
        
        # The following retrieves *indices* into 
        # our list of sample_ids.
        (self.train_sample_idxs, self.validate_sample_idxs) = \
            next(self.folds_iter)
        
        # Get actual sample_ids from the indices
        # into the sample ids:
        self.train_sample_ids = [self.sample_ids[idx] for idx in self.train_sample_idxs]
        self.validate_sample_ids = [self.sample_ids[idx] for idx in self.validate_sample_idxs]
        
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
    # get_db 
    #-------------------

    @classmethod    
    def get_db(cls, sqlite_filename):
        '''
        If db exists, open it and return
        the connection instance. Else, 
        create the db, and table Samples.
        
        This is a classmethod, so that unittests
        can obtain a properly initialized db instance
        without needing to instantiate a SpectrogramDataset. 
        
        @param sqlite_filename: file name of Sqlite3 db
        @type sqlite_filename: str
        '''
        
        db = sqlite3.connect(sqlite_filename)
        db.row_factory = sqlite3.Row
        
        tbl_list = db.execute(f'''
            SELECT name
              FROM sqlite_master
             WHERE type='table' AND name='Samples';
             ''').fetchall()
        samples_tbl_exists = len(tbl_list) > 0
        
        if not samples_tbl_exists:
            # Create the samples table:
            # By test above, tbl doesn't exist,
            # but the DROP makes me feel good:
            
            db.execute('''DROP TABLE IF EXISTS Samples;''')
            db.execute('''CREATE TABLE Samples(
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
        # Same for DirsAndFiles table where processed files
        # and directories are listed so that partial work
        # can be picked up after failure:
        tbl_list = db.execute(f'''
            SELECT name
              FROM sqlite_master
             WHERE type='table' AND name='DirsAndFiles';
             ''').fetchall()
        dirs_and_files_tbl_exists = len(tbl_list) > 0
        if not dirs_and_files_tbl_exists:
            # Create the samples table:
            db.execute('''CREATE TABLE DirsAndFiles(
                        dir_or_file_name varchar(1000)
                        );
                        ''')
        db.commit()
        return db
