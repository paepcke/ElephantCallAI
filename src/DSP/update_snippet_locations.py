#!/usr/bin/env python
'''
Created on Aug 5, 2020

@author: paepcke
'''
import os
import sqlite3
from sqlite3 import OperationalError as DatabaseError
import sys
import re

import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dsp_utils import DSPUtils

class SnippetPathChanger(object):
    '''
    Used with sqlite snippet databases. Those
    hold one record for each spectrogram snippet,
    including their file location. If the snippets
    are moved or copied, those names are out of date.
    This utility updates all the file names in the
    snippets db.
    
    You may wish to copy the affected sqlite db file
    first for the ability to recover the original names.
    
    Two methods of use: provide a list of existing 
    directories of the snippet_filename values, and
    an equal-length list of corresponding values.
    
    Or: if snippet subdirectories have a common name,
    and are distinguished only be a trailing number,
    can provide the directory names with the smallest
    and largest number, and the single destination root
    for the corresponding subdirs at the destination
    
    Use --listdirs for a list of directories of 
    snippet files currently in the snippet_filename
    column
    
    Examples invocation:
	   ../../src/DSP/update_snippet_locations.py \
	   --min_dir Spectrograms/Training/Threshold_-30_MinFreq_20_MaxFreq_40_FreqCap_30_snippets_0 \
	   --max_dir Spectrograms/Training/Threshold_-30_MinFreq_20_MaxFreq_40_FreqCap_30_snippets_19 \
	   --minmax_dst /Users/paepcke/EclipseWorkspacesNew/ElephantCallAI/Spectrograms/Training \
	   -- \
	   /Users/paepcke/EclipseWorkspacesNew/ElephantCallAI/Spectrograms/Training/joint_chop_info.sqlite
	       '''

    def __init__(self, sqlite_db_path):
        '''
        Constructor
        '''
        self.db = sqlite3.connect(sqlite_db_path)
        self.db.row_factory = sqlite3.Row

    #------------------------------------
    # map_dir_by_dir
    #-------------------
    
    def map_dir_by_dir(self, old_dirs, new_dirs):
        '''
        Change directories in destination db 
        by specifying explicit pairs of subdirectories
        to convert:
        
        @param old_dirs: list of directories in the db's 
            snippet_filename column
        @type old_dirs: [str]
        @param new_dirs: list of corresponding directories 
            to which to change column volumns 
        @type new_dirs: [str]
        '''

        dir_name_map = {}
        for (old_dir, new_dir) in zip(old_dirs, new_dirs):
            dir_name_map[old_dir] = new_dir

        DSPUtils.map_sample_filenames(self.db, dir_name_map)
        
    #------------------------------------
    # map_by_dir_num 
    #-------------------
    
    def map_by_dir_num(self, 
                       min_dir, 
                       max_dir,
                       minmax_dst):
        '''
        In simple cases snippets are distributed across
        subdirectories that only differ by an index number
        at the end:
        
          Spectrograms/Training/my_snippets_0/<bunch of snippet.pickle>
          Spectrograms/Training/my_snippets_1/<bunch of snippet.pickle>
                ...
          Spectrograms/Training/my_snippets_n/<bunch of snippet.pickle>
          
        Given the min and max subdir, and a destination parent,
        modify the db.
        
        Example:  given Spectrograms/Training/my_snippets_0
                        Spectrograms/Training/my_snippets_8
                        /Users/me/all_my_snippets
                        
                  will convert all snippet_filename values in the
                  db that match Spectrograms/Training/my_snippets_[0-9]*$
                  to
                       /Users/me/all_my_snippets/my_snippets_<number>
                       
        Note: provide all chars for min_dir up to
              the last underscore. I.e. before the digits. 
              Else the query used may pull the wrong file names.
              That's b/c Sqlite does not support regexp.

        '''
        # Regex: all file path chars excluding trailing digits
        # form one result group. The digits form another:
        dir_and_num_pat = re.compile(r'(.*_)([0-9]*)$')
        
        min_dir_match = dir_and_num_pat.search(min_dir)
        max_dir_match = dir_and_num_pat.search(max_dir)
        if min_dir_match is None or max_dir_match is None:
            print("Both min_dir and max_dir must have form '/foo/bar_<n>'; note end of '_<n>'")
            return
        (min_dir_stump, min_dir_num) = min_dir_match.groups()
        (max_dir_stump, max_dir_num) = max_dir_match.groups()
        
        # Work with ints, not the stringified ints:
        min_dir_num = int(min_dir_num)
        max_dir_num = int(max_dir_num)

        cmd = f'''
                SELECT snippet_filename
                  FROM Samples
                 WHERE snippet_filename like '{min_dir_stump}%'
                '''
        try:
            rows = self.db.execute(cmd)
        except Exception as e:
            raise DatabaseError(f"Cannot retrieve snippet filenames: {repr(e)}") from e
        
        dir_map = {}
        # Go through each snippet info row in the sqlite db.
        # Check whether the directory of the snippet_filename
        # 
        for row in rows:
            
            whole_path = row['snippet_filename']
            
            dir_name = os.path.dirname(whole_path)
            # Chop off the number at the end:
            match = dir_and_num_pat.search(dir_name)
            if match is None:
                # Was not a directory of the right form
                continue
            
            # Get the dir's index number:
            src_dir_num   = int(match.groups()[1])
            # If num is not in range of what caller
            # wants to replace:
            if src_dir_num < min_dir_num or src_dir_num > max_dir_num:
                continue
            
            # Get the last part of the src, i.e.
            # from /old/dir/path_18/snippet.pickle
            # get path_18:
            src_direct_parent = os.path.basename(max_dir_stump)
            
            # Already found one of these?
            try:
                dir_map[dir_name]
            except KeyError:
                # Nope, not yet:
                # Create: /new/dir/old_parent_<seqnum>:
                dir_map[dir_name] = os.path.join(minmax_dst, 
                                                 src_direct_parent + str(src_dir_num))
        
        # Now batch replace:
        DSPUtils.map_sample_filenames(self.db, dir_map)
        
    #------------------------------------
    # list_dirs
    #-------------------

    def list_dirs(self):
        cmd = '''SELECT DISTINCT TRIM(
                        SUBSTR(snippet_filename, 
                               1, 
                               INSTR(snippet_filename, recording_site)-1),
                        '\n')
                    AS dir_name
                   FROM Samples;
                '''
        try:
            rows = self.db.execute(cmd)
        except Exception as e:
            print(f"Could not retrieve snippet filenames from db: {repr(e)}")
            sys.exit(1)
            
        for row in rows:
            print(row['dir_name'])

# ------------------------ Main ------------

if __name__ == '__main__':
    
        parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                         formatter_class=argparse.RawTextHelpFormatter,
                                         description="Update spectrogram snippet locations in sqlite snippet file."
                                         )
     
        parser.add_argument('--old_dirs',
                            type=str,
                            nargs='+',
                            default=[],
                            help='Repeatable: old directories in the db file')
        parser.add_argument('--new_dirs',
                            type=str,
                            nargs='+',
                            default=[],
                            help='Repeatable: replacement directories in the db file')
        parser.add_argument('--min_dir',
                            type=str,
                            help='For replace by dir index: smallest-numbered dir that contains snippets')
        parser.add_argument('--max_dir',
                            type=str,
                            help='For replace by dir index: highest-numbered dir that contains snippets')
        parser.add_argument('--minmax_dst',
                            type=str,
                            help='For replace by dir index: destination parent dir for the snippet subdirs')
 
        parser.add_argument('-l', '--listdirs',
                            action='store_true',
                            default=False,
                            help='For a list of unique directories in the snippet db')
 
        parser.add_argument('dbpath',
                            type=str,
                            help='Path to the sqlite snippet db file')
 
        args = parser.parse_args();

        #***************
#         class Args(object):
#             pass
#         args = Args()
#         args.min_dir = 'Spectrograms/Training/Threshold_-30_MinFreq_20_MaxFreq_40_FreqCap_30_snippets_0'
#         args.max_dir = 'Spectrograms/Training/Threshold_-30_MinFreq_20_MaxFreq_40_FreqCap_30_snippets_19'
#         args.minmax_dst = '/Users/paepcke/EclipseWorkspacesNew/ElephantCallAI/Spectrograms/Training'
#         args.dbpath = '/Users/paepcke/EclipseWorkspacesNew/ElephantCallAI/Spectrograms/Training/joint_chop_info.sqlite' 
#         args.old_dirs = []
#         args.new_dirs = []
#         args.listdirs = None
        #***************

        old_dirs    = args.old_dirs
        new_dirs    = args.new_dirs
           
        min_dir     = args.min_dir
        max_dir     = args.max_dir
        
        minmax_dst = args.minmax_dst
        
        listdirs    = args.listdirs
        
        dbpath      = args.dbpath
        
        # Enforce exclusivity of pairwise vs. map-by-num vs. listdirs:
        try:
            if listdirs:
                assert(len(old_dirs) == len(new_dirs) == 0)
                assert(min_dir is None and max_dir is None and minmax_dst is None)
        except AssertionError:
            print("Cannot use listdirs with any other options")
            sys.exit(1)
                

        try:
            if len(old_dirs) > 0 or len(new_dirs) > 0:
                assert(len(old_dirs) == len(new_dirs))
                assert(min_dir is None and max_dir is None and minmax_dst is None)
                assert(not listdirs)
        except AssertionError:
            print("Using old-dirs/new-dirs requires the two to have equal length;\n"
                  "and no other option is allowed")
            sys.exit(1)

        try:
            if min_dir is not None or max_dir is not None or minmax_dst is not None:
                # If one is non-None, all must be:
                assert(not [val for val in (min_dir,max_dir,min_dir) if val is None])
                assert(len(old_dirs) == len(new_dirs) == 0)
                assert(not listdirs)
        except AssertionError:
            print("If using min_dir/max_dir/minmax_dst, all other options are not valid")
            sys.exit(1)

        if not os.path.exists(dbpath):
            print(f"Cannot find db path: {dbpath}")
            sys.exit(1)

        updater = SnippetPathChanger(dbpath)
        if listdirs:
            updater.list_dirs()
            sys.exit(0)
        if len(old_dirs) > 0:
            updater.map_dir_by_dir(old_dirs, new_dirs)
        else:
            updater.map_by_dir_num(min_dir, max_dir, minmax_dst)
            sys.exit(0)