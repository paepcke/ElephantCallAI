'''
Created on Aug 2, 2020

@author: paepcke
'''
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
import unittest


sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dsp_utils import DSPUtils

TEST_ALL = True
#TEST_ALL = False

class TestDSPUtils(unittest.TestCase):

    #------------------------------------
    # setUpClass 
    #-------------------
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        
        cls.curr_dir = os.path.dirname(__file__)

    #------------------------------------
    # setUp 
    #-------------------

    def setUp(self):
        dir_obj = tempfile.TemporaryDirectory(prefix='FilenameMapping')
        dir_name = dir_obj.name
        db_name = os.path.join(dir_name, 'test_db.sqlite')
        db = sqlite3.connect(db_name)
        db.row_factory = sqlite3.Row
        
        db.execute(f'''
                    CREATE TABLE Samples (sample_id int,
                                          trash float default 0.0,
                                          snippet_filename varchar(255));
                    ''')
        db.execute('''
                   INSERT INTO Samples (sample_id, snippet_filename)
                           VALUES (1, '/baddir/badsubdir/file1.txt'),
                                  (2, '/baddir/badsubdir/file2.txt'),
                                  (3, '/anotherBad/file3.txt'),
                                  (4, '/anotherBad/file4.txt')
                   ''')
        db.commit()
        self.true_tuples = [(1,'/tmp/file1.txt'),
                            (2,'/tmp/file2.txt'),
                            (3,'/tmp/file3.txt'),
                            (4,'/tmp/file4.txt')
                            ]
        self.dir_name = dir_name
        self.dir_obj  = dir_obj
        self.db_path  = db_name
        self.db = db
        
    #------------------------------------
    # tearDown 
    #-------------------

    def tearDown(self):
        try:
            shutil.rmtree(str(self.tmp_dir_obj))
        except Exception:
            # Best effort:
            pass
        
        try:
            shutil.rmtree(self.dir_obj)
        except Exception:
            pass
            

    #------------------------------------
    # testFind 
    #-------------------

    def testFind(self):
        
        # Create a test directory structure:
        
        # ---TmpDir Below Test Dir
        #   --- level0 
        #          --- no_match0_1.foo
        #          --- match0_1.txt
        #   --- level1
        #   --- level2
        #          --- no_match2_1.foo
        #          --- match2_1.txt
        #   --- level3
        #          --- match3_1.txt
        #          --- match3_2.txt
       
        self.tmp_dir_obj = tempfile.TemporaryDirectory(prefix='find_cmd_tmp', 
                                                       dir=self.curr_dir)
        tmp_dir = self.tmp_dir_obj.name
        os.makedirs(os.path.join(tmp_dir,'level0/level1/level2/level3'))
        lev0 = Path(os.path.join(tmp_dir,'level0'))
        _lev1 = Path(os.path.join(tmp_dir,'level0/level1'))
        lev2 = Path(os.path.join(tmp_dir,'level0/level1/level2'))
        lev3 = Path(os.path.join(tmp_dir,'level0/level1/level2/level3'))
                    
        Path(lev0).joinpath('no_match0_1.foo').touch()
        
        match0_1_path = Path(lev0).joinpath('match0_1.txt') 
        match0_1_path.touch()

        Path(lev2).joinpath('no_match2_1.foo').touch()

        match2_1_path = Path(lev2).joinpath('match2_1.txt')
        match2_1_path.touch()
        
        match3_1_path = Path(lev3).joinpath('match3_1.txt')
        match3_1_path.touch()
        
        match3_2_path = Path(lev3).joinpath('match3_2.txt')
        match3_2_path.touch()
        
        full_path_set = set(DSPUtils.unix_find(lev0, r'.*\.txt'))
        self.assertSetEqual(full_path_set, 
                          set([str(match0_1_path),
                              str(match2_1_path),
                              str(match3_1_path),
                              str(match3_2_path)
                              ]))
        
    #------------------------------------
    # test_map_sample_filenames 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_map_sample_filenames(self):
        
            # Test subtree before files, and lack of 
            # slash at end for dir part:
            
            DSPUtils.map_sample_filenames(self.db,{'/baddir/badsubdir' : '/tmp/'})
            rows = self.db.execute(f'''
                              SELECT sample_id, snippet_filename
                                FROM Samples
                              ORDER BY sample_id;
                              ''')
            tuples = []
            for row in rows:
                tuples.extend([(row['sample_id'], row['snippet_filename'])])
            
            # In this test, the '/anotherBad/' rows were
            # not change, so change ground truth:
            self.true_tuples[2] = (3, '/anotherBad/file3.txt')
            self.true_tuples[3] = (4, '/anotherBad/file4.txt')
            
            for (i, new_tuple) in enumerate(tuples):
                self.assertTupleEqual(new_tuple, self.true_tuples[i])

    #------------------------------------
    # test_one_dir_mapping 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_one_dir_mapping(self):
                
            # Test one dir before filename:
            
            DSPUtils.map_sample_filenames(self.db, {'/anotherBad' : '/tmp'})
            rows = self.db.execute(f'''
                              SELECT sample_id, snippet_filename
                                FROM Samples
                              ORDER BY sample_id;
                              ''')
            tuples = []
            for row in rows:
                tuples.extend([(row['sample_id'], row['snippet_filename'])])
            
            # In this test, the '/baddir/badsubdir' rows were
            # not change, so change ground truth:
            
            self.true_tuples[0] = (1, '/baddir/badsubdir/file1.txt')
            self.true_tuples[1] = (2, '/baddir/badsubdir/file2.txt')

            for (i, new_tuple) in enumerate(tuples):
                self.assertTupleEqual(new_tuple, self.true_tuples[i])

    #------------------------------------
    # test_no_dir_match_mapping 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_no_dir_match_mapping(self):

            # Test no directory matches:
            DSPUtils.map_sample_filenames(self.db, {'foodle' : '/tmp'})
            rows = self.db.execute(f'''
                              SELECT sample_id, snippet_filename
                                FROM Samples
                              ORDER BY sample_id;
                              ''')
            self.true_tuples = [(1, '/baddir/badsubdir/file1.txt'),
                                (2, '/baddir/badsubdir/file2.txt'),
                                (3, '/anotherBad/file3.txt'),
                                (4, '/anotherBad/file4.txt')
                                ]
            tuples = []
            for row in rows:
                tuples.extend([(row['sample_id'], row['snippet_filename'])])
            
            for (i, new_tuple) in enumerate(tuples):
                self.assertTupleEqual(new_tuple, self.true_tuples[i])

    #------------------------------------
    # test_two_changes_mapping 
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_two_changes_mapping(self):
            # Test two sets of changes at once:
            # Empty the table...

            DSPUtils.map_sample_filenames(self.db, 
                                          {'/baddir/badsubdir' : '/tmp/',
                                           '/anotherBad'        : '/tmp'})
            rows = self.db.execute(f'''
                              SELECT sample_id, snippet_filename
                                FROM Samples
                              ORDER BY sample_id;
                              ''')
            tuples = []
            for row in rows:
                tuples.extend([(row['sample_id'], row['snippet_filename'])])

            for (i, new_tuple) in enumerate(tuples):
                self.assertTupleEqual(new_tuple, self.true_tuples[i])

#     #------------------------------------
#     # test_snippet_path_changer
#     #-------------------
#     
#     Having trouble with the subprocess call below. Not working.
#
#     @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
#     def test_snippet_path_changer(self):
#         
#         # Test the command line facility that uses
#         # some of the above facilities to change
#         # snippet file name in sqlite snippet dbs.
#         
#         # Mapping {'/baddir/badsubdir' : '/tmp/',
#         #          '/anotherBad'       : '/tmp'}
#         
#         # Location of the update program:
#         dir_nm_mapper = os.path.join(os.path.dirname(__file__),
#                                      '../update_snippet_locations.py'
#                                      )
#         
#         # Application takes list of old dirs, followed by 
#         # list of new dirs:
# #         completed = subprocess.run(f"{dir_nm_mapper} "
# #                                    "--old_dirs /baddir/badsubdir /anotherBad "
# #                                    "--new_dirs /tmp /tmp "
# #                                    f"{self.db_path}"
# #                                    )
#         cmd = f"{dir_nm_mapper} --old_dirs /baddir/badsubdir /anotherBad --new_dirs /tmp /tmp {self.db_path}"
#         #completed = subprocess.call(cmd, shell=True)
#         completed = subprocess.run([cmd], shell=True)
# #         completed = subprocess.run([f"{dir_nm_mapper}",
# #                                    "--old_dirs",
# #                                    "baddir/badsubdir",
# #                                    "/anotherBad",
# #                                    "--new_dirs",
# #                                    "/tmp",
# #                                    "/tmp",
# #                                    f"{self.db_path}"
# #                                    ],
# #                                     shell=True
# #                                    )
#         #****self.assertEqual(completed.returncode, 0)
#         rows = self.db.execute(f'''
#                           SELECT sample_id, snippet_filename
#                             FROM Samples
#                           ORDER BY sample_id;
#                           ''')
#         tuples = []
#         for row in rows:
#             tuples.extend([(row['sample_id'], row['snippet_filename'])])
# 
#         for (i, new_tuple) in enumerate(tuples):
#             self.assertTupleEqual(new_tuple, self.true_tuples[i])


# --------------- Main -----------

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()