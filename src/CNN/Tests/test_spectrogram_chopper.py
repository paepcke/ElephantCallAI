'''
Created on Jul 29, 2020

@author: paepcke
'''
import glob
import os, sys
import shutil
import unittest

from CNN.chop_spectrograms import SpectrogramChopper
from CNN.spectrogram_dataset import SpectrogramDataset
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))



TEST_ALL = True
#TEST_ALL = False


class Test(unittest.TestCase):


    #------------------------------------
    # setUpClass
    #-------------------
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Create Sqlite3 db for spectrogram snippet info:
        cls.test_db_path = os.path.join(os.path.dirname(__file__),
                                        'SamplesTests.sqlite')
        # Start with a fresh Samples db/table:
        try:
            os.remove(cls.test_db_path)
        except Exception:
            # Wasn't there:
            pass

        # Path to a small Raven label file:
        cls.label_file = os.path.join(os.path.dirname(__file__),
                                      'labels_for_testing.txt')

        # Path where we sometimes pretend a 24 hour spectrogram resides:
        cls.snippet_outdir = os.path.dirname(__file__)
        
        cls.test_dir = os.path.dirname(__file__)

    #------------------------------------
    # tearDownClass
    #-------------------
    
    @classmethod
    def tearDownClass(cls):
        for file in glob.glob('test_spectro*.pickle'):
            os.remove(file)
            
        for file in glob.glob('test_spectro*.sqlite'):
            os.remove(file)
            
        for file in glob.glob('labels_for_testing_*.pickle'):
            os.remove(file)
        
        try:
            os.remove('test_spectroA.txt')
        except Exception:
            pass
        try:
            os.remove('test_spectroB.txt')
        except Exception:
            pass
        try:
            os.remove('SamplesTests.sqlite')
        except Exception:
            pass

    #------------------------------------
    # setUp 
    #-------------------

    def setUp(self):


        self.db = SpectrogramDataset.get_db(self.test_db_path)
        # Empty the test db Samples DirsAndFiles tables:
        self.db.execute("DELETE FROM Samples;")
        self.db.execute("DELETE FROM DirsAndFiles;")
        self.db.commit()

        self.files_to_delete = []

        #rows = self.db.execute("SELECT sample_id FROM Samples;")
        #self.sample_ids = [row['sample_id'] for row in rows]
        self.sample_ids = []
        self.db.close()
        
    #------------------------------------
    # tearDown 
    #-------------------


    def tearDown(self):
        self.db.close()
        for file in glob.glob('test_spectro*.sqlite'):
            os.remove(file)
        files_to_delete = glob.glob('spectroA*')
        files_to_delete.extend(glob.glob('spectroB*'))
        for file in files_to_delete:
            os.remove(file)


    #------------------------------------
    # testChopOneSpectrogram 
    #-------------------
    
    #*******@unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def testChopOneSpectrogram(self):
        
        spectrogram = pd.DataFrame([[  1,  2,  3,  4,  5,  6,  7],
                                    [ 10, 20, 30, 40, 50, 60, 70],
                                    [100,200,300,400,500,600,700],
                                    ], dtype=int,
                                    columns=[0,2,4,6,8,10,12],
                                    index=[10,30,50]
                                    )
        spectro_file1 = os.path.join(self.test_dir, 'spectroA_spectrogram.pickle')
        spectrogram.to_pickle(spectro_file1)
        self.files_to_delete.append(spectro_file1)
        
        spectroA_label_file = os.path.join(self.test_dir, 'spectroA.txt')
        spectroB_label_file = os.path.join(self.test_dir, 'spectroB.txt')
        
        # Use the same label file for both spectrograms:
        shutil.copyfile(self.label_file, spectroA_label_file)
        shutil.copyfile(self.label_file, spectroB_label_file)

        try:
            chopper = SpectrogramChopper (
                     spectro_file1,
                     sqlite_db_path=self.test_db_path,
                     recurse=False,
                     snippet_outdir=self.test_dir,
                     num_workers=0,
                     this_worker=0
                     )
        except Exception as e:
            print(f"Chopping failed: {repr(e)}")
            return
        
        self.db = chopper.dataset.db
        
        # Should have 2 rows in Sample table:
        rows = self.db.execute("SELECT * FROM Samples;").fetchall()
        self.assertEqual(len(rows), 2)
        # First snippet:
        row = rows[0]
        self.assertEqual(row['sample_id'], 1)
        self.assertEqual(row['label'], 1)
        self.assertEqual(row['start_time_tick'], 0)
        self.assertEqual(row['end_time_tick'], 2)
        self.assertEqual(row['start_time'], 0)
        self.assertEqual(row['end_time'], 4)
        self.assertEqual(row['parent_low_freqs_energy'], 4.0)
        self.assertEqual(row['parent_med_freqs_energy'], 40.0)
        self.assertEqual(row['parent_high_freqs_energy'], 400.0)
        self.assertEqual(row['snippet_low_freqs_energy'], 1.5)
        self.assertEqual(row['snippet_med_freqs_energy'], 15.0)
        self.assertEqual(row['snippet_high_freqs_energy'], 150.0)
        
        self.assertEqual(os.path.basename(row['snippet_filename']),
                         'spectroA_1_spectrogram.pickle')

        # Second snippet:
        row = rows[1]
        self.assertEqual(row['sample_id'], 2)
        self.assertEqual(row['label'], 1)
        self.assertEqual(row['start_time_tick'], 2)
        self.assertEqual(row['end_time_tick'], 4)
        self.assertEqual(row['start_time'], 4)
        self.assertEqual(row['end_time'], 8)
        self.assertEqual(row['parent_low_freqs_energy'], 4.0)
        self.assertEqual(row['parent_med_freqs_energy'], 40.0)
        self.assertEqual(row['parent_high_freqs_energy'], 400.0)
        self.assertEqual(row['snippet_low_freqs_energy'], 3.5)
        self.assertEqual(row['snippet_med_freqs_energy'], 35.0)
        self.assertEqual(row['snippet_high_freqs_energy'], 350.0)

        self.assertEqual(os.path.basename(row['snippet_filename']),
                         'spectroA_2_spectrogram.pickle')


# ----------------- Main --------------

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()