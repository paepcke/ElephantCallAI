'''
Created on Jul 14, 2020

@author: paepcke
'''
import os, sys
import shutil
import sqlite3
import unittest
import glob

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from spectrogram_dataset import SpectrogramDataset, AudioType

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
        
        os.remove('test_spectroA.txt')
        os.remove('test_spectroB.txt')
        os.remove('SamplesTests.sqlite')


    #------------------------------------
    # setUp 
    #-------------------

    def setUp(self):

        # Instance of a SpectrogramDataset that
        # does nothing in its init method
        self.spectr_dataset = SpectrogramDataset(testing=True)
        
        self.db = self.spectr_dataset.get_db(self.test_db_path)
        # Empty the test db Samples table:
        self.db.execute("DELETE FROM Samples;")
        self.db.commit()
        
    #------------------------------------
    # tearDown 
    #-------------------


    def tearDown(self):
        self.db.close()

    #------------------------------------
    # testLabelForSnippet 
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def testLabelForSnippet(self):
        
        # No overlap:
        snippet_interval = pd.Interval(left=50.1, right=55.5)
        is_ele_call = self.spectr_dataset.label_for_snippet(snippet_interval,
                                                       self.label_file,
                                                       required_overlap_percentage=None
                                                       )
        self.assertFalse(is_ele_call)
        
        # Partial overlap:
        snippet_interval = pd.Interval(left=60.3, right=70)
        is_ele_call = self.spectr_dataset.label_for_snippet(snippet_interval,
                                                       self.label_file,
                                                       required_overlap_percentage=None
                                                       )
        self.assertTrue(is_ele_call)

        # Full overlap, i.e. containment:
        snippet_interval = pd.Interval(left=93, right=94)
        is_ele_call = self.spectr_dataset.label_for_snippet(snippet_interval,
                                                       self.label_file,
                                                       required_overlap_percentage=None
                                                       )
        self.assertTrue(is_ele_call)
        
        # Percentage requirement percent as fraction 0-1:
        snippet_interval = pd.Interval(left=93, right=94)
        required = 0.5 # %
        is_ele_call = self.spectr_dataset.label_for_snippet(
            snippet_interval,
            self.label_file,
            required_overlap_percentage=required
            )
        
        if is_ele_call:
            self.assertTrue(1/(92.793 - 95.41599999999232) >= required)
        else:
            self.assertTrue(1/(92.793 - 95.41599999999232) < 0.5)
        
        
        # Percentage requirement zero percent:
        snippet_interval = pd.Interval(left=93, right=94)
        required = 0
        is_ele_call = self.spectr_dataset.label_for_snippet(
            snippet_interval,
            self.label_file,
            required_overlap_percentage=required
            )
        if is_ele_call:
            self.assertTrue(1/(95.41599999999232 - 92.793) >= required)
        else:
            self.assertTrue(1/(95.41599999999232 - 92.793) < required)

    #------------------------------------
    # testDecodeFilename 
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def testDecodeFilename(self):
        
        res_dict = self.spectr_dataset.decode_filename('/foo/bar.txt')
        expected = {'file_type': AudioType.LABEL,
                    'file_root': 'bar',
                    'path': '/foo',
                    'wav': 'bar.wav',
                    'label': 'bar.txt',
                    'spectro': 'bar_spectrogram.pickle'}
        self.assertDictContainsSubset(expected, res_dict)
        
        res_dict = self.spectr_dataset.decode_filename('bar.txt')
        expected = {'file_type': AudioType.LABEL,
                    'file_root': 'bar',
                    'path': None,
                    'wav': 'bar.wav',
                    'label': 'bar.txt',
                    'spectro': 'bar_spectrogram.pickle',
                    'snippet_id': None
                    }
        self.assertDictContainsSubset(expected, res_dict)

        res_dict = self.spectr_dataset.decode_filename('bar_spectrogram.pickle')
        expected = {'file_type': AudioType.SPECTRO,
                    'file_root': 'bar',
                    'path': None,
                    'wav': 'bar.wav',
                    'label': 'bar.txt',
                    'spectro': 'bar_spectrogram.pickle',
                    'snippet_id': None
                    }

        self.assertDictContainsSubset(expected, res_dict)
   
        res_dict = self.spectr_dataset.decode_filename('/foo/fum/bar.wav')
        expected = {'file_type': AudioType.WAV,
                    'file_root': 'bar',
                    'path': '/foo/fum',
                    'wav': 'bar.wav',
                    'label': 'bar.txt',
                    'spectro': 'bar_spectrogram.pickle',
                    'snippet_id': None
                    }

        self.assertDictContainsSubset(expected, res_dict)
        
        res_dict = self.spectr_dataset.decode_filename('bar_2_spectrogram.pickle')
        expected = {'file_type': AudioType.SNIPPET,
                    'file_root': 'bar',
                    'path': None,
                    'wav': 'bar.wav',
                    'label': 'bar.txt',
                    'spectro': 'bar_spectrogram.pickle',
                    'snippet_id': 2
                    }
        
        self.assertDictContainsSubset(expected, res_dict)
        
    #------------------------------------
    # testChopOneSpectrogram 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def testChopOneSpectrogram(self):
        
        spectrogram = pd.DataFrame([[  1,  2,  3,  4,  5,  6,  7],
                                    [ 10, 20, 30, 40, 50, 60, 70],
                                    [100,200,300,400,500,600,700],
                                    ], dtype=int,
                                    columns=[0,2,4,6,8,10,12],
                                    index=[1,2,3]
                                    )
        curr_file_family = self.spectr_dataset.decode_filename(self.label_file)
        self.spectr_dataset.chop_one_spectrogram(spectrogram,
                                                 self.label_file,
                                                 self.snippet_outdir,
                                                 curr_file_family
                                                 )
        # Times 0,4 sec
        #   Matches label 1.293    6.464999999991385
        # ==> Label: 1
        
        # Should have 2 rows in Sample table:
        rows = self.db.execute("SELECT * FROM Samples;").fetchall()
        self.assertEqual(len(rows), 2)
        # First snippet:
        row = rows[0]
        self.assertEqual(row['sample_id'], 0)
        self.assertEqual(row['label'], 1)
        self.assertEqual(row['start_time_tick'], 0)
        self.assertEqual(row['end_time_tick'], 2)
        self.assertEqual(row['start_time'], 0)
        self.assertEqual(row['end_time'], 4)
        self.assertTrue(row['snippet_filename'].endswith('labels_for_testing_0_spectrogram.pickle'))

        # Second snippet:
        row = rows[1]
        self.assertEqual(row['sample_id'], 1)
        self.assertEqual(row['label'], 1)
        self.assertEqual(row['start_time_tick'], 2)
        self.assertEqual(row['end_time_tick'], 4)
        self.assertEqual(row['start_time'], 4)
        self.assertEqual(row['end_time'], 8)
        self.assertTrue(row['snippet_filename'].endswith('labels_for_testing_1_spectrogram.pickle'))
        
        # Times 4,8
        #   Matches label 1.293    6.464999999991385
        # ==> Label: 1
        #***** Test for error when calling again
        
    #------------------------------------
    # testChopMultipleSpectrograms
    #-------------------
     
    def testChopMultipleSpectrograms(self):
        spectro1 = pd.DataFrame([[  1,  2,  3,  4,  5,  6,  7],
                                 [ 10, 20, 30, 40, 50, 60, 70],
                                 [100,200,300,400,500,600,700],
                                 ], dtype=float,
                                 columns=[60,70,80,90,100,110,120],
                                 index=[0,1,2]
                                 )
                                  
        spectro2 = pd.DataFrame([[1000,2000,3000,4000,5000,6000,7000],
                                 [2000,3000,4000,5000,6000,7000,8000],
                                 [3000,4000,5000,6000,7000,8000,9000],
                                 ], dtype=float,
                                 columns=[79,80,81,82,83,84,85],
                                 index=[0,1,2]
                                 )
    
        curr_dir = os.path.dirname(__file__)
        spectroA_file = os.path.join(curr_dir, 
                                     'test_spectroA_spectrogram.pickle')
        spectroB_file = os.path.join(curr_dir,
                                     'test_spectroB_spectrogram.pickle')
        spectro1.to_pickle(spectroA_file)
        spectro2.to_pickle(spectroB_file)

        # Use the same label file for both spectrograms:
        spectroA_label_file = os.path.join(curr_dir, 'test_spectroA.txt')
        spectroB_label_file = os.path.join(curr_dir, 'test_spectroB.txt')
                
        shutil.copyfile(self.label_file, spectroA_label_file)
        shutil.copyfile(self.label_file, spectroB_label_file)

        spectro_labels_dict = {spectroA_file : spectroA_label_file,
                               spectroB_file : spectroB_label_file
                               }
    
        self.spectr_dataset.chop_spectograms(spectro_labels_dict)
        print('foo')


# ----------------- Main --------------


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()