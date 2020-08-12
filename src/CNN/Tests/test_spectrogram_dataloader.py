'''
Created on Jul 14, 2020

@author: paepcke
'''
import os, sys
import unittest
import glob
from pathlib import Path

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from spectrogram_dataset import SpectrogramDataset
from spectrogram_dataloader import SpectrogramDataloader

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
        # Empty the test db Samples table:
        self.db.execute("DELETE FROM Samples;")
        self.db.commit()

        # Instance of a SpectrogramDataset that
        # does nothing in its init method
        self.spectr_dataset = SpectrogramDataset(testing=True,
                                                 test_db=self.db)

        self.files_to_delete = self.prepare_kfolds_test()
    
        rows = self.db.execute("SELECT sample_id FROM Samples;")
        self.sample_ids = [row['sample_id'] for row in rows]
        
        # Make the dataset's __len__() method work in spite
        # of having been created with testing == True:
        self.spectr_dataset.num_samples = len(self.sample_ids)

        # The frequency bands in which energy means
        # are computed are usually betwen 0 and 60,
        # But our test dfs only have three lines with
        # index 1,2,3. So artificially change the 
        # dataset instance's freq ranges:
        self.spectr_dataset.LOW_FREQ_BAND = pd.Interval(left=0, right=1)
        self.spectr_dataset.MED_FREQ_BAND = pd.Interval(left=1, right=2)
        self.spectr_dataset.HIGH_FREQ_BAND = pd.Interval(left=2, right=3)
        
    #------------------------------------
    # tearDown 
    #-------------------


    def tearDown(self):
        self.db.close()
        for file in glob.glob('test_spectro*.sqlite'):
            os.remove(file)

    #------------------------------------
    # testLen 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def testLen(self):
        
        dataloader = SpectrogramDataloader(self.spectr_dataset)
        num_samples = len(dataloader)
        self.assertEqual(num_samples, 21)

    #------------------------------------
    # testGetitem 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def testGetitem(self):
        dataloader = SpectrogramDataloader(self.spectr_dataset)
        spectro_label_dict = dataloader[14]
        sample_id = spectro_label_dict['spectrogram'].iloc[0,0]
        self.assertEqual(sample_id, 14)

    #------------------------------------
    # testGetNSplit 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def testGetNSplit(self):
        try:
            files_to_delete = self.prepare_kfolds_test()
            # Next: do 5 folds, repeated and pull sample-dfs/labels
            # until all folds have been exhausted:

            self.spectr_dataset.kfold_stratified(
                   n_splits=5,
                   n_repeats=2,
                   shuffle=False,
                   random_state=None
                   )
            num_folds = self.spectr_dataset.get_n_splits()
            # Should be 10 folds: 2 repetitions * 5 folds-per-rep
            self.assertEqual(num_folds, 10)
            
        finally:
            for file in files_to_delete:
                os.remove(file)        



# ----------------- Utils -------------

    #------------------------------------
    # prepare_kfolds_test 
    #-------------------
    
    def prepare_kfolds_test(self):
        
        # Create 20 fake spectrogram snippet files:

        curr_dir = os.path.dirname(__file__)
        files_to_delete = []
        for i in range(0,21):
            spectro_filename = Path(curr_dir).joinpath(f'fake_spectro{i}_spectrogram.pickle')
            df = pd.DataFrame({'foo' : i, 'bar' : -i}, index=[0,1])
            df.to_pickle(spectro_filename)
            files_to_delete.append(spectro_filename)
            
        self.create_samples_in_db(files_to_delete)
        
        # Since we set 'testing' to True when creating
        # the spectrogrammer, we need to set its' sample_ids
        # explicitly here:

        self.spectr_dataset.sample_ids = list(range(0,21))
        
        return files_to_delete



    #------------------------------------
    # create_samples_in_db 
    #-------------------

    def create_samples_in_db(self, files_to_delete):
            self.db.execute("DELETE FROM Samples")
            cmd = f'''INSERT INTO Samples (
                 sample_id,
                 recording_site,
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
                 snippet_high_freqs_energy,
                 snippet_filename
                 )
                 VALUES (0, "nouab", 1, 20, 30, 40, 50, 100, 200, 300, 400, 500, 600, "{files_to_delete[0]}"),
                        (1, "nouab", 0, 20, 30, 40, 50, 100, 200, 300, 400, 500, 600, "{files_to_delete[1]}"),
                        (2, "nouab", 1, 20, 30, 40, 50, 100, 200, 300, 400, 500, 600, "{files_to_delete[2]}"),
                        (3, "nouab", 0, 20, 30, 40, 50, 100, 200, 300, 400, 500, 600, "{files_to_delete[3]}"),
                        (4, "nouab", 1, 20, 30, 40, 50, 100, 200, 300, 400, 500, 600, "{files_to_delete[4]}"),
                        (5, "nouab", 0, 20, 30, 40, 50, 100, 200, 300, 400, 500, 600, "{files_to_delete[5]}"),
                        (6, "nouab", 1, 20, 30, 40, 50, 100, 200, 300, 400, 500, 600, "{files_to_delete[6]}"),
                        (7, "nouab", 0, 20, 30, 40, 50, 100, 200, 300, 400, 500, 600, "{files_to_delete[7]}"),
                        (8, "nouab", 1, 20, 30, 40, 50, 100, 200, 300, 400, 500, 600, "{files_to_delete[8]}"),
                        (9, "nouab", 0, 20, 30, 40, 50, 100, 200, 300, 400, 500, 600, "{files_to_delete[9]}"),
                        (10, "nouab", 1, 20, 30, 40, 50, 100, 200, 300, 400, 500, 600, "{files_to_delete[10]}"),
                        (11, "nouab", 0, 20, 30, 40, 50, 100, 200, 300, 400, 500, 600, "{files_to_delete[11]}"),
                        (12, "nouab", 1, 20, 30, 40, 50, 100, 200, 300, 400, 500, 600, "{files_to_delete[12]}"),
                        (13, "nouab", 0, 20, 30, 40, 50, 100, 200, 300, 400, 500, 600, "{files_to_delete[13]}"),
                        (14, "nouab", 1, 20, 30, 40, 50, 100, 200, 300, 400, 500, 600, "{files_to_delete[14]}"),
                        (15, "nouab", 0, 20, 30, 40, 50, 100, 200, 300, 400, 500, 600, "{files_to_delete[15]}"),
                        (16, "nouab", 1, 20, 30, 40, 50, 100, 200, 300, 400, 500, 600, "{files_to_delete[16]}"),
                        (17, "nouab", 0, 20, 30, 40, 50, 100, 200, 300, 400, 500, 600, "{files_to_delete[17]}"),
                        (18, "nouab", 1, 20, 30, 40, 50, 100, 200, 300, 400, 500, 600, "{files_to_delete[18]}"),
                        (19, "nouab", 0, 20, 30, 40, 50, 100, 200, 300, 400, 500, 600, "{files_to_delete[19]}"),
                        (20, "nouab", 1, 20, 30, 40, 50, 100, 200, 300, 400, 500, 600, "{files_to_delete[20]}");
            '''
            self.db.execute(cmd)


# ----------------- Main --------------


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()