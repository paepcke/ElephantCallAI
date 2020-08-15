'''
Created on Aug 15, 2020

@author: paepcke
'''
import os
import unittest

from torch import tensor

from CNN.train import SpectrogramTrainer


#*********TEST_ALL = True
TEST_ALL = False

class TestCNNTrain(unittest.TestCase):

    snippet_dir = os.path.join(os.path.dirname(__file__), 'TestSnippets')
    snippet_db_path = os.path.join(os.path.dirname(__file__), 'tiny_chop_info.sqlite')

    #------------------------------------
    # setUp
    #-------------------

    def setUp(self):
        pass

    #------------------------------------
    # tearDown
    #-------------------


    def tearDown(self):
        pass

    #------------------------------------
    # testTrainEpoch 
    #-------------------
    
    #******@unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def testTrainEpoch(self):

        trainer = SpectrogramTrainer(
                        self.snippet_dir,
                        self.snippet_db_path,
                        batch_size = 16)
        
        # Make all runs of equal input return the same results:
        trainer.set_seed(42)
        
        # Start a new k-fold cross_validation run for
        # each epoch:
        trainer.dataloader.kfold_stratified(shuffle=False)

        res_dict_tensors = trainer.train_epoch()
        # Losses are too eratic for testing equality:
        del res_dict_tensors['train_epoch_loss']
        # Turn tensors into floats:
        res_dict = {key: res_dict_tensors[key].item() 
                        for key in res_dict_tensors.keys()}
        
        # For testing equality, round the result:
        res_dict['train_epoch_fscore'] = round(res_dict['train_epoch_fscore'], 4)
         
        true_res_dict_A = {'train_epoch_acc': 0.5000,
                           'train_epoch_fscore': 0.6667,
                           'train_epoch_precision': 0.5,
                           'train_epoch_recall': 1.
                           }
        
        true_res_dict_B = {'train_epoch_acc': 0.5000,
                           'train_epoch_fscore': 0.,
                           'train_epoch_precision': 1.,
                           'train_epoch_recall': 0.
                          }
        try:
            self.assertDictEqual(res_dict, true_res_dict_A)
        except AssertionError:
            self.assertDictEqual(res_dict, true_res_dict_B)
            
        print('foo')


    #------------------------------------
    # testName
    #-------------------

    def test2EpocsBatchSize16(self):
        
        trainer = SpectrogramTrainer(
                        self.snippet_dir,
                        self.snippet_db_path,
                        batch_size = 16)
                        



# ------------------ Main -----------------

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()