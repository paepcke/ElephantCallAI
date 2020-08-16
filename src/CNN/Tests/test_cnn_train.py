'''
Created on Aug 15, 2020

@author: paepcke
'''
import os
import unittest

from CNN.train import SpectrogramTrainer, TrainResult


TEST_ALL = True
#TEST_ALL = False

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
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
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

    #------------------------------------
    # test1EpochBatchSize1Train
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test1EpochBatchSize1Train(self):
        
        trainer = SpectrogramTrainer(
                        self.snippet_dir,
                        self.snippet_db_path,
                        batch_size = 1)

        res_obj_tensors = trainer.train(num_epochs=1)

        res_obj = TrainResult()
        
        res_obj.best_valid_acc       = res_obj_tensors.best_valid_acc.item()
        res_obj.best_valid_fscore    = res_obj_tensors.best_valid_fscore.item()
        res_obj.best_valid_precision = res_obj_tensors.best_valid_precision.item()
        res_obj.best_valid_recall    = res_obj_tensors.best_valid_recall.item()
        
        # Two result possibilities (in spite of trying
        # to suppress randomness :-( ):
        
        true_res_obj_A = TrainResult()
         
        true_res_obj_A.best_valid_acc = 0.5000
        true_res_obj_A.best_valid_fscore = 0.6667
        true_res_obj_A.best_valid_precision = 0.5
        true_res_obj_A.best_valid_recall = 1.
        
        true_res_obj_B = TrainResult() 
        
        true_res_obj_B.best_valid_acc = 0.5000
        true_res_obj_B.best_valid_fscore = 0.
        true_res_obj_B.best_valid_precision = 1.
        true_res_obj_B.best_valid_recall = 0.

        self.assertTrue(res_obj == true_res_obj_A or
                        res_obj == true_res_obj_B)

# ------------------ Main -----------------

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()