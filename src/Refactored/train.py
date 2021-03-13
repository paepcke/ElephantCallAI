import torch
import numpy as np
import time
import pdb
import sys
from collections import deque
import faulthandler; faulthandler.enable()

# Local file imports
import parameters
from model_utils import Model_Utils

class Train_Pipeline(object):
    """
        Let us try doing a class like Andreas suggests! Basically capture everything that is here
        but in a class! First let us just make all of these class method

        Later for curriculum learning for example we may consider having the training have a state

        However, NO my thinking is that this training pipeline is just responsible for 
        training one model for a given number of epochs!!!

        It should not be responsible for multiple trainings or anything!!


        KEEP IT SIMPLE. THE TASKS WE NEED TO COMPLETE
            - Take in for now just a train/test.
            - Train a given model for x epochs or until "convergence"!
                - For each epoch do one iteration over the Train data
                - If we are evaluating then do an iteration over that as well!!!!
            - Do we want to add a logger??? Maybe later!! Screw that for now!!

    """
    
    early_stop_criteria_map = {
                               'acc': 'best_valid_acc', 
                               'fscore': 'best_valid_fscore',
                               }


    # FOR NOW DO NOT HAVE AN INIT SINCE THIS CLASS SHOULD NOT BE INSTANTIATED
    # Question to think about:
    #   Is there a reason we would want to keep a state during / after training??
    #   Maybe honestly it can clean up the code! Where we instantiate a training
    #   pipeline with all of our things and then call train!

    # LET US TRY THIS OUT 
    # For now be kinda adhoc and assume that this stuff is just going to be changed.
    # but in theory this could allow us to change the data and keep training a model 
    # for example! But for now we should just leave this!!
    def __init__(self, dataloaders, model, loss_func, optimizer, 
                scheduler, writer, save_path, early_stop_criteria="acc"):
        super(Train_Pipeline, self).__init__()

        # Step 1) Get the dataloaders
        self.train_dataloader = dataloaders['train']
        self.test_dataloader = dataloaders['valid']

        # Step 2) Save the model we are training!
        self.model = model

        # Step 3) Save the model optimizers
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Step 4) Model save path
        self.save_path = save_path

        # Step 5) Save the summary writer
        self.writer = writer

        # Step 6) Save the early stopping criteria
        self.early_stop_criteria = early_stop_criteria


    #-----------------------------
    # run_epoch 
    #--------------

    # Try having this essentially be private!
    def run_epoch(self, dataloader, train=True):
        """
            Run an epoch! Note let us try combining the logic of training
            and eval epochs with the train flag to change execution!
        """
        # Set the model to the corresponding train/test mode
        epoch_name = "Train" if train else "Val"

        # Make sure model is in correct mode!
        model.train(train)

        epoch_stats = {
                       'running_loss': 0.0,
                       'running_corrects': 0,
                       'running_samples': 0,
                       'running_tp': 0,
                       'running_tp_fp': 0,
                       'running_tp_fn': 0,
                        }

        print ("Num batches:", len(dataloader))
        for idx, batch in enumerate(dataloader):
            # Training specific settings
            if train:
                optimizer.zero_grad()

            # Help track the training pipeline
            if (idx % 250 == 0) and parameters.VERBOSE:
                print ("Batch number {} of {}".format(idx, len(dataloader)))

            # Cast variables to the correct types and put on the correct torch device
            # clone for now but may remove!!
            inputs = batch[0].clone().float()
            labels = batch[1].clone().float()
            inputs = inputs.to(parameters.device)
            labels = labels.to(parameters.device)

            # Forward pass
            logits = model(inputs).squeeze(-1)
            loss = self.loss_func(logits, labels)

            if train:
                self.loss.backward()
                self.optimizer.step()

            self.update_epoch_stats(epoch_stats, loss=loss.item(), logits=logits, labels=labels)
     

        # Update the schedular
        if train:
            self.scheduler.step()
        
        return self.epoch_summary(epoch_stats, len(dataloader), name=epoch_name):

    #-----------------------------
    # train 
    #--------------

    def train(self, num_epochs):
        """
            Runs num_epochs of training for the class model!

            @TODO WRITE CODE!!!
        """
        train_start_time = time.time()

        best_valid_stats = {
                            'best_valid_acc': 0.0,
                            'best_valid_fscore': 0.0,
                            'best_valid_precision': 0.0,
                            'best_valid_recall': 0.0,
                            'best_valid_loss': 0.0,
                            }

        # Use early stopping module
        early_stopping = EarlyStopping(larger_is_better=True, patience=patience, verbose=True, path=self.save_path)

        # Include a try catch loop to allow for 'ctrl C' early stopping
        try:
            for epoch in range(num_epochs):
                print ('Epoch [{}/{}]'.format(epoch + 1, num_epochs))

                # Run a training epoch
                train_epoch_results = self.run_epoch(self.train_dataloader, train=True)

                ## Write train metrics to tensorboard
                self.update_writer(train_epoch_results, epoch, lr=scheduler.lr())

                # Evaluate the model
                if Model_Utils.is_eval_epoch(epoch):

                    val_epoch_results = self.run_epoch(self.test_dataloader, train=False)

                    # Write val metrics to tensorboard
                    self.update_writer(val_epoch_results, epoch)
                    # Update best evaluation metrics
                    self.track_best_performance(val_epoch_results, best_valid_stats)

                    # Check if we should stop early!
                    early_stopping(best_valid_stats[early_stop_criteria_map[self.early_stop_criteria]], model)
                    if early_stopping.early_stop:
                        print("Early stopping")
                        break

                print('Finished Epoch [{}/{}] - Total Time: {}.'.format(epoch + 1, num_epochs, (time.time()-train_start_time)/60))

        except KeyboardInterrupt:
            print("Early stopping due to keyboard intervention")

        print('Best val Acc: {:4f}'.format(best_valid_stats['best_valid_acc']))
        print('Best val F-score: {:4f} with Precision: {:4f} and Recall: {:4f}'.format(best_valid_stats['best_valid_fscore'], 
                                                                    best_valid_stats['best_valid_precision'],
                                                                    best_valid_stats['best_valid_recall']))
        print ('Best val Loss: {:6f}'.format(best_valid_stats['best_valid_loss']))

        # Return the best model weights stored in the EarlyStopping object
        return early_stopping.best_model_wts


    ########################
    #### Helper Methods ####
    ########################

    def update_epoch_stats(self, stats, loss, logits, labels):
        """
            Helps to break up the code in running the epoch!
        """
        stats['running_loss'] += loss
        stats['running_corrects'] += Model_Utils.num_correct(logits, labels)

        # Update precision/recall trackers
        tp, tp_fp, tp_fn = Model_Utils.get_precission_recall_values(logits, labels)
        stats['running_tp'] += tp
        stats['running_tp_fp'] += tp_fp
        stats['running_tp_fn'] += tp_fn

        # Update the number of time slices predicted over for accuracy calculations
        stats['running_samples'] += logits.shape[0] * logits.shape[1] 

    def epoch_summary(self, stats, num_batches, name='Train'):
        """
            Compute the summary of the epoch and return a dict
        """
        epoch_loss = stats['running_loss'] / num_batches
        epoch_acc = float(stats['running_corrects']) / stats['running_samples']

        # If this is zero print a warning
        epoch_precision = stats['running_tp'] / stats['running_tp_fp'] if stats['running_tp_fp'] > 0 else 1
        epoch_recall = stats['running_tp'] / stats['running_tp_fn']
        if epoch_precision + epoch_recall > 0:
            epoch_fscore = (2 * epoch_precision * epoch_recall) / (epoch_precision + epoch_recall)
        else:
            epoch_fscore = 0

        print('{} loss: {:.6f}, acc: {:.4f}, p: {:.4f}, r: {:.4f}, f-score: {:.4f}'.format(
            name, epoch_loss, epoch_acc, epoch_precision, epoch_recall, epoch_fscore))

        return {f'{name}_epoch_acc': epoch_acc, f'{name}_epoch_fscore': epoch_fscore, 
                f'{name}_epoch_loss': epoch_loss, f'{name}_epoch_precision':epoch_precision, 
                f'{name}_epoch_recall': epoch_recall} 


    def update_writer(self, stats, epochs, lr=None):
        """
            Given a dict with keys representing tensorboard fields,
            output to the tensorboard writer the stats!

            @TODO update this
        """
        for key, value in stats.item():
            writer.add_scalar(key, value, epoch)

        # Add the learning rate as well if lr is not None
        if lr:
            writer.add_scalar('learning_rate', lr, epoch)

    def track_best_performance(self, epoch_stats, best_stats):
        """
            Helper function to update the best validation statistics base on
            a recent epoch
        """
        # Update the best accuracy
        if epoch_stats['valid_epoch_acc'] > best_stats['best_valid_acc']:
            best_stats['best_valid_acc'] = epoch_stats['valid_epoch_acc']
        
        # Update the best f_score and save corresponding P and R
        if epoch_stats['valid_epoch_fscore'] > best_stats['best_valid_fscore']:
            best_stats['best_valid_fscore'] = epoch_stats['valid_epoch_fscore']
            best_stats['best_valid_precision'] = epoch_stats['valid_epoch_precision']
            best_stats['best_valid_recall'] = epoch_stats['valid_epoch_recall']
            
        # Update the best loss function
        if epoch_stats['valid_epoch_loss'] < best_stats['best_valid_loss']:
            best_stats['best_valid_loss'] = epoch_stats['valid_epoch_loss'] 

        

################################
#### Early stopping tracker ####
################################

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, larger_is_better=True, patience=30, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            larger_is_better (bool): Indicates whether having a larger objective value is
                            better (e.g. 'accuracy' or 'fscore'). If True x > y means
                            x is better than y, else reversed.
            patience (int): How long to wait after last time validation loss improved.
                            Default: 30
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.larger_equal_better = larger_equal_better
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_criteria = None
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.best_model_wts = None


    def __call__(self, criteria, model):
        """
            Update the best criteria and any counter involved with early stopping
        """
        # We have not tracked any progress
        if self.best_criteria is None:
            self.save_checkpoint(criteria, model)
            self.best_criteria = criteria

        # Want criteria to be smaller (e.g. loss)
        elif (not self.larger_is_better) and (criteria < self.best_criteria - self.delta):
            self.save_checkpoint(criteria, model)
            self.best_criteria = criteria
            self.counter = 0

        # Want criteria to be larger (e.g. acc)
        elif (self.larger_is_better) and criteria > self.best_criteria + self.delta
            self.save_checkpoint(criteria, model)
            self.best_criteria = criteria
            self.counter = 0

        # Criteria did not improve
        else: 
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        

    def save_checkpoint(self, new_best, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation criteria improved ({self.best_criteria:.6f} --> {new_best:.6f}).  Saving model ...')
        
        self.best_model_wts = model.state_dict()




