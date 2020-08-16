#!/usr/bin/env python
'''
Created on Jul 7, 2020 based heavily on
Jonathan and Nikita's Code.

@author: paepcke
'''


import argparse
from collections import deque
import sys, os
import time

import numpy as np

from torch import cuda
from torch import optim
import torch
from torch.nn import BCELoss
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.resnet import ResNet, BasicBlock

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from elephant_utils.logging_service import LoggingService
from spectrogram_dataloader import SpectrogramDataloader
from spectrogram_dataset import SpectrogramDataset
from spectrogram_models import ModelSelector
from spectrogram_parameters import Defaults
from spectrogram_parameters import HyperParameters
import torch.nn as nn

import faulthandler; faulthandler.enable()

class SpectrogramTrainer(object):
    
    # Special mode for testing the code:
    #******TESTING=False
    #TESTING=True
    
    
    SPECTRO_WIDTH  = 117
    SPECTRO_HEIGHT = 2048
    
    KERNEL_HEIGHT  = 128  # Evenly divides into 2048 (128)
    KERNEL_WIDTH   = 13   # Evenly divides into 117 (9)
    
    STRIDE = 2
    NUM_CHANNELS = 1

    #------------------------------------
    # Constructor
    #-------------------
    
    def __init__(self,
                 dirs_and_files,
                 sqlite_db_path,
                 batch_size=Defaults.BATCH_SIZE,
                 decision_threshold=Defaults.THRESHOLD,
                 logfile=None
                 ):
        self.log = LoggingService(logfile)
        self.decision_threshold = decision_threshold
        
        self.device = torch.device('cuda' if cuda.is_available() else 'cpu')
        self.cuda   = torch.device('cuda')
        self.cpu    = torch.device('cpu')

        dataset  = SpectrogramDataset(dirs_and_files,
                                      sqlite_db_path=sqlite_db_path,
                                      recurse=True
                                      )
        self.dataloader = SpectrogramDataloader(dataset, 
                                                batch_size=batch_size)

        # Make a one-channel model with probability
        # decision boundary found in the parameters file:
        self.model = Resnet18Grayscale(num_classes=1)
        # Move to GPU if available:
        self.to_best_device(self.model)
        
        # Hyper parameter for chosen model
        model_hypers = HyperParameters['resnet18']
        
        # Loss function:
        if Defaults.LOSS == 'BCELoss':
            self.loss_func = BCELoss()
        else:
            raise ValueError(f"Loss function {Defaults.LOSS} not imported.")
        
        # Default dest dir: ./runs
        self.writer = SummaryWriter(log_dir=None)

        # Optimizer:
        self.optimizer = optim.SGD(self.model.parameters(), 
                                   model_hypers['lr'],
                                   momentum=model_hypers['momentum'],
                                   weight_decay=model_hypers['weight_decay']
                                   )
        
        # Scheduler:
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)

    #------------------------------------
    # to_best_device 
    #-------------------
    
    def to_best_device(self, item):
        if self.device == 'cuda':
            item.to(device=self.cuda)
        else:
            item.to(device=self.cpu)

    #------------------------------------
    # zero_tallies 
    #-------------------
    
    def zero_tallies(self):
        '''
        Returns a dict with the running 
        tallies that are maintained in both
        train_epoch and eval_epoch.
        
        @return: dict with float zeros
        @rtype: {str : float}
        '''
        tallies = {'running_loss' : 0.0,
                   'running_corrects' : 0.0,
                   'running_samples' : 0.0,
                   # True positives
                   'running_tp' : 0.0,
                   # True positives + false positives
                   'running_tp_fp' : 0.0,
                   # True positives + false negatives
                   'running_tp_fn' : 0.0,
                   # For focal loss purposes
                   'running_true_non_zero' : 0.0
            }
        return tallies

    #------------------------------------
    # tally_result
    #-------------------

    def tally_result(self, 
                     labels_tns, 
                     pred_prob_tns, 
                     loss,
                     tallies
                     ):
        '''
        Given a set of batch results, and a dict
        of running tallies, update and return the
        tallies. All quantities, even counts, are 
        kept as float tensors.
        
        This method is shared between train_epoch() and eval_epoch()
        
        Tallies is expected to contain:
            running_loss,
            running_true_non_zero,
            running_corrects, 
            running_samples, 
            running_tp_fp, 
            running_tp, 
            running_tp_fn

        @param labels_tns: ground truth labels for batch
        @type labels_tns: torch.Tensor (floats)
        @param pred_prob_tns: model prediction probabilities for this batch
        @type pred_prob_tns: torch.Tensor
        @param loss: result of loss function for one batch
        @type loss: torch.Tensor (float)
        @param tallies: running tallies to be updated
        @type tallies: dict
        '''
        
        # Turn into 1 or 0, depending on the
        # threshold noted in the parameters file:
        
        pred_tns = torch.where(pred_prob_tns > self.decision_threshold, torch.tensor(1.), torch.tensor(0.))
        
        num_positive_labels = torch.sum(labels_tns)
        tallies['running_true_non_zero'] += num_positive_labels
        tallies['running_loss'] += loss.item()
        tp = sum(torch.logical_and(labels_tns, pred_tns)).float()
        if len(pred_tns.shape) == 2:
            # Count the number slices for accuracy calculations
            tallies['running_samples'] += (torch.tensor(pred_tns.shape[0] * pred_tns.shape[1])).float()
        else:
            # For the binary window classification
            tallies['running_samples'] += torch.tensor(pred_tns.shape[0]).float()
        
        # For fp: where pred is 0, and labels is 1.
        # Element-wise subtract labels from pred. Where 
        # result is zero: True positive. Where result
        # is -1: FN. Where result is 1: FP: 
        
        diff = torch.sub(pred_tns,labels_tns)
        # Tie the -1s to 0; sum all the 1s. Finally, get an int:
        fp = torch.sum(torch.max(torch.tensor(0.), diff))
        # Count the -1s, then get an int:
        fn = torch.sum(diff == -1).float()
        
        tn = sum(torch.logical_and(pred_tns==0, labels_tns==0)).float()

        tp_fp = tp + fp
        tp_fn = tp + fn
        
        tallies['running_corrects'] += tp + tn

        tallies['running_tp'] += tp
        tallies['running_tp_fp'] += tp_fp
        tallies['running_tp_fn'] += tp_fn
        
        return tallies
    
    #------------------------------------
    # train_epoch 
    #-------------------

    def train_epoch(self, 
                    include_boundaries=False):
        
        # Set model to train mode:
        self.model.train(True)
        self.dataloader.switch_to_split('train')
        
        time_start = time.time()

        # Get a zeroed out set of running tallies:
        tallies = self.zero_tallies()
    
        self.log.info (f"Num batches: {len(self.dataloader)}")
        for idx, (spectros_tns, labels_tns) in enumerate(self.dataloader):
            
            self.optimizer.zero_grad()
            if (idx % 250 == 0) and Defaults.VERBOSE:
                self.log.info (f"Batch number {idx} of {len(self.dataloader)}")
                self.log.info (f"Total correct predictions {tallies['running_corrects']}, Total true positives {tallies['running_tp']}")
    
            # Put input and labels to where the
            # model is:
                 
            spectros_tns.to(self.model.device())
            labels_tns.to(self.model.device())
    
            # Forward pass
            # The unsqueeze() adds a dimension
            # for holding the batch_size?
            pred_prob_tns = self.model(spectros_tns)
            
            # The Binary Cross Entropy function wants 
            # equal datatypes for prediction and targets:
            
            labels_tns = labels_tns.float()
            loss =  self.loss_func(pred_prob_tns, labels_tns)
            loss.backward()
            self.optimizer.step()
    
            # Free GPU memory:
            spectros_tns.to('cpu')
            labels_tns.to('cpu')
            pred_prob_tns.to('cpu')

            tallies = self.tally_result(labels_tns, pred_prob_tns, loss, tallies) 

            train_epoch_loss = tallies['running_loss'] / (idx + 1)
            train_epoch_acc = float(tallies['running_corrects']) / tallies['running_samples']
        
            # If denom is zero: no positive sample was
            # fed into the training, and the classifier
            # correctly identified none as positive:
            
            train_epoch_precision = tallies['running_tp'] / tallies['running_tp_fp'] \
                if tallies['running_tp_fp'] > 0 else torch.tensor(1.)
            
            # If denom is zero: no positive samples 
            # were encountered, and the classifier did not
            # claim that any samples were positive:
            
            train_epoch_recall = tallies['running_tp'] / tallies['running_tp_fn'] \
                if tallies['running_tp_fn'] > 0 else torch.tensor(1.)
            
            if train_epoch_precision + train_epoch_recall > 0:
                train_epoch_fscore = (2 * train_epoch_precision * train_epoch_recall) / (train_epoch_precision + train_epoch_recall)
            else:
                train_epoch_fscore = torch.tensor(0.)
        
        self.log.info(f"Epoch train time: {(time.time() - time_start) / 60}")
        return {'train_epoch_acc': train_epoch_acc, 'train_epoch_fscore': train_epoch_fscore, 
                'train_epoch_loss': train_epoch_loss, 'train_epoch_precision':train_epoch_precision, 
                'train_epoch_recall': train_epoch_recall} 
    
    
    #------------------------------------
    # eval_epoch 
    #-------------------

    def eval_epoch(self, include_boundaries=False):
        
        # Set model to evaluation mode:
        self.model.eval()
        self.dataloader.switch_to_split('validate')
        
        time_start = time.time()
        # Get a zeroed out set of running tallies:
        tallies = self.zero_tallies()
    
        self.log.info (f"Num batches: {len(self.dataloader)}")
        with torch.no_grad():
             
            for idx, (spectros_tns, labels_tns) in enumerate(self.dataloader):
                
                self.optimizer.zero_grad()
                if (idx % 250 == 0) and Defaults.VERBOSE:
                    self.log.info (f"Batch number {idx} of {len(self.dataloader)}")
                    self.log.info (f"Total correct predictions {tallies['running_tp']}, Total true positives {tallies['running_tp']}")
        
                # Put input and labels to where the
                # model is:
                     
                spectros_tns.to(self.model.device())
                labels_tns.to(self.model.device())
        
                # Forward pass
                # The unsqueeze() adds a dimension
                # for holding the batch_size?
                pred_prob_tns = self.model(spectros_tns)
                
                # The Binary Cross Entropy function wants 
                # equal datatypes for prediction and targets:
                
                labels_tns = labels_tns.float()
                loss =  self.loss_func(pred_prob_tns, labels_tns)
                
                # Free GPU memory:
                spectros_tns.to('cpu')
                labels_tns.to('cpu')
                pred_prob_tns.to('cpu')

                tallies = self.tally_result(labels_tns, pred_prob_tns, loss, tallies) 

        valid_epoch_loss = tallies['running_loss'] / (idx + 1)
        valid_epoch_acc = float(tallies['running_corrects']) / tallies['running_samples']
        #valid_epoch_fscore = running_fscore / (idx + 1)
    
        # If this is zero issue a warning
        valid_epoch_precision = tallies['running_tp'] / tallies['running_tp_fp'] \
            if tallies['running_tp_fp'] > 0 else torch.tensor(1.)
        valid_epoch_recall = tallies['running_tp'] / tallies['running_tp_fn']
        if valid_epoch_precision + valid_epoch_recall > 0:
            valid_epoch_fscore = (2 * valid_epoch_precision * valid_epoch_recall) / (valid_epoch_precision + valid_epoch_recall)
        else:
            valid_epoch_fscore = torch.tensor(0.)
        
        self.log.info(f"Epoch validation time: {(time.time() - time_start) / 60}")
        return {'valid_epoch_acc': valid_epoch_acc, 'valid_epoch_fscore': valid_epoch_fscore, 
                'valid_epoch_loss': valid_epoch_loss, 'valid_epoch_precision':valid_epoch_precision, 
                'valid_epoch_recall': valid_epoch_recall}

    #------------------------------------
    # train 
    #-------------------

    def train(self,
              num_epochs=Defaults.NUM_EPOCHS, 
              starting_epoch=0, 
              resume=False,
              include_boundaries=False): 
               
        
        train_start_time = time.time()
        res_obj = TrainResult()
        
    
        # Check this
        previous_validation_accuracies = deque(maxlen=Defaults.TRAIN_STOP_ITERATIONS)
        previous_validation_fscores = deque(maxlen=Defaults.TRAIN_STOP_ITERATIONS)
        previous_validation_losses = deque(maxlen=Defaults.TRAIN_STOP_ITERATIONS)
    
        try:
            for epoch in range(starting_epoch, num_epochs):
                
                # Start a new k-fold cross_validation run for
                # each epoch:
                self.dataloader.kfold_stratified(shuffle=True)
                
                self.log.info (f'Epoch [{epoch + 1}/{num_epochs}]')
    
                train_epoch_results = self.train_epoch(include_boundaries)

                #Logging
                msg = (f"Epoch [{epoch + 1}/{num_epochs}] Training:\n"
                       f"Training loss        {train_epoch_results['train_epoch_loss']:.6f}\n"
                       f"Accuracy             {train_epoch_results['train_epoch_acc'].item():.4f}\n"
                       f"Precision            {train_epoch_results['train_epoch_precision'].item():.4f}\n"
                       f"Recall               {train_epoch_results['train_epoch_recall'].item():.4f}\n"
                       f"f-score              {train_epoch_results['train_epoch_fscore'].item():.4f}"
                       )
                self.log.info(msg)

                ## Write train metrics to tensorboard
                self.writer.add_scalar('train_epoch_loss', train_epoch_results['train_epoch_loss'], epoch)
                self.writer.add_scalar('train_epoch_acc', train_epoch_results['train_epoch_acc'], epoch)
                self.writer.add_scalar('train_epoch_fscore', train_epoch_results['train_epoch_fscore'], epoch)
                #self.writer.add_scalar('learning_rate', self.scheduler.get_lr(), epoch)
                self.writer.add_scalar('learning_rate', self.get_lr(self.scheduler), epoch)
    
                if self.is_eval_epoch(epoch):
                    
                    val_epoch_results = self.eval_epoch(include_boundaries) 

                    # Update the schedular
                    self.scheduler.step(val_epoch_results['valid_epoch_loss'])

                    #Logging
                    msg = (f"Epoch [{epoch + 1}/{num_epochs}] Validation:\n"
                           f"Validation loss      {val_epoch_results['valid_epoch_loss']:.6f}\n"
                           f"Accuracy             {val_epoch_results['valid_epoch_acc'].item():.4f}\n"
                           f"Precision            {val_epoch_results['valid_epoch_precision'].item():.4f}\n"
                           f"Recall               {val_epoch_results['valid_epoch_recall'].item():.4f}\n"
                           f"f-score              {val_epoch_results['valid_epoch_fscore'].item():.4f}"
                           )
                    self.log.info(msg)
                    
                    ## Write val metrics to tensorboard
                    self.writer.add_scalar('valid_epoch_loss', val_epoch_results['valid_epoch_loss'], epoch)
                    self.writer.add_scalar('valid_epoch_acc', val_epoch_results['valid_epoch_acc'], epoch)
                    self.writer.add_scalar('valid_epoch_fscore', val_epoch_results['valid_epoch_fscore'], epoch)
    
                    # Update eval tracking statistics!
                    previous_validation_accuracies.append(val_epoch_results['valid_epoch_acc'])
                    previous_validation_fscores.append(val_epoch_results['valid_epoch_fscore'])
                    previous_validation_losses.append(val_epoch_results['valid_epoch_loss'])
    
                    curr_best_acc = res_obj.best_valid_acc
                    res_obj.best_valid_acc = max(val_epoch_results['valid_epoch_acc'],
                                                 res_obj.best_valid_acc)
                    if Defaults.TRAIN_MODEL_SAVE_CRITERIA.lower() == 'acc' \
                            and curr_best_acc != res_obj.best_valid_acc:
                        best_model_wts = self.model.state_dict()
    
                    curr_best_fs = res_obj.best_valid_fscore
                    res_obj.best_valid_fscore = torch.max(val_epoch_results['valid_epoch_fscore'],
                                                          res_obj.best_valid_fscore)
                    if Defaults.TRAIN_MODEL_SAVE_CRITERIA.lower() == 'fscore' and \
                            curr_best_fs != val_epoch_results['best_valid_fscore']:
                        best_model_wts = self.model.state_dict()
                    
                    res_obj.best_valid_precision = torch.max(val_epoch_results['valid_epoch_precision'],
                                                     res_obj.best_valid_precision)
                    res_obj.best_valid_recall = torch.max(val_epoch_results['valid_epoch_recall'],
                                                          res_obj.best_valid_recall)
    
                    res_obj.best_valid_loss = min(val_epoch_results['valid_epoch_loss'],
                                                  res_obj.best_valid_loss)
    
                    # Check whether to early stop due to decreasing validation acc or f-score
                    if Defaults.TRAIN_MODEL_SAVE_CRITERIA.lower() == 'acc':
                        if all([val_accuracy < res_obj.best_valid_acc for val_accuracy in previous_validation_accuracies]):
                            self.log.info(f"Early stopping because last {Defaults.TRAIN_STOP_ITERATIONS} "
                                          f"validation accuracies have been {previous_validation_accuracies} " 
                                          f"and less than best val accuracy {res_obj.best_valid_acc}")
                            break
                    elif Defaults.TRAIN_MODEL_SAVE_CRITERIA.lower() == 'fscore':
                        if all([val_fscore < res_obj.best_valid_fscore for val_fscore in previous_validation_fscores]):
                            self.log.info(f"Early stopping because last {Defaults.TRAIN_STOP_ITERATIONS} "
                                          f"validation f-scores have been {previous_validation_fscores} "
                                          f"and less than best val f-score {res_obj.best_valid_fscore}")
                            break
    
                self.log.info(f'Finished Epoch [{epoch + 1}/{num_epochs}] - Total Time: {(time.time()-train_start_time)/60}')
    
        except KeyboardInterrupt:
            self.log.info("Early stopping due to keyboard intervention")
    
        msg = (f"\nBest val Acc    {res_obj.best_valid_acc.item():.4f}\n"
               f"Best val fscore    {res_obj.best_valid_fscore.item():.4f}\n"
               f"    with precision {res_obj.best_valid_precision.item():.4f}\n"
               f"    and recall {res_obj.best_valid_recall.item():.4f}\n"
               f"Best val loss    {res_obj.best_valid_loss:.6f}"
               )
        self.log.info(msg)
    
        res_obj.best_model_wts = best_model_wts
        return res_obj

    # ------------- Utils -----------

    #------------------------------------
    # is_eval_epoch 
    #-------------------

    def is_eval_epoch(self, cur_epoch):
        """Determines if the model should be evaluated at the current epoch."""
        return (
            (cur_epoch + 1) % Defaults.EVAL_PERIOD == 0 or
            cur_epoch == 0 or
            (cur_epoch + 1) == Defaults.NUM_EPOCHS
        )

    #------------------------------------
    # get_lr 
    #-------------------
    
    def get_lr(self, scheduler):
        '''
        Given a scheduler instance, return its
        current learning rate. All schedulers but
        one have a get_lr() method themselves. But
        the auto-lr-reducing ReduceLROnPlateau does
        not. It maintains multiple learning rates,
        one for each 'group'. So the result of a
        get_lr() method would be a list, which would
        be contrary to all other schedulers. This
        method masks the difference. 
        
        We only have one group, so we return the
        first (and only) lr if the scheduler is an
        instance of the problem class.
        
        
        @param scheduler: scheduler whose learning rate 
            is to be retrieved
        @type scheduler:torch.optim.lr_scheduler
        @return: the scheduler's current learning rate
        @rtype: float
        '''
        
        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            learning_rates = [ group['lr'] for group in scheduler.optimizer.param_groups ]
            lr = learning_rates[0]
        else:
            lr = scheduler.get_lr()
        return lr

    #------------------------------------
    # get_prec_rec_components
    #-------------------
    
    def get_prec_rec_components(self, pred_tns, label_tns):
        '''
        Given tensor of finalized predictions (0s and 1s),
        and the label tensor from the dataloader, return
        numbers of true positive, the sum of true positive
        and false positive, and the sum of true positive,
        and false negatives (tp, tp_fp, tp_fn)
        
        @param pred_tns: tensors of 1.0 and 0.0 stacked batch_size high
        @type pred_tns: torch.Tensor [batch_size, 1]
        @param label_tns: 1.0 or 2.0 true labels stacked batch_size high
        @type label_tns: torch.Tensor [batch_size, 1]
        @return: precision values tp, tp_fp, tp_fn
        @rtype: [int,int,int]
        '''
        
        # Get the values from tensor [batch_size, label_val]:
        label_vals = label_tns.view(-1)
        tp         = torch.sum(label_vals)
        pred_pos = torch.sum(pred_tns)
        fp = max(0, pred_pos - tp)
        fn = max(0, tp - pred_pos)
        tp_fp = tp + fp
        tp_fn = tp + fn
        
        return (tp, tp_fp, tp_fn)
        

    #------------------------------------
    # set_seed  
    #-------------------

    def set_seed(self, seed):
        '''
        Set the seed across all different necessary platforms
        to allow for comparison of different models and runs
        
        @param seed: random seed to set for all random num generators
        @type seed: int
        '''
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Not totally sure what these two do!
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        #random.seed = seed
    
# ---------------------- Resnet18Grayscale ---------------

class Resnet18Grayscale(ResNet):
    '''
    A Resnet18 variant that accepts single-channel
    grayscale images instead of RGB.
    
    Using this class saves space from not having 
    to replicate our single-layer spectrograms three 
    times to pretend they are RGB images.
    '''

    #------------------------------------
    # Constructor 
    #-------------------
    
    def __init__(self, *args, **kwargs):
        '''
        Args and kwargs as per https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
        class ResNet.__init__()
        
        '''
        # The [2,2,2,2] is an instruction to the
        # superclass' __init__() for how many layers
        # of each type to create. This info makes the
        # ResNet into a ResNet18:
        super().__init__(BasicBlock, [2,2,2,2], *args, **kwargs)
        
        # Change expected channels from 3 to 1
        # The superclass created first layer
        # with the first argument being a 3.
        # We just replace the first layer:
        self.inplanes = 64 #******* Should be batch size?
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        
    #------------------------------------
    # forward 
    #-------------------
    
    def forward(self, x):
        out_logit = super().forward(x)

        # Since we have binary classification,
        # the Sigmoid function does what a 
        # softmax would do for multi-class:

        out_probs  = nn.Sigmoid()(out_logit)
        return out_probs

    #------------------------------------
    # device 
    #-------------------
    
    def device(self):
        '''
        Returns device where model resides.
        Can use like this to move a tensor
        to wherever the model is:
        
            some_tensor.to(<model_instance>.device())

        '''
        return next(self.parameters()).device


# ----------------------- Class Train Results -----------

class TrainResult(object):
    '''
    Instances of this class hold training results
    accumulated during method train(). The method
    returns such an instance. Used e.g. for unittests
    '''
    
    #------------------------------------
    # Constructor 
    #-------------------

    def __init__(self):
        self.best_valid_acc       = torch.tensor(0.0)
        self.best_valid_fscore    = torch.tensor(0.0)
        # Best precision and recall, which reflect
        # the best fscore
        self.best_valid_precision = torch.tensor(0.0)
        self.best_valid_recall    = torch.tensor(0.0)
        self.best_valid_loss      = torch.tensor(float("inf"))
        self.best_model_wts       = None

    #------------------------------------
    # __eq__ 
    #-------------------
    
    def __eq__(self, other):
        '''
        Return True if given TrainResult instance
        is equal to self in all but loss and weights
        @param other: instance to compare to
        @type other: TrainResult
        @return: True for equality
        @rtype: bool
        '''
        if not isinstance(other, TrainResult):
            return False
        
        if  round(self.best_valid_acc,4)       ==  round(other.best_valid_acc,4)         and \
            round(self.best_valid_fscore,4)    ==  round(other.best_valid_fscore,4)      and \
            round(self.best_valid_precision,4) ==  round(other.best_valid_precision,4)   and \
            round(self.best_valid_recall,4)    ==  round(other.best_valid_recall,4):
            return True
        else:
            return False

    #------------------------------------
    # print 
    #-------------------
    
    def print(self, include_weights=False):
        msg = (f"best_valid_acc      : {self.best_valid_acc}\n"
               f"best_valid_fscore   : {self.best_valid_fscore}\n"
               f"best_valid_precision: {self.best_valid_precision}\n"
               f"best_valid_recall   : {self.best_valid_recall}\n"
               f"best_valid_loss     : {self.best_valid_loss}\n"
               )
        if include_weights:
            msg += f"best_valid_weights: {self.best_valid_wts}\n" 
        print(msg)

# ------------------------ Main ------------
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Train from a set of spectrogram snippets."
                                     )
# 
#     parser.add_argument('-l', '--logfile',
#                         help='fully qualified log file name to which info and error messages \n' +\
#                              'are directed. Default: stdout.',
#                         default=None);
#     parser.add_argument('-b', '--batchsize',
#                         type=int,
#                         help='how many sample to submit to training machinery together; default: 1',
#                         default=1)
#     parser.add_argument('-e', '--epochs',
#                         type=int,
#                         help=f'how many epochs to run; default: Defaults.NUM_EPOCHS',
#                         default=Defaults.NUM_EPOCHS)
#     parser.add_argument('snippet_db_path',
#                         type=str,
#                         help='path to sqlite db file holding info about each snippet')
#     parser.add_argument('files_and_dirs',
#                         type=str,
#                         nargs='+',
#                         help='Repeatable: directories/files containing .pickle spectrogram dataframes, '
#                              'and corresponding .txt label files')
# 
    args = parser.parse_args();
    
    #***********
    args.snippet_db_path = '/Users/paepcke/EclipseWorkspacesNew/ElephantCallAI/Spectrograms/Training//tiny_chop_info.sqlite'
    args.files_and_dirs  = '/Users/paepcke/EclipseWorkspacesNew/ElephantCallAI/Spectrograms/Training/Threshold_-30_MinFreq_20_MaxFreq_40_FreqCap_30_snippets_0'
    #args.batchsize=2
    args.batchsize=16
    args.epochs=2
    args.logfile =None
    #***********
    
    if not os.path.exists(args.snippet_db_path):
        print(f"Snippet information db path {args.snippet_db_path} not found")
        sys.exit(1)

    SpectrogramTrainer(args.files_and_dirs,
                       args.snippet_db_path,
                       batch_size=args.batchsize,
                       logfile=args.logfile
                       ).train(args.epochs)
