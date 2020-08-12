#!/usr/bin/env python
'''
Created on Jul 7, 2020 based heavily on
Jonathan and Nikita's Code.

@author: paepcke
'''


from collections import deque
import sys, os
import time

import argparse
import torch
import torch.nn as nn
from torch.nn import BCELoss
from torch import cuda
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from torchvision.models.resnet import ResNet, BasicBlock

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from elephant_utils.logging_service import LoggingService
from spectrogram_parameters import Defaults
from spectrogram_parameters import HyperParameters
from spectrogram_dataloader import SpectrogramDataloader
from spectrogram_dataset import SpectrogramDataset
from utils import get_precission_recall_values
from utils import num_correct 
from utils import num_non_zero, get_f_score

from spectrogram_models import ModelSelector


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
    DEFAULT_BATCH_SIZE = 1
    NUM_CHANNELS = 1

    #------------------------------------
    # Constructor
    #-------------------
    
    def __init__(self,
                 dirs_and_files,
                 sqlite_db_path,
                 batch_size=1,
                 logfile=None
                 ):
        self.log = LoggingService(logfile)
        if batch_size is None:
            batch_size = self.DEFAULT_BATCH_SIZE
        self.device = torch.device('cuda' if cuda.is_available() else 'cpu')
        self.cuda   = torch.device('cuda')
        self.cpu    = torch.device('cpu')

        dataset  = SpectrogramDataset(dirs_and_files,
                                      sqlite_db_path=sqlite_db_path,
                                      recurse=True
                                      )
        self.dataloader = SpectrogramDataloader(dataset, 
                                                batch_size=batch_size)
        
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
    # train_epoch 
    #-------------------

    def train_epoch(self, 
                    include_boundaries=False):
        
        # Set model to train mode:
        self.model.train(True)
        self.dataloader.switch_to_split('train')
        
        time_start = time.time()
    
        running_loss = 0.0
        running_corrects = 0
        running_samples = 0
        # True positives
        running_tp = 0
        # True positives + false positives
        running_tp_fp = 0
        # True positives + false negatives
        running_tp_fn = 0
        #running_fscore = 0.0
        #running_precission = 0.0
        #running_recall = 0.0
    
        # For focal loss purposes
        running_true_non_zero = 0
    
        self.log.info (f"Num batches: {len(self.dataloader)}")
        for idx, (spectros_tns, labels_tns) in enumerate(self.dataloader):
            
            self.optimizer.zero_grad()
            if (idx % 250 == 0) and Defaults.VERBOSE:
                self.log.info (f"Batch number {idx} of {len(self.dataloader)}")
                self.log.info (f"Total correct predictions {running_tp}, Total true positives {running_tp}")
    
            # Put input and labels to where the
            # model is:
                 
            spectros_tns.to(self.model.device())
            labels_tns.to(self.model.device())
    
            # Forward pass
            # The unsqueeze() adds a dimension
            # for holding the batch_size?
            logits = self.model(spectros_tns)
            
            # The Binary Cross Entropy function wants 
            # equal datatypes for prediction and targets:
            
            labels_tns = labels_tns.float()
            pred = nn.Softmax(0)(logits)
            loss =  self.loss_func(pred, labels_tns)
            loss.backward()
            self.optimizer.step()
    
            # Free GPU memory:
            spectros_tns.to('cpu')
            labels_tns.to('cpu')

            running_true_non_zero += torch.sum(labels_tns).item()
            running_loss += loss.item()
            running_corrects += num_correct(pred, labels_tns)
            if len(logits.shape) == 2:
                # Count the number slices for accuracy calculations
                running_samples += logits.shape[0] * logits.shape[1] 
            else: # For the binary window classification
                running_samples += logits.shape[0]
            #running_fscore += get_f_score(logits, labels)
            tp, tp_fp, tp_fn = get_precission_recall_values(logits, labels_tns)
            running_tp += tp
            running_tp_fp += tp_fp
            running_tp_fn += tp_fn 
            
    
            train_epoch_loss = running_loss / (idx + 1)
            train_epoch_acc = float(running_corrects) / running_samples
            #train_epoch_fscore = running_fscore / (idx + 1)
        
            # If denom is zero: no positive sample was
            # fed into the training, and the classifier
            # correctly identified none as positive:
            
            train_epoch_precision = running_tp / running_tp_fp if running_tp_fp > 0 else 1
            
            # If denom is zero: no positive samples 
            # were encountered, and the classifier did not
            # claim that any samples were positive:
            
            train_epoch_recall = running_tp / running_tp_fn if running_tp_fn > 0 else 1
            
            if train_epoch_precision + train_epoch_recall > 0:
                train_epoch_fscore = (2 * train_epoch_precision * train_epoch_recall) / (train_epoch_precision + train_epoch_recall)
            else:
                train_epoch_fscore = 0
        
        #Logging
        self.log.info('Training loss: {:.6f}, acc: {:.4f}, p: {:.4f}, r: {:.4f}, f-score: {:.4f}, time: {:.4f}'.format(
            train_epoch_loss, train_epoch_acc, train_epoch_precision, train_epoch_recall, train_epoch_fscore ,(time.time()-time_start)/60))
        
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
    
        running_loss = 0.0
        running_corrects = 0
        running_samples = 0
        running_fscore = 0.0
        # True positives
        running_tp = 0
        # True positives, false positives
        running_tp_fp = 0
        # True positives, false negatives
        running_tp_fn = 0
    
        # For focal loss purposes
        running_true_non_zero = 0
    
        self.log.info (f"Num batches: {len(self.dataloader)}")
        with torch.no_grad():
             
            for idx, (spectros_tns, labels_tns) in enumerate(self.dataloader):
                
                self.optimizer.zero_grad()
                if (idx % 250 == 0) and Defaults.VERBOSE:
                    self.log.info (f"Batch number {idx} of {len(self.dataloader)}")
                    self.log.info (f"Total correct predictions {running_tp}, Total true positives {running_tp}")
        
                # Cast the variables to the correct type and 
                # put on the correct torch device
                     
                spectros_tns.to(Defaults.device)
                labels_tns.to(Defaults.device)
        
                # Forward pass
                # The unsqueeze() adds a dimension
                # for holding the batch_size?
                logits = self.model(spectros_tns)
                
                # The Binary Cross Entropy function wants 
                # equal datatypes for prediction and targets:
                
                labels_tns = labels_tns.float()
                pred = nn.Softmax(0)(logits)
                loss =  self.loss_func(pred, labels_tns)
    
                running_true_non_zero += torch.sum(labels_tns).item()
                running_loss += loss.item()
                running_corrects += num_correct(logits, labels_tns)

                if len(logits.shape) == 2:
                    # Count the number slices for accuracy calculations
                    running_samples += logits.shape[0] * logits.shape[1] 
                else: # For the binary window classification
                    running_samples += logits.shape[0]
                #running_fscore += get_f_score(logits, labels)
                tp, tp_fp, tp_fn = get_precission_recall_values(logits, labels_tns)
                running_tp += tp
                running_tp_fp += tp_fp
                running_tp_fn += tp_fn 

        valid_epoch_loss = running_loss / (idx + 1)
        valid_epoch_acc = float(running_corrects) / running_samples
        #valid_epoch_fscore = running_fscore / (idx + 1)
    
        # If this is zero issue a warning
        valid_epoch_precision = running_tp / running_tp_fp if running_tp_fp > 0 else 1
        valid_epoch_recall = running_tp / running_tp_fn
        if valid_epoch_precision + valid_epoch_recall > 0:
            valid_epoch_fscore = (2 * valid_epoch_precision * valid_epoch_recall) / (valid_epoch_precision + valid_epoch_recall)
        else:
            valid_epoch_fscore = 0
    
        #Logging
        self.log.info('Validation loss: {:.6f}, acc: {:.4f}, p: {:.4f}, r: {:.4f}, f-score: {:.4f}, time: {:.4f}'.format(
                valid_epoch_loss, valid_epoch_acc, valid_epoch_precision, valid_epoch_recall,
                valid_epoch_fscore, (time.time()-time_start)/60))
    
    
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
        
        best_valid_acc = 0.0
        best_valid_fscore = 0.0
        # Best precision and recall reflect
        # the best fscore
        best_valid_precision = 0.0
        best_valid_recall = 0.0
        best_valid_loss = float("inf")
        best_model_wts = None
    
        # Check this
        last_validation_accuracies = deque(maxlen=Defaults.TRAIN_STOP_ITERATIONS)
        last_validation_fscores = deque(maxlen=Defaults.TRAIN_STOP_ITERATIONS)
        last_validation_losses = deque(maxlen=Defaults.TRAIN_STOP_ITERATIONS)
    
        try:
            for epoch in range(starting_epoch, num_epochs):
                
                # Start a new k-fold cross_validation run for
                # each epoch:
                self.dataloader.kfold_stratified(shuffle=True)
                
                self.log.info (f'Epoch [{epoch + 1}/{num_epochs}]')
    
                train_epoch_results = self.train_epoch(include_boundaries)

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
    
                    ## Write val metrics to tensorboard
                    self.writer.add_scalar('valid_epoch_loss', val_epoch_results['valid_epoch_loss'], epoch)
                    self.writer.add_scalar('valid_epoch_acc', val_epoch_results['valid_epoch_acc'], epoch)
                    self.writer.add_scalar('valid_epoch_fscore', val_epoch_results['valid_epoch_fscore'], epoch)
    
                    # Update eval tracking statistics!
                    last_validation_accuracies.append(val_epoch_results['valid_epoch_acc'])
                    last_validation_fscores.append(val_epoch_results['valid_epoch_fscore'])
                    last_validation_losses.append(val_epoch_results['valid_epoch_loss'])
    
                    if val_epoch_results['valid_epoch_acc'] > best_valid_acc:
                        best_valid_acc = val_epoch_results['valid_epoch_acc']
                        if Defaults.TRAIN_MODEL_SAVE_CRITERIA.lower() == 'acc':
                            best_model_wts = self.model.state_dict()
    
                    if val_epoch_results['valid_epoch_fscore'] > best_valid_fscore:
                        best_valid_fscore = val_epoch_results['valid_epoch_fscore']
                        best_valid_precision = val_epoch_results['valid_epoch_precision']
                        best_valid_recall = val_epoch_results['valid_epoch_recall']
                        if Defaults.TRAIN_MODEL_SAVE_CRITERIA.lower() == 'fscore':
                            best_model_wts = self.model.state_dict()
    
                    if val_epoch_results['valid_epoch_loss'] < best_valid_loss:
                        best_valid_loss = val_epoch_results['valid_epoch_loss'] 
    
                    # Check whether to early stop due to decreasing validation acc or f-score
                    if Defaults.TRAIN_MODEL_SAVE_CRITERIA.lower() == 'acc':
                        if all([val_accuracy < best_valid_acc for val_accuracy in last_validation_accuracies]):
                            self.log.info(f"Early stopping because last {Defaults.TRAIN_STOP_ITERATIONS} "
                                          f"validation accuracies have been {last_validation_accuracies} " 
                                          f"and less than best val accuracy {best_valid_acc}")
                            break
                    elif Defaults.TRAIN_MODEL_SAVE_CRITERIA.lower() == 'fscore':
                        if all([val_fscore < best_valid_fscore for val_fscore in last_validation_fscores]):
                            self.log.info(f"Early stopping because last {Defaults.TRAIN_STOP_ITERATIONS} "
                                          f"validation f-scores have been {last_validation_fscores} "
                                          f"and less than best val f-score {best_valid_fscore}")
                            break
    
                self.log.info(f'Finished Epoch [{epoch + 1}/{num_epochs}] - Total Time: {(time.time()-train_start_time)/60}')
    
        except KeyboardInterrupt:
            self.log.info("Early stopping due to keyboard intervention")
    
        self.log.info('Best val Acc: {:4f}'.format(best_valid_acc))
        self.log.info('Best val F-score: {:4f} with Precision: {:4f} and Recall: {:4f}'.format(best_valid_fscore, best_valid_precision, best_valid_recall))
        self.log.info ('Best val Loss: {:6f}'.format(best_valid_loss))
    
        return best_model_wts

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
