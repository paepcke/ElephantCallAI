import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_curve
from torchsummary import summary
import time
from tensorboardX import SummaryWriter
import sklearn
import sys
import copy
import os
import torch.nn.functional as F
import argparse


#import torchvision.models as models
#from torchvision import transforms
#import matplotlib
#import matplotlib.pyplot as plt
#from collections import deque

# Local file imports
import parameters
from models import get_model
from loss import get_loss
from train import Train_Pipeline
from model_utils import Model_Utils


parser = argparse.ArgumentParser()

parser.add_argument('--local_files', dest='local_files', action='store_true',
    help='Flag specifying to read data from the local elephant_dataset directory.'
    'The default is to read from the quatro data directory.')
parser.add_argument('--save_local', dest='save_local', action='store_true',
    help='Flag specifying to save model run information to the local models directory.'
    'The default is to save to the quatro data directory.')
parser.add_argument('--pre_train', type=str, 
    help='Specifies the model path for the pre-trained model')
parser.add_argument('--use_generated', actions='store_true', 
    help="Use generated positive data in addition to just the positive data")
parser.add_argument('--generated_path', type=str,
    help="Path to generated positive data")


"""
    What needs to happen here???

    - First let us write it like a main file. However, later I think that having it set up like a class 
    will be especially useful when we do two stage or curriculum training! Namely, since we can track 
    internal states as we go!

    - But for now let us do a couple of things

    1) Load the dataset that we want!
    2) Create the model
    3) Get the loss, optimizer, and schedular
    4) Create a training class
    5) Run train!
"""


def main():
    args = parser.parse_args()

    # Step 1) Get the paths to the training and test datafolders
    train_data_path, test_data_path = Model_Utils.get_dataset_paths(local_files=args.local_files)
    
    # Step 2) Create the train and test dataset 
    # WE SHOULD DO SOMETHING IF WE WANT TO INCORPERATE GENERATED DATA!!!!
    # Add here including the generated data!
    # To the dataset class:
    #   - Function to include new set of positive data!
    #   - Potentially re-randomly sample negative data to match ratio!
    train_dataset = Subsampled_ElephantDataset(train_data_path, neg_ratio=parameters.NEG_SAMPLES, 
                                        normalization=parameters.NORM, log_scale=parameters.SCALE, seed=8)
    if args.use_generated:
        train_dataset.add_positive_examples_from_dir(args.generated_path)
        train_dataset.undersample_negative_features_to_balance()

    test_dataset = Subsampled_ElephantDataset(test_data_path, neg_ratio=parameters.TEST_NEG_SAMPLES, 
                                        normalization=parameters.NORM, log_scale=parameters.SCALE, seed=8)

    # Step 3) Create the dataloaders
    train_loader = Model_Utils.get_loader(train_dataset, parameters.BATCH_SIZE, shuffle=True)
    test_loader = Model_Utils.get_loader(test_dataset, parameters.BATCH_SIZE, shuffle=False)

    dataloaders = {'train':train_loader, 'valid':test_loader}

    # Step 4) 
    save_path = Model_Utils.create_save_path(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()), args.save_local)

    ## Training - INCLUDE PRE-train later!1
    # Step 5) Create the model!
    model = get_model(parameters.MODEL_ID)
    model.to(parameters.device)
    print(model)

    # Step 6) Set up the tensorboard summary writer
    writer = SummaryWriter(save_path)
    writer.add_scalar('batch_size', parameters.BATCH_SIZE)
    writer.add_scalar('weight_decay', parameters.HYPERPARAMETERS[parameters.MODEL_ID]['l2_reg'])

    # Step 7) Get the loss function. The loss.py could be cleaner
    loss_func, _ = get_loss()


    # Step 8) Create the optimizer and schedular
    # Honestly probably do not need to have hyper-parameters per model, but leave it for now.
    optimizer = torch.optim.Adam(model.parameters(), lr=parameters.HYPERPARAMETERS[parameters.MODEL_ID]['lr'],
                                 weight_decay=parameters.HYPERPARAMETERS[parameters.MODEL_ID]['l2_reg'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, parameters.HYPERPARAMETERS[parameters.MODEL_ID]['lr_decay_step'], 
                                            gamma=parameters.HYPERPARAMETERS[parameters.MODEL_ID]['lr_decay'])

    # Step 9) Train the model!
    start_time = time.time()

    train_pipeline = Train_Pipeline(dataloaders, model, loss_func, optimizer, 
                scheduler, writer, save_path, early_stop_criteria=parameters.TRAIN_MODEL_SAVE_CRITERIA.lower())
    model_wts = train_pipeline.train(parameters.NUM_EPOCHS)

    if model_wts:
        model.load_state_dict(model_wts)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_path = save_path + "/" + "model.pt"
        torch.save(model, save_path)
        print('Saved best model based on {} to path {}'.format(parameters.TRAIN_MODEL_SAVE_CRITERIA.upper(), save_path))
    else:
        print('For some reason I don\'t have a model to save')

    print('Training time: {:10f} minutes'.format((time.time()-start_time)/60))

    writer.close()


if __name__ == '__main__':
    main()


