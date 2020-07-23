import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
from torchsummary import summary
import time
from tensorboardX import SummaryWriter
import sys
import copy
import os

import parameters
import argparse
from visualization import visualize
import torchvision.models as models
from torchvision import transforms
import matplotlib
import matplotlib.pyplot as plt
from collections import deque

from models import get_model
from loss import get_loss
from train import train
from utils import create_save_path, create_dataset_path
from data import get_loader, get_loader_fuzzy

parser = argparse.ArgumentParser()

parser.add_argument('--local_files', dest='local_files', action='store_true',
    help='Flag specifying to read data from the local elephant_dataset directory.'
    'The default is to read from the quatro data directory.')
parser.add_argument('--save_local', dest='save_local', action='store_true',
    help='Flag specifying to save model run information to the local models directory.'
    'The default is to save to the quatro data directory.')


def main():
    args = parser.parse_args()

    # Use the 24hr datasets!

    if args.local_files:
        if parameters.LOSS.upper() == "FOCAL":
            train_data_path = parameters.LOCAL_FULL_TRAIN
            test_data_path = parameters.LOCAL_FULL_TEST
        else:
            train_data_path = parameters.LOCAL_TRAIN_FILES
            test_data_path = parameters.LOCAL_TEST_FILES
    else:
        if parameters.LOSS.upper() == "FOCAL":
            train_data_path = parameters.REMOTE_FULL_TRAIN
            test_data_path = parameters.REMOTE_FULL_TEST
        else:
            train_data_path = parameters.REMOTE_TRAIN_FILES
            test_data_path = parameters.REMOTE_TEST_FILES

    if parameters.LOSS.upper() != "FOCAL":
        train_data_path, _ = create_dataset_path(train_data_path, neg_samples=parameters.NEG_SAMPLES, 
                                                                    call_repeats=parameters.CALL_REPEATS, 
                                                                    shift_windows=parameters.SHIFT_WINDOWS)
        test_data_path, _ = create_dataset_path(test_data_path, neg_samples=parameters.TEST_NEG_SAMPLES, 
                                                                call_repeats=1)
    else:
        if parameters.SHIFT_WINDOWS:
            train_data_path += '_OversizeCalls'
    
    
    train_loader = get_loader_fuzzy(train_data_path, parameters.BATCH_SIZE, random_seed=parameters.DATA_LOADER_SEED, 
                                        norm=parameters.NORM, scale=parameters.SCALE, 
                                        shift_windows=parameters.SHIFT_WINDOWS, full_window_predict=True)
    test_loader = get_loader_fuzzy(test_data_path, parameters.BATCH_SIZE, random_seed=parameters.DATA_LOADER_SEED, 
                                        norm=parameters.NORM, scale=parameters.SCALE, full_window_predict=True)

    # For now model 18 signifies this!!!
    save_path = create_save_path(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()), args.save_local)

    dloaders = {'train':train_loader, 'valid':test_loader}

    ## Training
    model = get_model(parameters.MODEL_ID)
    model.to(parameters.device)

    print(model)

    writer = SummaryWriter(save_path)
    writer.add_scalar('batch_size', parameters.BATCH_SIZE)
    writer.add_scalar('weight_decay', parameters.HYPERPARAMETERS[parameters.MODEL_ID]['l2_reg'])

    loss_func, include_boundaries = get_loss()

    # Honestly probably do not need to have hyper-parameters per model, but leave it for now.
    optimizer = torch.optim.Adam(model.parameters(), lr=parameters.HYPERPARAMETERS[parameters.MODEL_ID]['lr'],
                                 weight_decay=parameters.HYPERPARAMETERS[parameters.MODEL_ID]['l2_reg'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, parameters.HYPERPARAMETERS[parameters.MODEL_ID]['lr_decay_step'], 
                                            gamma=parameters.HYPERPARAMETERS[parameters.MODEL_ID]['lr_decay'])

    start_time = time.time()
    model_wts = None

    model_wts = train(dloaders, model, loss_func, optimizer, scheduler, 
                    writer, parameters.NUM_EPOCHS, include_boundaries=include_boundaries)

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