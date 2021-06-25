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


# Local file imports
import parameters
from models import get_model
from loss import get_loss
from train import Train_Pipeline
from curriculum_train import Curriculum_Strategy
from model_utils import Model_Utils
from datasets import Subsampled_ElephantDataset, Full_ElephantDataset


parser = argparse.ArgumentParser()

parser.add_argument('--local_files', dest='local_files', action='store_true',
    help='Flag specifying to read data from the local elephant_dataset directory.'
    'The default is to read from the quatro data directory.')
parser.add_argument('--save_local', dest='save_local', action='store_true',
    help='Flag specifying to save model run information to the local models directory.'
    'The default is to save to the quatro data directory.')
parser.add_argument('--pre_train', type=str, 
    help='Specifies the model path for the pre-trained model')
parser.add_argument('--use_generated', action='store_true', 
    help="Use generated positive data in addition to just the positive data")
parser.add_argument('--generated_path', type=str, default=None,
    help="Path to generated positive data if we want to use it!")


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

def read_adversarial_files(adversarial_file_path):
    """
        Read in the contents of a previously created train and test adversarial file!!
    """
    # We should also consider trying using the examples from "BEST_HIERARCH"
    # Step 1) Read in the training adversarial files
    adversarial_files = []
    with open(adversarial_file_path, 'r') as f:
        files = f.readlines()
        for file_pair in files:
            # Split by ', ' to get the data and the label
            file_pair = file_pair.strip()
            split_pair = file_pair.split(', ')
            adversarial_files.append((split_pair[0], split_pair[1]))


    return adversarial_files


def main():
    args = parser.parse_args()

    # What datasets to we need for curriculum learning
    # The training dataset starts just with the subsampled
    # training dataset that would be used for solo / 2-stage model
    # We need the full training set as well.
    # and we need the test_set to be some test set. For now we will
    # just use the randomly sampled test_set. Later we can use
    # the 2-stage model

    # Step 1) Create the training model that we will pass to the curriculum training engine

    # Step  1) Get the paths to the training dataset that will be dynamically
    # updated by sir curriculum
    train_data_path, test_data_path = Model_Utils.get_dataset_paths(local_files=args.local_files)

    # Step 2) Create the subsampled datasets
    train_dataset = Subsampled_ElephantDataset(train_data_path, neg_ratio=parameters.NEG_SAMPLES, 
                                        normalization=parameters.NORM, log_scale=parameters.SCALE, 
                                        gaussian_smooth=parameters.LABEL_SMOOTH, seed=8)
    

    test_dataset = Subsampled_ElephantDataset(test_data_path, neg_ratio=parameters.TEST_NEG_SAMPLES, 
                                        normalization=parameters.NORM, log_scale=parameters.SCALE, 
                                        gaussian_smooth=parameters.LABEL_SMOOTH, seed=8)
    # For the test dataset, we inject the hard negative adversarial 
    # examples discovered during the 2 stage model learning process
    # NOTE: For local test do not include this for now
    if not args.local_files:
        adversarial_files = read_adversarial_files(parameters.ADVERSARIAL_TEST_FILES)
        test_dataset.add_hard_neg_examples(adversarial_files, combine_data=True)


    # Step 3) Get the full training dataset to supervise the curriculum model
    full_train_dataset = Full_ElephantDataset(train_data_path, only_negative=True,
                                                        normalization=parameters.NORM, 
                                                        log_scale=parameters.SCALE, 
                                                        gaussian_smooth=False, seed=8)

    # Step 4) Get the dataloaders
    train_loader = Model_Utils.get_loader(train_dataset, parameters.BATCH_SIZE, shuffle=True)
    test_loader = Model_Utils.get_loader(test_dataset, parameters.BATCH_SIZE, shuffle=False)
    full_train_loader = Model_Utils.get_loader(full_train_dataset, parameters.BATCH_SIZE, shuffle=False)

    dataloaders = {
                    'train': train_loader,
                    'valid': test_loader,
                    'full_train_loader': full_train_loader
                    }

    # Step 5) Create the save path maybe? Unclear for now.we should create this for now!
    # Probably need to have a new save path type 
    save_path = Model_Utils.create_save_path(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()), args.save_local, save_prefix='Curriculum_')
    print ("save_path", save_path)

    ## Training - INCLUDE PRE-train later!1
    # Step 6) Create the model!
    model = get_model(parameters.MODEL_ID)
    model.to(parameters.device)
    print(model)

    # Step 7) Set up the tensorboard summary writer
    writer = SummaryWriter(save_path)
    writer.add_scalar('batch_size', parameters.BATCH_SIZE)
    writer.add_scalar('weight_decay', parameters.HYPERPARAMETERS[parameters.MODEL_ID]['l2_reg'])

    # Step 8) Get the loss function. The loss.py could be cleaner
    loss_func, _ = get_loss()


    # Step 9) Create the optimizer and schedular
    # Honestly probably do not need to have hyper-parameters per model, but leave it for now.
    optimizer = torch.optim.Adam(model.parameters(), lr=parameters.HYPERPARAMETERS[parameters.MODEL_ID]['lr'],
                                 weight_decay=parameters.HYPERPARAMETERS[parameters.MODEL_ID]['l2_reg'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, parameters.HYPERPARAMETERS[parameters.MODEL_ID]['lr_decay_step'], 
                                            gamma=parameters.HYPERPARAMETERS[parameters.MODEL_ID]['lr_decay'])

    # Step 10) Bring everything together to create the training pipeline that the 
    # curriculum model supervises
    start_time = time.time()

    train_pipeline = Train_Pipeline(dataloaders, model, loss_func, optimizer, 
                scheduler, writer, save_path, early_stop_criteria=parameters.TRAIN_MODEL_SAVE_CRITERIA.lower())

    # Step 11) Create and run the curriculum pipeline!
    curriculum_strategy = Curriculum_Strategy(train_pipeline, dataloaders, save_path, 
                                    num_epochs_per_era=parameters.NUM_EPOCHS_PER_ERA, eras=parameters.ERAS,  
                                    rand_keep_ratio=parameters.RAND_KEEP_RATIO, hard_keep_ratio=parameters.HARD_KEEP_RATIO, 
                                    hard_vs_rand_ratio=parameters.HARD_VS_RAND_RATIO, 
                                    hard_increase_factor=parameters.HARD_INCREASE_FACTOR,
                                    hard_vs_rand_ratio_max=parameters.HARD_VS_RAND_RATIO_MAX,
                                    hard_sample_size_factor=parameters.HARD_SAMPLE_SIZE_FACTOR)
    model_wts = curriculum_strategy.curriculum_train()

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


