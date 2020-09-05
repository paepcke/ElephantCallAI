from tensorboardX import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
import sys
import time
import os
import argparse

import parameters
from data import get_loader, get_loader_fuzzy
from utils import create_save_path, create_dataset_path
from models import * # Note for some reason we need to import the models as well
from loss import get_loss
from train import train_curriculum

### THINGS THAT I WANT TO DO
"""
    Where do we want to start writing this code:
        - We can still leverage the train_epoch and val_epoch functions in 
        train.py. Note later if we really want to do the batches or iteration
        level than we can re-write this. Overall, the train.py file should
        really just be responsible for either training one epoch, training
        several epochs, or training several iterations. A separate file 
        should be responsible for re-sampling the data and then calling
        the necessary train functions. 
        - In this class we should define the outward framework for doing the curriculum
        learning, including the curriculum scheduling and curriculum defining
        - Methods to write here:
        ...
    Look at some profiling:
        - After five epochs keep a histogram of how many get x segments incorrect
        - keep track of the variance of number of wrong for each example! Namely,
        for each example, see how many wrong were for that example after 
        5, 10, 15, 20, 25 epochs and then calculate the # wrong variance.
        - Also keep track of the variance of "avg" confidence like in focal loss
        - Methods to write:
            - Profiling method that trains a model for 5 then 10 then 15 then ...
            epochs and at each time calls a helper method that runs through the
            full training data to compute statistics based on the full training data.
            - Full data "scoring" statistics computation. Takes some training model
            and runs it over the full dataset to compute per window statistics such
            as:
                - the number of incorrect chunks
                - the avg prediction cofidence for the correct slice class (i.e. 
                think the chunk focal loss)
                - 
"""
parser = argparse.ArgumentParser()

parser.add_argument('--local_files', dest='local_files', action='store_true',
    help='Flag specifying to read data from the local elephant_dataset directory.'
    'The default is to read from the quatro data directory.')
parser.add_argument('--save_local', dest='save_local', action='store_true',
    help='Flag specifying to save model run information to the local models directory.'
    'The default is to save to the quatro data directory.')

# Just so numpy does not print rediculously un-readible stuff
np.set_printoptions(precision=2)


def model_statistics(model, full_dataloaders, threshold=0.5):
    """
        Full data "scoring" statistics computation. Takes a model
        and runs it over the full datasets to compute per window statistics such
        as:
            - the number of incorrect chunks
            - the avg prediction cofidence for the correct slice class (i.e. 
            think the chunk focal loss)

        NOTE: Make sure these are not shuffled datasets!
    """
    # Used for computing the avg of 1 - correct class pred probabilities
    bce = nn.BCEWithLogitsLoss(reduction='none')
    total_window_errors = {'train': np.zeros(0), 'valid': np.zeros(0)}
    total_window_inv_avg_predictions = {'train': np.zeros(0), 'valid': np.zeros(0)}
    for phase in ['train', 'valid']:
        dataloader = full_dataloaders[phase]
        # Run the model over the data
        print ("Num batches:", len(dataloader))
        for idx, batch in enumerate(dataloader):
            if idx % 1000 == 0:
                print("Gone through {} batches".format(idx))
            
            inputs = batch[0].clone().float()
            labels = batch[1].clone().float()
            inputs = inputs.to(parameters.device)
            labels = labels.to(parameters.device)
            
            # ONLY Squeeze the last dim!
            logits = model(inputs).squeeze(-1) # Shape - (batch_size, seq_len)

            # Now for each chunk we want to see whether it should be flagged as 
            # a true false positive. For now do "approx" by counting number pos samples
            predictions = torch.sigmoid(logits)
            # Pre-compute the number of pos. slices in each chunk
            # Threshold the predictions - May add guassian blur
            binary_preds = torch.where(predictions > threshold, torch.tensor(1.0).to(parameters.device), torch.tensor(0.0).to(parameters.device))
            
            window_errors = torch.sum(binary_preds != labels, axis = 1).cpu().detach().numpy()
            total_window_errors[phase] = np.concatenate((total_window_errors[phase], window_errors))

            # Get for each chunk the pred prob for the correct class
            bce_loss = bce(logits, labels)    
            pts = torch.exp(-bce_loss)
            # Now the difficulty is 1 - pts
            # i.e. hard examples have high hardness score as
            # the model is not confident for many slices (low pts)
            # so (1-low) = high
            window_inv_avg_predictions = torch.mean(1 - pts, axis = 1).cpu().detach().numpy()
            total_window_inv_avg_predictions[phase] = np.concatenate((total_window_inv_avg_predictions[phase], window_inv_avg_predictions))

        #total_window_errors[phase] = np.expand_dims(total_window_errors[phase], axis=0)
        #total_window_inv_avg_predictions[phase] = np.expand_dims(total_window_inv_avg_predictions[phase], axis=0)

    # Note for ease of concatenation later expand the second dim!
    stats = {'window_errors': total_window_errors,
             'window_inv_avg_predictions': total_window_inv_avg_predictions}
    
    return stats   
       



def curriculum_profiling(model, train_dataloaders, full_dataloaders, loss_func, optimizer, 
                        scheduler, writer, include_boundaries=False):
    """
        Trains a model for 5 then 10 then 15 then ... epochs 
        and at each time calls a helper method that runs through the
        full training data to compute statistics based on the full training data.
    """
    # Things to profile
    curriculum_file = '../Curriculum_profiling/'
    train_window_errors = None
    train_inv_avg_predictions = None
    test_window_errors = None
    test_inv_avg_predictions = None
    # Train 5, 10, 15, 20, 25 epochs
    for i in range(1):
        # In train curriculum, for now do not return model based on best performance
        # but simply return the model at the end of that training loop
        model_weights = train_curriculum(model, train_dataloaders, loss_func, optimizer,
                                        scheduler, writer, epochs=5, include_boundaries=include_boundaries)
        # Technically model will already have the weights we want since we are returning
        # the model weights after 5 epochs not the best epoch run; however, maybe later this
        # will change
        model.load_state_dict(model_weights)

        # Profile the model over the full training dataset and test dataset to see 
        # window difficulties and variations.
        model_stats = model_statistics(model, full_dataloaders)
        train_window_error_i = np.expand_dims(model_stats['window_errors']['train'], axis=0)
        train_inv_avg_prediction_i = np.expand_dims(model_stats['window_inv_avg_predictions']['train'], axis=0)
        test_window_error_i = np.expand_dims(model_stats['window_errors']['valid'], axis=0)
        test_inv_avg_prediction_i = np.expand_dims(model_stats['window_inv_avg_predictions']['valid'], axis=0)
        if i == 0:
            train_window_errors = train_window_error_i
            train_inv_avg_predictions = train_inv_avg_prediction_i
            test_window_errors = test_window_error_i
            test_inv_avg_predictions = test_inv_avg_prediction_i
        else:
            # Concatenate these together so that we can get std info
            train_window_errors = np.concatenate((train_window_errors, train_window_error_i))
            train_inv_avg_predictions = np.concatenate((train_inv_avg_predictions, train_inv_avg_prediction_i))
            test_window_errors = np.concatenate((test_window_errors, test_window_error_i))
            test_inv_avg_predictions = np.concatenate((test_inv_avg_predictions, test_inv_avg_prediction_i))

        # Save the histograms so that we can open them in jupyter
        print ("Saving Histograms for Iteration i:", i)
        # Number of incorrect slices distribution
        n, bins, _ = plt.hist(train_window_error_i[0], bins=25)
        plt.title('Train - Number incorrect slices iteration' + str((i + 1) * 5))
        plt.savefig(curriculum_file + "Train_Num_Incorrect_i-" + str((i+1) * 5) + ".png")
        # Print out to visually inspect
        print ('Train - Number incorrect slices iteration' + str((i + 1) * 5))
        print ('Vals:', n)
        print ('Bins:', bins)
        print ('Number Incorrect > 15:', np.sum(train_window_error_i[0] > 15))
        print ('Number Incorrect > 25:', np.sum(train_window_error_i[0] > 25))
        print('------------------------------')
        plt.clf()
    
        n, bins, _ = plt.hist(test_window_error_i[0], bins=25)
        plt.title('Valid - Number incorrect slices iteration' + str((i + 1) * 5))
        plt.savefig(curriculum_file + "Valid_Num_Incorrect_i-" + str((i+1) * 5) + ".png")
        print ('Valid - Number incorrect slices iteration' + str((i + 1) * 5))
        print ('Vals:', n)
        print ('Bins:', bins)
        print ('Number Incorrect > 15:', np.sum(test_window_error_i[0] > 15))
        print ('Number Incorrect > 25:', np.sum(train_window_error_i[0] > 25))
        print('------------------------------')
        plt.clf()

        # 1 - avg. prediction confidence distribution
        n, bins, _ = plt.hist(train_inv_avg_prediction_i[0], bins=25)
        plt.title('Train - (1 - avg. prediction confidence) iteration' + str((i + 1) * 5))
        plt.savefig(curriculum_file  + "Train_pred_condfidence_i-" + str((i+1) * 5) + ".png")
        print ('Train - (1 - avg. prediction confidence)  iteration' + str((i + 1) * 5))
        print ('Vals:', n)
        print ('Bins:', bins)
        print('------------------------------')
        plt.clf()
    
        n, bins, _ = plt.hist(test_inv_avg_predictions[0], bins=25)
        plt.title('Valid - (1 - avg. prediction confidence) iteration' + str((i + 1) * 5))
        plt.savefig(curriculum_file + "Valid_pred_condfidence_i-" + str((i+1) * 5) + ".png")
        print ('Valid - (1 - avg. prediction confidence) iteration' + str((i + 1) * 5))
        print ('Vals:', n)
        print ('Bins:', bins)
        print('------------------------------')
        plt.clf()

        # Look at the distribution of variances across the 
        # trails until now!
        if i != 0:
            # Now do calculations of the variance and shit
            # Let us do this part a bit later!
            std_train_window_errors = np.std(train_window_errors, axis=0)
            std_train_inv_avg_predictions = np.std(train_inv_avg_predictions, axis=0)
            std_test_window_errors = np.std(test_window_errors, axis=0)
            std_test_inv_avg_predictions = np.std(test_inv_avg_predictions, axis=0)

            n, bins, _ = plt.hist(std_train_window_errors, bins=20)
            plt.title('Train - STD incorrect slices after iteration' + str((i + 1) * 5))
            plt.savefig(curriculum_file  + "Train_std_window_errors_i-" + str((i+1) * 5) + ".png")
            print ('Train - STD incorrect slices after iteration' + str((i + 1) * 5))
            print ('Vals:', n)
            print ('Bins:', bins)
            print('------------------------------')
            plt.clf()
            
            n, bins, _ = plt.hist(std_train_inv_avg_predictions, bins=20)
            plt.title('Train - STD (1 - avg. prediction confidence) after iteration' + str((i + 1) * 5))
            plt.savefig(curriculum_file  + "Train_std_pred_condfidence_i-" + str((i+1) * 5) + ".png")
            print ('Valid - STD (1 - avg. prediction confidence) after iteration' + str((i + 1) * 5))
            print ('Vals:', n)
            print ('Bins:', bins)
            print('------------------------------')
            plt.clf()

    # We should also save the actual saved stats to look at later!
    np.save(curriculum_file + 'train_window_errors', train_window_errors)
    np.save(curriculum_file + 'train_inv_avg_predictions', train_inv_avg_predictions)
    np.save(curriculum_file + 'test_window_errors', test_window_errors)
    np.save(curriculum_file + 'test_inv_avg_predictions', test_inv_avg_predictions)
    print ("Completed")

    

def main():
    args = parser.parse_args()


    if args.local_files:
        train_data_path = parameters.LOCAL_TRAIN_FILES
        test_data_path = parameters.LOCAL_TEST_FILES
        full_train_path = parameters.LOCAL_FULL_TRAIN
        full_test_path = parameters.LOCAL_FULL_TEST
    else:
        if parameters.DATASET.lower() == "noab":
            train_data_path = parameters.REMOTE_TRAIN_FILES
            test_data_path = parameters.REMOTE_TEST_FILES
            full_train_path = parameters.REMOTE_FULL_TRAIN
            full_test_path = parameters.REMOTE_FULL_TEST
        else:
            train_data_path = parameters.REMOTE_BAI_TRAIN_FILES
            test_data_path = parameters.REMOTE_BAI_TEST_FILES
            full_train_path = parameters.REMOTE_FULL_TRAIN_BAI
            full_test_path = parameters.REMOTE_FULL_TEST_BAI

    
    
    train_data_path, include_boundaries = create_dataset_path(train_data_path, neg_samples=parameters.NEG_SAMPLES, 
                                                                    call_repeats=parameters.CALL_REPEATS, 
                                                                    shift_windows=parameters.SHIFT_WINDOWS)
    test_data_path, _ = create_dataset_path(test_data_path, neg_samples=parameters.TEST_NEG_SAMPLES, 
                                                                call_repeats=1)
    
    train_loader = get_loader_fuzzy(train_data_path, parameters.BATCH_SIZE, random_seed=parameters.DATA_LOADER_SEED, 
                                        norm=parameters.NORM, scale=parameters.SCALE, 
                                        include_boundaries=include_boundaries, shift_windows=parameters.SHIFT_WINDOWS)
    test_loader = get_loader_fuzzy(test_data_path, parameters.BATCH_SIZE, random_seed=parameters.DATA_LOADER_SEED, 
                                        norm=parameters.NORM, scale=parameters.SCALE, include_boundaries=include_boundaries)

    # For now we don't need to save the model
    save_path = create_save_path(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()), args.save_local)

    train_dataloaders = {'train':train_loader, 'valid':test_loader}

    # Load the full data sets - SET SHUFFLE = False
    full_train_loader = get_loader_fuzzy(full_train_path, parameters.BATCH_SIZE, shuffle=False, 
                                        norm=parameters.NORM, scale=parameters.SCALE, 
                                        include_boundaries=False, shift_windows=False,
                                        is_full_dataset=True)
    full_test_loader = get_loader_fuzzy(full_test_path, parameters.BATCH_SIZE, shuffle=False, 
                                        norm=parameters.NORM, scale=parameters.SCALE, include_boundaries=False)
    full_dataloaders = {'train':full_train_loader, 'valid': full_test_loader}

    
    model = get_model(parameters.MODEL_ID)
    model.to(parameters.device)

    print(model)

    writer = SummaryWriter(save_path)
    writer.add_scalar('batch_size', parameters.BATCH_SIZE)
    writer.add_scalar('weight_decay', parameters.HYPERPARAMETERS[parameters.MODEL_ID]['l2_reg'])

    # Want to use focal loss! Next thing to check on!
    loss_func, include_boundaries = get_loss()

    # Honestly probably do not need to have hyper-parameters per model, but leave it for now.
    optimizer = torch.optim.Adam(model.parameters(), lr=parameters.HYPERPARAMETERS[parameters.MODEL_ID]['lr'],
                                 weight_decay=parameters.HYPERPARAMETERS[parameters.MODEL_ID]['l2_reg'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, parameters.HYPERPARAMETERS[parameters.MODEL_ID]['lr_decay_step'], 
                                            gamma=parameters.HYPERPARAMETERS[parameters.MODEL_ID]['lr_decay'])

    start_time = time.time()

    curriculum_profiling(model, train_dataloaders, full_dataloaders, loss_func, optimizer, scheduler, writer)

    print('Training time: {:10f} minutes'.format((time.time()-start_time)/60))

    writer.close()


if __name__ == '__main__':
    main()




