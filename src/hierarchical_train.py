from tensorboardX import SummaryWriter
import numpy as np
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
from models import get_model
from loss import get_loss
from train import train


parser = argparse.ArgumentParser()

parser.add_argument('--local_files', dest='local_files', action='store_true',
    help='Flag specifying to read data from the local elephant_dataset directory.'
    'The default is to read from the quatro data directory.')
parser.add_argument('--save_local', dest='save_local', action='store_true',
    help='Flag specifying to save model run information to the local models directory.'
    'The default is to save to the quatro data directory.')
parser.add_argument('--full', dest='full_pipeline', action='store_true',
    help='Flag specifying to run the full hierarchical model pipeline.')
parser.add_argument('--adversarial', dest='adversarial', action='store_true',
    help='Flag specifying to generate a new set of adversarial_examples.')
parser.add_argument('--model_1', dest='model1', action='store_true',
    help='Flag specifying to just train Model_1.')
parser.add_argument('--models_path', type=str,
    help='When running \'adversarial\' or \'model1\' we must provide the folder with model_0')

"""
    General approach ideas. Train the first model, we will call this model
    the noise detector. Then for now let us run the model over the full train
    dataset (note this is not maybe as good / easy as running over the actual
    full spectograms)! For all of the samples that are false positives (for now 
    define false positive as > x 1's predicted) keep track of those samples. 
    Lastly, write a function for the dataset function that either re-initializes
    the negative samples of the dataset, or creates a new dataset! For now let
    us try the re-initializes the negative samples:
"""
# Option 2:
#   We have multiple options in this file that given different flags does different things.
#   1) the option to do the full pipeline. Train Model_0 save it, get the adversarial examples
#   and write a file containing the adversarial files and the threshold used, and finally train
#   Model_1. Note, we should experiment with both training a Model_1 from scratch and 
#   continuing to train Model_0 but on the new dataset. Overall we need to create a parameters
#   variable that dictates what model to use for Model_1.
#
#   2) Allow for given Model_0 get a new set of adversarial examples with for example a different
#   threhold! 
#
#   3) Train just Model_1. This involves initializing the datasets as if they were for Model_0
#   but then replacing the negative samples with the saved negative samples in the adversarial
#   examples file!
#
#   NOTE: The parameter file specifications must match that of Model_0 in order to execute option
#   2 and 3 as expected.


def adversarial_discovery_helper(dataloader, model, min_length, threshold=0.5, num_files_to_return=-1):
    """
        Given a trained model, we identify and save false negative data chunks. 
        These false negatives are then used as the negative examples for the 
        training of a second heirarchical model. By default return all false
        positive data chunks (i.e. num_files_to_return = -1). 
        We define a false positive data chunk "loosely" for now as having 
        more than 'min_length' predicted slices (prediction = 1)
    """
    # Note there may be edge cases where an adversarial example exists right
    # near an elephant call and is not included in the training dataset because
    # of the way that the chunks are created for training. i.e. the chunks in 
    # the training dataset may not have included the adversarial examples, but
    # when creating chunks for the 24hrs the chunks may be aligned differently
    adversarial_examples = []
    # Put in eval mode!!
    model.eval()
    print ("Num batches:", len(dataloader))
    for idx, batch in enumerate(dataloader):
        if idx % 1000 == 0:
            print("Adversarial search has gotten through {} batches".format(idx))
        # Allows for subsampling of adversarial examples.
        # -1 indicates collect all
        if num_files_to_return != -1 and len(adversarial_examples) >= num_files_to_return:
            break

        inputs = batch[0].clone().float()
        labels = batch[1].clone().float()
        inputs = inputs.to(parameters.device)
        labels = labels.to(parameters.device)
        # Get the data_file locations for each chunk
        data_files = np.array(batch[2])

        logits = model(inputs).squeeze() # Shape - (batch_size, seq_len)

        # Now for each chunk we want to see whether it should be flagged as 
        # a true false positive. For now do "approx" by counting number pos samples
        predictions = torch.sigmoid(logits)
        # Pre-compute the number of pos. slices in each chunk
        # Threshold the predictions - May add guassian blur
        binary_preds = torch.where(predictions > threshold, torch.tensor(1.0).to(parameters.device), torch.tensor(0.0).to(parameters.device))
        pred_counts = torch.sum(binary_preds, dim=1).squeeze().cpu().detach().numpy() # Shape - (batch_size)
        # Get ground truth label counts
        gt_counts = torch.sum(labels, dim=1).cpu().detach().numpy() # Shape - (batch_size)
        
        # We want to look for chunks that have gt_counts = 0
        # and pred_counts > min_length. Create masks for each
        gt_empty = (gt_counts == 0)
        predicted_chunks = (pred_counts >= min_length)

        epoch_adversarial_examples = list(data_files[gt_empty & predicted_chunks])
        adversarial_examples += epoch_adversarial_examples

        # Visualize every 100 selected examples
        # NEED to figure this out a bit
        if parameters.VERBOSE:
            adversarial_features = inputs[torch.tensor(gt_empty & predicted_chunks)]
            adversarial_predictions = predictions[torch.tensor(gt_empty & predicted_chunks)]
            adversarial_label = labels[torch.tensor(gt_empty & predicted_chunks)]
            for idx, data_file in enumerate(epoch_adversarial_examples):
                if (idx + 1) % 100 == 0:
                    print ("Adversarial Example:", (idx + 1))
                    features = adversarial_features[idx].cpu().detach().numpy()
                    output = adversarial_predictions[idx].cpu().detach().numpy()
                    label = adversarial_label[idx].cpu().detach().numpy()

                    visualize(features, output, label, title=data_file)

    print (len(adversarial_examples))
    return adversarial_examples


def initialize_training(model_id, save_path):
    # The get_model method is in charge of 
    # setting the same seed for each loaded model.
    # Thus, for each inner loop we train the same initialized model
    # Load model_0 to continue training with it
    if str(model_id).lower() == 'same':
        final_slash  = save_path.rindex('/')
        model_0_path = os.path.join(save_path[:final_slash], "Model_0/model.pt")
        model = torch.load(model_0_path, map_location=parameters.device)
    else:
        model = get_model(model_id).to(parameters.device)

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

    return model, loss_func, include_boundaries, optimizer, scheduler, writer

def train_model_1(adversarial_train_files, adversarial_test_files, train_loader, test_loader, save_path):

    print ("++===============================++")
    print ("++Training Error Correcting Model++") 
    print ("++===============================++")
    # Update initialize_training to allow for loading back model_0!!
    # Update the negative examples of the training and validation datasets
    train_loader.dataset.set_neg_features(adversarial_train_files)
    test_loader.dataset.set_neg_features(adversarial_test_files)
    dloaders = {'train':train_loader, 'valid':test_loader}

    model_name = "Model_1_Type-" + str(parameters.HIERARCHICAL_MODEL) + '_CallRepeats-' + str(parameters.HIERARCHICAL_REPEATS).lower()
    # Add if we are using shifting windows
    if parameters.HIERARCHICAL_SHIFT_WINDOWS:
        model_name += '_OversizeCalls'

    second_model_save_path = os.path.join(save_path, model_name)
    if not os.path.exists(second_model_save_path):
            os.makedirs(second_model_save_path)

    # For now just use same model for 0 and 1
    start_time = time.time()
    model_1, loss_func, include_boundaries, optimizer, scheduler, writer = initialize_training(parameters.HIERARCHICAL_MODEL, 
                                                                                second_model_save_path)
    model_1_wts = train(dloaders, model_1, loss_func, optimizer, scheduler, 
                    writer, parameters.NUM_EPOCHS, include_boundaries=include_boundaries)

    if model_1_wts:
        model_1.load_state_dict(model_1_wts)
        model_save_path = os.path.join(second_model_save_path, "model.pt")
        torch.save(model_1, model_save_path)
        print('Saved best Model 1 based on {} to path {}'.format(parameters.TRAIN_MODEL_SAVE_CRITERIA.upper(), second_model_save_path))
    else:
        print('For some reason I don\'t have a model to save')

    print('Training time: {:10f} minutes'.format((time.time()-start_time)/60))
    writer.close()

def adversarial_discovery(full_train_path, full_test_path, model_0, save_path):
    """
        Collect the adversarial - false positives based on model_0
        for the train and validation set.
    """
    print ('++================================================++')
    print ("++ Beginning False Positive Adversarial Discovery ++")
    print ('++================================================++')
    # Do not include boundary uncertainty in full train loader. We only need the model predictions, we do not
    # calculate the loss! Use the HIERARCH_SHIFT flag along to decide if the Heirarchical model will use
    # randomly shifted windows. Note, we flag that this is the full dataset to make sure that during 
    # adversarial discovery we alwas sample the midlle of oversized windows
    full_train_loader = get_loader_fuzzy(full_train_path, parameters.BATCH_SIZE, random_seed=parameters.DATA_LOADER_SEED, 
                                        norm=parameters.NORM, scale=parameters.SCALE, 
                                        include_boundaries=False, shift_windows=parameters.HIERARCHICAL_SHIFT_WINDOWS,
                                        is_full_dataset=True)
    full_test_loader = get_loader_fuzzy(full_test_path, parameters.BATCH_SIZE, random_seed=parameters.DATA_LOADER_SEED, 
                                        norm=parameters.NORM, scale=parameters.SCALE, include_boundaries=False)

    # For now let us try including all of the false negatives!
    train_adversarial_file = "model_0-False_Pos_Train.txt"
    if parameters.HIERARCHICAL_SHIFT_WINDOWS:
        train_adversarial_file = "model_0-False_Pos_Train_Shift.txt"
    adversarial_train_files = adversarial_discovery_helper(full_train_loader, model_0, min_length=parameters.FALSE_NEGATIVE_THRESHOLD)
    adversarial_train_save_path = os.path.join(save_path, train_adversarial_file)
    with open(adversarial_train_save_path, 'w') as f:
        for file in adversarial_train_files:
            f.write('{}\n'.format(file))

    train_adversarial_file = "model_0-False_Pos_Test.txt"
    if parameters.HIERARCHICAL_SHIFT_WINDOWS:
        train_adversarial_file = "model_0-False_Pos_Test_Shift.txt"
    adversarial_test_files = adversarial_discovery_helper(full_test_loader, model_0, min_length=parameters.FALSE_NEGATIVE_THRESHOLD)
    adversarial_test_save_path = os.path.join(save_path, "model_0-False_Pos_Test.txt")
    with open(adversarial_test_save_path, 'w') as f:
        for file in adversarial_test_files:
            f.write('{}\n'.format(file))

    return adversarial_train_files, adversarial_test_files

def train_model_0(train_loader, test_loader, save_path):
    """
        Train the "sound" detector - Model_0
    """
    print ("++================================++")
    print ("++ Training initial call detector ++")
    print ("++================================++")
    first_model_save_path = os.path.join(save_path, "Model_0")
    if not os.path.exists(first_model_save_path):
            os.makedirs(first_model_save_path)

    dloaders = {'train':train_loader, 'valid':test_loader}
    start_time = time.time()
    model_0, loss_func, include_boundaries, optimizer, scheduler, writer = initialize_training(parameters.MODEL_ID, first_model_save_path)
    model_0_wts = train(dloaders, model_0, loss_func, optimizer, scheduler, 
                    writer, parameters.NUM_EPOCHS, include_boundaries=include_boundaries)

    if model_0_wts:
        model_0.load_state_dict(model_0_wts)
        model_save_path = os.path.join(first_model_save_path, "model.pt")
        torch.save(model_0, model_save_path)
        print('Saved best Model 0 based on {} to path {}'.format(parameters.TRAIN_MODEL_SAVE_CRITERIA.upper(), first_model_save_path))
    else:
        print('For some reason I don\'t have a model to save!!')
        quit()

    print('Training time: {:10f} minutes'.format((time.time()-start_time)/60))
    writer.close()

    # Check this, but it should have the trained model_0
    return model_0


def main():
    args = parser.parse_args()

    # What do we need to do across all of the settings!
    # Get the data loaders!
    args = parser.parse_args()

    if args.local_files:
        train_data_path = parameters.LOCAL_TRAIN_FILES
        test_data_path = parameters.LOCAL_TEST_FILES
        full_train_path = parameters.LOCAL_FULL_TRAIN
        full_test_path = parameters.LOCAL_FULL_TEST
    else:
        train_data_path = parameters.REMOTE_TRAIN_FILES
        test_data_path = parameters.REMOTE_TEST_FILES
        full_train_path = parameters.REMOTE_FULL_TRAIN
        full_test_path = parameters.REMOTE_FULL_TEST

    if parameters.HIERARCHICAL_SHIFT_WINDOWS:
            full_train_path += '_OversizeCalls'

    model_0_train_data_path, include_boundaries = create_dataset_path(train_data_path, neg_samples=parameters.NEG_SAMPLES, 
                                                                    call_repeats=parameters.CALL_REPEATS, 
                                                                    shift_windows=parameters.SHIFT_WINDOWS)
    model_0_test_data_path, _ = create_dataset_path(test_data_path, neg_samples=parameters.TEST_NEG_SAMPLES, 
                                                                call_repeats=1)
    

    # Check if a different dataset is being used for Model_1
    model_1_train_data_path = model_0_train_data_path
    model_1_test_data_path = model_0_test_data_path
    if str(parameters.HIERARCHICAL_REPEATS).lower() != "same":
        # SHould prob just have neg samples x1 since doesnt matter!!
        model_1_train_data_path, _ = create_dataset_path(train_data_path, neg_samples=parameters.NEG_SAMPLES, 
                                                        call_repeats=parameters.HIERARCHICAL_REPEATS,
                                                        shift_windows=parameters.HIERARCHICAL_SHIFT_WINDOWS)
    
    
    # Model 0 Loaders
    model_0_train_loader = get_loader_fuzzy(model_0_train_data_path, parameters.BATCH_SIZE, random_seed=parameters.DATA_LOADER_SEED, 
                                        norm=parameters.NORM, scale=parameters.SCALE, 
                                        include_boundaries=include_boundaries, shift_windows=parameters.SHIFT_WINDOWS)
    model_0_test_loader = get_loader_fuzzy(model_0_test_data_path, parameters.BATCH_SIZE, random_seed=parameters.DATA_LOADER_SEED, 
                                        norm=parameters.NORM, scale=parameters.SCALE, include_boundaries=include_boundaries)
    
    # Model 1 Loaders
    model_1_train_loader = get_loader_fuzzy(model_1_train_data_path, parameters.BATCH_SIZE, random_seed=parameters.DATA_LOADER_SEED, 
                                        norm=parameters.NORM, scale=parameters.SCALE, 
                                        include_boundaries=include_boundaries, shift_windows=parameters.HIERARCHICAL_SHIFT_WINDOWS)
    model_1_test_loader = get_loader_fuzzy(model_1_test_data_path, parameters.BATCH_SIZE, random_seed=parameters.DATA_LOADER_SEED, 
                                        norm=parameters.NORM, scale=parameters.SCALE, include_boundaries=include_boundaries)
    

    if args.models_path is None:
        save_path = create_save_path(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()), args.save_local, save_prefix='Hierarchical_')
    else:
        save_path = args.models_path

    # Case 1) Do the entire pipeline! Can break now the pipeline into 3 helper functions!
    if args.full_pipeline:
        # Train and save model_0
        #model_0 = train_model_0(dloaders, save_path)
        model_0 = train_model_0(model_0_train_loader, model_0_test_loader , save_path)
        # Do the adversarial discovery
        adversarial_train_files, adversarial_test_files = adversarial_discovery(full_train_path, full_test_path, model_0, save_path)
        # Train and save model 1
        train_model_1(adversarial_train_files, adversarial_test_files, model_1_train_loader, model_1_test_loader, save_path)
    
    # Just generate new adversarial examples 
    elif args.adversarial:
        # Load model_0
        model_0_path = os.path.join(save_path, "Model_0/model.pt")
        model_0 = torch.load(model_0_path, map_location=parameters.device)
        adversarial_discovery(full_train_path, full_test_path, model_0, save_path)

    # Train just model_1
    elif args.model1:
        # Read in the adversarial files
        adversarial_train_save_path = os.path.join(save_path, "model_0-False_Pos_Train.txt")
        adversarial_train_files = []
        with open(adversarial_train_save_path, 'r') as f:
            files = f.readlines()
            for file in files:
                adversarial_train_files.append(file.strip())

        adversarial_test_save_path = os.path.join(save_path, "model_0-False_Pos_Test.txt")
        adversarial_test_files = []
        with open(adversarial_test_save_path, 'r') as f:
            files = f.readlines()
            for file in files:
                adversarial_test_files.append(file.strip())

        train_model_1(adversarial_train_files, adversarial_test_files, model_1_train_loader, model_1_test_loader, save_path)

    else:
        print ("Invalid running mode!")




if __name__ == "__main__":
    main()

