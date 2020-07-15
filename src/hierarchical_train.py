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
from utils import create_save_path
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
parser.add_argument('--model_path', type=str,
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

    Things to do:
    1) In the dataset, keep track seperately of the positive and negative samples,
    as well as a combination of them.
    2) Write function for re-initializing the negative examples!
    3) Need to think about how to incorperate CALL REPEATS! Likely need to 
    save datasamples larger than 256 and then allow for a transform to be
    applied in the dataset class ---- or we could do the false positive 
    discovery over the full spectograms (though this is likely super slow)!!!
"""
# Want to have two modes!!! One train the whole pipeline. 2 train just the hierarchical
# part of the pipeline. First part is done! What is needed for the second part. 
# Option 1:
#   After training the "sound" detector and getting the adversarial examples, we could
#   Create a new dataset that is saved to elephant datasets and then can be used to directly
#   train a new model. Why I don't like this! One, we want to explicitly keep Model_0 
#   (the "sound" detector) and Model_1 (the "specialist") linked. Namely, Model_1 is 
#   explicitally trained based on Model_0's performance. Additionally, it feels a bit
#   much to create many datasets over this.
#
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

    # Train the first model!
    second_model_save_path = save_path + '/' + "Model_1_" + str(parameters.HIERARCHICAL_MODEL) 
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
        model_save_path = second_model_save_path + "/" + "model.pt"
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
    # calculate the loss!
    full_train_loader = get_loader_fuzzy(full_train_path, parameters.BATCH_SIZE, random_seed=parameters.DATA_LOADER_SEED, 
                                        norm=parameters.NORM, scale=parameters.SCALE, include_boundaries=False)
    full_test_loader = get_loader_fuzzy(full_test_path, parameters.BATCH_SIZE, random_seed=parameters.DATA_LOADER_SEED, 
                                        norm=parameters.NORM, scale=parameters.SCALE, include_boundaries=False)

    # For now let us try including all of the false negatives!
    adversarial_train_files = adversarial_discovery_helper(full_train_loader, model_0, min_length=parameters.FALSE_NEGATIVE_THRESHOLD)
    adversarial_train_save_path = save_path + "/" + "model_0-False_Pos_Train"+ ".txt"
    with open(adversarial_train_save_path, 'w') as f:
        for file in adversarial_train_files:
            f.write('{}\n'.format(file))

    adversarial_test_files = adversarial_discovery_helper(full_test_loader, model_0, min_length=parameters.FALSE_NEGATIVE_THRESHOLD)
    adversarial_test_save_path = save_path + "/" + "model_0-False_Pos_Test"+ ".txt"
    with open(adversarial_test_save_path, 'w') as f:
        for file in adversarial_test_files:
            f.write('{}\n'.format(file))

    return adversarial_train_files, adversarial_test_files

def train_model_0(dloaders, save_path):
    """
        Train the "sound" detector - Model_0
    """
    print ("++================================++")
    print ("++ Training initial call detector ++")
    print ("++================================++")
    first_model_save_path = save_path + '/' + "Model_0"
    if not os.path.exists(first_model_save_path):
            os.makedirs(first_model_save_path)

    start_time = time.time()
    model_0, loss_func, include_boundaries, optimizer, scheduler, writer = initialize_training(parameters.MODEL_ID, first_model_save_path)
    model_0_wts = train(dloaders, model_0, loss_func, optimizer, scheduler, 
                    writer, parameters.NUM_EPOCHS, include_boundaries=include_boundaries)

    if model_0_wts:
        model_0.load_state_dict(model_0_wts)
        model_save_path = first_model_save_path + "/" + "model.pt"
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

    train_data_path += 'Neg_Samples_x' + str(parameters.NEG_SAMPLES) + "_Seed_" + str(parameters.RANDOM_SEED) + \
                        "_CallRepeats_" + str(parameters.CALL_REPEATS)
    # Probably make call repeats and neg samples default to 1 for test data!!!!
    test_data_path += "Neg_Samples_x" + str(parameters.TEST_NEG_SAMPLES) + "_Seed_" + str(parameters.RANDOM_SEED) + \
                    "_CallRepeats_" + str(1)
    
    # Include boundary uncertainty in training
    include_boundaries = False
    if parameters.LOSS == "BOUNDARY":
        include_boundaries = True
        train_data_path += "_FudgeFact_" + str(parameters.BOUNDARY_FUDGE_FACTOR) + "_Individual-Boarders_" + str(parameters.INDIVIDUAL_BOUNDARIES)
        test_data_path += "_FudgeFact_" + str(parameters.BOUNDARY_FUDGE_FACTOR) + "_Individual-Boarders_" + str(parameters.INDIVIDUAL_BOUNDARIES)

    train_loader = get_loader_fuzzy(train_data_path, parameters.BATCH_SIZE, random_seed=parameters.DATA_LOADER_SEED, 
                                        norm=parameters.NORM, scale=parameters.SCALE, include_boundaries=include_boundaries)
    test_loader = get_loader_fuzzy(test_data_path, parameters.BATCH_SIZE, random_seed=parameters.DATA_LOADER_SEED, 
                                        norm=parameters.NORM, scale=parameters.SCALE, include_boundaries=include_boundaries)
    
    dloaders = {'train':train_loader, 'valid':test_loader}

    if args.model_path is None:
        save_path = create_save_path(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()), args.save_local, save_prefix='Hierarchical_')
    else:
        save_path = args.model_path

    # Case 1) Do the entire pipeline! Can break now the pipeline into 3 helper functions!
    if args.full_pipeline:
        # Train and save model_0
        model_0 = train_model_0(dloaders, save_path)
        # Do the adversarial discovery
        adversarial_train_files, adversarial_test_files = adversarial_discovery(full_train_path, full_test_path, model_0, save_path)
        # Train and save model 1
        train_model_1(adversarial_train_files, adversarial_test_files, train_loader, test_loader, save_path)
    
    # Just generate new adversarial examples 
    elif args.adversarial:
        # Load model_0
        model_0_path = os.path.join(save_path, "Model_0/model.pt")
        model_0 = torch.load(model_0_path, map_location=parameters.device)
        adversarial_discovery(full_train_path, full_test_path, model_0, save_path)

    # Train just model_1
    elif args.model1:
        # Read in the adversarial files
        adversarial_train_save_path = save_path + "/" + "model_0-False_Pos_Train"+ ".txt"
        adversarial_train_files = []
        with open(adversarial_train_save_path, 'r') as f:
            files = f.readlines()
            for file in files:
                adversarial_train_files.append(file.strip())

        adversarial_test_save_path = save_path + "/" + "model_0-False_Pos_Test"+ ".txt"
        adversarial_test_files = []
        with open(adversarial_test_save_path, 'r') as f:
            files = f.readlines()
            for file in files:
                adversarial_test_files.append(file.strip())

        train_model_1(adversarial_train_files, adversarial_test_files, train_loader, test_loader, save_path)

    else:
        print ("Invalid running mode!")




if __name__ == "__main__":
    main()

