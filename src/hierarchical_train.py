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
from utils import create_save_path, create_dataset_path, hierarchical_model_1_path
from models import * # Note for some reason we need to import the models as well
from loss import get_loss
from train import train


parser = argparse.ArgumentParser()

parser.add_argument('--local_files', dest='local_files', action='store_true',
    help='Flag specifying to read data from the local elephant_dataset directory.'
    'The default is to read from the quatro data directory.')
parser.add_argument('--save_local', dest='save_local', action='store_true',
    help='Flag specifying to save model run information to the local models directory.'
    'The default is to save to the quatro data directory.')

# Running Flages
parser.add_argument('--full', dest='full_pipeline', action='store_true',
    help='Flag specifying to run the full hierarchical model pipeline.')
parser.add_argument('--adversarial', dest='adversarial', action='store_true',
    help='Flag specifying to generate a new set of adversarial_examples.')
parser.add_argument('--model_1', dest='model1', action='store_true',
    help='Flag specifying to just train Model_1.')
parser.add_argument('--visualize', action='store_true',
    help='Visualize the adversarial examples and Model_0 and Model_1 predictions on them')

# Model Paths
parser.add_argument('--path', type=str,
    help='When running \'adversarial\' or \'model1\' we must provide the folder with model_0 and other models')
# Note this pre-loads model_0
parser.add_argument('--model_0', type=str,
    help='Provide a path to a pre-trained model_0 that will be saved to model_0 and used for adversarial discovery')

# Pre train flags
parser.add_argument('--pre_train_0', type=str,
    help='Use a pre-trained model to initialize model 0')
parser.add_argument('--pre_train_1', type=str,
    help='Use a pre-trained model to initialize model 1')


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
    # One thing to consider would be instead to rank examples! I.e allowing for 
    # the case where there may not be enough adversarial examples and then we want
    # to actually just pick the top X "hardest."
    adversarial_examples = []
    # Only keep the FP model predictions
    model_0_FP_predictions = []

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

        # ONLY Squeeze the last dim!
        logits = model(inputs).squeeze(-1) # Shape - (batch_size, seq_len)

        # Now for each chunk we want to see whether it should be flagged as 
        # a true false positive. For now do "approx" by counting number pos samples
        predictions = torch.sigmoid(logits)
        # Pre-compute the number of pos. slices in each chunk
        # Threshold the predictions - May add guassian blur
        binary_preds = torch.where(predictions > threshold, torch.tensor(1.0).to(parameters.device), torch.tensor(0.0).to(parameters.device))

        pred_counts = torch.sum(binary_preds, dim=1).squeeze().cpu().detach().numpy() # Shape - (batch_size)
        # Get ground truth label counts
        gt_counts = torch.sum(labels, dim=1).cpu().detach().numpy() # Shape - (batch_size)
        
        # Find FP chunks:
        # Chunks with gt_counts = 0 and pred_counts > min_length
        gt_empty = (gt_counts == 0)
        predicted_chunks = (pred_counts >= min_length)

        epoch_adversarial_examples = list(data_files[gt_empty & predicted_chunks])
        adversarial_examples += epoch_adversarial_examples

        # Add the model predictions for FP chunks
        # Hopefully this doesn't slow things down a ton!
        binary_preds = binary_preds.cpu().detach().numpy()
        fp_preds = binary_preds[gt_empty & predicted_chunks]
        for i in range(len(epoch_adversarial_examples)):
            model_0_FP_predictions.append(fp_preds[i])


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
    return adversarial_examples, model_0_FP_predictions


def model_0_Elephant_Predictions(dataloader, model, threshold=0.5):
    """
        !!!!! WE NEED TO RE-FACTOR THIS TO CONSIDER THE CASE WHEN WE ADD REPEATS AND SUCH TO THE MODEL_1 DATASET!!
    """
    elephant_examples = []
    # Only keep the FP model predictions
    model_0_predictions = []
    gt_labels = []

    # Put in eval mode!!
    model.eval()
    print ("Num batches:", len(dataloader))
    for idx, batch in enumerate(dataloader):
        if idx % 1000 == 0:
            print("Adversarial search has gotten through {} batches".format(idx))

        inputs = batch[0].clone().float()
        labels = batch[1].clone().float()
        inputs = inputs.to(parameters.device)
        labels = labels.to(parameters.device)
        # Get the data_file locations for each chunk
        data_files = np.array(batch[2])

        # ONLY Squeeze the last dim!
        logits = model(inputs).squeeze(-1) # Shape - (batch_size, seq_len)

        # Now for each chunk we want to see whether it should be flagged as 
        # a true false positive. For now do "approx" by counting number pos samples
        predictions = torch.sigmoid(logits)
        # Pre-compute the number of pos. slices in each chunk
        # Threshold the predictions - May add guassian blur
        binary_preds = torch.where(predictions > threshold, torch.tensor(1.0).to(parameters.device), torch.tensor(0.0).to(parameters.device))
        # Used to tell if this is an elephant call window
        gt_counts = torch.sum(labels, dim=1).cpu().detach().numpy() # Shape - (batch_size)

        # Model predictions for TP chunks
        gt_elephant = (gt_counts > 0)

        epoch_true_pos_examples = list(data_files[gt_elephant])
        elephant_examples += epoch_true_pos_examples
        # Collect GT and Model_0 preds
        binary_preds = binary_preds.cpu().detach().numpy()
        tp_preds = binary_preds[gt_elephant]
        gt_labeling = labels[gt_elephant].cpu().detach().numpy()
        for i in range(len(epoch_true_pos_examples)):
            model_0_predictions.append(tp_preds[i])
            gt_labels.append(gt_labeling[i])

    return elephant_examples, model_0_predictions, gt_labels



def initialize_training(model_id, save_path, model_type=0, pre_train_path=None):
    # The get_model method is in charge of 
    # setting the same seed for each loaded model.
    # Thus, for each inner loop we train the same initialized model
    # Load model_0 to continue training with it
    if str(model_id).lower() == 'same':
        final_slash  = save_path.rindex('/')
        model_0_path = os.path.join(save_path[:final_slash], "Model_0/model.pt")
        model = torch.load(model_0_path, map_location=parameters.device)
    elif parameters.PRE_TRAIN and model_type == 0: # Load a pre-trained model
        print ("Loading Pre-Trained Model 0")
        model = torch.load(pre_train_path, map_location=parameters.device)
    elif parameters.HIERARCHICAL_PRE_TRAIN and model_type == 1:
        print ("Loading Pre-Trained Model 1")
        model = torch.load(pre_train_path, map_location=parameters.device)
    else:
        model = get_model(model_id).to(parameters.device)

    print(model)
    writer = SummaryWriter(save_path)
    writer.add_scalar('batch_size', parameters.BATCH_SIZE)
    writer.add_scalar('weight_decay', parameters.HYPERPARAMETERS[parameters.MODEL_ID]['l2_reg'])

    # Include whether we are using the second stage model
    second_stage = (model_type == 1)
    loss_func, include_boundaries = get_loss(is_second_stage=second_stage)

    # Honestly probably do not need to have hyper-parameters per model, but leave it for now.
    optimizer = torch.optim.Adam(model.parameters(), lr=parameters.HYPERPARAMETERS[parameters.MODEL_ID]['lr'],
                                 weight_decay=parameters.HYPERPARAMETERS[parameters.MODEL_ID]['l2_reg'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, parameters.HYPERPARAMETERS[parameters.MODEL_ID]['lr_decay_step'], 
                                            gamma=parameters.HYPERPARAMETERS[parameters.MODEL_ID]['lr_decay'])

    return model, loss_func, include_boundaries, optimizer, scheduler, writer

def visualize_adversarial(adversarial_train_files, train_loader, model_0, model_1):
    print ("++================================++")
    print ("++Visualizing Adversarial Examples++") 
    print ("++================================++")
    # Load the adversarial examples and zero out the true positives
    train_loader.dataset.set_featues(pos_features=[], neg_features=adversarial_train_files)

    # Get the model predictions over the adversarial examples
    model_0.eval()
    model_1.eval()
    with torch.no_grad(): 
        for idx, batch in enumerate(train_loader):
            print ("Batch number {} of {}".format(idx, len(train_loader)))
            # Cast the variables to the correct type and 
            # put on the correct torch device
            inputs = batch[0].clone().float()
            labels = batch[1].clone().float()
            inputs = inputs.to(parameters.device)
            labels = labels.to(parameters.device)

            # Forward pass
            logits_0 = model_0(inputs).squeeze(-1) # Shape - (batch_size, seq_len)
            logits_1 = model_1(inputs).squeeze(-1) # Shape - (batch_size, seq_len)

            # Visualize the individual example
            for i in range(len(inputs)):
                features = inputs[i].cpu().detach().numpy()
                output_0 = torch.sigmoid(logits_0[i]).cpu().detach().numpy()
                output_1 = torch.sigmoid(logits_1[i]).cpu().detach().numpy()
                label = labels[i].cpu().detach().numpy()
                # Maybe also show the masks

                # Go a bit against the actual visualization inputs.
                # To visualize model_0 and model_1 predictions just put
                # model_1 as what would be the binary (second) set of predictions
                visualize(features, outputs=output_0, labels=label, binary_preds=output_1)


def train_model_1(adversarial_train_files, adversarial_test_files, train_loader, test_loader, save_path, pre_train_path=None):

    print ("++===============================++")
    print ("++Training Error Correcting Model++") 
    print ("++===============================++")
    # Update initialize_training to allow for loading back model_0!!
    # Update the negative examples of the training and validation datasets
    print ("Updating Negative Features")
    if parameters.HIERARCHICAL_ADD_FP:
        train_loader.dataset.add_neg_features(adversarial_train_files)
        test_loader.dataset.add_neg_features(adversarial_test_files)
    else:
        train_loader.dataset.set_neg_features(adversarial_train_files)
        test_loader.dataset.set_neg_features(adversarial_test_files)

    # If we are using the extra class data then we need to update the labels
    # of the trainnig data accordingly
    if parameters.EXTRA_LABEL:
        # Given the save path update the labels based on the folders 
        # in hierarchical
        new_pos_train_labels_dir = os.path.join(save_path, 'transformed_model_0_tp_train_preds')
        new_neg_train_labels_dir = os.path.join(save_path, 'transformed_model_0_fp_train_preds')
        train_loader.dataset.update_labels(new_pos_labels_dir=new_pos_train_labels_dir, 
                                            new_neg_labels_dir=new_neg_train_labels_dir)
        # UPDATE THE TEST LABELS!!
        new_pos_test_labels_dir = os.path.join(save_path, 'transformed_model_0_tp_test_preds')
        new_neg_test_labels_dir = os.path.join(save_path, 'transformed_model_0_fp_test_preds')
        test_loader.dataset.update_labels(new_pos_labels_dir=new_pos_test_labels_dir, 
                                            new_neg_labels_dir=new_neg_test_labels_dir)

    # If we are using model_0 predictions as additional features for the model
    if parameters.MODEL_0_FEATURES:
        # Add train features
        model_0_train_pos_dir = os.path.join(save_path, 'model_0_tp_train_preds')
        model_0_train_neg_dir = os.path.join(save_path, 'model_0_fp_train_preds')
        train_loader.dataset.add_model_0_preds(model_0_pos_dir=model_0_train_pos_dir, 
                                                model_0_neg_dir=model_0_train_neg_dir)
        # Add test features
        model_0_test_pos_dir = os.path.join(save_path, 'model_0_tp_test_preds')
        model_0_test_neg_dir = os.path.join(save_path, 'model_0_fp_test_preds')
        test_loader.dataset.add_model_0_preds(model_0_pos_dir=model_0_test_pos_dir, 
                                                model_0_neg_dir=model_0_test_neg_dir)


    # Create repeated dataset with fixed indeces
    # WITH THE EXTRA LABELS THIS DOES NOT WORK RIGHT NOW!
    #if parameters.HIERARCHICAL_REPEATS > 1 or parameters.HIERARCHICAL_REPEATS_POS > 1 or parameters.HIERARCHICAL_REPEATS_NEG > 1:
    if parameters.HIERARCHICAL_REPEATS_POS != 1  or parameters.HIERARCHICAL_REPEATS_NEG != 1:
        # Include Twice as many repeats for the positive examples!
        print ("Re-scaling num features")
        train_loader.dataset.scale_features(parameters.HIERARCHICAL_REPEATS_POS, parameters.HIERARCHICAL_REPEATS_NEG)
        # Only include shifted windows if repeats is larger > 1 for one of them!
        if parameters.HIERARCHICAL_REPEATS_POS > 1  or parameters.HIERARCHICAL_REPEATS_NEG > 1:
            train_loader.dataset.create_fixed_windows()

    dloaders = {'train':train_loader, 'valid':test_loader}

    model_name = hierarchical_model_1_path()

    second_model_save_path = os.path.join(save_path, model_name)
    if not os.path.exists(second_model_save_path):
            os.makedirs(second_model_save_path)

    # For now just use same model for 0 and 1
    start_time = time.time()
    model_1, loss_func, include_boundaries, optimizer, scheduler, writer = initialize_training(parameters.HIERARCHICAL_MODEL, 
                                                                                second_model_save_path, model_type=1,
                                                                                pre_train_path=pre_train_path)
    model_1_wts = train(dloaders, model_1, loss_func, optimizer, scheduler, 
                    writer, parameters.NUM_EPOCHS, include_boundaries=include_boundaries, multi_class=parameters.EXTRA_LABEL)

    if model_1_wts:
        model_1.load_state_dict(model_1_wts)
        model_save_path = os.path.join(second_model_save_path, "model.pt")
        torch.save(model_1, model_save_path)
        print('Saved best Model 1 based on {} to path {}'.format(parameters.TRAIN_MODEL_SAVE_CRITERIA.upper(), second_model_save_path))
    else:
        print('For some reason I don\'t have a model to save')

    print('Training time: {:10f} minutes'.format((time.time()-start_time)/60))
    writer.close()

def save_model_0_predictions(save_path, window_files, model_0_predictions, folder_name):
    """
        Creates and saves a sub-folder with the given model_0 predictions
        corresponding to the provided window / feature files
    """
    # Save model_0 predictions
    directory = os.path.join(save_path, folder_name)
    # Create the directory if does not exist
    if not os.path.isdir(directory):
            os.mkdir(directory)

    for i in range(len(window_files)):
        # Strip off just the label tag
        window_file = window_files[i].split('/')[-1]
        # Make the file name reflect it is a label file
        model_prediction_file = window_file.replace("features", "labels")
        window_pred = model_0_predictions[i]
        pred_path = os.path.join(directory, model_prediction_file)
        np.save(pred_path, window_pred)

def transform_model_0_predictions(model_0_predictions, gt_labels=None):
    """
        Adds an extra class '2' label for FP model_0 predictions. 
        For windows with an elephant call present, we do not count
        FP predictions that occur do to boarder issues in predicting
        the actual call (i.e. mis-alignment).

        If gt_labels = None, then this is a negative window and has no elephant calls

    """
    transformed_preds = []
    for i in range(len(model_0_predictions)):
        prediction = model_0_predictions[i]
        if gt_labels is not None:
            labels = gt_labels[i]
        else:
            labels = np.zeros_like(prediction)

        # Create mask for when the prediction is a 1, but the GT label is 0
        pred_ones = (prediction == 1)
        gt_zeros = (labels == 0)


        new_pred = labels
        new_pred[pred_ones & gt_zeros] = 2

        # Correction is not needed if the window is a negative window
        if gt_labels is None:
            transformed_preds.append(new_pred)
            continue

        # For windows with elephant calls, heuristically make
        # all '2' labels touching '1' labels be '0' (i.e. normal background)
        # Do a forward and backward pass
        ranges = [range(new_pred.shape[0]), reversed(range(new_pred.shape[0]))]
        for direction in ranges:
            in_call = False
            for j in direction:
                # Check if we are entering a call
                if new_pred[j] == 1:
                    in_call = True
                # Exiting call
                elif new_pred[j] == 0:
                    in_call = False

                # If we have not exited a call and have hit '2'
                # predictions zero these out
                if in_call and new_pred[j] == 2:
                    new_pred[j] = 0

        transformed_preds.append(new_pred)

    return transformed_preds



def adversarial_discovery(full_train_path, full_test_path, model_1_train_loader, model_1_test_loader, model_0, save_path):
    """
        Collect the adversarial - false positives based on model_0
        for the train and validation set.
    """
    # Test our little function
    print ('++================================================++')
    print ("++ Beginning False Positive Adversarial Discovery ++")
    print ('++================================================++')
    # Do not include boundary uncertainty in full train loader. We only need the model predictions, we do not
    # calculate the loss! Use the HIERARCH_SHIFT flag along to decide if the Heirarchical model will use
    # randomly shifted windows. Note, we flag that this is the full dataset to make sure that during 
    # adversarial discovery we alwas sample the midlle of oversized windows
    #shift_windows = parameters.HIERARCHICAL_SHIFT_WINDOWS # or parameters.HIERARCHICAL_REPEATS > 1
    shift_windows = parameters.HIERARCHICAL_REPEATS_POS > 1 or parameters.HIERARCHICAL_REPEATS_NEG > 1
    full_train_loader = get_loader_fuzzy(full_train_path, parameters.BATCH_SIZE, random_seed=parameters.DATA_LOADER_SEED, 
                                        norm=parameters.NORM, scale=parameters.SCALE, 
                                        include_boundaries=False, shift_windows=shift_windows,
                                        is_full_dataset=True)
    full_test_loader = get_loader_fuzzy(full_test_path, parameters.BATCH_SIZE, random_seed=parameters.DATA_LOADER_SEED, 
                                        norm=parameters.NORM, scale=parameters.SCALE, include_boundaries=False)


    # For now let us try including all of the false negatives!
    train_adversarial_file = "model_0-False_Pos_Train.txt"
    if shift_windows:
        train_adversarial_file = "model_0-False_Pos_Train_Shift.txt"

    adversarial_train_files, model_0_fp_train_preds  = adversarial_discovery_helper(full_train_loader,
                                                             model_0, min_length=parameters.FALSE_POSITIVE_THRESHOLD)
    # Save the adversarial feature file paths
    adversarial_train_save_path = os.path.join(save_path, train_adversarial_file)
    with open(adversarial_train_save_path, 'w') as f:
        for file in adversarial_train_files:
            f.write('{}\n'.format(file))

    # Save model_0 FP train predictions both raw and transformed with additional label
    print ("Saving model_0 FP predictions on the train data")
    save_model_0_predictions(save_path, adversarial_train_files, model_0_fp_train_preds, "model_0_fp_train_preds")
    transformed_model_0_fp_train_preds = transform_model_0_predictions(model_0_fp_train_preds)
    save_model_0_predictions(save_path, adversarial_train_files, transformed_model_0_fp_train_preds, "transformed_model_0_fp_train_preds")

    # Save model_0 TP train predictions both raw and transformed with additional label
    print ("Saving model_0 TP predictions on the train data")
    elephant_train_files, model_0_tp_train_preds, gt_train_labels = model_0_Elephant_Predictions(model_1_train_loader, model_0)
    save_model_0_predictions(save_path, elephant_train_files, model_0_tp_train_preds, "model_0_tp_train_preds")
    transformed_model_0_tp_train_preds = transform_model_0_predictions(model_0_tp_train_preds, gt_train_labels)
    save_model_0_predictions(save_path, elephant_train_files, transformed_model_0_tp_train_preds, "transformed_model_0_tp_train_preds")

    test_adversarial_files = "model_0-False_Pos_Test.txt"
    adversarial_test_files, model_0_fp_test_preds = adversarial_discovery_helper(full_test_loader, 
                                                             model_0, min_length=parameters.FALSE_POSITIVE_THRESHOLD)
    adversarial_test_save_path = os.path.join(save_path, test_adversarial_files)
    with open(adversarial_test_save_path, 'w') as f:
        for file in adversarial_test_files:
            f.write('{}\n'.format(file))

    # Save model_0 FP test predictions both raw and transformed with additional label
    print ("Saving model_0 FP predictions on the test data")
    save_model_0_predictions(save_path, adversarial_test_files, model_0_fp_test_preds, "model_0_fp_test_preds")
    transformed_model_0_fp_test_preds = transform_model_0_predictions(model_0_fp_test_preds)
    save_model_0_predictions(save_path, adversarial_test_files, transformed_model_0_fp_test_preds, "transformed_model_0_fp_test_preds")

    # Save model_0 TP train predictions both raw and transformed with additional label
    print ("Saving model_0 TP predictions on the test data")
    elephant_test_files, model_0_tp_test_preds, gt_test_labels = model_0_Elephant_Predictions(model_1_test_loader, model_0)
    save_model_0_predictions(save_path, elephant_test_files, model_0_tp_test_preds, "model_0_tp_test_preds")
    transformed_model_0_tp_test_preds = transform_model_0_predictions(model_0_tp_test_preds, gt_test_labels)
    save_model_0_predictions(save_path, elephant_test_files, transformed_model_0_tp_test_preds, "transformed_model_0_tp_test_preds")

    return adversarial_train_files, adversarial_test_files

def train_model_0(train_loader, test_loader, save_path, pre_train_path=None):
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
    model_0, loss_func, include_boundaries, optimizer, scheduler, writer = initialize_training(parameters.MODEL_ID, 
                                                                                            first_model_save_path, model_type=0,
                                                                                            pre_train_path=pre_train_path)
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
    # What do we need to do across all of the settings!
    # Get the data loaders!
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


    # Get oversized calls if shifting windows or repeating for model 2
    # We should try to remove both of these This is an issue too!
    if parameters.HIERARCHICAL_SHIFT_WINDOWS: # or parameters.HIERARCHICAL_REPEATS > 1:
        full_train_path += '_OversizeCalls'

    # For model 2 we need to have oversized calls to generate the randomly located repeats
    if parameters.HIERARCHICAL_REPEATS_POS > 1 or parameters.HIERARCHICAL_REPEATS_NEG > 1:
        full_train_path += '_OversizeCalls'

    model_0_train_data_path, include_boundaries = create_dataset_path(train_data_path, neg_samples=parameters.NEG_SAMPLES, 
                                                                    call_repeats=parameters.CALL_REPEATS, 
                                                                    shift_windows=parameters.SHIFT_WINDOWS)
    model_0_test_data_path, _ = create_dataset_path(test_data_path, neg_samples=parameters.TEST_NEG_SAMPLES, 
                                                                call_repeats=1)
    

    # Check if a different dataset is being used for Model_1
    model_1_train_data_path = model_0_train_data_path
    model_1_test_data_path = model_0_test_data_path
    # Remove this same thing!
    #if str(parameters.HIERARCHICAL_REPEATS).lower() != "same" or parameters.HIERARCHICAL_REPEATS_POS > 1  or parameters.HIERARCHICAL_REPEATS_NEG > 1:
    if parameters.HIERARCHICAL_REPEATS_POS > 1  or parameters.HIERARCHICAL_REPEATS_NEG > 1:
        # SHould prob just have neg samples x1 since doesnt matter!!
        # For now set call repeats to 1, but get shifting windows so we later can do call repeats!
        #shift_windows = parameters.HIERARCHICAL_REPEATS > 1 or parameters.HIERARCHICAL_SHIFT_WINDOWS
        # For now should make shift windows just be true! Because it does not make a lot of sense to do 
        # repeats without shifting windows since we can only repeat the pos examples
        shift_windows = True
        # Set this to 1 because we take care of this later!!!!
        call_repeats = 1
        model_1_train_data_path, _ = create_dataset_path(train_data_path, neg_samples=parameters.NEG_SAMPLES, 
                                                        call_repeats=call_repeats,
                                                        shift_windows=shift_windows)
    
    
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
    

    if args.path is None:
        save_path = create_save_path(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()), args.save_local, save_prefix='Hierarchical_')
    else:
        save_path = args.path

    # Case 1) Do the entire pipeline! Can break now the pipeline into 3 helper functions!
    if args.full_pipeline:
        # Train and save model_0
        if args.model_0 == None:
            model_0 = train_model_0(model_0_train_loader, model_0_test_loader , save_path, args.pre_train_0)
        else: # Load and save model_0
            model_0 = torch.load(args.model_0, map_location=parameters.device)
            first_model_save_path = os.path.join(save_path, "Model_0")
            if not os.path.exists(first_model_save_path):
                os.makedirs(first_model_save_path)

            model_save_path = os.path.join(first_model_save_path, "model.pt")
            torch.save(model_0, model_save_path)

        # Do the adversarial discovery
        adversarial_train_files, adversarial_test_files = adversarial_discovery(full_train_path, full_test_path, 
                                                                model_0_train_loader, model_0_test_loader, model_0, save_path)
        # Train and save model 1
        train_model_1(adversarial_train_files, adversarial_test_files, model_1_train_loader, model_1_test_loader, 
                                                save_path, args.pre_train_1)
    
    # Just generate new adversarial examples 
    elif args.adversarial:
        # Load model_0
        model_0_path = os.path.join(save_path, "Model_0/model.pt")
        model_0 = torch.load(model_0_path, map_location=parameters.device)
        adversarial_discovery(full_train_path, full_test_path, model_0_train_loader, model_0_test_loader, model_0, save_path)

    # Train just model_1
    elif args.model1:
        # Read in the adversarial files
        train_adversarial_file = "model_0-False_Pos_Train.txt"
        #if parameters.HIERARCHICAL_SHIFT_WINDOWS or parameters.HIERARCHICAL_REPEATS > 1:
        if parameters.HIERARCHICAL_REPEATS_POS > 1  or parameters.HIERARCHICAL_REPEATS_NEG > 1:
            train_adversarial_file = "model_0-False_Pos_Train_Shift.txt"

        adversarial_train_save_path = os.path.join(save_path, train_adversarial_file)
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

        train_model_1(adversarial_train_files, adversarial_test_files, model_1_train_loader, 
                        model_1_test_loader, save_path, args.pre_train_1)

    elif args.visualize:
        model_0_path = os.path.join(save_path, "Model_0/model.pt")
        model_0 = torch.load(model_0_path, map_location=parameters.device)
        model_1_name = hierarchical_model_1_path()
        model_1_path = os.path.join(save_path, model_1_name+'/model.pt')
        model_1 = torch.load(model_1_path, map_location=parameters.device)

        # Read in the adversarial files
        train_adversarial_file = "model_0-False_Pos_Train.txt"
        #if parameters.HIERARCHICAL_SHIFT_WINDOWS or parameters.HIERARCHICAL_REPEATS > 1:
        if parameters.HIERARCHICAL_REPEATS_POS > 1  or parameters.HIERARCHICAL_REPEATS_NEG > 1:
            train_adversarial_file = "model_0-False_Pos_Train_Shift.txt"

        adversarial_train_save_path = os.path.join(save_path, train_adversarial_file)
        adversarial_train_files = []
        with open(adversarial_train_save_path, 'r') as f:
            files = f.readlines()
            for file in files:
                adversarial_train_files.append(file.strip())

        visualize_adversarial(adversarial_train_files, model_1_train_loader, model_0, model_1)

    else:
        print ("Invalid running mode!")




if __name__ == "__main__":
    main()

