from tensorboardX import SummaryWriter
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import sys
import time
import os
import argparse

# Local file imports
import parameters

# from models import * # Note for some reason we need to import the models as well
from models import get_model
from loss import get_loss
from train import Train_Pipeline
from model_utils import Model_Utils
from datasets import Subsampled_ElephantDataset, Full_ElephantDataset


parser = argparse.ArgumentParser()

parser.add_argument('--local_files', dest='local_files', action='store_true',
    help='Flag specifying to read data from the local elephant_dataset directory.'
    'The default is to read from the quatro data directory.')
parser.add_argument('--save_local', dest='save_local', action='store_true',
    help='Flag specifying to save model run information to the local models directory.'
    'The default is to save to the quatro data directory.')

# Running Flages
parser.add_argument('--run_type', type=str,
    help="Specify what parts we want to run [full, adversarial, stage_2]")

parser.add_argument('--generated_path', type=str, default=None,
    help="Path to generated positive data if we want to use it!")

# Model Paths
parser.add_argument('--path', type=str,
    help='When running \'adversarial\' or \'stage_2\' we must provide the folder with stage_1 and other models')

# NEED TO ADD THESE BELOW LATER

# Note this pre-loads model_0
parser.add_argument('--model_0', type=str,
    help='Provide a path to a pre-trained model_0 that will be saved to model_0 and used for adversarial discovery')

# Pre train flags
parser.add_argument('--pre_train_0', type=str,
    help='Use a pre-trained model to initialize model 0')
parser.add_argument('--pre_train_1', type=str,
    help='Use a pre-trained model to initialize model 1')


"""
    NEW FILE THINKING!!

    I think that having a class for this would be pretty hype. Especially when 
    we start doing curriculum shit!! But for now let us set up the framework
    to just have a 2 stage model!

    Things that need to happen:
        - Get the initial model stuff:
            - Create the initial undersampled train / test datasets
            - Create the full 24hr dataset for "2 stage evaluation"
            - Create optimizers and such!

        - Two steps of training  
            1) Train the first model, just like in the normal main_train. Get the 
            model weights

            2) Evaluate the first model on the 24hr dataset, and do 2 possible things:
                - Store an array that holds the "number wrong" for each data example
                in the 24hr dataset: This seems better because then we can exactly
                control the ratio of new negatives added

                - Store just a list of the "false-positives" 

            3) Train the second model, again like the first but now we have updated
            the negative data that we are using, as well as having the option to
            add new positive "generated" data

"""

class TwoStage_Model(object):
    """docstring for TwoStage_Model"""

    # Pass in the train/test dataloaders
    # Pass in model_types that we want for model_1/2
    # String indicating how much of the two stage process
    # we want to do! (i.e. full_pipeline, get adversarial, just second stage)


    """
        1) Things we need to still iron out. Make sure the save paths actually make sense!!!!!
            - Save path should initially be the path to the root directory of the two stage model.
            so let us just follow what was done before!!

    """
    def __init__(self, dataloaders, save_path, stage_one_id=17, stage_two_id=17,
                    add_generated_data=None, adversarial_neg_ratio=1, run_type="Full"):
        super(TwoStage_Model, self).__init__()
        
        # Step 1) Unpack the dataloaders
        self.train_loader = dataloaders['train_loader']
        self.test_loader = dataloaders['test_loader']
        self.full_train_loader = dataloaders['full_train_loader']
        self.full_test_loader = dataloaders['full_test_loader']

        # Step 2) Save other model information
        self.save_path = save_path
        self.stage_one_id = stage_one_id
        self.stage_two_id = stage_two_id

        # Step 3) Create the adversarial file names
        adversarial_file_prefix = self.generate_adversarial_file_name(add_generated_data, adversarial_neg_ratio)
        self.train_adversarial_files = os.path.join(self.save_path, adversarial_file_prefix + "Train.txt")
        self.test_adversarial_files = os.path.join(self.save_path, adversarial_file_prefix + "Test.txt")

        # Initialize the actual models
        # NOTE here is where we can consider loading in a pre_defined models
        self.stage_one = None
        self.stage_two = None
        # Figure out the run type! Assume for now that we run the full pipelines
        if run_type == "full":
            # Step 1) Train the first model stage
            self.train_stage_1()

            # Step 1b) 
            # This is where we would want to add the generated positves!!
            if add_generated_data is not None:
                self.train_loader.dataset.add_positive_examples_from_dir(add_generated_data)
                self.train_loader.dataset.undersample_negative_features_to_balance()

            # Step 2) Do the adversarial discovery. In this we will
            # produce a ranking vector of "how" wrong each example 
            # was by the first model
            self.adversarial_train_weighting, adversarial_train_files, \
                    self.adversarial_test_weighting, adversarial_test_files = self.adversarial_discovery(adversarial_neg_ratio)
            
            # Step 3) Update the datasets by adding the new negative samples!
            self.update_datasets(adversarial_train_files, adversarial_test_files)

            # Step 4) Train and save model 1
            self.train_stage_2()

        elif run_type == "adversarial":
            """
                We need to load the stage 1 model
            """
            # Step 1) Load stage 1
            stage_one_path = os.path.join(save_path, "Model_0/model.pt")
            self.stage_one = torch.load(model_0_path, map_location=parameters.device)
            self.adversarial_discovery(adversarial_neg_ratio)

        # Train just the second stage
        elif run_type == "stage_2":
            # Just run the second stage!
            # This involves reading in the adversarial generated files!
            # LET US FIRST FIGURE OUT WHERE TO SAVE THESE!!
            if add_generated_data is not None:
                self.train_loader.dataset.add_positive_examples_from_dir(add_generated_data)
                self.train_loader.dataset.undersample_negative_features_to_balance()

            adversarial_train_files, adversarial_test_files = self.read_adversarial_files()

            # Step 3) Update the datasets by adding the new negative samples!
            self.update_datasets(adversarial_train_files, adversarial_test_files)

            # Step 4) Train and save model 1
            self.train_stage_2()

        else:
            print ("Invalid running mode!")
   

    def create_and_train(self, model_id, save_path):
        """
            This represents the generic process of creating a model given a certain
            model type, getting the optimizer, etc. and then training it on a given 
            dataset. 

            ** The general training process of stage 1 and 2 is the same the only difference is the data **
            training of a model with a provided dataset!

            Inputs:
                - model_type
                - dataloader
                - ???

            # THINGS TO STILL DO
                - Allow for loading a model or doing the same model etc!!!
                - Make sure the save path makes sense!!
        """
        # Start the clock!
        start_time = time.time()

        # Step 1) Create the model - for now ignore having the same model or a pre-trained model! 
        # We can circle back and add this! Basically should allow for accessing previous models etc.
        # Gonna want to think about how to have the path and such make sense
        print ("Creating model type:", model_id)
        model = get_model(model_id).to(parameters.device)

        # Step 2) Setup summary writers - Kinda adhoc for now
        # FIGURE OUT THIS SAVE PATH STUFF!!!!!
        writer = SummaryWriter(save_path)
        writer.add_scalar('batch_size', parameters.BATCH_SIZE)
        writer.add_scalar('weight_decay', parameters.HYPERPARAMETERS[model_id]['l2_reg'])

        # Step 3) Get the model loss function
        loss_func, _ = get_loss()

        optimizer = torch.optim.Adam(model.parameters(), lr=parameters.HYPERPARAMETERS[model_id]['lr'],
                                     weight_decay=parameters.HYPERPARAMETERS[model_id]['l2_reg'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, parameters.HYPERPARAMETERS[model_id]['lr_decay_step'], 
                                                gamma=parameters.HYPERPARAMETERS[model_id]['lr_decay'])

        train_loaders = {'train': self.train_loader, 'valid':self.test_loader}
        train_pipeline = Train_Pipeline(train_loaders, model, loss_func, optimizer, 
                scheduler, writer, save_path, early_stop_criteria=parameters.TRAIN_MODEL_SAVE_CRITERIA.lower())
        
        model_wts, _ = train_pipeline.train(parameters.NUM_EPOCHS)

        # Save our model!
        if model_wts:
            model.load_state_dict(model_wts)
            model_save_path = os.path.join(save_path, "model.pt")
            torch.save(model, model_save_path)
            print('Saved best Model 0 based on {} to path {}'.format(parameters.TRAIN_MODEL_SAVE_CRITERIA.upper(), save_path))
        else:
            print('For some reason I don\'t have a model to save!!')
            quit()

        print('Training time: {:10f} minutes'.format((time.time()-start_time)/60))

        writer.close()

        return model


    def train_stage_1(self):
        """
            Helper for training stage 1
        """
        print ("++==========================++")
        print ("++== Training Stage 1 Model ++")
        print ("++==========================++")
        # Step 1a) Update the models save path
        stage_one_path = Model_Utils.join_paths(self.save_path, "Stage_1")
        self.stage_one = self.create_and_train(self.stage_one_id, stage_one_path)

    def train_stage_2(self):
        """
            Helper for training stage 2
        """
        print ("++===============================++")
        print ("++Training Error Correcting Model++") 
        print ("++===============================++")
        stage_two_path = Model_Utils.stage_2_model_path()
        stage_two_save_path = Model_Utils.join_paths(self.save_path, stage_two_path)
        self.stage_two = self.create_and_train(self.stage_two_id, stage_two_save_path)


    def update_datasets(self, train_adversarial_examples, test_adversarial_examples):
        """
            @TODO Add comment
        """
        self.update_dataset(self.train_loader, train_adversarial_examples)
        self.update_dataset(self.test_loader, test_adversarial_examples)

    def update_dataset(self, dataloader, adversarial_examples):
        """
            Update the dynamic dataset to have the new adversarial files
        """
        dataloader.dataset.set_neg_examples(adversarial_examples)

    def discover_adversarial(self, model_dataloader, full_dataloader, data_weightings, adversarial_neg_ratio):
        """
            For a given dataset, do:
                - Compute the number of adversarials to sample
                - Sample the k largest weighted samples
                - Save and return these adversarial examples
        """
        # Step 1) Compute k adversarial examples to sample
        num_sample = len(model_dataloader.dataset.pos_features) * adversarial_neg_ratio

        # Step 2) Get a sorted version of the data_weighting with the corresponding
        # indeces so we can sample these indeces from the dataset!
        sorted_weight_indeces = np.argsort(data_weightings)

        # Step 2a) From the dataset, sample a list of tuples of form:
        # [(feature_file, label_file)]
        adversarial_examples = []
        for idx in range(1, num_sample + 1):
            data_file_idx = sorted_weight_indeces[-idx]
            adversarial_examples.append((full_dataloader.dataset.data[data_file_idx], full_dataloader.dataset.labels[data_file_idx]))

        # Step 3) We should save this but let us do this later!!

        return adversarial_examples

    def adversarial_discovery_helper(self, dataloader, model, threshold=0.5):
        """
            We are now changing this up a bit! Rather than getting the exact files, we are 
            just going to return a list ranking how wrong the each file is. Note since 
            we have shuffle equal false we can use this later!!!

            NOTE IN REALITY THIS DATASET SHOULD JUST BE OVER THE NEGATIVE SAMPLEEESSS!!!
        """
        # NOTE: The last batch may be incomplete so let us keep a data counter
        adversarial_rankings = np.zeros(len(dataloader.dataset))

        # Put in eval mode!!
        model.eval()

        print ("Num batches:", len(dataloader))
        for idx, batch in enumerate(dataloader):
            # Do basic logging
            if idx % 1000 == 0:
                print("Adversarial search has gotten through {} batches".format(idx))
        
            # Step 1) Evaluate model on data
            inputs = batch[0].clone().float()
            labels = batch[1].clone().float()
            inputs = inputs.to(parameters.device)
            labels = labels.to(parameters.device)

            logits = model(inputs).squeeze(-1)

            # Step 2) Now compute the adversarial weighting of each example.
            # NOTE: FOR NOW WE SIMPLY DO NUMBER INCORRECT!!
            predictions = torch.sigmoid(logits)
            binary_preds = torch.where(predictions > threshold, torch.tensor(1.0).to(parameters.device), torch.tensor(0.0).to(parameters.device))
            # Compute the number of predictions made in the window
            pred_counts = torch.sum(binary_preds, dim=1).squeeze().cpu().detach().numpy() # Shape - (batch_size)
            # Get ground truth label counts - Note for empty windows this is zero!!
            gt_counts = torch.sum(labels, dim=1).cpu().detach().numpy() # Shape - (batch_size)
            
            # Make the weight of each example the number of incorrect predictions!!
            # NOTE: If the window includes an elephant call make this weight 0 for now!
            # i.e. zero out these pred counts
            gt_calls = (gt_counts > 0) # Find the index of windows with a call!
            pred_counts[gt_calls] = 0 # Zero the adversarial weight for these windows

            # Set the weights for the corresponding examples
            curr_batch_size = pred_counts.shape[0]
            adversarial_rankings[idx * parameters.BATCH_SIZE: idx * parameters.BATCH_SIZE + curr_batch_size] = pred_counts
        
        return adversarial_rankings


    def adversarial_discovery(self, adversarial_neg_ratio=1):
        """
            Use the adversarial discover helper to get the file rankings!
        """     
        """
            Collect the adversarial - false positives based on model_0
            for the train and validation set.
        """
        # Test our little function
        print ('++=========================================++')
        print ("++ Running Adversarial Weighting Discovery ++")
        print ('++=========================================++')
        
        # Step 1) Run stage 1 over the full training data
        adversarial_train_weighting = self.adversarial_discovery_helper(self.full_train_loader, self.stage_one, threshold=0.5)

        # Step 1a) Later maybe we want to save these rankings???

        # Step 2) Run stage 1 over the full test data
        adversarial_test_weighting = self.adversarial_discovery_helper(self.full_test_loader, self.stage_one, threshold=0.5)

        # Step 2a) Later maybe we want to save these rankings???

        # Step 3) Use these rankings to extract and save the adversarial examples we will use for training the second stage model
        adversarial_train_files = self.discover_adversarial(self.train_loader, self.full_train_loader, 
                                                        adversarial_train_weighting, adversarial_neg_ratio)
        adversarial_test_files = self.discover_adversarial(self.test_loader, self.full_test_loader, 
                                                        adversarial_test_weighting, adversarial_neg_ratio)

        # Step 4) Save the files that we were generated!! Note, save with some information
        # about them. Basically just include the ratio and whether we included generated data!
        # Read in the adversarial files
        # Save the training data
        with open(self.train_adversarial_files, 'w') as f:
            for data, label in adversarial_train_files:
                f.write('{}, {}\n'.format(data, label))

        # Save the test data
        with open(self.test_adversarial_files, 'w') as f:
            for data, label in adversarial_test_files:
                f.write('{}, {}\n'.format(data, label))
    
        return adversarial_train_weighting, adversarial_train_files, adversarial_test_weighting, adversarial_test_files

    def generate_adversarial_file_name(self, add_generated_data=None ,adversarial_neg_ratio=1):
        """
            Returns the prefix used for saving the generated adversarial 
            example files 

            @TODO make comment better!
        """
        prefix = "Adversarial_"
        if add_generated_data is not None:
            prefix += "Add-Generated-Data_"

        prefix += "Neg-Ratio-" + str(adversarial_neg_ratio) + "_"

        return prefix

    def read_adversarial_files(self):
        """
            Read in the contents of a previously created train and test adversarial file!!
        """
        # Step 1) Read in the training adversarial files
        adversarial_train_files = []
        with open(self.train_adversarial_files, 'r') as f:
            files = f.readlines()
            for file_pair in files:
                # Split by ', ' to get the data and the label
                file_pair = file_pair.strip()
                split_pair = file_pair.split(', ')
                adversarial_train_files.append((split_pair[0], split_pair[1]))
        
        # Step 2) Read in the test adversarial files
        adversarial_test_files = []
        with open(self.test_adversarial_files, 'r') as f:
            files = f.readlines()
            for file_pair in files:
                # Split by ', ' to get the data and the label
                file_pair = file_pair.strip()
                split_pair = file_pair.split(', ')
                adversarial_test_files.append((split_pair[0], split_pair[1]))


        return adversarial_train_files, adversarial_test_files


def main():
    # What do we need to do across all of the settings!
    # Get the data loaders!
    args = parser.parse_args()

    # Step 1) Get the paths to the training and test datafolders. Note
    # these data folder contain the full 24hr data so we can use them to
    # create both the train/test datasets and the full datasets
    train_data_path, test_data_path = Model_Utils.get_dataset_paths(local_files=args.local_files)

    # Step 2) Get the initial subsampled train/test datasets
    train_dataset = Subsampled_ElephantDataset(train_data_path, neg_ratio=parameters.NEG_SAMPLES, 
                                        normalization=parameters.NORM, log_scale=parameters.SCALE, 
                                        gaussian_smooth=parameters.LABEL_SMOOTH, seed=8)
    test_dataset = Subsampled_ElephantDataset(test_data_path, neg_ratio=parameters.TEST_NEG_SAMPLES, 
                                        normalization=parameters.NORM, log_scale=parameters.SCALE, 
                                        gaussian_smooth=parameters.LABEL_SMOOTH, seed=8)

    # Step 3) Get the complete datasets for adversarial discovery
    # Because the full dataset is just used for adversarial discovery where
    # we want to just know if the window is a negative window, we do not need 
    # to smooth the labels
    full_train_dataset = Full_ElephantDataset(train_data_path, normalization=parameters.NORM, 
                                                        log_scale=parameters.SCALE, 
                                                        gaussian_smooth=False, seed=8)
    full_test_dataset = Full_ElephantDataset(test_data_path, normalization=parameters.NORM, 
                                                        log_scale=parameters.SCALE, 
                                                        gaussian_smooth=False, seed=8)

    # Step 4) Get the dataloaders
    train_loader = Model_Utils.get_loader(train_dataset, parameters.BATCH_SIZE, shuffle=True)
    test_loader = Model_Utils.get_loader(test_dataset, parameters.BATCH_SIZE, shuffle=False)
    full_train_loader = Model_Utils.get_loader(full_train_dataset, parameters.BATCH_SIZE, shuffle=False)
    full_test_loader = Model_Utils.get_loader(full_test_dataset, parameters.BATCH_SIZE, shuffle=False)
    dataloaders = {
                    'train_loader': train_loader,
                    'test_loader': test_loader,
                    'full_train_loader': full_train_loader,
                    'full_test_loader': full_test_loader
                    }
    
    # Step 5) Create the save path for all of the model information
    if args.path is None:
        save_path = Model_Utils.create_save_path(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()), args.save_local, save_prefix='Two-Stage_')
    else:
        save_path = args.path

    # Step 6) Instantiate the 2_Stage model class and let it do its thing!
    TwoStage_Model(dataloaders, save_path, stage_one_id=parameters.MODEL_ID, stage_two_id=parameters.HIERARCHICAL_MODEL,
                    add_generated_data=args.generated_path, adversarial_neg_ratio=1, run_type=args.run_type)


if __name__ == "__main__":
    main()

