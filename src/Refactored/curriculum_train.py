"""
    For now let this represent very similar to the 2_stage model,
    a basic class to lay out the groundworks for creating a curriculum
    training class for a model!!
"""

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


"""
    
    THINGS THAT NEED TO BE INCLUDED TO MAKE A CURRICULUM TRAINING CLASS
        - Need a method that just handles training a given model for a certain number
        of epochs. Here it seems like we can use the already built in training methods
        that we have. This basically just would take a model and a dataset and train
        for x epochs (simple)
        - Given a trained model, evaluate that model over all of the training data (i.e.
        the full training dataset). In this method we want to track a collection of
        different data statistics about how the model ran on each example. e.g.
            - Number of incorrect predicted slices
            - Similar to the focal loss, we could measure 1 - confidence in correct class.
            So for the negative windows we would basically compute sigmoid(logit) and this
            would give (1 - prob not elephant call). So doing an avg. over this sigmoid
            would give us the "likilihood of having a call" - when in fact we know the window
            should be empty!! So here a more difficult example will have a higher likelihood
            of a call, meaning more slices the model got confused. 
            - Difficulty metric (could be distance from 50%) i.e. if we 
            are guessing either way near 1 or way near 0 then it is too easy
            or too hard but if guessing near 50% at times is medium difficult.
            We can also somehow make it so that after medium we have hard than
            easy by scaling the upper distance from 0.5

        - Now we need a method that basically take the summarized run over the full 
        training data and saves / updates some class tracking variables. 
            - At the most simple level, we want to just have a variable that 
            takes the rankings of the data and sorts them to inform the model 
            on what to sample (i.e. the hardest) 
            - We could also sample based on the relative data weightings, though
            this seems very costly.
            - The maybe more gold standard is to track the variance of examples
            by testing every 3 iterations. This would be take for example number of predicted
            wrong and compute the variance over x evaluation iterations. The higher
            variance ones are those that we want to sample!


        - Lastly, we need to update the dataset! This is likely another method or 
        just fully leverages the dataset class methods



    Skeleton:

        - Setup the class variables. Key vars:
            - The data loaders:
                - Train loader (dynamic)!
                - Test loader (unsure if we will have it be dynamic. I like
                the idea of having it be the "difficult" examples from
                the two stage model)
                - full train loader (May just want to include non-calls!!!)
                - full test loader (potentially)

            - The model we are training - Because this is a training strategy
            I think that it makes sense that we are given a model. The job 
            of the curriculum strategy class is to set the curriculum traning
            strategy of our model!

            - Data statistics for the full dataset. 
                - General ranking of the data array
                - Variance of the data
                - Other things?? We can think!

        - 

    How to craft the test set:
        - Use the false positives from the 2 stage model! + 


"""


class Curriculum_Strategy(object):
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
    # We should pass in a training class and just try that for now. Maybe it all doesn't make sense but fuck it
    def __init__(self, train_model, full_loaders, save_path, 
            num_epochs_per_era=3, eras=20, neg_ratio=1, 
            rand_keep_ratio=0.5, hard_keep_ratio=0.0, hard_vs_rand_ratio=0.5,
            run_type="Simple", incorrect_scoring_method="slices"):
    
        super(Curriculum_Strategy, self).__init__()
        
        # Step 1) Setup the training class that the curriculum model 
        # supervises
        self.train_model = train_model
        
        # Step 2) The full collection of negative examples
        self.full_train_loader = full_loaders['full_train_loader']
    

        # Step 3) Model save information
        self.save_path = save_path
        # Create Save path for the dynamically sampled examples
        self.sampled_data_path = os.path.join(self.save_path, "Sampled_Data")
        

        # Step 4) Save curriculum specific parameters
        self.eras = eras
        self.num_epochs_per_era = num_epochs_per_era

        # Compute the different sampled dataset sizes based on the 
        # different ratios
        self.num_negatives = len(self.full_train_loader.dataset.neg_features)
        self.num_important_negatives = int(num_negatives * hard_vs_rand_ratio)
        self.num_rand_negatives = num_negatives - num_important_negatives

        self.num_new_hard = self.num_important_negatives - int(hard_keep_ratio * self.num_important_negatives) 
        self.num_new_rand = self.num_rand_negatives - int(rand_keep_ratio * self.num_rand_negatives)


        self.incorrect_scoring_method = incorrect_scoring_method

        # Figure out the run type! Assume for now that we run the full pipelines
        self.curriculum_train(run_type)
        
   
    """
        Methods that I think we need

        - A method called something like curriculum_train
        that does the training outer loops. Sets up the
        skeleton for the general learning procedure
            - The basic loop is over eras
            - In each era we train the model for a small number of epochs
            like 3 / 5 epochs.
            - After training run the method that evaluates the model over
            the full training dataset
            - Update our internal representations of each of the data windows
            - The curriculum model should be the one in charge of selecting
            the new data!! I think we need to have a method that does this.
            So for example we want in the simple case to just select a random sampling
            of data examples based on the most difficult!

        I may just consider writing the logic of training and eval over here? It feels very wasteful?
    """
    def curriculum_train(run_type):

        # Should we set it up 
        # For now leave out the extra class vars
        # Start super fucking simple
        # How can we start? We just need to basically:
        # 1) train for a small number of epochs
        # 2) Evaluate over the full data
        
        for era in range(self.eras):
            print(f"Curriculum Era: {era + 1}\n")
            # Step 1) Train the model on the current state of the dataset
            print ("Training Model")
            self.train_model.train(self.num_epochs_per_era)
            print ("Finished Era Training")

            # Step 2) Run the model over the entire dataset
            # This can be made better but let us just hash her out
            print ("Evaluating Model Over Full Data")
            window_scores = self.full_data_eval()

            # Step 3) We need to adjust the dataset!!!!!!!
            # Get the new hard and random negatives
            new_hard_negatives, new_rand_negatives = self.re_sample_data(window_scores, era)


            # Step 4) Adjust the datasets baby for the next round of
            # training!
            self.update_dataset(new_hard_negatives, new_rand_negatives)


            # Step 5) Figure out exactly how we want to end???
            # We obviously need a way to end this? The question is how?
            # I guess that will be when the result on the test set that is
            # in the training model doesn't decrease!! Let us just mess around
            # a bit with it. This may have to come from the training model!
            # Also we do run over the entire training set, so maybe running
            # over the entire test set is not a bad idea to track stopping
            # metrics? Like maybe we should just after an era test over
            # the full dataset with some modified metrics??


    def full_data_eval(self):
        """
            @TODO more comments!!
            This is where we use the current state of our model to compute
            the difficulty / errors we make over the full colleciton of negative data.
        """

        # Step 1) Create scoring vector for each example in the dataset
        window_scores = np.zeros(len(self.full_train_loader.dataset))
        
        # Put the model in eval mode!!
        self.train_model.model.eval()

        # Step 2) Evaluate over every example in the full dataset
        print ("Num batches:", len(self.full_train_loader))
        for idx, batch in enumerate(self.full_train_loader):
            if idx % 1000 == 0:
                print("Full data evaluation has gotten through {} batches".format(idx))
        
            # Step 2a) Evaluate model on data
            inputs = batch[0].clone().float()
            labels = batch[1].clone().float()
            inputs = inputs.to(parameters.device)
            labels = labels.to(parameters.device)

            logits = self.train_model.model(inputs).squeeze(-1)
            predictions = torch.sigmoid(logits)

            # Step 2b) compute the difficulty score of each window
            # based on the specified scoring method
            if self.incorrect_scoring_method == "slices":
                # Compute the number of incorrest slices.
                batch_scores = self.compute_incorrect_slices(predictions)
            
            batch_size = inputs.shape[0]
            # We should set this to be a batch size for the complete loaders!!!
            window_scores[idx * parameters.BATCH_SIZE: idx * parameters.BATCH_SIZE + curr_batch_size] = batch_scores
        
        return window_scores



    def compute_incorrect_slices(self, preds, threshold=0.5):
        """
            @TODO Commments bitch!
        """
        # Number incorrect
        binary_preds = torch.where(preds > threshold, torch.tensor(1.0).to(parameters.device), torch.tensor(0.0).to(parameters.device))
        # Compute the number of predictions made in the window
        pred_counts = torch.sum(binary_preds, dim=1).squeeze().cpu().detach().numpy() # Shape - (batch_size)
        
        return pred_counts

    def re_sample_data(self, window_scores, era):
        """
            Sample new hard and random negative samples.
            To allow for maximal flexability we leverage 
            'self.rand_keep_ratio' and 'self.hard_keep_ratio' 
            to determine the ratio of added vs. kepy hard/random
            examples. Moreover, we use 'self.hard_vs_rand_ratio' 
            to determine the ratio of hard to random examples.
        """

        # Here we need to do a couple of things!
        # Firstly we need to do this seperately
        # for the hard and negative
        # Let us do the hard first. This basically
        # entails sorting the scores of the window
        # and computing how many we want to sample
        # num_hard - int(hard_keep_ratio * num_hard)
        # num_hard we should just compute as class variables in the beginning!!!!

        # Step 1) argsort the scores to order the "most" difficult
        # examples for the current model 
        sorted_weight_indeces = np.argsort(window_scores)


        # Step 2) Sample num_new_hard of the "most" difficult examples.
        # NOTE: Depending on the curriculum run type most difficult 
        # can take on different meaning (i.e. most on the boarder of 
        # classification)
        new_hard_negatives = []
        for idx in range(1, self.num_new_hard + 1):
            data_file_idx = sorted_weight_indeces[-idx]
            new_hard_negatives.append((self.full_train_loader.dataset.data[data_file_idx], \
                            self.full_train_loader.dataset.labels[data_file_idx]))

        # Step 2a) Save the sampled hard negatives
        sampled_hard_negatives = sorted_weight_indeces[-self.num_new_hard:]
        self.save_sampled_examples(sampled_hard_negatives, window_scores[sampled_hard_negatives], "Hard-Negatives_Era-" + str(era))

        # Step 3) Sample num_new_rand random examples. 
        # These examples are sampled from the remaining examples
        # not sampled as the new hard training examples - 
        # indeces in the range [0 : sorted_weight_indeces.shape[0] - num_new_hard]
        new_rand_negatives = []
        rand_data_idxs = np.random.choice(np.arange(sorted_weight_indeces.shape[0] - self.num_new_hard), size=self.num_new_rand, replace=False)
        # Now we want to add these new random datapoints
        for idx in rand_data_idxs:
            data_file_idx = sorted_weight_indeces[idx]
            new_rand_negatives.append((self.full_train_loader.dataset.data[data_file_idx], \
                            self.full_train_loader.dataset.labels[data_file_idx]))

        # Step 2a) Save the sampled hard negatives
        self.save_sampled_examples(rand_data_idxs, window_scores[rand_data_idxs], "Random-Negatives_Era-" + str(era) + ".txt")

        return new_hard_negatives, new_rand_negatives

    def save_sampled_examples(sampled_data, data_scores, title):
        """
            Save a record of the data sampled one per line:
            (idx, score, file name)
        """
        with open(self.train_adversarial_files, 'w') as f:
            for data, label in adversarial_train_files:
                f.write('{}, {}\n'.format(data, label))
        # Save a file for the given ERA 
        file_path = os.path.join(self.sampled_negatives_path, title)
        with open(file_path, 'w') as f:
            for i in range(1, sampled_data.shape[0] + 1):
                data_file_idx = sampled_data[-i]
                f.write(f'{sampled_data[-i]}, {data_scores[-i]}, {self.full_train_loader.dataset.data[data_file_idx]}\n')


    def update_dataset(self, new_hard_negatives, new_rand_negatives, file_title):
        """
            Update the dynamic dataset to have the new adversarial files
        """
        # Things to get done. 
        # Update the negative samples of the dataset to include new_rand_negatives
        # while keeping a 'self.rand_keep_ratio' of negatives.
        #
        # Update the hard negatives to include new_hard_negatives
        # while keeping 'self.hard_keep_ratio' of hard negs.
        
        # Step 1) Update the random negatives 
        self.train_model.train_dataloader.dataset.update_neg_examples(new_rand_negatives, \
                                                 keep_ratio=self.rand_keep_ratio, combine_data=False)

        # Step 2) Update the hard negatives
        self.train_model.train_dataloader.dataset.update_hard_neg_examples(new_hard_negatives, \
                                                 keep_ratio=self.hard_keep_ratio, combine_data=True)

    