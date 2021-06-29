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
from visualizer import visualize


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
    def __init__(self, train_model, dataloaders, save_path, 
            num_epochs_per_era=3, eras=20,  
            rand_keep_ratio=0.5, hard_keep_ratio=0.25, hard_vs_rand_ratio=0.5,
            hard_increase_factor=0.0,
            hard_vs_rand_ratio_max=0.5,
            hard_sample_size_factor=5,
            difficulty_scoring_method="slices"):
    
        super(Curriculum_Strategy, self).__init__()
        
        # Step 1) Setup the training class that the curriculum model 
        # supervises
        self.train_model = train_model
        
        # Step 2) The full collection of ONLY negative examples
        self.full_train_loader = dataloaders['full_train_loader']
    

        # Step 3) Model save information
        self.save_path = save_path
        # Create Save path for the dynamically sampled examples
        self.sampled_data_path = os.path.join(self.save_path, "Sampled_Data")
        # Create the new directory
        os.makedirs(self.sampled_data_path)
        

        # Step 4) Save curriculum specific parameters
        self.eras = eras
        self.num_epochs_per_era = num_epochs_per_era

        # Step 5) Compute the breakdown of difficult vs. random negatives 
        #################################################################
        # Step 5a) Number of negatives that exist in the "train dataset" for train model
        self.num_negatives = len(self.train_model.train_dataloader.dataset.neg_features)
        # Step 5b) Number of difficult negatives that we want will train with.
        self.num_hard_negatives = int(self.num_negatives * hard_vs_rand_ratio)
        # Step 5c) Number of random negatives that we include to "respect" the distribution.
        self.num_rand_negatives = self.num_negatives - self.num_hard_negatives

        # Step 6) How many NEW difficult vs. random negatives we will include after each era
        #self.num_keep_hard = int(hard_keep_ratio * self.num_hard_negatives)
        self.num_new_hard = self.num_hard_negatives - int(hard_keep_ratio * self.num_hard_negatives) 
        #self.num_keep_rand = int(rand_keep_ratio * self.num_rand_negatives)
        self.num_new_rand = self.num_rand_negatives - int(rand_keep_ratio * self.num_rand_negatives)

        # Step 7) Save the parameters for growing the hard samples
        self.hard_keep_ratio = hard_keep_ratio
        self.rand_keep_ratio = rand_keep_ratio
        self.hard_increase_factor = hard_increase_factor
        self.hard_vs_rand_ratio_max = hard_vs_rand_ratio_max
        self.hard_sample_size_factor = hard_sample_size_factor


        # Step 8) Define the curriculum strategy
        self.difficulty_scoring_method = difficulty_scoring_method
        
        # Step 9) Create Dummy dataloader that we will use to save current model performances
        # on the newly sampled hard datapoints. Note we will clear out the pos and neg so these
        # are just dummy values
        print ("Creating Dummy Dataset")
        performance_tracking_dataset = Subsampled_ElephantDataset(self.train_model.train_dataloader.dataset.data_path,
                                        neg_ratio=1, normalization=parameters.NORM, log_scale=parameters.SCALE, 
                                        gaussian_smooth=parameters.LABEL_SMOOTH, seed=8)
        # Clear out the positive samples and negative samples
        performance_tracking_dataset.update_pos_examples([], num_keep=0)
        performance_tracking_dataset.update_neg_examples([], num_keep=0)
        self.performance_tracking_loader = Model_Utils.get_loader(performance_tracking_dataset, parameters.BATCH_SIZE, shuffle=False)
   
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
    def curriculum_train(self):

        # Should we set it up 
        # For now leave out the extra class vars
        # Start super fucking simple
        # How can we start? We just need to basically:
        # 1) train for a small number of epochs
        # 2) Evaluate over the full data
        best_model_wts = None
        try:
            for era in range(self.eras):
                print(f"Curriculum Era: {era + 1}\n")
                # Step 1) Train the model on the current state of the dataset
                print ("Training Model")
                best_model_wts, early_stop = self.train_model.train(self.num_epochs_per_era)
                print ("Finished Era Training")
                if early_stop:
                    print("Early stopping!!")
                    break

                # Step 2) Run the model over the entire dataset
                # This can be made better but let us just hash her out
                print ("Evaluating Model Over Full Data")
                window_difficulty_scores = self.full_data_eval()

                # Step 3) We need to adjust the dataset!!!!!!!
                # Get the new hard and random negatives
                print ("Re-Sampling the data")
                new_hard_negatives, new_rand_negatives = self.re_sample_data(window_difficulty_scores, era)


                # Step 4) Adjust the datasets baby for the next round of
                # training!
                print ("Updating the datasets with new examples")
                self.update_dataset(new_hard_negatives, new_rand_negatives)

                # Step ?) Figure out exactly how we want to end???
                # We obviously need a way to end this? The question is how?
                # I guess that will be when the result on the test set that is
                # in the training model doesn't decrease!! Let us just mess around
                # a bit with it. This may have to come from the training model!
                # Also we do run over the entire training set, so maybe running
                # over the entire test set is not a bad idea to track stopping
                # metrics? Like maybe we should just after an era test over
                # the full dataset with some modified metrics??

                # Step 5) Update curriculum parameters
                self.update_curriculum_params()

                # Step 6) Save the current model
                self.save_current_model(era)

        except KeyboardInterrupt:
            print("Early stopping due to keyboard intervention")

        # I think that this makes sense?
        return best_model_wts


    def save_current_model(self, era):
        """

        """
        # Just double check
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        model_path = os.path.join(self.save_path, "model_era-" + str(era) + ".pt")
        torch.save(self.train_model.model, model_path)
        print('Saved current model for era {}'.format(era))

    def update_curriculum_params(self):
        """

        """
        current_hard_vs_rand_ratio = self.num_hard_negatives / self.num_negatives
        if self.hard_increase_factor > 0 and current_hard_vs_rand_ratio < self.hard_vs_rand_ratio_max: 
            # Update how many hard negatives we are going to sample
            self.num_hard_negatives = int(self.num_hard_negatives * self.hard_increase_factor)
            self.num_new_hard = self.num_hard_negatives - int(self.hard_keep_ratio * self.num_hard_negatives)

            # Update how many rand negatives
            self.num_rand_negatives = self.num_negatives - self.num_hard_negatives
            self.num_new_rand = self.num_rand_negatives - int(self.rand_keep_ratio * self.num_rand_negatives)

        print ("++++++++++++++++++++++++++++")
        print ("New Curriculum Parameters")
        print ("Total Number of Negatives:", self.num_negatives)
        print ("Total Number of Hard Negatives:", self.num_hard_negatives)
        print (f"Total Hard Keep: {self.num_hard_negatives - self.num_new_hard} and Hard Sampled: {self.num_new_hard}")
        print ("Total Number of Rand Negatives:", self.num_rand_negatives)
        print (f"Total Rand Keep: {self.num_rand_negatives - self.num_new_rand} and Rand Sampled: {self.num_new_rand}")



    def full_data_eval(self):
        """
            @TODO more comments!!
            This is where we use the current state of our model to compute
            the difficulty / errors we make over the full colleciton of negative data.
        """

        # Step 1) Create scoring vector for each example in the dataset
        window_difficulty_scores = np.zeros(len(self.full_train_loader.dataset))

        # Put the model in eval mode!!
        self.train_model.model.eval()

        # Step 2) Evaluate over every example in the full dataset
        print ("Num batches:", len(self.full_train_loader))
        for idx, batch in enumerate(self.full_train_loader):
            if idx % 3000 == 0:
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
            if self.difficulty_scoring_method == "slices":
                # Compute the number of incorrest slices.
                batch_scores = self.compute_incorrect_slices(predictions)
            elif self.difficulty_scoring_method == "uncertainty_weighting":
                batch_scores = self.compute_uncertainty_weighting_score(predictions)
            
            batch_size = inputs.shape[0]
            # We should set this to be a batch size for the complete loaders!!!
            window_difficulty_scores[idx * parameters.BATCH_SIZE: idx * parameters.BATCH_SIZE + batch_size] = batch_scores
        
        return window_difficulty_scores


    def compute_incorrect_slices(self, preds, threshold=0.5):
        """
            @TODO Commments bitch!
        """
        # Number incorrect
        # TEST THIS!
        pred_counts = torch.sum(preds > threshold, dim=1).squeeze().cpu().detach().numpy()
        #binary_preds = torch.where(preds > threshold, torch.tensor(1.0).to(parameters.device), torch.tensor(0.0).to(parameters.device))
        # Compute the number of predictions made in the window
        #pred_counts = torch.sum(binary_preds, dim=1).squeeze().cpu().detach().numpy() # Shape - (batch_size)
        
        return pred_counts

    def compute_uncertainty_weighting_score(self, preds, threshold=0.5):
        """
            General Idea:
            Rather than focus soley on the hardest examples or the ones that
            we get the most wrong, we actually want to focus on the "middle"
            difficulty examples and work our way up in difficulty.

            As a proxy for measuring relative difficulty, we look to quantify
            those examples that are on the current decision boundary (i.e. those
            that are close to being correctly classified but show quite some
            uncertainty). To do so we quantify the uncertainty of each slice
            through the function: f(pred) = (-|0.5 - pred| + pred / 2 + 0.5) / 0.75.
            This scoring measure captures the relative distance from the decision
            threshold '0.5', where the highest score is for highly uncertain scores
            near 0.5 and we assymetrically assign higher scores to
            predictions above 0.5. The scores range from [0, 1], 
            where additionally if pred < 0.2 set score to be 0. 

            NOTE: We are only evaluating over negative windows that are expected to
            have all slices with label 0.
        """
        preds = preds.cpu().detach().numpy()
        scores = (- np.abs(0.5 - preds) + preds / 2. + 0.5) / 0.75
        # Zero out very low scores with pred <= 0.2 and score <= 0.4.
        # This will help the mean not get dominated!
        scores[preds < 0.2] = 0
        # Take the mean across each window
        return np.mean(scores, dim=1)

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

        # Step 1) Save also the window scores so that we can visualize this
        # change over times
        self.save_era_scores(window_scores, "Negative-Scores_Era-" + str(era))

        # argsort the scores to order the "most" difficult
        # examples for the current model.
        sorted_weight_indeces = np.argsort(window_scores)


        # Step 2) Sample num_new_hard of the "most" difficult examples.
        # NOTE: Depending on the curriculum run type most difficult 
        # can take on different meaning (i.e. most on the boarder of 
        # classification)
        new_hard_negatives = []

        # Note that if this is the first era there are no hard negatives yet so we
        # want to incorperate all of the hard negatives. 
        # Basically in the first era there are just pos | start_neg
        # want to go to pos | neg | hard_neg --> where neg + hard_neg = start_neg
        # So initially we want the number of re-sampled hard_neg to be all 
        num_sample_hard = self.num_new_hard
        if era == 0:
            num_sample_hard = self.num_hard_negatives

        # Rather than sampling the hardest we will sample randomly from a window
        # of the top "hard_sample_size_factor * num_sample_hard"
        num_sample_hard_window = num_sample_hard * self.hard_sample_size_factor
        print (f"Sampling {num_sample_hard} hard samples from the top {num_sample_hard_window} hard samples")
        sample_data_pool = sorted_weight_indeces[-num_sample_hard_window: ]
        print (sample_data_pool.shape)
        if self.hard_sample_size_factor == 1:
            # Sample the indeces in reverse order to keep the hardest at the top!
            hard_sample_idxs = np.arange(sample_data_pool.shape[0])[:: -1]
        else:
            hard_sample_idxs = np.random.choice(np.arange(sample_data_pool.shape[0]), size=num_sample_hard, replace=False)

        for idx in hard_sample_idxs:
            data_file_idx = sample_data_pool[idx]
            new_hard_negatives.append((self.full_train_loader.dataset.data[data_file_idx], \
                            self.full_train_loader.dataset.labels[data_file_idx]))

        # Step 2a) Save the sampled hard negative indeces
        #sampled_hard_negatives = sorted_weight_indeces[-num_sample_hard:]
        sampled_hard_negatives = sample_data_pool[hard_sample_idxs]
        self.save_sampled_examples(sampled_hard_negatives, window_scores[sampled_hard_negatives], "Hard-Negatives_Era-" + str(era) + ".txt")
        self.save_current_model_performance(new_hard_negatives, "Model-Performance_Hard-Negatives_Era-" + str(era))
        print ("Finished sampling hard examples and saving current model performance")

        # Step 3) Sample num_new_rand random examples. 
        # These examples are sampled from the remaining examples
        # not sampled as the new hard training examples - 
        # indeces in the range [0 : sorted_weight_indeces.shape[0] - num_new_hard]
        new_rand_negatives = []
        # Random.choice may be slow?
        rand_data_idxs = np.random.choice(np.arange(sorted_weight_indeces.shape[0] - num_sample_hard_window), size=self.num_new_rand, replace=False)
        # Now we want to add these new random datapoints
        for idx in rand_data_idxs:
            data_file_idx = sorted_weight_indeces[idx]
            new_rand_negatives.append((self.full_train_loader.dataset.data[data_file_idx], \
                            self.full_train_loader.dataset.labels[data_file_idx]))

        # Step 2a) Save the sampled hard negatives
        self.save_sampled_examples(rand_data_idxs, window_scores[rand_data_idxs], "Random-Negatives_Era-" + str(era) + ".txt")

        return new_hard_negatives, new_rand_negatives

    def save_era_scores(self, scores, title):
        """
            Save the score distribution for the era so we can track how the scores
            evolve over time!
        """
        file_path = os.path.join(self.sampled_data_path, title)
        np.save(file_path, title)

    def save_current_model_performance(self, new_hard_negatives, title):
        """
            After sampling a set of new hard negatives, we want to track
            and save the current models performance over these examples.
            Basically, we want to run and save the current models' predictions 
            over these examples!
        """
        # Update the global evaluation dataloader by switching out the negatives
        # for the new sampled hard negatives
        self.performance_tracking_loader.dataset.update_hard_neg_examples(new_hard_negatives, num_keep=0, combine_data=True)
        # Run our current model to get predictions
        current_model_preds = None

        # Put the model in eval mode!!
        self.train_model.model.eval()

        # Step 2) Evaluate over every example in the full dataset
        print ("Evaluating on new difficult examples")
        for idx, batch in enumerate(self.performance_tracking_loader):
            # Step 2a) Evaluate model on data
            inputs = batch[0].clone().float()
            labels = batch[1].clone().float()
            inputs = inputs.to(parameters.device)
            labels = labels.to(parameters.device)

            logits = self.train_model.model(inputs).squeeze(-1)
            predictions = torch.sigmoid(logits)

            # Step 2b) Update and save the model predictions
            if current_model_preds is None:
                current_model_preds = np.zeros((len(self.performance_tracking_loader.dataset), predictions.shape[1]))

            batch_size = inputs.shape[0]
            current_model_preds[idx * parameters.BATCH_SIZE: idx * parameters.BATCH_SIZE + batch_size] = predictions.cpu().detach().numpy()

        # Step 3) Save these predictions so that we can later visualize and better understand everything
        file_path = os.path.join(self.sampled_data_path, title)
        np.save(file_path, current_model_preds)


    def save_sampled_examples(self, sampled_data, data_scores, title):
        """
            Save a record of the data sampled one per line:
            (idx, score, data_file_name, label_file_name). 
            NOTE: The examples are ordered in ascending difficulty
        """
        # Save a file for the given ERA 
        file_path = os.path.join(self.sampled_data_path, title)
        with open(file_path, 'w') as f:
            # Save in reverse order to preserve the difficulty ranking
            for i in range(sampled_data.shape[0]):
            #for i in range(1, sampled_data.shape[0] + 1):
                data_file_idx = sampled_data[i]
                f.write(f'{sampled_data[i]}, {data_scores[i]}, {self.full_train_loader.dataset.data[data_file_idx]}, {self.full_train_loader.dataset.labels[data_file_idx]}\n')


    def update_dataset(self, new_hard_negatives, new_rand_negatives):
        """
            Update the dynamic dataset to have the new adversarial files
        """
        # Things to get done. 
        # Update the negative samples of the dataset to include new_rand_negatives
        # while keeping a 'self.rand_keep_ratio' of negatives.
        #
        # Update the hard negatives to include new_hard_negatives
        # while keeping 'self.hard_keep_ratio' of hard negs.

        # Do we need to pass the keep ratio? We have already used this to determine the size of the
        # new. So the size of the keep could be determined and saved?? This would solve potentially
        # most of the issues. and then it needs to be only calculated once!!!!!!
        # We can just compute how many we want to keep
        # num_total - len(new). 
        # Cases:
        #   - num_total = len(new) ==> num_keep = 0 ) itr 1
        #   - num_new = len(new) ==> num_keep = expected ) iter 2+

        # Step 1) Update the random negatives 
        rand_keep = self.num_rand_negatives - len(new_rand_negatives)
        self.train_model.train_dataloader.dataset.update_neg_examples(new_rand_negatives, \
                                                 num_keep=rand_keep, combine_data=False)

        # Step 2) Update the hard negatives
        hard_keep = self.num_hard_negatives - len(new_hard_negatives)
        self.train_model.train_dataloader.dataset.update_hard_neg_examples(new_hard_negatives, \
                                                 num_keep=hard_keep, combine_data=True)


# Create a class just for visualizing the results of one Curriculum Strategy
class Visualize_Curriculum(object):
    """
        For now make this just ad-hoc so we can see what is going on
    """
    def __init__(self, base_path, era):
        super(Visualize_Curriculum, self).__init__()

        # Step 1) Generate the different paths
        data_path, predictions_path, model_path = self.create_paths(base_path, era)
        
        # Step 1) Read in the data into a list of tuples
        features, labels, difficulty_scores = self.read_data(data_path)

        # Step 2) Read the matching predictions for the model at the
        # time the samples were created 
        model_preds = np.load(predictions_path)

        # Step 3) Load the model 
        model = torch.load(model_path, map_location=parameters.device)
        # Set to eval mode
        model.eval()

        # Step 4) Visualize the hard examples with the saved
        # predictions as well as the current prediction
        self.visualize_examples(features, labels, difficulty_scores, model_preds, model)

    def create_paths(self, base_path, era):
        """
            
        """
        sampled_data_folder = os.path.join(base_path, "Sampled_Data")

        data_path = os.path.join(sampled_data_folder, "Hard-Negatives_Era-" + str(era) + ".txt")
        model_preds = os.path.join(sampled_data_folder, "Model-Performance_Hard-Negatives_Era-" + str(era) + ".npy")
        model_path = os.path.join(base_path, "model.pt")

        return data_path, model_preds, model_path

    def read_data(self, data_path):
        """
            Read the new sampled hard negatives into:

                [(feature, label, ...), ...]
        """
        features = []
        labels = []
        difficulty_scores = []
        with open(data_path, 'r') as f:
            examples = f.readlines()
            for example in examples:
                example = example.strip()
                split = example.split(', ')
                features.append(split[2])
                labels.append(split[3])
                difficulty_scores.append(split[1])

        return features, labels, difficulty_scores


    def visualize_examples(self, features, labels, difficulty_scores, model_preds, model):
        """
            Step through example by example and do the following:
                - Load the spectrogram
                - Load the labels
                - Load the save predictions
                - Generate new predictions based on the current model
                - Visualize
        """
        for i in range(len(features)):
            # Load data
            spect = np.load(features[i])
            label = np.load(labels[i])
            scores = difficulty_scores[i]

            saved_pred = model_preds[i]

            # Run the model over this data example!
            log_spect = 10 * np.log10(spect)
            spect_expand = np.expand_dims(log_spect, axis=0)
            # Transform the slice!!!! 
            spect_expand = (spect_expand - np.mean(spect_expand)) / np.std(spect_expand)
            spect_expand = torch.from_numpy(spect_expand).float()
            spect_expand = spect_expand.to(parameters.device)

            outputs = model(spect_expand).view(-1, 1).squeeze()
            # Apply sigmoid 
            model_pred = torch.sigmoid(outputs).cpu().detach().numpy()

            # Visualize!!
            visualize(log_spect, [saved_pred, model_pred], label, title="Score: " + str(scores) + " " + features[i])
   

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--base_path', type=str,
        help='The path to the model folder with all the good information!')
    parser.add_argument('--era', type=int,
        help='Which era do we want to explore!')

    args = parser.parse_args()

    Visualize_Curriculum(args.base_path, args.era)
    


if __name__ == '__main__':
    main()







    