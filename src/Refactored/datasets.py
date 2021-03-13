import numpy as np
import torch
from torch.utils import data
from torchvision import transforms
import os
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import glob
import parameters

class Subsampled_ElephantDataset(data.Dataset):
    """
        Dataset used for training or testing a subsampled elephant dataset.
        To deal with the massive imbalance present within the elephant data
        (i.e. negative data >> positive data) we must strategically re-balance 
        the data distribution. This dataset class allows for different means
        of data class re-balancing. 

        By default we use random majority class undersampling!

        Also allows for manual setting of the negative or positive data!

        THINGS THAT A SUBSAMPLED DATSET SHOULD DO!!!!!
        - This dataset is meant to be used to train the model and evaluate the model DURING TRAINING
        - A dataset class should be responsible for basic data things
            - Random undersampling
            - Setting the pos and negative features
            - representing the current dataset for either train/test runs!
            - transforming the data
        - Takes in the directory that has all of the windows!
        - Collect all of the positive samples
        - Sample a set of negative samples:
            - By default sub-sample from the full set of negatives.
            - If passed a set of negatives use those!!!!!
        - Allow for changing the positive and negative features (this allows for updating the dataset we are on!)
            - Here we should assume that we are passed an array of tuples with (feat_file, label_file,...). Basically,
            each tuple contains what we need to generate the necessary new neg or pos features
            - In this way we can avoid this glob stuff! Except in the very beginning

        - Then that should be everything!!!!!

    """
    def __init__(self, data_path, neg_ratio=1, neg_features=None, normalization="norm", 
                log_scale=True, transform=None, 
                shift_windows=False, seed=8):
        """
            @TODO: add comments for these!!!
        """
        # SET THE SEED???

        self.user_transforms = transform
        self.normalization = normalization
        self.log_scale = log_scale 
        self.shift_windows = shift_windows
        self.neg_ratio = neg_ratio

        # Step 1) Initialize the positive examples
        self.pos_features = None
        self.pos_labels = None
        self.init_positive_examples()

        # Step 2) Initialize the negative examples, 
        # either through direct assignment or through
        # random undersampling.
        self.neg_features = None
        self.neg_labels = None
        if neg_features is not None:
            self.set_neg_examples(neg_features)
        # By default randomly undersample to get the negative features
        else: 
            self.undersample_negative_features()
        
        # Step 3) Combine the pos and neg features to get the complete data
        self.combine_data()

        # Check for consistancy
        assert len(self.data) == len(self.labels)

        print("================================")
        print("=======  ElephantDataset =======")
        print("================================")
        print()
        print("Number of examples for training / evaluation".format(len(self.data)))
        print("Number of positive examples {}".format(len(self.pos_features)))
        print("Number of negative examples {}".format(len(self.neg_features)))
        print("Total number of negative examples {}.".format(len(self.all_neg_features)))
        print('Normalizing with {} and scaling {}'.format(preprocess, scale))


    def init_positive_examples(self, data_path):
        """
            Grabs the positive features / labels from the data_path dir

            @TODO: Make this more clear later
        """
        self.pos_features = glog.glob(os.path.join(data_path, "*_pos-features_*"), recursive=True)
        # Now collect the corresponding labels!
        self.pos_labels = []
        for feature_path in self.pos_features:
            feature_parts = feature_path.split("pos-features")
            self.pos_labels.append(feature_parts[0] + "pos-labels" + feature_parts[1])


    def undersample_negative_features(self):
        """
            Perform majority class random undersampling
        """
        num_neg_samples = self.neg_ratio * len(self.pos_features)

        # Collect all of the negative features and sample a random set of them.
        # For now let us just keep them around! (MAY CHANGE!!)
        self.all_neg_features = glog.glob(os.path.join(data_path, "*_neg-features_*"), recursive=True)
        # Sample without replacement
        sampled_idxs = np.random.choice(len(self.all_neg_features), num_neg_samples, replace=False)

        self.neg_features = self.all_neg_features[sampled_idxs]

    
    def undersample_negative_features_to_balance(self):
        """
            Perform majority class random undersampling to balance
            the number of pos and neg features to be a factor 
            of 'ratio' different.
        """
        # Step 1) Compute how many negative examples to sample to rebalance
        # as: len(pos) * neg_ratio = len(neg)
        num_neg_samples = self.neg_ratio * len(self.pos_features) - len(self.neg_features)

        # Step 2) Collect all of the negative features and discard the negative features
        # we have already sampled
        neg_features_to_sample = list(np.setdiff1d(self.all_neg_features, self.neg_features))

        # Step 3) Sample without replacement
        sampled_idxs = np.random.choice(len(neg_features_to_sample), num_neg_samples, replace=False)

        # Step 4) Append these new negative features
        self.neg_features += neg_features_to_sample[sampled_idxs]
    

    def combine_data(self):
        """
            Combines the pos and neg data examples! Outputs some just logging if 
            we are changing the data
        """
        # Logging
        if self.data is not None:
            print("Total number of samples was {} and is now {}".format(len(self.data), len(self.pos_features) + len(self.neg_features)))

        # Combine
        self.data = self.pos_features + self.neg_features
        self.labels = self.pos_labels + self.neg_labels


    def set_pos_examples(self, pos_examples):
        """
            Unlike before, assume that pos_features is in the following form:

                [(feature, label, ...), ...]

            Namely, we have a list of each data example as a tuple 
            that includes the feature, the label, and any other things we want!
        """
        print("Length of pos_features was {} and is now {} ".format(len(self.pos_features), len(pos_examples)))
        print("Length of neg_features is {}".format(len(self.neg_features)))
        self.pos_features = []
        self.pos_labels = []
        for feature, label in pos_examples:
            self.pos_features.append(features)
            self.pos_labels.append(label)
        
        self.combine_data()

    def add_positive_examples_from_dir(self, data_dir):
        """ 
            In certain cases we want to do the actual collection of
            the positive examples form a different directory using glob, etc.
            (e.g. generated calls!!)
        """
        # Graph the postive features
        pos_feature_paths = glog.glob(os.path.join(data_dir, "*_spectro.npy"), recursive=True)
        pos_examples = []
        for feature_path in pos_feature_paths:
            # Get the corresponding label file
            feature_start = feature_path[:-12]
            label_path = feature_start + "_labels.npy"

            # Add the example
            pos_examples.append((feature_path, label_path))

        # Call class function to add examples
        self.add_pos_examples(pos_examples)


    def add_pos_examples(self, pos_examples):
        """
            Unlike before, assume that pos_features is in the following form:

                [(feature, label, ...), ...]

            Namely, we have a list of each data example as a tuple 
            that includes the feature, the label, and any other things we want!

            @TODO comments
        """
        print("Length of pos_features was {} and is now {} ".format(len(self.pos_features), len(self.pos_features) + len(pos_examples)))
        print("Length of neg_features is {}".format(len(self.neg_features)))
        for feature, label in pos_examples:
            self.pos_features.append(features)
            self.pos_labels.append(label)
        
        self.combine_data()

    def set_neg_examples(self, neg_examples):
        """
            Unlike before, assume that neg_features is in the following form:

                [(feature, label, ...), ...]

            Namely, we have a list of each data example as a tuple 
            that includes the feature, the label, and any other things we want!
        """
        print("Length of neg_features was {} and is now {} ".format(len(self.neg_features), len(neg_examples)))
        print("Length of pos_features is {}".format(len(self.pos_features)))
        self.neg_features = []
        self.neg_labels = []
        for feature, label in neg_examples:
            self.neg_features.append(features)
            self.neg_labels.append(label)
        
        self.combine_data() 

    def add_neg_examples(self, neg_examples):
        """
            Unlike before, assume that pos_features is in the following form:

                [(feature, label, ...), ...]

            Namely, we have a list of each data example as a tuple 
            that includes the feature, the label, and any other things we want!

            @TODO comments
        """
        print("Length of neg_features was {} and is now {} ".format(len(self.neg_features), len(self.neg_features) + len(neg_examples)))
        print("Length of pos_features is {}".format(len(self.pos_features)))
        for feature, label in pos_examples:
            self.neg_features.append(features)
            self.neg_labels.append(label)
        
        self.combine_data()


    def __len__(self):
        return len(self.data)

    """
    Return a single element at provided index
    """
    def __getitem__(self, index):
        feature = np.load(self.data[index])
        label = np.load(self.labels[index])

        # This we need to update more!
        feature = self.apply_transforms(feature)

        if self.user_transforms:
            feature = self.user_transforms(feature)

        # Honestly may be worth pre-process this
        feature = torch.from_numpy(feature).float()  
        label = torch.from_numpy(label).float()

        return feature, label, (self.features[index], self.labels[index]) # Include the data files!


    def apply_transforms(self, data):
        # Apply a log transform to the spectrogram! This is equivalent to the convert to db
        if self.scale:
            data = 10 * np.log10(data)

        # Normalize Features
        if self.preprocess == "norm":
            data = (data - np.mean(data)) / np.std(data)
        elif self.preprocess == "feature": # Look into the normalize we used in CS224S
            data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

        return data


# THis class is just for the full dataset!!!!
class Full_ElephantDataset(data.Dataset):
    """
        Dataset used for training or testing a subsampled elephant dataset.
        To deal with the massive imbalance present within the elephant data
        (i.e. negative data >> positive data) we must strategically re-balance 
        the data distribution. This dataset class allows for different means
        of data class re-balancing. 

        By default we use random majority class undersampling!

        Also allows for manual setting of the negative or positive data!

        Things that we want to do here!!
            - This class is just for the complete elephant dataset!
            - 

    """
    def __init__(self, data_path, normalization="norm", log_scale=True, transform=None, 
            shift_windows=False, seed=8):

        # Should look into this with Vrinda about how we want to deal with data augmentation transforms!
        self.user_transforms = transform
        self.normalization = normalization
        self.log_scale = log_scale 
        self.shift_windows = shift_windows

        # Init positive and negative data complete data
        self.init_data(data_path)

        assert len(self.data) == len(self.labels)

        print("================================")
        print("=======  ElephantDataset =======")
        print("================================")
        print()
        print("Number of total examples".format(len(self.data)))
        print("Number of positive examples {}".format(len(self.pos_features)))
        print("Number of negative examples {}".format(len(self.neg_features)))
        print('Normalizing with {} and scaling {}'.format(preprocess, scale))


    def init_data(self, data_path):
        """
            Intialize the positive and neg features (complete neg features)

            @TODO be more specificc
        """
        self.pos_features = self.pos_features = glog.glob(os.path.join(data_path, "*_pos-features_*"), recursive=True)
        self.pos_labels = []
        for feature_path in self.pos_features:
            feature_parts = feature_path.split("pos-features")
            self.pos_labels.append(feature_parts[0] + "pos-labels" + feature_parts[1])

        self.neg_features = glog.glob(os.path.join(data_path, "*_neg-features_*"), recursive=True)
        self.neg_labels = []
        for feature_path in self.neg_features:
            feature_parts = feature_path.split("neg-features")
            self.neg_labels.append(feature_parts[0] + "neg-labels" + feature_parts[1])

        # Combine the positive and negative examples!
        self.data = self.pos_features + self.neg_features
        self.labels = self.pos_labels + self.neg_labels




    def __len__(self):
        return len(self.data)

    """
    Return a single element at provided index
    """
    def __getitem__(self, index):
        feature = np.load(self.data[index])
        label = np.load(self.labels[index])

        feature = self.apply_transforms(feature)

        if self.user_transforms:
            feature = self.user_transforms(feature)

        # Honestly may be worth pre-process this
        feature = torch.from_numpy(feature).float()  
        label = torch.from_numpy(label).float()

        return feature, label, (self.features[index], self.labels[index]) # Include the data files!


    def apply_transforms(self, data):
        # Apply a log transform to the spectrogram! This is equivalent to the convert to db
        if self.scale:
            data = 10 * np.log10(data)

        # Normalize Features
        if self.preprocess == "norm":
            data = (data - np.mean(data)) / np.std(data)
        elif self.preprocess == "feature":
            data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

        return data
