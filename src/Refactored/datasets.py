import numpy as np
import torch
from torch.utils import data
from torchvision import transforms
import os
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import glob
import parameters

from utils import set_seed

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

        # Should look into this with Vrinda about how we want to deal with data augmentation transforms!
        self.user_transforms = transform
        self.normalization = normalization
        self.log_scale = log_scale 
        self.shift_windows = shift_windows

        # Initialize the positive and subsampled negative data
        self.pos_features = glog.glob(os.path.join(data_path, "*_pos-features_*"), recursive=True)
        # Use provided neg_features
        if neg_features is not None:
            self.neg_features = neg_features
        # By default randomly undersample to get the negative features
        else: 
            self.undersample_negative_features(neg_ratio)
        
        self.intialize_data(init_pos=True, init_neg=True)

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

    def undersample_negative_features(self, neg_ratio):
        """
            Perform majority class random undersampling
        """
        num_neg_samples = neg_ratio * len(self.pos_features)

        # Collect all of the negative features and sample a random set of them.
        # For now let us just keep them around! (MAY CHANGE)
        self.all_neg_features = glog.glob(os.path.join(data_path, "*_neg-features_*"), recursive=True)
        # Sample without replacement
        sampled_idxs = np.random.choice(len(self.all_neg_features), num_neg_samples, replace=False)

        self.neg_features = self.all_neg_features[sampled_idxs]


    def init_data(self, init_pos=True, init_neg=True):
        """
            Initialize the positive and negative labels depending on
            the initialization flags 'init_pos' and 'init_neg'. 
            After initializing any necessary data, combine the positive and negative examples!
        """
        # Initialize the positive examples
        if init_pos:
            self.pos_labels = []
            for feature_path in self.pos_features:
                feature_parts = feature_path.split("pos-features")
                self.pos_labels.append(glob.glob(feature_parts[0] + "pos-labels" + feature_parts[1])[0])

        # Initialize the negative examples
        if init_neg:
            # Initialize the 
            self.neg_labels = []
            for feature_path in self.neg_features:
                feature_parts = feature_path.split("neg-features")
                self.neg_labels.append(glob.glob(feature_parts[0] + "neg-labels" + feature_parts[1])[0])

        # Combine the positive and negative examples!
        self.data = self.pos_features + self.neg_features
        self.labels = self.pos_labels + self.neg_labels


    def set_pos_features(self, pos_features):
        print("Length of pos_features was {} and is now {} ".format(len(self.pos_features), len(pos_features)))
        print("Length of neg_features is {}".format(len(self.neg_features)))
        self.pos_features = pos_features
        self.intialize_data(init_pos=True, init_neg=False)

    def set_neg_features(self, neg_features):
        print("Length of neg_features was {} and is now {} ".format(len(self.neg_features), len(neg_features)))
        print("Length of pos_features is {}".format(len(self.pos_features)))
        self.neg_features = neg_features
        self.intialize_data(init_pos=False, init_neg=True)

    def add_neg_features(self, neg_features):
        print("Length of neg_features was {} and grew to {} ".format(len(self.neg_features), len(neg_features) + len(self.neg_features)))
        self.neg_features += neg_features
        self.intialize_data(init_pos=False, init_neg=True)

    def set_featues(self, pos_features, neg_features):
        print("Length of pos_features was {} and is now {} ".format(len(self.pos_features), len(pos_features)))
        print("Length of neg_features was {} and is now {} ".format(len(self.neg_features), len(neg_features)))
        self.pos_features = pos_features
        self.neg_features = neg_features
        self.intialize_data(init_pos=True, init_neg=True)


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
        elif self.preprocess == "globalnorm":
            data = (data - 132.228) / 726.319 # Calculated these over the training dataset 
        elif self.preprocess == "feature":
            data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

        return data


# THis class is just for the full dataset!!!!
class Subsampled_ElephantDataset(data.Dataset):
    """
        Dataset used for training or testing a subsampled elephant dataset.
        To deal with the massive imbalance present within the elephant data
        (i.e. negative data >> positive data) we must strategically re-balance 
        the data distribution. This dataset class allows for different means
        of data class re-balancing. 

        By default we use random majority class undersampling!

        Also allows for manual setting of the negative or positive data!
    """
    def __init__(self, data_path, neg_ratio=1, neg_features=None, normalization="norm", log_scale=True, transform=None, 
            shift_windows=False, seed=8):

        # Should look into this with Vrinda about how we want to deal with data augmentation transforms!
        self.user_transforms = transform
        self.normalization = normalization
        self.log_scale = log_scale 
        self.shift_windows = shift_windows

        # Initialize the positive and subsampled negative data
        self.pos_features = glog.glob(os.path.join(data_path, "*_pos-features_*"), recursive=True)
        # Use provided neg_features
        if neg_features is not None:
            self.neg_features = neg_features
        # By default randomly undersample to get the negative features
        else: 
            self.undersample_negative_features(neg_ratio)
        
        self.intialize_data(init_pos=True, init_neg=True)

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

    def undersample_negative_features(self, neg_ratio):
        """
            Perform majority class random undersampling
        """
        num_neg_samples = neg_ratio * len(self.pos_features)

        # Collect all of the negative features and sample a random set of them.
        # For now let us just keep them around! (MAY CHANGE)
        self.all_neg_features = glog.glob(os.path.join(data_path, "*_neg-features_*"), recursive=True)
        # Sample without replacement
        sampled_idxs = np.random.choice(len(self.all_neg_features), num_neg_samples, replace=False)

        self.neg_features = self.all_neg_features[sampled_idxs]


    def init_data(self, init_pos=True, init_neg=True):
        """
            Initialize the positive and negative labels depending on
            the initialization flags 'init_pos' and 'init_neg'. 
            After initializing any necessary data, combine the positive and negative examples!
        """
        # Initialize the positive examples
        if init_pos:
            self.pos_labels = []
            for feature_path in self.pos_features:
                feature_parts = feature_path.split("pos-features")
                self.pos_labels.append(glob.glob(feature_parts[0] + "pos-labels" + feature_parts[1])[0])

        # Initialize the negative examples
        if init_neg:
            # Initialize the 
            self.neg_labels = []
            for feature_path in self.neg_features:
                feature_parts = feature_path.split("neg-features")
                self.neg_labels.append(glob.glob(feature_parts[0] + "neg-labels" + feature_parts[1])[0])

        # Combine the positive and negative examples!
        self.data = self.pos_features + self.neg_features
        self.labels = self.pos_labels + self.neg_labels


    def set_pos_features(self, pos_features):
        print("Length of pos_features was {} and is now {} ".format(len(self.pos_features), len(pos_features)))
        print("Length of neg_features is {}".format(len(self.neg_features)))
        self.pos_features = pos_features
        self.intialize_data(init_pos=True, init_neg=False)

    def set_neg_features(self, neg_features):
        print("Length of neg_features was {} and is now {} ".format(len(self.neg_features), len(neg_features)))
        print("Length of pos_features is {}".format(len(self.pos_features)))
        self.neg_features = neg_features
        self.intialize_data(init_pos=False, init_neg=True)

    def add_neg_features(self, neg_features):
        print("Length of neg_features was {} and grew to {} ".format(len(self.neg_features), len(neg_features) + len(self.neg_features)))
        self.neg_features += neg_features
        self.intialize_data(init_pos=False, init_neg=True)

    def set_featues(self, pos_features, neg_features):
        print("Length of pos_features was {} and is now {} ".format(len(self.pos_features), len(pos_features)))
        print("Length of neg_features was {} and is now {} ".format(len(self.neg_features), len(neg_features)))
        self.pos_features = pos_features
        self.neg_features = neg_features
        self.intialize_data(init_pos=True, init_neg=True)


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
        elif self.preprocess == "globalnorm":
            data = (data - 132.228) / 726.319 # Calculated these over the training dataset 
        elif self.preprocess == "feature":
            data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

        return data
