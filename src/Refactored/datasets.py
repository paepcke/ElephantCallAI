import numpy as np
import torch
from torch.utils import data
from torchvision import transforms
import os
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import glob
import parameters
from scipy.ndimage import gaussian_filter1d


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
    def __init__(self, data_path, neg_ratio=1, neg_features=None, 
                #seperate_hard_samples=False, 
                normalization="norm", 
                log_scale=True, gaussian_smooth=0, transform=None, 
                shift_windows=False, seed=8):
        """
            Params
            ------
            @param seperate_hard_samples: Flag indicating that we
            want to have seperate arrays for hard and regular negative
            samples.
            @type spect_file: bool. 
        """
        # SET THE SEED???

        self.user_transforms = transform
        self.normalization = normalization
        self.log_scale = log_scale 
        self.shift_windows = shift_windows
        self.neg_ratio = neg_ratio
        self.data_path = data_path
        # Apply gaussian smoothing to the labels
        self.gaussian_smooth = gaussian_smooth

        # Step 1) Initialize the positive examples
        self.pos_features = None
        self.pos_labels = None
        self.init_positive_examples(data_path)

        # Step 2) Initialize the negative examples, 
        # either through direct assignment or through
        # random undersampling.
        self.neg_features = None
        self.neg_labels = None
        if neg_features is not None:
            self.set_neg_examples(neg_features)
        # By default randomly undersample to get the negative features
        else: 
            self.undersample_negative_features(data_path)

        # Step 3) Initialize empty array to seperately store the hard negative
        # samples. NOTE: for now this is only used in curriculum training and
        # these are empty lists otherwise!
        #self.seperate_hard_samples = seperate_hard_samples
        #if self.seperate_hard_samples:
        self.hard_neg_features = []
        self.hard_neg_labels = []
        
        # Step 3) Combine the pos and neg features to get the complete data
        self.data = None
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
        print('Normalizing with {} and scaling {}'.format(normalization, log_scale))


    def init_positive_examples(self, data_path):
        """
            Grabs the positive features / labels from the data_path dir

            @TODO: Make this more clear later
        """
        self.pos_features = glob.glob(os.path.join(data_path, "*_pos-features_*"), recursive=True)
        # Now collect the corresponding labels!
        self.pos_labels = []
        for feature_path in self.pos_features:
            feature_parts = feature_path.split("pos-features")
            self.pos_labels.append(feature_parts[0] + "pos-labels" + feature_parts[1])


    def undersample_negative_features(self, data_path):
        """
            Perform majority class random undersampling
        """
        num_neg_samples = self.neg_ratio * len(self.pos_features)

        # Collect all of the negative features and sample a random set of them.
        # For now let us just keep them around! (MAY CHANGE!!)
        self.all_neg_features = glob.glob(os.path.join(data_path, "*_neg-features_*"), recursive=True)
        # Sample without replacement
        sampled_idxs = np.random.choice(len(self.all_neg_features), num_neg_samples, replace=False)

        self.neg_features = [self.all_neg_features[idx] for idx in sampled_idxs]
        # Set the corresponding negative labels
        self.neg_labels = []
        for feature_path in self.neg_features:
            feature_parts = feature_path.split("neg-features")
            self.neg_labels.append(feature_parts[0] + "neg-labels" + feature_parts[1])

    
    def undersample_negative_features_to_balance(self):
        """
            Perform majority class random undersampling to balance
            the number of pos and neg features to be a factor 
            of 'ratio' different.
        """
        # Step 1) Compute how many negative examples to sample to rebalance.
        # Note that here we keep any neg_features that already exist.
        # as: len(pos) * neg_ratio = len(neg)
        num_neg_samples = self.neg_ratio * len(self.pos_features) - len(self.neg_features)

        # Step 2) Collect all of the negative features and discard the negative features
        # we have already sampled
        neg_features_to_sample = list(np.setdiff1d(self.all_neg_features, self.neg_features))

        # Step 3) Sample without replacement
        sampled_idxs = np.random.choice(len(neg_features_to_sample), num_neg_samples, replace=False)

        # Step 4) Append these new negative features
        self.neg_features += [neg_features_to_sample[idx] for idx in sampled_idxs]
        # Set the corresponding negative labels
        for feature_path in self.neg_features:
            feature_parts = feature_path.split("neg-features")
            self.neg_labels.append(feature_parts[0] + "neg-labels" + feature_parts[1])

        # Make sure to combine the data to reflect the added negative features
        self.combine_data()
        print("Undersampling more negative features to match ratio!")
        print ("New number of negative features is {}".format(len(self.neg_features)))
    
    """
        Things that could be done easiest to most involved:
        - Simply have the curriculum model sample a certain ratio of hard examples
        and new negatives.
        - Have the curriculum model sample a set of hard examples and then use
        the dataset to upsample negative features to balance?
        - Would we want to keep around some data and what data would that be:
            - There are two types of data fed to the model: hard negatives and
            regular background. 
            - Way to make it the mosssttt flexible
                - Have an array for the hard examples and an array for the
                random background examples
                - Have a ratio for how much of each we keep and replace!
                Naemly, we have a hard_keep ratio and a rand_keep 
                ratio. 
                - Then we do two things!!! 
                    - For both we keep a random subset of 
                    hard and random samples
                    - Then add a new set of hard and a 
                    new set of random!

            - Things we need
                - Method that subsamples / keeps a random portion
                of either the hard or random samples.
                - Method that sets the hard and random negative samples
                (These will be fed by the curriculum model) even the
                random sampling of the remaining data! 
    """

    def combine_data(self):
        """
            Combines the pos and neg data examples! Outputs some just logging if 
            we are changing the data
        """
        # Logging
        if self.data is not None:
            print("Total number of samples was {} and is now {}".format(len(self.data), len(self.pos_features) \
                                                            + len(self.neg_features) + len(self.hard_neg_features)))

        # Combine - NOTE: that if unused the hard feats/labels are empty!
        self.data = self.pos_features + self.neg_features + self.hard_neg_features 
        self.labels = self.pos_labels + self.neg_labels + self.hard_neg_labels


    # Next idea!!!!! Define the method with a keep ratio!
    # Basically, keep ratio defines how much is kept 
    # which we randomly sample and then add too. Special
    # cases are 0% and 100% where we empty the bish
    # or keep all of it!
    def update_examples(self, new_examples, features, labels, num_keep):#, combine_data=True):

        # First we need to incorperate keep ratio
        new_features = []
        new_labels = []
        if len(new_examples) > 0 and num_keep > 0:
            # Sample the ones we want to keep
            kept_idxs = np.random.choice(np.arange(len(features)), num_keep, replace=False)

            new_features = [features[idx] for idx in kept_idxs]
            new_labels = [labels[idx] for idx in kept_idxs] 
        elif len(new_examples) == 0:
            new_features = features
            new_labels = labels

        # Now we want to actually add the new examples
        for feature, label in new_examples:
            new_features.append(feature)
            new_labels.append(label)
        
        return new_features, new_labels 


    def update_pos_examples(self, pos_examples, num_keep, combine_data=True):
        """
            Unlike before, assume that pos_features is in the following form:

                [(feature, label, ...), ...]

            Namely, we have a list of each data example as a tuple 
            that includes the feature, the label, and any other things we want!

            @param combine_data: flag indicating whether we should in the
            end combine the new data. Set this to false if we plan to
            do multiple operations on the dataset and want to call
            combine yourself! 
        """
        print("Length of pos_features was {} and is now {} ".format(len(self.pos_features), len(pos_examples)))
        print("Length of neg_features is {}".format(len(self.neg_features) + len(self.hard_neg_features)))
        self.pos_features, self.pos_labels = self.update_examples(pos_examples, self.pos_features, \
                                                    self.pos_labels, num_keep)

        if combine_data:
            self.combine_data()

    def update_neg_examples(self, neg_examples, num_keep, combine_data=True):
        """
            Unlike before, assume that neg_examples is in the following form:

                [(feature, label, ...), ...]

            Namely, we have a list of each data example as a tuple 
            that includes the feature, the label, and any other things we want!
        """
        new_features_length = num_keep + len(neg_examples) + len(self.hard_neg_features)
        print("Length of neg_features was {} and is now {} ".format(len(self.neg_features)+ len(self.hard_neg_features),\
                                                             new_features_length))
        print("Length of pos_features is {}".format(len(self.pos_features)))
        self.neg_features, self.neg_labels = self.update_examples(neg_examples, self.neg_features, \
                                                    self.neg_labels, num_keep)

        if combine_data:
            self.combine_data()

    def update_hard_neg_examples(self, hard_neg_examples, num_keep, combine_data=True):
        """
            Unlike before, assume that neg_examples is in the following form:

                [(feature, label, ...), ...]

            Namely, we have a list of each data example as a tuple 
            that includes the feature, the label, and any other things we want!
        """
        new_features_length = num_keep + len(hard_neg_examples) + len(self.neg_features)
        print("Length of neg_features was {} and is now {} ".format(len(self.neg_features)+ len(self.hard_neg_features),\
                                                             new_features_length))
        print("Length of pos_features is {}".format(len(self.pos_features)))
        self.hard_neg_features, self.hard_neg_labels = self.update_examples(hard_neg_examples, self.hard_neg_features, \
                                                    self.hard_neg_labels, num_keep)

        if combine_data:
            self.combine_data()


    """
        This stuff below is a bit antiquated but we keep it around so that past code still works!!!!!!!!!
    """
    # Let us try something to unify some of the logic here!
    def add_examples(self, new_examples, features, labels, combine_data=True):
        """
            Generic wrapper function for setting the examples
            one of the instance variable arrays of the dataset class.

            @param features: feature list that can be one of three option
            (self.pos_features, self.neg_features, self.hard_neg_features)
            @param labels: labels list that can be one of three option
            (self.pos_labels, self.neg_labels, self.hard_neg_labels)
            @param combine_data: flag indicating whether we should in the
            end combine the new data. Set this to false if we plan to
            do multiple operations on the dataset and want to call
            combine yourself! 
        """
        for feature, label in new_examples:
            features.append(feature)
            labels.append(label)
        
        if combine_data:
            self.combine_data()

    
    def set_pos_examples(self, pos_examples, combine_data=True):
        """
            Unlike before, assume that pos_features is in the following form:

                [(feature, label, ...), ...]

            Namely, we have a list of each data example as a tuple 
            that includes the feature, the label, and any other things we want!

            @param combine_data: flag indicating whether we should in the
            end combine the new data. Set this to false if we plan to
            do multiple operations on the dataset and want to call
            combine yourself! 
        """
        print("Length of pos_features was {} and is now {} ".format(len(self.pos_features), len(pos_examples)))
        print("Length of neg_features is {}".format(len(self.neg_features) + len(self.hard_neg_features)))
        self.pos_features = []
        self.pos_labels = []
        self.add_examples(pos_examples, self.pos_features, self.pos_labels, combine_data=combine_data)
        
    

    '''
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
            self.pos_features.append(feature)
            self.pos_labels.append(label)
        
        self.combine_data()
    '''

    # Need to update this!!!!!
    def add_positive_examples_from_dir(self, data_dir):
        """ 
            In certain cases we want to do the actual collection of
            the positive examples form a different directory using glob, etc.
            (e.g. generated calls!!)
        """
        # Graph the postive features
        print ("Adding generated data!")
        pos_feature_paths = glob.glob(os.path.join(data_dir, "*_spectro.npy"), recursive=True)
        pos_examples = []
        for feature_path in pos_feature_paths:
            # Get the corresponding label file
            feature_start = feature_path[:-12]
            label_path = feature_start + "_labels.npy"

            # Add the example
            pos_examples.append((feature_path, label_path))

        # Call class function to add examples
        self.add_pos_examples(pos_examples)


    def add_pos_examples(self, pos_examples, combine_data=True):
        """
            Unlike before, assume that pos_features is in the following form:

                [(feature, label, ...), ...]

            Namely, we have a list of each data example as a tuple 
            that includes the feature, the label, and any other things we want!

            @param combine_data: flag indicating whether we should in the
            end combine the new data. Set this to false if we plan to
            do multiple operations on the dataset and want to call
            combine yourself! 
        """
        print("Length of pos_features was {} and is now {} ".format(len(self.pos_features), len(self.pos_features) + len(pos_examples)))
        print("Length of neg_features is {}".format(len(self.neg_features) + len(self.hard_neg_features)))
        self.add_examples(pos_examples, self.pos_features, self.pos_labels, combine_data=combine_data)
        

    '''
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
            self.pos_features.append(feature)
            self.pos_labels.append(label)
        
        self.combine_data()
    '''


    def set_neg_examples(self, neg_examples, combine_data=True):
        """
            Unlike before, assume that neg_examples is in the following form:

                [(feature, label, ...), ...]

            Namely, we have a list of each data example as a tuple 
            that includes the feature, the label, and any other things we want!
        """
        print("Length of neg_features was {} and is now {} ".format(len(self.neg_features)+ len(self.hard_neg_features),\
                                                             len(neg_examples) + len(self.hard_neg_features)))
        print("Length of pos_features is {}".format(len(self.pos_features)))
        self.neg_features = []
        self.neg_labels = []
        self.add_examples(neg_examples, self.neg_features, self.neg_labels, combine_data=combine_data)


    '''
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
            self.neg_features.append(feature)
            self.neg_labels.append(label)
        
        self.combine_data() 
    '''

    def add_neg_examples(self, neg_examples, combine_data=True):
        """
            Unlike before, assume that neg_examples is in the following form:

                [(feature, label, ...), ...]

            Namely, we have a list of each data example as a tuple 
            that includes the feature, the label, and any other things we want!
        """
        print("Length of neg_features was {} and is now {} ".format(len(self.neg_features)+ len(self.hard_neg_features),\
                                                             len(self.neg_features) + len(neg_examples) + len(self.hard_neg_features)))
        print("Length of pos_features is {}".format(len(self.pos_features)))
        self.add_examples(neg_examples, self.neg_features, self.neg_labels, combine_data=combine_data)

    '''
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
            self.neg_features.append(feature)
            self.neg_labels.append(label)
        
        self.combine_data()
    '''

    def set_hard_neg_examples(self, hard_neg_examples, combine_data=True):
        """
            Unlike before, assume that neg_examples is in the following form:

                [(feature, label, ...), ...]

            Namely, we have a list of each data example as a tuple 
            that includes the feature, the label, and any other things we want!
        """
        print("Length of neg_features was {} and is now {} ".format(len(self.neg_features)+ len(self.hard_neg_features),\
                                                             len(hard_neg_examples) + len(self.neg_features)))
        print("Length of pos_features is {}".format(len(self.pos_features)))
        self.hard_neg_features = []
        self.hard_neg_labels = []
        self.add_examples(hard_neg_examples, self.hard_neg_features, self.hard_neg_labels, combine_data=combine_data)

    def add_hard_neg_examples(self, hard_neg_examples, combine_data=True):
        """
            Unlike before, assume that neg_examples is in the following form:

                [(feature, label, ...), ...]

            Namely, we have a list of each data example as a tuple 
            that includes the feature, the label, and any other things we want!
        """
        print("Length of neg_features was {} and is now {} ".format(len(self.neg_features)+ len(self.hard_neg_features),\
                                                             len(self.neg_features) + len(hard_neg_examples) + len(self.hard_neg_features)))
        print("Length of pos_features is {}".format(len(self.pos_features)))
        self.add_examples(hard_neg_examples, self.hard_neg_features, self.hard_neg_labels, combine_data=combine_data)




    def __len__(self):
        return len(self.data)

    """
    Return a single element at provided index
    """
    def __getitem__(self, index):
        feature = np.load(self.data[index])
        label = np.load(self.labels[index])

        # This we need to update more!
        feature = self.apply_data_transforms(feature)
        label = self.apply_label_transforms(label)

        if self.user_transforms:
            feature = self.user_transforms(feature)

        # Honestly may be worth pre-process this
        feature = torch.from_numpy(feature).float() 
        # DO FOR NOW 
        if feature.shape[0] == 77:
            feature = feature.T
        label = torch.from_numpy(label).float()

        return feature, label, (self.data[index], self.labels[index]) # Include the data files!

    def apply_label_transforms(self, label):
        # Gaussian smooth the labels!
        if self.gaussian_smooth != 0:
            label = gaussian_filter1d(label,sigma=self.gaussian_smooth)

        return label


    def apply_data_transforms(self, data):
        # Look into librosa.utils.normalize!

        # Apply a log transform to the spectrogram! This is equivalent to the convert to db
        if self.log_scale:
            data = 10 * np.log10(data)

        # Normalize Features
        if self.normalization == "norm":
            data = (data - np.mean(data)) / np.std(data)
        elif self.normalization == "feature": # Look into the normalize we used in CS224S
            data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)


        return data


# THis class is just for the full dataset!!!!
class Full_ElephantDataset(data.Dataset):
    """
        Dataset class for the full elephant dataset.

        Importantly, if we set the 'only_negative' flage 
        we only include the negative samples primarily 
        in the case of dynamic dataset training
        as in the 2-stage model and Curriculum setting!
    """
    def __init__(self, data_path, only_negative=False, normalization="norm", log_scale=True, transform=None, 
            shift_windows=False, gaussian_smooth=0, seed=8):

        # Should look into this with Vrinda about how we want to deal with data augmentation transforms!
        self.user_transforms = transform
        self.normalization = normalization
        self.log_scale = log_scale 
        self.shift_windows = shift_windows
        # Apply gaussian smoothing to the labels
        self.gaussian_smooth = gaussian_smooth

        # Init positive and negative data complete data
        self.init_data(data_path, only_negative)

        assert len(self.data) == len(self.labels)

        print("================================")
        print("=======  ElephantDataset =======")
        print("================================")
        print()
        print("Number of total examples".format(len(self.data)))
        print("Number of positive examples {}".format(len(self.pos_features)))
        print("Number of negative examples {}".format(len(self.neg_features)))
        print('Normalizing with {} and scaling {}'.format(normalization, log_scale))


    def init_data(self, data_path, only_negative):
        """
            Intialize the positive and neg features (complete neg features)

            @TODO be more specificc
        """
        self.pos_features = []
        self.pos_labels = []
        if not only_negative:
            self.pos_features = self.pos_features = glob.glob(os.path.join(data_path, "*_pos-features_*"), recursive=True)
            for feature_path in self.pos_features:
                feature_parts = feature_path.split("pos-features")
                self.pos_labels.append(feature_parts[0] + "pos-labels" + feature_parts[1])

        self.neg_features = glob.glob(os.path.join(data_path, "*_neg-features_*"), recursive=True)
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

        feature = self.apply_data_transforms(feature)
        label = self.apply_label_transforms(label)

        if self.user_transforms:
            feature = self.user_transforms(feature)

        # Honestly may be worth pre-process this
        feature = torch.from_numpy(feature).float()  
        label = torch.from_numpy(label).float()

        return feature, label, (self.data[index], self.labels[index]) # Include the data files!

    def apply_label_transforms(self, label):
        # Gaussian smooth the labels!
        if self.gaussian_smooth != 0:
            label = gaussian_filter1d(label,sigma=self.gaussian_smooth)

        return label


    def apply_data_transforms(self, data):
        # Apply a log transform to the spectrogram! This is equivalent to the convert to db
        if self.log_scale:
            data = 10 * np.log10(data)

        # Normalize Features
        if self.normalization == "norm":
            data = (data - np.mean(data)) / np.std(data)
        elif self.normalization == "feature":
            data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

        return data

