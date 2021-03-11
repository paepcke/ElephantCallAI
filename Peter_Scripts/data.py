import matplotlib.pyplot as plt
import numpy as np
import torch
import aifc
from scipy import signal
from torch.utils import data
#from torchvision import transforms
import os
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import glob
import parameters

from utils import set_seed

Noise_Stats_Directory = "../elephant_dataset/eleph_dataset/Noise_Stats/"

def get_loader(data_dir,
               batch_size,
               random_seed=8,
               norm="norm",
               scale=False,
               augment=False,
               shuffle=True,
               num_workers=16,
               pin_memory=False):
    """
    Utility function for loading and returning train and valid
    multi-process iterators.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - random_seed: fix seed for reproducibility.
    - augment: whether data augmentation scheme. Only applied on the train split.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    - data_file_paths: If you know what particular data file names you want to load, 
      pass them in as a list of strings.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    # Note here we could do some data preprocessing!
    # define transform
    # Set the dataloader seed
    set_seed(parameters.DATA_LOADER_SEED)

    dataset = ElephantDataset(data_dir, preprocess=norm, scale=scale)
    
    print('Size of dataset at {} is {} samples'.format(data_dir, len(dataset)))

    # Set the data_loader random seed for reproducibility.
    # Should do some checks on this
    def _init_fn(worker_id):
        # We probably do not want every worker to have 
        # the same random seed or else they may do the same 
        # thing?
        np.random.seed(int(random_seed) + worker_id)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
        shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, worker_init_fn=_init_fn)

    return data_loader

def get_loader_fuzzy(data_dir,
               batch_size,
               random_seed=8,
               norm="norm",
               scale=False,
               include_boundaries=False,
               shift_windows=False,
               is_full_dataset=False,
               full_window_predict=False,
               augment=False,
               shuffle=True,
               num_workers=16,
               pin_memory=False):
    """
    Utility function for loading and returning train and valid
    multi-process iterators.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - random_seed: fix seed for reproducibility.
    - augment: whether data augmentation scheme. Only applied on the train split.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    - data_file_paths: If you know what particular data file names you want to load, 
      pass them in as a list of strings.

    -is_full_dataset: Is important for when we are shifting the windows, because
    when using the full 24 hr dataset for adversarial discover we always want to 
    use the middle of the oversized window!
    
    -fixed_repeat: Used for training the second model in a heirarchical setting.
    Repeat sliding windows but save fixed random slices for each window

    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.

    """
    # Note here we could do some data preprocessing!
    # define transform
    # Set the dataloader seed
    print ("DataLoader Seed:", parameters.DATA_LOADER_SEED)
    set_seed(parameters.DATA_LOADER_SEED)

    dataset = ElephantDatasetFuzzy(data_dir, preprocess=norm, scale=scale, include_boundaries=include_boundaries, 
                        shift_windows=shift_windows, is_full_dataset=is_full_dataset, 
                        full_window_predict=full_window_predict)
    
    print('Size of dataset at {} is {} samples'.format(data_dir, len(dataset)))

    # Set the data_loader random seed for reproducibility.
    # Should do some checks on this
    def _init_fn(worker_id):
        # Assign each worker its own seed
        np.random.seed(int(random_seed) + worker_id)
        # Is this bad??
        # This seems bad as each epoch will be the same order of data! 
        #torch.manual_seed(int(random_seed) + worker_id)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
        shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, worker_init_fn=_init_fn)

    return data_loader


class ElephantDatasetFuzzy(data.Dataset):
    def __init__(self, data_path, preprocess="norm", scale=False, transform=None, include_boundaries=False, 
            shift_windows=False, is_full_dataset=False, full_window_predict=False):
        # Plan: Load in all feature and label names to create a list
        self.data_path = data_path
        self.user_transforms = transform
        self.preprocess = preprocess
        self.scale = scale
        self.include_boundaries = include_boundaries
        self.shift_windows = shift_windows
        self.is_full_dataset = is_full_dataset
        self.full_window_predict = full_window_predict
        # This is only used if we want to generate fixed repeated
        # windows during hierarchical training
        self.fixed_indeces = None
        # By default this is False 
        # and only True for the special case where
        # we incorperate model_0 predictions into 
        # the 2-stage model 
        self.model_0_feature = False

        '''
        self.features = glob.glob(data_path + "/" + "*features*", recursive=True)
        self.initialize_labels()
        '''

        self.pos_features = glob.glob(data_path + "/" + "*_features_*", recursive=True)
        self.neg_features = glob.glob(data_path + "/" + "*_neg-features_*", recursive=True)
        self.intialize_data(init_pos=True, init_neg=True)

        assert len(self.features) == len(self.labels)
        if self.include_boundaries:
            assert len(self.features) == len(self.boundary_masks)

        print("ElephantDataset number of features {} and number of labels {}".format(len(self.features), len(self.labels)))
        print('Normalizing with {} and scaling {}'.format(preprocess, scale))

    def initialize_labels(self):
        self.labels = []
        self.boundary_masks = []
        for feature_path in self.features:
            feature_parts = feature_path.split("features")
            self.labels.append(glob.glob(feature_parts[0] + "labels" + feature_parts[1])[0])
            if self.include_boundaries:
                self.boundary_masks.append(glob.glob(feature_parts[0] + "boundary-masks" + feature_parts[1])[0])


    def set_pos_features(self, pos_features):
        print("Length of pos_features was {} and is now {} ".format(len(self.pos_features), len(pos_features)))
        self.pos_features = pos_features
        self.intialize_data(init_pos=True, init_neg=False)

    def set_neg_features(self, neg_features):
        print("Length of neg_features was {} and is now {} ".format(len(self.neg_features), len(neg_features)))
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

    def scale_features(self, pos_factor, neg_factor):
        print("Length of pos_features was {} and is now {} ".format(len(self.pos_features), int(pos_factor * len(self.pos_features))))
        print("Length of neg_features was {} and is now {} ".format(len(self.neg_features), int(neg_factor * len(self.neg_features))))
        # Add in a feature to undersample as well!
        # Could consider also giving hardness to these to help with selection.
        # Let us do random for now
        if pos_factor < 1:
            indeces = np.arange(len(self.pos_features))
            pos_inds = np.random.choice(indeces, int(indeces.shape[0] * pos_factor))
            self.pos_features = list(np.array(self.pos_features)[pos_inds])
            self.pos_labels = list(np.array(self.pos_labels)[pos_inds])
        else:
            self.pos_features *= pos_factor
            self.pos_labels *= pos_factor

        if neg_factor < 1:
            indeces = np.arange(len(self.neg_features))
            neg_inds = np.random.choice(indeces, int(indeces.shape[0] * neg_factor))
            self.neg_features = list(np.array(self.neg_features)[neg_inds])
            self.neg_labels = list(np.array(self.neg_labels)[neg_inds])
        else:
            self.neg_features *= neg_factor
            self.neg_labels *= neg_factor

        # Re-form the feature and data set
        self.features = self.pos_features + self.neg_features
        self.labels = self.pos_labels + self.neg_labels

    def update_labels(self, new_pos_labels_dir, new_neg_labels_dir):
        """
            Kinda an adhoc method, but currently we are using this in
            the new 3rd label dataset. For the given features / windows
            in the dataset, replace the corresponding labels with the
            new 3 class labels. 
            Implemenation: Since the new label names should match the
            training example names, go through each training example
            and get the new label path from either pos/neg label dir.

            @ Params
            @ new_pos_labels_dir - The folder that contains the new positive window labels
            @ new_neg_labels_dir - The folder that contains the new negative window labels
        """
        # Replace the labels for the positive examples
        new_pos_labels = []
        for pos_feat in self.pos_features:
            data_id = pos_feat.split('/')[-1]
            new_pos_label = os.path.join(new_pos_labels_dir, data_id.replace('features', 'labels'))
            new_pos_labels.append(new_pos_label)

        self.pos_labels = new_pos_labels

        # Replace the labels for the negative examples
        new_neg_labels = []
        for neg_feat in self.neg_features:
            data_id = neg_feat.split('/')[-1]
            new_neg_label = os.path.join(new_neg_labels_dir, data_id.replace('features', 'labels'))
            new_neg_labels.append(new_neg_label)

        self.neg_labels = new_neg_labels

        # Re-set self.labels
        self.labels = self.pos_labels + self.neg_labels

    def add_model_0_preds(self, model_0_pos_dir, model_0_neg_dir):
        """
            Add the additional feature of the model_0 predictions for
            each training window.  
            Implemenation: Since the new label names should match the
            training example names, go through each training example
            and get the new label path from either pos/neg label dir.

            @ Params
            @ model_0_pos_dir - The folder that contains the model_0 positive window preds
            @ model_0_neg_dir - The folder that contains the model_0 negative window preds
        """
        # Replace the labels for the positive examples
        self.model_0_pos_preds = []
        for pos_feat in self.pos_features:
            data_id = pos_feat.split('/')[-1]
            new_pos_label = os.path.join(model_0_pos_dir, data_id.replace('features', 'labels'))
            self.model_0_pos_preds.append(new_pos_label)

        # Replace the labels for the negative examples
        self.model_0_neg_preds = []
        for neg_feat in self.neg_features:
            data_id = neg_feat.split('/')[-1]
            new_neg_label = os.path.join(model_0_neg_dir, data_id.replace('features', 'labels'))
            self.model_0_neg_preds.append(new_neg_label)

        # Re-set self.labels
        self.model_0_preds = self.model_0_pos_preds + self.model_0_neg_preds
        self.model_0_feature = True


    def create_fixed_windows(self):
        self.fixed_indeces = []

        # Generate the fixed indeces
        for i in range(len(self.features)):
            feature = np.load(self.features[i])
            label = np.load(self.labels[i])

            # Sample a random start index to save
            call_length = -(label.shape[0] - 2 * parameters.CHUNK_SIZE)
            # Use torch.randint because of weird numpy seeding issues
            start_slice = torch.randint(0, parameters.CHUNK_SIZE - call_length, (1,))[0].item()
            self.fixed_indeces.append(start_slice)

    def intialize_data(self, init_pos=True, init_neg=True):
        """
            Initialize both the positive and negative label and boundary
            mask data arrays if indicated by the initialization flags 
            'init_pos' and 'init_neg'. After initializing any necessary
            data, combine the positive and negative examples!
        """
        # Initialize the positive examples
        if init_pos:
            self.pos_labels = []
            self.pos_boundary_masks = []
            for feature_path in self.pos_features:
                feature_parts = feature_path.split("features")
                self.pos_labels.append(glob.glob(feature_parts[0] + "labels" + feature_parts[1])[0])
                if self.include_boundaries:
                    self.pos_boundary_masks.append(glob.glob(feature_parts[0] + "boundary-masks" + feature_parts[1])[0])


        # Initialize the negative examples
        if init_neg:
            self.neg_labels = []
            self.neg_boundary_masks = []
            for feature_path in self.neg_features:
                feature_parts = feature_path.split("features")
                self.neg_labels.append(glob.glob(feature_parts[0] + "labels" + feature_parts[1])[0])
                if self.include_boundaries:
                    self.neg_boundary_masks.append(glob.glob(feature_parts[0] + "boundary-masks" + feature_parts[1])[0])

        # Combine the positive and negative examples!
        self.features = self.pos_features + self.neg_features
        self.labels = self.pos_labels + self.neg_labels
        if self.include_boundaries:
            self.boundary_masks = self.pos_boundary_masks + self.neg_boundary_masks

        print ("Len Pos Features:", len(self.pos_features))
        print ("Len Neg Features:", len(self.neg_features))


    def __len__(self):
        return len(self.features)

    """
    Return a single element at provided index
    """
    def __getitem__(self, index):
        feature = np.load(self.features[index])
        label = np.load(self.labels[index])

        # Load the model_0 predictions and incorperate
        # them into the data transform
        if self.model_0_feature:
            model_0_pred = np.load(self.model_0_preds[index])
            feature = self.apply_transforms(feature, model_0_pred)
        else:
            feature = self.apply_transforms(feature)

        if self.shift_windows:
            feature, label = self.sample_chunk(feature, label)

        # Select fixed random crop
        if self.fixed_indeces is not None:
            start_index = self.fixed_indeces[index]
            feature = feature[start_index: start_index + parameters.CHUNK_SIZE, :]
            label = label[start_index: start_index + parameters.CHUNK_SIZE]

        if self.user_transforms:
            feature = self.user_transforms(feature)

        # Honestly may be worth pre-process this
        feature = torch.from_numpy(feature).float()
        if self.full_window_predict:
            # Make the label a binary 0/1 if an elephant 
            # call is present (May be some weird boundary cases
            # with call being on the edge, but we'll cross that
            # bridge later).
            label = 1. if np.sum(label) > 0 else 0.
        else:    
            label = torch.from_numpy(label).float()

        # Return the boundary masks
        if self.include_boundaries:
            masks = np.load(self.boundary_masks[index])
            # Cast to a bool tensor to allow for array masking
            masks = torch.from_numpy(masks) == 1

            return feature, label, masks, self.features[index]
        else:
            return feature, label, self.features[index] # Include the data file

    def sample_chunk(self, feature, label):
        """
            Selected a random chunk within the oversized window.
            Figure out the call length as: -(window_size - 2*256).
            Then sample starting slice as rand in range [0, 256 - call_length].

            Note: if the flag 'is_full_dataset' is set then return the middle
            256! This is for adversarial discovery mode
        """
        if self.is_full_dataset:
            # The full test set window sizes are 2 * (256 / normal)
            start_slice = label.shape[0] // 4
            end_slice = start_slice + label.shape[0] // 2
        else:
            call_length = -(label.shape[0] - 2 * parameters.CHUNK_SIZE)
            # Draw this out but it should be correct!
            # Use torch.randint because of weird numpy seeding issues
            start_slice = torch.randint(0, parameters.CHUNK_SIZE - call_length, (1,))[0].item()
            end_slice = start_slice + parameters.CHUNK_SIZE

        return feature[start_slice : end_slice, :], label[start_slice : end_slice]

    def apply_transforms(self, data, model_0_pred=None):
        if self.scale:
            data = 10 * np.log10(data)

        # Normalize Features
        if self.preprocess == "norm":
            data = (data - np.mean(data)) / np.std(data)
        elif self.preprocess == "globalnorm":
            data = (data - 132.228) / 726.319 # Calculated these over the training dataset 
        elif self.preprocess == "feature":
            data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

        # If model_0_pred is provided, then create a 3 channel
        # "image" where channels 1 and 2 are the spectrogram and 
        # the 3rd channel is a (-1, 1) valued image of model_0 preds.
        # Specifically, create a column of '1' for '1' predictions and
        # a column of '-1' for '0' preds
        if model_0_pred is not None:
            # Expand the channel dim of the spectrogram
            data = np.expand_dims(data, axis=0)
            # Create the prediction mask. First convert '0'
            # to '-1' value
            model_0_pred[model_0_pred == 0] = -1
            # Repeat the pred values along the feature axis
            model_0_pred = np.expand_dims(model_0_pred, axis=1)
            model_0_pred = np.repeat(model_0_pred, data.shape[2], axis=1)
            # Consider normalizing this input!!
            model_0_pred = (model_0_pred - np.mean(model_0_pred)) / np.std(model_0_pred)

            # Repeat the spectrogram data to creat 3 channels and then
            # make the final channel by the model_0_pred
            data = np.repeat(data, 3, axis=0)
            data[2, :, :] = model_0_pred

        return data

"""
    Notes
    - Preprocess = Norm, Scale = False ===> seems bad
    - Preprocess = Norm, Scale = True ===> Works well on small dataset!
    - Preprocess = Scale, Scale = False ===> Has quite a bit of trouble over fitting small dataset compared to other but eventually can
    - Preprocess = Scale, Scale = True ===> Has quite a bit of trouble over fitting small dataset compared to other and bad val acc!
    - Preprocess = ChunkNorm, Scale = False ===> Very slow and bad
    - Preprocess = ChunkNorm, Scale = True ===> Similar to Norm with scale
    - Preprocess = None, Scale = True ====> No worky
    - Preprocess = Scale range (-1, 1), Scale = True ===> Overfit but huge variance issue
"""
class ElephantDataset(data.Dataset):
    def __init__(self, data_path, transform=None, preprocess="norm", scale=False):
        # Plan: Load in all feature and label names to create a list
        self.data_path = data_path
        self.user_transforms = transform
        self.preprocess = preprocess
        self.scale = scale

        # Probably should not have + "**/" after data_path? It seems like 
        # we are passing the exact datapths anyways! Also why recursive?
        self.features = glob.glob(data_path + "/" + "*features*", recursive=False)
        self.initialize_labels()

        assert len(self.features) == len(self.labels)

        print("Dataset from path {}".format(data_path))
        print("ElephantDataset number of features {} and number of labels {}".format(len(self.features), len(self.labels)))
        print('Normalizing with {} and scaling {}'.format(preprocess, scale))
        print("Shape of a feature is {} and a label is {}".format(self[0][0].shape, self[0][1].shape))

    def initialize_labels(self):
        self.labels = []
        for feature_path in self.features:
            feature_parts = feature_path.split("features")
            self.labels.append(glob.glob(feature_parts[0] + "labels" + feature_parts[1])[0])


    def __len__(self):
        return len(self.features)

    """
    Return a single element at provided index
    """
    def __getitem__(self, index):
        feature = np.load(self.features[index])
        label = np.load(self.labels[index])

        feature = self.apply_transforms(feature)
        if self.user_transforms:
            feature = self.user_transforms(feature)
            
        # Honestly may be worth pre-process this
        feature = torch.from_numpy(feature).float()
        label = torch.from_numpy(label).float()


        return feature, label, self.features[index] # Include the data file

    def apply_transforms(self, data):
        if self.scale:
            data = 10 * np.log10(data)

        # Normalize Features
        if self.preprocess == "norm":
            data = (data - np.mean(data)) / np.std(data)
        elif self.preprocess == "globalnorm":
            data = (data - 132.228) / 726.319 # Calculated these over the training dataset 

        return data

        # elif self.preprocess == "Scale":
        #     scaler = MinMaxScaler()
        #     # Scale features for each training example
        #     # to be within a certain range. Preserves the
        #     # relative distribution of each feature. Here
        #     # each feature is the different frequency band
        #     for i in range(self.features.shape[0]):
        #         self.features[i, :, :] = scaler.fit_transform(self.features[i,:,:].astype(np.float32))
        #     #num_ex = self.features.shape[0]
        #     #seq_len = self.features.shape[1]
        #     #self.features = self.features.reshape(num_ex * seq_len, -1)
        #     #self.features = scaler.fit_transform(self.features)
        #     #self.features = self.features.reshape(num_ex, seq_len, -1)
        # elif self.preprocess == "ChunkNorm":
        #     for i in range(self.features.shape[0]):
        #         self.features[i, :, :] = (self.features[i, :, :] - np.mean(self.features[i, :, :])) / np.std(self.features[i, :, :])
        # elif self.preprocess == "BackgroundS":
        #     # Load in the pre-calculated mean,std,etc.
        #     if not scale:
        #         mean_noise = np.load(Noise_Stats_Directory + "mean.npy")
        #         std_noise = np.load(Noise_Stats_Directory + "std.npy")
        #     else:
        #         mean_noise = np.load(Noise_Stats_Directory + "mean_log.npy")
        #         std_noise = np.load(Noise_Stats_Directory + "std_log.npy")

        #     self.features = (self.features - mean_noise) / std_noise
        # elif self.preprocess == "BackgroundM":
        #     # Load in the pre-calculated mean,std,etc.
        #     if not scale:
        #         mean_noise = np.load(Noise_Stats_Directory + "mean.npy")
        #         median_noise = np.load(Noise_Stats_Directory + "median.npy")
        #     else:
        #         mean_noise = np.load(Noise_Stats_Directory + "mean_log.npy")
        #         median_noise = np.load(Noise_Stats_Directory + "median_log.npy")

        #     self.features = (self.features - mean_noise) / median_noise
        # elif self.preprocess == "FeatureNorm":
        #     self.features = (self.features - np.mean(self.features, axis=(0, 1))) / np.std(self.features, axis=(0,1))





"""
    Dataset for full test length audio
    NEED TO FIX THIS!!
"""
class ElephantDatasetFull(data.Dataset):
    def __init__(self, spectrogram_files, label_files, gt_calls, preprocess="norm", 
                    scale=True, only_preds=False):

        self.specs = spectrogram_files
        # Note there may not actually be files associated with these files
        self.labels = label_files
        self.gt_calls = gt_calls # This is the .txt file that contains start and end times of calls
        self.preprocess = preprocess
        self.scale = scale
        self.only_preds = only_preds
        
        print('Normalizing with {} and scaling {}'.format(preprocess, scale))


    def __len__(self):
        return len(self.specs)


    def transform(self, spectrogram): # We need to fix this probably!!!
        # Potentially include other transforms
        if self.scale:
            spectrogram = 10 * np.log10(spectrogram)

        # Quite janky, but for now we will do the normalization 
        # seperately!
        '''
        # Normalize Features
        if self.preprocess == "norm": # Only have one training example so is essentially chunk norm
            spectrogram = (spectrogram - np.mean(spectrogram)) / np.std(spectrogram)
        elif preprocess == "Scale":
            scaler = MinMaxScaler()
            # Scale features for each training example
            # to be within a certain range. Preserves the
            # relative distribution of each feature. Here
            # each feature is the different frequency band
            spectrogram = scaler.fit_transform(spectrogram.astype(np.float32))
        elif self.preprocess == "ChunkNorm":
            spectrogram = (spectrogram - np.mean(spectrogram)) / np.std(spectrogram)
        elif self.preprocess == "BackgroundS":
            # Load in the pre-calculated mean,std,etc.
            if not scale:
                mean_noise = np.load(Noise_Stats_Directory + "mean.npy")
                std_noise = np.load(Noise_Stats_Directory + "std.npy")
            else:
                mean_noise = np.load(Noise_Stats_Directory + "mean_log.npy")
                std_noise = np.load(Noise_Stats_Directory + "std_log.npy")

            spectrogram = (spectrogram - mean_noise) / std_noise
        elif self.preprocess == "BackgroundM":
            # Load in the pre-calculated mean,std,etc.
            if not scale:
                mean_noise = np.load(Noise_Stats_Directory + "mean.npy")
                median_noise = np.load(Noise_Stats_Directory + "median.npy")
            else:
                mean_noise = np.load(Noise_Stats_Directory + "mean_log.npy")
                median_noise = np.load(Noise_Stats_Directory + "median_log.npy")

            spectrogram = (spectrogram - mean_noise) / median_noise
        elif self.preprocess == "FeatureNorm":
            spectrogram = (spectrogram - np.mean(spectrogram, axis=1)) / np.std(spectrogram, axis=1)
        '''
        return spectrogram

    """
    Return a single element at provided index
    """
    def __getitem__(self, index):
        spectrogram_path = self.specs[index]
        label_path = self.labels[index]
        gt_call_path = self.gt_calls[index]

        spectrogram = np.load(spectrogram_path)
        spectrogram = self.transform(spectrogram)
            
        # Honestly may be worth pre-process this
        #spectrogram = torch.from_numpy(spectrogram)
        #label = torch.from_numpy(label)

        if self.only_preds:
            return spectrogram, None, gt_call_path
        else:   
            return spectrogram, label, gt_call_path


