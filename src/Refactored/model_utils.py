import parameters
import numpy as np
import torch
import torch.nn as nn
import sklearn
from sklearn.metrics import f1_score, precision_recall_fscore_support
import os


class Model_Utils(object):
    """
        General utility functions for training and evaluating the model
    """

    @classmethod
    def get_dataset_paths(cls, local_files=True):
        """
            Return train and test data paths
        """
        if args.local_files:
            train_data_path = parameters.LOCAL_TRAIN_FILES
            test_data_path = parameters.LOCAL_TEST_FILES
        else:
            if parameters.DATASET.lower() == "noab":
                train_data_path = parameters.REMOTE_TRAIN_FILES
                test_data_path = parameters.REMOTE_TEST_FILES
            else:
                train_data_path = parameters.REMOTE_BAI_TRAIN_FILES
                test_data_path = parameters.REMOTE_BAI_TEST_FILES

        return train_data_path, test_data_path

    @classmethod
    def get_loader(cls, dataset,
                       batch_size,
                       random_seed=8,
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
        # Set the dataloader seed
        set_seed(parameters.DATA_LOADER_SEED)
        print('Size of dataset is {} samples'.format(len(dataset)))

        # Set the data_loader random seed for reproducibility.
        def _init_fn(worker_id):
            np.random.seed(int(random_seed) + worker_id)

        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
            shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, worker_init_fn=_init_fn)

        return data_loader

    @classmethod
    def set_seed(cls, seed):
        """
            Set the seed across all different necessary platforms
            to allow for comparrsison of different model seeding. 
            Additionally, in the adversarial discovery, we want to
            initialize and train each of the models with the same
            seed.
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Not totally sure what these two do!
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)


    @classmethod
    def create_save_path(cls, save_time, save_local=False, save_prefix=None):
        save_path = ''
        if save_local:
            save_path += parameters.LOCAL_SAVE_PATH
        else:
            save_path += parameters.REMOTE_SAVE_PATH

        # Save prefix represents intial tags to add such as
        # 'AdversarialTraining_' or 'Hierarchical_'
        if save_prefix is not None:
            save_path += save_prefix

        if parameters.DATASET.lower() == 'bai':
            save_path += 'BAI-Pre-Train_'

        # TIME TO GET RID OF THE CALL_DATASET STUFF!
        save_path += 'Model-' + str(parameters.MODEL_ID) + '_'

        if parameters.PRE_TRAIN:
            save_path += 'Pre-Trained_'

        # Add the downsampling sizes and the number of filters as model specific parameters
        if parameters.MODEL_ID == 26 or parameters.MODEL_ID == 27:
            save_path += 'DownSample-['
            compress_factors = parameters.HYPERPARAMETERS[parameters.MODEL_ID]['compress_factors']
            for factor in compress_factors:
                save_path += str(factor) + '-'

            save_path = save_path[:-1] + ']_'

            save_path += 'Filters-['
            num_filters = parameters.HYPERPARAMETERS[parameters.MODEL_ID]['num_filters']
            for filter in num_filters:
                save_path += str(filter) + '-'

            save_path = save_path[:-1] + ']_'

        # Add the model random seed only if the Dataset Random Seed
        # and the Model Random seed differ. Hacky for now!
        if parameters.MODEL_SEED != parameters.DATASET_SEED:
            save_path += 'DataSeed-' + str(parameters.DATASET_SEED) + '_'
            save_path += 'ModelSeed-' + str(parameters.MODEL_SEED) + '_'

        save_path += "Norm-" + parameters.NORM + "_"
        save_path += "NegFactor-x" + str(parameters.NEG_SAMPLES) + "_"
        save_path += "CallRepeats-" + str(parameters.CALL_REPEATS) + "_"
        # Note should include something potentially if using focal loss
        save_path += "TestNegFactor-x" + str(parameters.TEST_NEG_SAMPLES) + "_"
        save_path += "Loss-" + parameters.LOSS + "_"

        if parameters.CHUNK_SIZE != 256:
            save_path += "WindowSize-" + str(parameters.CHUNK_SIZE) + "_"

        if parameters.LOSS.upper() == "FOCAL":
            save_path += "Alpha-" + str(parameters.FOCAL_ALPHA) + "_"
            save_path += "Gamma-" + str(parameters.FOCAL_GAMMA) + "_"
            save_path += "WeightInit-" + str(parameters.FOCAL_WEIGHT_INIT) + "_"

        if parameters.LOSS == "BOUNDARY":
            save_path += "BoundaryFudgeFac-" + str(parameters.BOUNDARY_FUDGE_FACTOR) + "_"
            save_path += "IndividualBoundaries-" + str(parameters.INDIVIDUAL_BOUNDARIES) + "_"
            save_path += "BoundaryLoss-" + str(parameters.BOUNDARY_LOSS) + "_"
            if parameters.BOUNDARY_LOSS.lower() == "weight":
                save_path += "BoundaryWeight-" + str(parameters.BOUNDARY_WEIGHT) + "_"

        if parameters.SHIFT_WINDOWS:
            save_path += "ShiftWindows_"

        save_path +=  str(save_time)
        
        return save_path

    @classmethod
    def hierarchical_model_1_path(cls):
        model_name = "Model_1_Type-" + str(parameters.HIERARCHICAL_MODEL) + "_"

        if parameters.HIERARCHICAL_PRE_TRAIN:
            model_name += 'BAI-Pre-Train_'

        # Add the downsampling sizes and the number of filters as model specific parameters
        if parameters.HIERARCHICAL_MODEL == 26 or parameters.HIERARCHICAL_MODEL == 27:
            model_name += 'DownSample-['
            compress_factors = parameters.HYPERPARAMETERS[parameters.HIERARCHICAL_MODEL]['compress_factors']
            for factor in compress_factors:
                model_name += str(factor) + '-'

            model_name = model_name[:-1] + ']_'

            model_name += 'Filters-['
            num_filters = parameters.HYPERPARAMETERS[parameters.HIERARCHICAL_MODEL]['num_filters']
            for filter in num_filters:
                model_name += str(filter) + '-'

            model_name = model_name[:-1] + ']_'

        # Fix this later, but fow now do this for cross versions!
        if parameters.HIERARCHICAL_REPEATS_POS != 1 or parameters.HIERARCHICAL_REPEATS_NEG != 1:
            model_name += "PosRepeats-" + str(parameters.HIERARCHICAL_REPEATS_POS) + "_NegRepeats-" + str(parameters.HIERARCHICAL_REPEATS_NEG)
        else:
            model_name += 'CallRepeats-' + str(parameters.HIERARCHICAL_REPEATS).lower()

        # Add if we are using shifting windows # Should get rid of this!
        if parameters.HIERARCHICAL_SHIFT_WINDOWS:
            model_name += '_OversizeCalls'

        # For now, if FALSE_POSITIVE_THRESHOLD != 15 include it
        if parameters.FALSE_POSITIVE_THRESHOLD != 15:
            model_name += '_FalsePosThreshold-' + str(parameters.FALSE_POSITIVE_THRESHOLD)

        if parameters.HIERARCHICAL_ADD_FP:
            model_name += "_AddingFPs"

        # Just for now quickly
        #if parameters.EXTRA_LABEL:
        #    model_name += '_MULTI-CLASS'

        return model_name

    @classmethod
    def create_dataset_path(cls, init_path, neg_samples=1, call_repeats=1, shift_windows=False):
        init_path += 'Neg_Samples_x' + str(neg_samples) + "_Seed_" + str(parameters.DATASET_SEED) + \
                            "_CallRepeats_" + str(call_repeats)
        # Include boundary uncertainty in training
        include_boundaries = False
        if parameters.LOSS == "BOUNDARY":
            include_boundaries = True
            init_path += "_FudgeFact_" + str(parameters.BOUNDARY_FUDGE_FACTOR) + "_Individual-Boarders_" + str(parameters.INDIVIDUAL_BOUNDARIES)

        if shift_windows:
            init_path += '_OversizeCalls'

        if parameters.CHUNK_SIZE != 256:
            init_path += '_WindowSize-' + str(parameters.CHUNK_SIZE)

        return init_path, include_boundaries

    @classmethod
    def is_eval_epoch(cls, cur_epoch):
        """Determines if the model should be evaluated at the current epoch."""
        return (
            (cur_epoch + 1) % parameters.EVAL_PERIOD == 0 or
            cur_epoch == 0 or
            (cur_epoch + 1) == parameters.NUM_EPOCHS
        )

    @classmethod
    def sigmoid(cls, x):                                        
        return 1 / (1 + np.exp(-x))

    @classmethod
    def calc_accuracy(cls, binary_preds, labels):
        accuracy = (binary_preds == labels).sum() / labels.shape[0]
        return accuracy

    @classmethod
    def num_correct(cls, logits, labels):
        sig = nn.Sigmoid()
        with torch.no_grad():
            pred = sig(logits)
            binary_preds = pred > parameters.THRESHOLD # This should likely be 0
            # Cast to proper type!
            binary_preds = binary_preds.float()
            num_correct = (binary_preds == labels).sum().item()

        return num_correct

    @classmethod
    def multi_class_num_correct(cls, logits, labels):
        """
            Treat '2' and '0' as the same for now! May want to profile this later.

            @ Pre-condition: assumes that the labels have already converted 
            the '2' labels back to the singular '0' value
        """
        softmax = nn.Softmax(dim=1)
        with torch.no_grad():
            pred = softmax(logits)
            class_preds = torch.argmax(pred, dim=1)

            # Treat all '2's as 0s
            two_pred = (class_preds == 2)
            class_preds[two_pred] = 0

            # Cast to proper type!
            class_preds = class_preds.float()
            num_correct = (class_preds == labels).sum().item()

        return num_correct

    @classmethod
    def num_non_zero(cls, logits, labels):
        sig = nn.Sigmoid()
        with torch.no_grad():
            pred = sig(logits)
            binary_preds = pred > parameters.THRESHOLD
            # Cast to proper type!
            binary_preds = binary_preds.float()
            num_non_zero = binary_preds.sum().item()

        return num_non_zero

    @classmethod
    def get_f_score(cls, logits, labels):
        sig = nn.Sigmoid()
        with torch.no_grad():
            pred = sig(logits)
            binary_preds = torch.where(pred > parameters.THRESHOLD, torch.tensor(1.0).to(parameters.device), torch.tensor(0.0).to(parameters.device))
            # Flatten the array for fscore
            binary_preds = binary_preds.view(-1)
            labels = labels.view(-1)

            # Cast to proper type!
            binary_preds = binary_preds.float()
            f_score = f1_score(labels.data.cpu().numpy(), binary_preds.data.cpu().numpy())

        return f_score

    @classmethod
    def get_precission_recall_fscore(cls, logits, labels):
        sig = nn.Sigmoid()
        with torch.no_grad():
            pred = sig(logits)
            binary_preds = torch.where(pred > parameters.THRESHOLD, torch.tensor(1.0).to(parameters.device), torch.tensor(0.0).to(parameters.device))
            # Flatten the array for fscore
            binary_preds = binary_preds.view(-1)
            labels = labels.view(-1)

            # Cast to proper type!
            binary_preds = binary_preds.float()
            p, r, f_score, _ = precision_recall_fscore_suppor(labels.data.cpu().numpy(), binary_preds.data.cpu().numpy(), average='binary')

        return p, r, f_score

    @classmethod
    def get_precission_recall_values(cls, logits, labels):
        sig = nn.Sigmoid()
        with torch.no_grad():
            pred = sig(logits)
            binary_preds = torch.where(pred > parameters.THRESHOLD, torch.tensor(1.0).to(parameters.device), torch.tensor(0.0).to(parameters.device))
            # Flatten the array for fscore
            binary_preds = binary_preds.view(-1)
            binary_preds = binary_preds.float()
            labels = labels.view(-1)

            # Number predicted
            tp_fp = torch.sum(binary_preds).item()
            # Number true positives
            tp = (binary_preds + labels) == 2
            tp = torch.sum(tp).item()
            # Number of actual calls
            tp_fn = torch.sum(labels).item()

        return tp, tp_fp, tp_fn

    @classmethod
    def multi_class_precission_recall_values(cls, logits, labels):
        """
            Treat '2' and '0' as the same for now! May want to profile this later.

            @ Pre-condition: assumes that the labels have already converted 
            the '2' labels back to the singular '0' value
        """
        softmax = nn.Softmax(dim=1)
        with torch.no_grad():
            pred = softmax(logits)
            class_preds = torch.argmax(pred, dim=1)

            # Treat all '2's as 0s
            two_pred = (class_preds == 2)
            class_preds[two_pred] = 0

            # Number predicted
            tp_fp = torch.sum(class_preds).item()
            # Number true positives - This works because
            # we have converted all '2' labels to '0'
            tp = (class_preds + labels) == 2
            tp = torch.sum(tp).item()
            # Number of actual calls
            tp_fn = torch.sum(labels).item()

        return tp, tp_fp, tp_fn









