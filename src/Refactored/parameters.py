import torch

################################################
#### Trainining Hyper-Parameters + Settings ####
################################################
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS = 1000
EVAL_PERIOD = 1
BATCH_SIZE = 32 
TRAIN_STOP_ITERATIONS = 30
THRESHOLD = 0.5 # Only used in training
VERBOSE = False
'''
    Which metric to track for early stopping
    and saving the best model
    Options:
    1) acc = accuracy
    2) fscore
'''
TRAIN_MODEL_SAVE_CRITERIA = 'acc'

##########################
#### Eval Parameters #####
##########################
''' Note This Should Match FALSE_POSITIVE_THRESHOLD '''
MIN_CALL_LENGTH = 15
EVAL_THRESHOLD = 0.5

''' Hop/stride length of model window when predicting over full spectrogram '''
PREDICTION_SLIDE_LENGTH = 128

##############################
#### Model_0 / Solo Model ####
##############################
''' 28 = window size 512 '''
MODEL_ID = 17
PRE_TRAIN = False

################################
#### Dataset Specifications ####
################################
''' This now represents whether to use noab or bai '''
DATASET = 'noab'

''' Data Augmentation / Undersampling values '''
NEG_SAMPLES = 2
TEST_NEG_SAMPLES = 2
CALL_REPEATS = 1
# New methods for smoothing labels and 
# excluding calls that are marginal 
# and data-augmentation
LABEL_SMOOTH = 0
EXCLUDE_MARGINALS = False
OVERSIZED_WINDOWS = False
####################################
NORM = "norm"
SCALE = True # Log scale the spectrograms
SHIFT_WINDOWS = False
CHUNK_SIZE = 256

#######################################
#### Curriculum Model Parameters ####
#######################################
ERAS = 20
NUM_EPOCHS_PER_ERA = 3

''' Parameters specifying how new data is kept/updated'''
RAND_KEEP_RATIO = 0.5
HARD_KEEP_RATIO = 0.25
HARD_VS_RAND_RATIO = 0.05
HARD_INCREASE_FACTOR = 1.25
HARD_VS_RAND_RATIO_MAX = 0.5 
HARD_SAMPLE_SIZE_FACTOR = 5

''' 
    Difficulty Scoring Method for each Window 
    Options:
        - slices
        - uncertainty_weighting
'''
DIFFICULTY_SCORING_METHOD = "slices"

''' Path of the adversarial test files from the 2-stage learning process'''
ADVERSARIAL_TEST_FILES = "/home/jgs8/CS224S/ElephantCallAI/models/Two-Stage_Model-17_Norm-norm_NegFactor-x2_CallRepeats-1_TestNegFactor-x2_Loss-CE_2021-03-14_01:07:55/Adversarial_Neg-Ratio-1_Test.txt"

#######################################
#### 2-Stage Model Parameters ####
#######################################
''' Threshold for a window to be considered a false positive'''
FALSE_POSITIVE_THRESHOLD = 15 

''' Specify 'same' to keep training Model_0 '''
HIERARCHICAL_MODEL = 17
HIERARCHICAL_PRE_TRAIN = False

''' Deptricated!!! '''
HIERARCHICAL_REPEATS = 1 # This should be deprecated!!!!!!
''' Repeats for Pos and Neg (False-Pos) examples seperately! '''
HIERARCHICAL_REPEATS_POS = 1
HIERARCHICAL_REPEATS_NEG = 1

# If true, append FP examples to given negative samples
HIERARCHICAL_ADD_FP = False

# Use '2' label for FP model_0 predictions
EXTRA_LABEL = False

# Use model_0 predictions as an extra feature
MODEL_0_FEATURES = False

''' Whether window shifting as data-aug is used in training '''
HIERARCHICAL_SHIFT_WINDOWS = False 



################################
#### Loss function criteria ####
################################
LOSS = "CE"
CHUNK_WEIGHTING = "avg"
FOCAL_WEIGHT_INIT = 0.01 
FOCAL_GAMMA = 2
FOCAL_ALPHA = 0.25

###############################################
#### Adversarial Inner Outer Loop Training ####
###############################################
ADVERSARIAL_LOOPS = 10
# Determines number of adversarial samples to discover.
# This is a number (0, 1] that calculates how
# many adversarial samples to find based on the size of the
# current dataset. 
# Note: -1 means find all adversarial samples!
ADVERSARIAL_SAMPLES = 0.5
# How many incorrect slices for a chunk to be considered
# an adversarial false positive
ADVERSARIAL_THRESHOLD = 0

#######################################
#### Boundary 'Fudging' Parameters ####
#######################################
# Flags for how to deal with boundaries!!
# If > 0 then use boundaries else no boundaries
BOUNDARY_FUDGE_FACTOR = 0
INDIVIDUAL_BOUNDARIES = True
# Determine how to incorperate the boundary
# into the loss
# WEIGHT = Re-weight the boundary slices
# EQUAL = Make the ground truth label match the class predicted
BOUNDARY_LOSS = 'EQUAL'
# How to weight the boundary slices in [0, 1]
BOUNDARY_WEIGHT = 0.5

##########################
#### Random Seed Info ####
########################## 
MODEL_SEED = 8
# Make these two below the same!
DATASET_SEED = 8
DATA_LOADER_SEED = 8


# This is adhoc for now!
LOCAL_TRAIN_FILES = '../../elephant_dataset/Train_chop/'
LOCAL_TEST_FILES = '../../elephant_dataset/Train_chop/'
LOCAL_FULL_TRAIN = '../elephant_dataset/Train/Full_24_hrs'
LOCAL_FULL_TEST =  '../elephant_dataset/Test/Full_24_hrs'

REMOTE_TRAIN_FILES = '/home/data/elephants/processed_data/Train_Chopped_nouab/'
REMOTE_TEST_FILES = "/home/data/elephants/processed_data/Test_Chopped_nouab/"
REMOTE_MARGINAL_TRAIN_FILES = '/home/data/elephants/processed_data/Train_Marginal_Chopped_nouab/'
REMOTE_MARGINAL_TEST_FILES = "/home/data/elephants/processed_data/Test_Marginal_Chopped_nouab/"
REMOTE_OVERSIZED_TRAIN_FILES = "/home/data/elephants/processed_data/Train_Oversized_Chopped_nouab/"
REMOTE_OVERSIZED_TEST_FILES = "/home/data/elephants/processed_data/Test_Chopped_nouab/"

REMOTE_BAI_TRAIN_FILES = '/home/data/elephants/processed_data/Train_bai/'
REMOTE_BAI_TEST_FILES = '/home/data/elephants/processed_data/Test_bai/'
REMOTE_FULL_TRAIN = '/home/data/elephants/processed_data/Train_nouab/Full_24_hrs'
REMOTE_FULL_TEST = '/home/data/elephants/processed_data/Test_nouab/Full_24_hrs'
REMOTE_FULL_TRAIN_BAI = '/home/data/elephants/processed_data/Train_bai/Full_24_hrs'
REMOTE_FULL_TEST_BAI = '/home/data/elephants/processed_data/Test_bai/Full_24_hrs'


#Local
#SAVE_PATH = '../models/'
# For the refactored
LOCAL_SAVE_PATH = '../../models/'
REMOTE_SAVE_PATH = '/home/data/elephants/models/'

INPUT_SIZE = 77
OUTPUT_SIZE = 1

HYPERPARAMETERS = {
0: {
        'lr': 1e-3,
        'lr_decay_step': 4,
        'lr_decay': 0.95,
        'l2_reg': 1e-5,
        },
1: {
        'lr': 1e-3,
        'lr_decay_step': 4,
        'lr_decay': 0.95,
        'l2_reg': 1e-5,
        },
2: {
        'lr': 1e-3,
        'lr_decay_step': 4,
        'lr_decay': 0.95,
        'l2_reg': 1e-5,
        },
3: {
        'lr': 1e-3,
        'lr_decay_step': 4,
        'lr_decay': 0.95,
        'l2_reg': 1e-5,
        },
4: {
        'lr': 1e-3,
        'lr_decay_step': 4,
        'lr_decay': 0.95,
        'l2_reg': 1e-5,
        },
5: {
        'lr': 1e-3,
        'lr_decay_step': 4,
        'lr_decay': 0.95,
        'l2_reg': 1e-5,
        },
6: {
        'lr': 1e-3,
        'lr_decay_step': 4,
        'lr_decay': 0.95,
        'l2_reg': 1e-5,
        },
7: {
        'lr': 1e-3,
        'lr_decay_step': 4,
        'lr_decay': 0.95,
        'l2_reg': 1e-5,
        },
8: {
        'lr': 1e-3,
        'lr_decay_step': 4,
        'lr_decay': 0.95,
        'l2_reg': 1e-5,
        },
9: {
        'lr': 1e-3,
        'lr_decay_step': 4,
        'lr_decay': 0.95,
        'l2_reg': 1e-5,
        },
10: {
        'lr': 1e-3,
        'lr_decay_step': 4,
        'lr_decay': 0.95,
        'l2_reg': 1e-5,
        },

11: {
        'lr': 1e-3,
        'lr_decay_step': 4,
        'lr_decay': 0.95,
        'l2_reg': 1e-5,
        },
12: {
        'lr': 1e-3,
        'lr_decay_step': 4,
        'lr_decay': 0.95,
        'l2_reg': 1e-5,
        },
13: {
        'lr': 1e-3,
        'lr_decay_step': 4,
        'lr_decay': 0.95,
        'l2_reg': 1e-5,
        },
14: {
        'lr': 1e-3,
        'lr_decay_step': 4,
        'lr_decay': 0.95,
        'l2_reg': 1e-5,
        },
15: {
        'lr': 1e-3,
        'lr_decay_step': 4,
        'lr_decay': 0.95,
        'l2_reg': 1e-5,
        },
16: {
        'lr': 1e-3,
        'lr_decay_step': 4,
        'lr_decay': 0.95,
        'l2_reg': 1e-5,
        },
17: {
        'lr': 1e-3,
        'lr_decay_step': 4,
        'lr_decay': 0.95,
        'l2_reg': 1e-5,
        },
18: {
        'lr': 1e-3,
        'lr_decay_step': 20, # Let us almost try not having this for now for the focal loss!
        'lr_decay': 0.95,
        'l2_reg': 1e-5,
        },
19: {
        'lr': 1e-3,
        'lr_decay_step': 4, 
        'lr_decay': 0.95,
        'l2_reg': 1e-5,
        },
20: {
        'lr': 1e-3,
        'lr_decay_step': 4, 
        'lr_decay': 0.95,
        'l2_reg': 1e-5,
        },
21: {
        'lr': 1e-3,
        'lr_decay_step': 4, 
        'lr_decay': 0.95,
        'l2_reg': 1e-5,
        },
22: {
        'lr': 1e-3,
        'lr_decay_step': 4, 
        'lr_decay': 0.95,
        'l2_reg': 1e-5,
        },
23: {
        'lr': 1e-3,
        'lr_decay_step': 4, 
        'lr_decay': 0.95,
        'l2_reg': 1e-5,
        },
24: {
        'lr': 1e-3,
        'lr_decay_step': 4, 
        'lr_decay': 0.95,
        'l2_reg': 1e-5,
        },
25: {
        'lr': 1e-3,
        'lr_decay_step': 4, 
        'lr_decay': 0.95,
        'l2_reg': 1e-5,
        },
26: {
        'lr': 1e-3,
        'lr_decay_step': 4, 
        'lr_decay': 0.95,
        'l2_reg': 1e-5,
        'compress_factors': [5, 4, 2],
        'num_filters': [32, 64, 64]
        },
27: {
        'lr': 1e-3,
        'lr_decay_step': 4, 
        'lr_decay': 0.95,
        'l2_reg': 1e-5,
        'compress_factors': [1, 2, 2, 2],
        'num_filters': [32, 64, 64, 128]
        },
28: {
        'lr': 1e-3,
        'lr_decay_step': 4, 
        'lr_decay': 0.95,
        'l2_reg': 1e-5,
        },
29: {
        'lr': 1e-3,
        'lr_decay_step': 4, 
        'lr_decay': 0.95,
        'l2_reg': 1e-5,
        },
30: {
        'lr': 1e-3,
        'lr_decay_step': 4, 
        'lr_decay': 0.95,
        'l2_reg': 1e-5,
        },
}

