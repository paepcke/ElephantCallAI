import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
THRESHOLD = 0.5
VERBOSE = False

NUM_EPOCHS = 1000
EVAL_PERIOD = 1

ADVERSARIAL_LOOPS = 10
TRAIN_STOP_ITERATIONS = 30
# Which metric to track for early stopping
# and saving the best model
# Options:
# 1) acc = accuracy
# 2) fscore
TRAIN_MODEL_SAVE_CRITERIA = 'acc'

# Determines number of adversarial samples to discover.
# This is a number (0, 1] that calculates how
# many adversarial samples to find based on the size of the
# current dataset. 
# Note: -1 means find all adversarial samples!
ADVERSARIAL_SAMPLES = 0.5
# How many incorrect slices for a chunk to be considered
# an adversarial false positive
ADVERSARIAL_THRESHOLD = 0

FALSE_NEGATIVE_THRESHOLD = 15 # Test this!
# Specify 'same' to keep training Model_0
HIERARCHICAL_MODEL = 18
# Specify the number of repeats for ONLY the 
# positive examples for model_1. 
# 'same' - use the same dataloader from model_0
HIERARCHICAL_REPEATS = 1
HIERARCHICAL_SHIFT_WINDOWS = False

# Model 18 = entire window classification
MODEL_ID = 18

# WE SHOULD PHASE THIS OUT!
DATASET = 'Call'
#DATASET = 'Activate'
#DATASET = 'MFCC_Call'

LOSS = "CE"
CHUNK_WEIGHTING = "count"
FOCAL_WEIGHT_INIT = 0.01 
FOCAL_GAMMA = 2
FOCAL_ALPHA = 0.75

NEG_SAMPLES = 1
TEST_NEG_SAMPLES = 1
CALL_REPEATS = 1
NORM = "norm"
SCALE = True
SHIFT_WINDOWS = False

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

RANDOM_SEED = 8
DATA_LOADER_SEED = 33

BATCH_SIZE = 32 # Was 32

LOCAL_TRAIN_FILES = '../elephant_dataset/Train/'
LOCAL_TEST_FILES = '../elephant_dataset/Test/'
LOCAL_FULL_TRAIN = '../elephant_dataset/Train/Full_24_hrs'
LOCAL_FULL_TEST =  '../elephant_dataset/Test/Full_24_hrs'

REMOTE_TRAIN_FILES = '/home/data/elephants/processed_data/Train_nouab/'
REMOTE_TEST_FILES = "/home/data/elephants/processed_data/Test_nouab/"
REMOTE_FULL_TRAIN = '/home/data/elephants/processed_data/Train_nouab/Full_24_hrs'
REMOTE_FULL_TEST = '/home/data/elephants/processed_data/Test_nouab/Full_24_hrs'


#Local
#SAVE_PATH = '../models/'
LOCAL_SAVE_PATH = '../models/'
REMOTE_SAVE_PATH = '/home/data/elephants/models/'

CHUNK_SIZE = 256
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
        }
}