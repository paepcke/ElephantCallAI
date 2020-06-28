import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
THRESHOLD = 0.5
VERBOSE = False

NUM_EPOCHS = 1000

ADVERSARIAL_LOOPS = 10
TRAIN_STOP_ITERATIONS = 30
# Determines number of adversarial samples to discover.
# This is a number (0, 1] that calculates how
# many adversarial samples to find based on the size of the
# current dataset. 
# Note: -1 means find all adversarial samples!
ADVERSARIAL_SAMPLES = 0.5
# How many incorrect slices for a chunk to be considered
# an adversarial false positive
ADVERSARIAL_THRESHOLD = 0

DATASET = 'Call'
#DATASET = 'Activate'
#DATASET = 'MFCC_Call'

LOSS = "CE"
CHUNK_WEIGHTING = "count"
FOCAL_WEIGHT_INIT = 0.5 
FOCAL_GAMMA = 15
FOCAL_ALPHA = 0.25

NEG_SAMPLES = 1
NORM = "norm"
SCALE = True

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
        'lr_decay_step': 4,
        'lr_decay': 0.95,
        'l2_reg': 1e-5,
        }
}

RANDOM_SEED = 8
DATA_LOADER_SEED = 33

BATCH_SIZE = 32

#Local
# SAVE_PATH = '../models/'
SAVE_PATH = '/home/data/elephants/models/'

INPUT_SIZE = 77
OUTPUT_SIZE = 1