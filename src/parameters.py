#DATASET = 'Call'
DATASET = 'Activate'
#DATASET = 'MFCC_Call'

NORM = "Norm"
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
}

RANDOM_SEED = 42

BATCH_SIZE = 32
NUM_EPOCHS = 1000

MODEL_SAVE_PATH = '../weights/'
LOGS_SAVE_PATH = './runs/'

INPUT_SIZE = 77
OUTPUT_SIZE = 1