import parameters
import numpy as np
import torch
import torch.nn as nn
import sklearn
from sklearn.metrics import f1_score


def create_save_path(save_time, save_local=False, save_prefix=None):
    save_path = ''
    if save_local:
        save_path += parameters.LOCAL_SAVE_PATH
    else:
        save_path += parameters.REMOTE_SAVE_PATH

    # Save prefix represents intial tags to add such as
    # 'AdversarialTraining_' or 'Hierarchical_'
    if save_prefix is not None:
        save_path += save_prefix

    # TIME TO GET RID OF THE CALL_DATASET STUFF!
    save_path += 'Model-' + str(parameters.MODEL_ID) + '_'
    save_path += "Norm-" + parameters.NORM + "_"
    save_path += "NegFactor-x" + str(parameters.NEG_SAMPLES) + "_"
    save_path += "CallRepeats-" + str(parameters.CALL_REPEATS) + "_"
    # Note should include something potentially if using focal loss
    save_path += "TestNegFactor-x" + str(parameters.TEST_NEG_SAMPLES) + "_"
    save_path += "Loss-" + parameters.LOSS + "_"

    if parameters.LOSS == "BOUNDARY":
        save_path += "BoundaryFudgeFac-" + str(parameters.BOUNDARY_FUDGE_FACTOR) + "_"
        save_path += "IndividualBoundaries-" + str(parameters.INDIVIDUAL_BOUNDARIES) + "_"
        save_path += "BoundaryLoss-" + str(parameters.BOUNDARY_LOSS) + "_"
        if parameters.BOUNDARY_LOSS.lower() == "weight":
            save_path += "BoundaryWeight-" + str(parameters.BOUNDARY_WEIGHT) + "_"

    save_path +=  str(save_time)
    
    return save_path

def is_eval_epoch(cur_epoch):
    """Determines if the model should be evaluated at the current epoch."""
    return (
        (cur_epoch + 1) % parameters.EVAL_PERIOD == 0 or
        cur_epoch == 0 or
        (cur_epoch + 1) == parameters.NUM_EPOCHS
    )

def sigmoid(x):                                        
    return 1 / (1 + np.exp(-x))

def num_correct(logits, labels):
    sig = nn.Sigmoid()
    with torch.no_grad():
        pred = sig(logits)
        binary_preds = pred > parameters.THRESHOLD
        # Cast to proper type!
        binary_preds = binary_preds.float()
        num_correct = (binary_preds == labels).sum()

    return num_correct

def num_non_zero(logits, labels):
    sig = nn.Sigmoid()
    with torch.no_grad():
        pred = sig(logits)
        binary_preds = pred > parameters.THRESHOLD
        # Cast to proper type!
        binary_preds = binary_preds.float()
        num_non_zero = binary_preds.sum()

    return num_non_zero


def get_f_score(logits, labels):
    sig = nn.Sigmoid()
    with torch.no_grad():
        pred = sig(logits)
        binary_preds = pred > parameters.THRESHOLD
        # Flatten the array for fscore
        binary_preds = binary_preds.view(-1)
        labels = labels.view(-1)

        # Cast to proper type!
        binary_preds = binary_preds.float()
        f_score = f1_score(labels.data.cpu().numpy(), binary_preds.data.cpu().numpy())

    return f_score
