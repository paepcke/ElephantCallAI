import numpy as np
import torch
import torch.nn as nn
import os
import argparse

import parameters
from data import get_loader_fuzzy
from utils import sigmoid, calc_accuracy, num_correct, num_non_zero, get_f_score, get_precission_recall_values

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str,
    help = 'Path to the model to test on')
parser.add_argument('--local_files', dest='local_files', action='store_true',
    help='Flag specifying to read data from the local elephant_dataset directory.'
    'The default is to read from the quatro data directory.')

# Should put into utils!
def loadModel(model_path):
    model = torch.load(model_path, map_location=parameters.device)
    print (model)
    # Get the model name from the path
    tokens = model_path.split('/')
    model_id = tokens[-2]
    return model, model_id

def eval_model(dataloader, model):
    model.eval()

    running_corrects = 0
    running_samples = 0
    # True positives
    running_tp = 0
    # True positives, false positives
    running_tp_fp = 0
    # True positives, false negatives
    running_tp_fn = 0

    print ("Num batches:", len(dataloader))
    with torch.no_grad(): 
        for idx, batch in enumerate(dataloader):
            if (idx % 1000 == 0):
                print ("Batch number {} of {}".format(idx, len(dataloader)))

            # Cast the variables to the correct type and 
            # put on the correct torch device
            inputs = batch[0].clone().float()
            labels = batch[1].clone().float()
            inputs = inputs.to(parameters.device)
            labels = labels.to(parameters.device)

            # Forward pass
            logits = model(inputs).squeeze() # Shape - (batch_size, seq_len)

            running_corrects += num_correct(logits, labels)
            running_samples += logits.shape[0]

            tp, tp_fp, tp_fn = get_precission_recall_values(logits, labels)
            running_tp += tp
            running_tp_fp += tp_fp
            running_tp_fn += tp_fn 

    full_acc = float(running_corrects) / running_samples

    # If this is zero print a warning
    full_precision = running_tp / running_tp_fp if running_tp_fp > 0 else 1
    full_recall = running_tp / running_tp_fn
    if full_precision + full_recall > 0:
        full_fscore = (2 * full_precision * full_recall) / (full_precision + full_recall)
    else:
        full_fscore = 0

    #Logging
    print('Full Accuracy: {:.4f}'.format(full_acc))
    print ('Full Precision: {:.4f}'.format(full_precision))
    print ('Full Recall: {:.4f}'.format(full_recall))
    print ('Full F-Score: {:.4f}'.format(full_fscore))


def main(args):
    """
    Example runs:

    """
    # Load Model
    model, model_id = loadModel(args.model)
    # Put in eval mode!
    print (model_id)
    
    # For the window classification setting we literally just want to test on the full_spect chunks
    if args.local_files:
        test_data_path = parameters.LOCAL_FULL_TEST
    else:
        test_data_path = parameters.REMOTE_FULL_TEST
        
    
    eval_loader = get_loader_fuzzy(test_data_path, parameters.BATCH_SIZE, random_seed=parameters.DATA_LOADER_SEED, 
                                        norm=parameters.NORM, scale=parameters.SCALE, full_window_predict=True)
       
    eval_model(eval_loader, model)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)






