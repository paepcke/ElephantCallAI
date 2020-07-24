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

def save_false_positives(files, path):
    final_slash = path.rindex('/')
    model_path = path[:final_slash]
    save_path = os.path.join(model_path, "false_positives.txt")

    with open(save_path, 'w') as f:
        for file in files:
            f.write('{}\n'.format(file))


def eval_model(dataloader, model, model_path):
    model.eval()

    running_corrects = 0
    running_samples = 0
    # True positives
    running_tp = 0
    # True positives, false positives
    running_tp_fp = 0
    # True positives, false negatives
    running_tp_fn = 0

    false_positives = []

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
            # Get the data_file locations for each chunk
            data_files = np.array(batch[2])

            # Forward pass
            logits = model(inputs).squeeze() # Shape - (batch_size, seq_len)

            # Determine and save false negative examples
            predictions = torch.sigmoid(logits)
            binary_preds = torch.where(predictions > parameters.THRESHOLD, torch.tensor(1.0).to(parameters.device), torch.tensor(0.0).to(parameters.device))
            binary_preds = binary_preds.cpu().detach().numpy()
            detach_labels = labels.cpu().detach().numpy()
            # We want to look for chunks where the prediction is
            # a false negative for the entire window
            gt_empty = (detach_labels == 0.)
            predicted_chunks = (binary_preds == 1.)

            batch_false_positives = list(data_files[gt_empty & predicted_chunks])
            false_positives += batch_false_positives

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

    save_false_positives(false_positives, model_path)


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
       
    eval_model(eval_loader, model, args.model)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)






