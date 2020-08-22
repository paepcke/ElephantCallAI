import torch
from utils import is_eval_epoch, num_correct, num_non_zero, get_f_score, get_precission_recall_values
import time
import parameters
import pdb
import sys
from collections import deque
import faulthandler; faulthandler.enable()


def train_epoch(dataloader, model, loss_func, optimizer, scheduler, writer, 
            include_boundaries=False):
    model.train(True)
    time_start = time.time()

    running_loss = 0.0
    running_corrects = 0
    running_samples = 0
    # May ditch these!
    running_non_zero = 0
    # True positives
    running_tp = 0
    # True positives, false positives
    running_tp_fp = 0
    # True positives, false negatives
    running_tp_fn = 0
    #running_fscore = 0.0
    #running_precission = 0.0
    #running_recall = 0.0

    # For focal loss purposes
    running_true_non_zero = 0

    print ("Num batches:", len(dataloader))
    for idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        if (idx % 250 == 0) and parameters.VERBOSE:
            print ("Batch number {} of {}".format(idx, len(dataloader)))
            print ("Total Non Zero Predicted {}, Total True Non Zero {}".format(running_non_zero, running_true_non_zero))

        # Cast the variables to the correct type and 
        # put on the correct torch device
        inputs = batch[0].clone().float()
        labels = batch[1].clone().float()
        inputs = inputs.to(parameters.device)
        labels = labels.to(parameters.device)

        # Forward pass
        # ONLY Squeeze the last dim!
        logits = model(inputs).squeeze()
        # Are we zeroing out the hidden state in the model???
        # Include boundary positions if necessary
        if include_boundaries:
            boundary_masks = batch[2]
            loss = loss_func(logits, labels, boundary_masks)
        else:
            loss = loss_func(logits, labels)

        #pdb.set_trace()
        loss.backward()
        optimizer.step()

        running_true_non_zero += torch.sum(labels).item()
        running_loss += loss.item()
        running_corrects += num_correct(logits, labels)
        running_non_zero += num_non_zero(logits, labels)
        if len(logits.shape) == 2:
            running_samples += logits.shape[0] * logits.shape[1] # Count the number slices for accuracy calculations
        else: # For the binary window classification
            running_samples += logits.shape[0]
        #running_fscore += get_f_score(logits, labels)
        tp, tp_fp, tp_fn = get_precission_recall_values(logits, labels)
        running_tp += tp
        running_tp_fp += tp_fp
        running_tp_fn += tp_fn 
        

    train_epoch_loss = running_loss / (idx + 1)
    train_epoch_acc = float(running_corrects) / running_samples
    #train_epoch_fscore = running_fscore / (idx + 1)
    train_non_zero = running_non_zero

    # If this is zero print a warning
    train_epoch_precision = running_tp / running_tp_fp if running_tp_fp > 0 else 1
    train_epoch_recall = running_tp / running_tp_fn
    if train_epoch_precision + train_epoch_recall > 0:
        train_epoch_fscore = (2 * train_epoch_precision * train_epoch_recall) / (train_epoch_precision + train_epoch_recall)
    else:
        train_epoch_fscore = 0

    # Update the schedular
    scheduler.step()

    #Logging
    print ('Train Non-Zero: {}'.format(train_non_zero))
    print('Training loss: {:.6f}, acc: {:.4f}, p: {:.4f}, r: {:.4f}, f-score: {:.4f}, time: {:.4f}'.format(
        train_epoch_loss, train_epoch_acc, train_epoch_precision, train_epoch_recall, train_epoch_fscore ,(time.time()-time_start)/60))
    
    return {'train_epoch_acc': train_epoch_acc, 'train_epoch_fscore': train_epoch_fscore, 
            'train_epoch_loss': train_epoch_loss, 'train_epoch_precision':train_epoch_precision, 
            'train_epoch_recall': train_epoch_recall} 


def eval_epoch(dataloader, model, loss_func, writer, include_boundaries=False):
    model.eval()
    time_start = time.time()

    running_loss = 0.0
    running_corrects = 0
    running_samples = 0
    # May ditch these!
    running_non_zero = 0
    running_fscore = 0.0
    # True positives
    running_tp = 0
    # True positives, false positives
    running_tp_fp = 0
    # True positives, false negatives
    running_tp_fn = 0

    # For focal loss purposes
    running_true_non_zero = 0

    print ("Num batches:", len(dataloader))
    with torch.no_grad(): 
        for idx, batch in enumerate(dataloader):
            if (idx % 250 == 0) and parameters.VERBOSE:
                print ("Batch number {} of {}".format(idx, len(dataloader)))
                print ("Total Non Zero Predicted {}, Total True Non Zero {}".format(running_non_zero, running_true_non_zero))
            # Cast the variables to the correct type and 
            # put on the correct torch device
            inputs = batch[0].clone().float()
            labels = batch[1].clone().float()
            inputs = inputs.to(parameters.device)
            labels = labels.to(parameters.device)

            # Forward pass
            # ONLY Squeeze the last dim!
            logits = model(inputs).squeeze(-1) # Shape - (batch_size, seq_len)
            # Are we zeroing out the hidden state in the model???

            if include_boundaries:
                boundary_masks = batch[2]
                loss = loss_func(logits, labels, boundary_masks)
            else:
                loss = loss_func(logits, labels)


            running_true_non_zero += torch.sum(labels).item()
            running_loss += loss.item()
            running_corrects += num_correct(logits, labels)
            running_non_zero += num_non_zero(logits, labels)
            if len(logits.shape) == 2:
                running_samples += logits.shape[0] * logits.shape[1] # Count the number slices for accuracy calculations
            else: # For the binary window classification
                running_samples += logits.shape[0]
            #running_fscore += get_f_score(logits, labels)
            tp, tp_fp, tp_fn = get_precission_recall_values(logits, labels)
            running_tp += tp
            running_tp_fp += tp_fp
            running_tp_fn += tp_fn 

    valid_epoch_loss = running_loss / (idx + 1)
    valid_epoch_acc = float(running_corrects) / running_samples
    #valid_epoch_fscore = running_fscore / (idx + 1)
    valid_non_zero = running_non_zero

    # If this is zero print a warning
    valid_epoch_precision = running_tp / running_tp_fp if running_tp_fp > 0 else 1
    valid_epoch_recall = running_tp / running_tp_fn
    if valid_epoch_precision + valid_epoch_recall > 0:
        valid_epoch_fscore = (2 * valid_epoch_precision * valid_epoch_recall) / (valid_epoch_precision + valid_epoch_recall)
    else:
        valid_epoch_fscore = 0

    #Logging
    print ('Val Non-Zero: {}'.format(valid_non_zero))
    print('Validation loss: {:.6f}, acc: {:.4f}, p: {:.4f}, r: {:.4f}, f-score: {:.4f}, time: {:.4f}'.format(
            valid_epoch_loss, valid_epoch_acc, valid_epoch_precision, valid_epoch_recall,
            valid_epoch_fscore, (time.time()-time_start)/60))


    return {'valid_epoch_acc': valid_epoch_acc, 'valid_epoch_fscore': valid_epoch_fscore, 
            'valid_epoch_loss': valid_epoch_loss, 'valid_epoch_precision':valid_epoch_precision, 
            'valid_epoch_recall': valid_epoch_recall}

def train(dataloaders, model, loss_func, optimizer, 
                        scheduler, writer, num_epochs, starting_epoch=0, include_boundaries=False):
    
    train_start_time = time.time()

    dataset_sizes = {'train': len(dataloaders['train'].dataset), 
                     'valid': len(dataloaders['valid'].dataset)}

    best_valid_acc = 0.0
    best_valid_fscore = 0.0
    # Best precision and recall reflect
    # the best fscore
    best_valid_precision = 0.0
    best_valid_recall = 0.0
    best_valid_loss = float("inf")
    best_model_wts = None

    # Check this
    last_validation_accuracies = deque(maxlen=parameters.TRAIN_STOP_ITERATIONS)
    last_validation_fscores = deque(maxlen=parameters.TRAIN_STOP_ITERATIONS)
    last_validation_losses = deque(maxlen=parameters.TRAIN_STOP_ITERATIONS)

    # See if this still works!
    try:
        for epoch in range(starting_epoch, num_epochs):
            print ('Epoch [{}/{}]'.format(epoch + 1, num_epochs))

            train_epoch_results = train_epoch(dataloaders['train'], model, loss_func, optimizer, scheduler, writer, include_boundaries)
            ## Write train metrics to tensorboard
            writer.add_scalar('train_epoch_loss', train_epoch_results['train_epoch_loss'], epoch)
            writer.add_scalar('train_epoch_acc', train_epoch_results['train_epoch_acc'], epoch)
            writer.add_scalar('train_epoch_fscore', train_epoch_results['train_epoch_fscore'], epoch)
            writer.add_scalar('learning_rate', scheduler.get_lr(), epoch)

            if is_eval_epoch(epoch):
                val_epoch_results = eval_epoch(dataloaders['valid'], model, loss_func, writer, include_boundaries) 
                ## Write val metrics to tensorboard
                writer.add_scalar('valid_epoch_loss', val_epoch_results['valid_epoch_loss'], epoch)
                writer.add_scalar('valid_epoch_acc', val_epoch_results['valid_epoch_acc'], epoch)
                writer.add_scalar('valid_epoch_fscore', val_epoch_results['valid_epoch_fscore'], epoch)

                # Update eval tracking statistics!
                last_validation_accuracies.append(val_epoch_results['valid_epoch_acc'])
                last_validation_fscores.append(val_epoch_results['valid_epoch_fscore'])
                last_validation_losses.append(val_epoch_results['valid_epoch_loss'])

                if val_epoch_results['valid_epoch_acc'] > best_valid_acc:
                    best_valid_acc = val_epoch_results['valid_epoch_acc']
                    if parameters.TRAIN_MODEL_SAVE_CRITERIA.lower() == 'acc':
                        best_model_wts = model.state_dict()

                if val_epoch_results['valid_epoch_fscore'] > best_valid_fscore:
                    best_valid_fscore = val_epoch_results['valid_epoch_fscore']
                    best_valid_precision = val_epoch_results['valid_epoch_precision']
                    best_valid_recall = val_epoch_results['valid_epoch_recall']
                    if parameters.TRAIN_MODEL_SAVE_CRITERIA.lower() == 'fscore':
                        best_model_wts = model.state_dict()

                if val_epoch_results['valid_epoch_loss'] < best_valid_loss:
                    best_valid_loss = val_epoch_results['valid_epoch_loss'] 

                # Check whether to early stop due to decreasing validation acc or f-score
                if parameters.TRAIN_MODEL_SAVE_CRITERIA.lower() == 'acc':
                    if all([val_accuracy < best_valid_acc for val_accuracy in last_validation_accuracies]):
                        print("Early stopping because last {} validation accuracies have been {} " 
                            "and less than best val accuracy {}".format(parameters.TRAIN_STOP_ITERATIONS, 
                            last_validation_accuracies, best_valid_acc))
                        break
                elif parameters.TRAIN_MODEL_SAVE_CRITERIA.lower() == 'fscore':
                    if all([val_fscore < best_valid_fscore for val_fscore in last_validation_fscores]):
                        print("Early stopping because last {} validation f-scores have been {} "
                            "and less than best val f-score {}".format(parameters.TRAIN_STOP_ITERATIONS, 
                            last_validation_fscores, best_valid_fscore))
                        break

            print('Finished Epoch [{}/{}] - Total Time: {}.'.format(epoch + 1, num_epochs, (time.time()-train_start_time)/60))

    except KeyboardInterrupt:
        print("Early stopping due to keyboard intervention")

    print('Best val Acc: {:4f}'.format(best_valid_acc))
    print('Best val F-score: {:4f} with Precision: {:4f} and Recall: {:4f}'.format(best_valid_fscore, best_valid_precision, best_valid_recall))
    print ('Best val Loss: {:6f}'.format(best_valid_loss))

    return best_model_wts
