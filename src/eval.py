from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_curve, f1_score, precision_recall_fscore_support
from data import get_loader, ElephantDatasetFull
from torchsummary import summary
import time
from tensorboardX import SummaryWriter
import sys
import copy
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.cm
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.ticker as plticker
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
import parameters
from model import num_correct
from model import Model0, Model1, Model2, Model3, Model4, Model5, Model6, Model7, Model8, Model9, Model10, Model11, Model14
from process_rawdata_new import generate_labels
from visualization import visualize, visualize_predictions
from scipy.io import wavfile
import math
from matplotlib import mlab as ml
import parameters
import sklearn
from scipy.ndimage import gaussian_filter1d
import csv
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--preds_path', type=str, dest='predictions_path', default='../Predictions',
    help = 'Path to the folder where we output the full test predictions')

parser.add_argument('--test_files', type=str, default='../elephant_dataset/Test_New/Neg_Samples_x2/files.txt')
# For quatro
#parser.add_argument('--test_files', type=str, default='../elephant_dataset/Test/files.txt')

parser.add_argument('--spect_path', type=str, default="../elephant_dataset/New_Data/Spectrograms", 
    help='Path to the processed spectrogram files')
# For quatro
#parser.add_argument('--spect_path', type=str, default="/home/data/elephants/rawdata/Spectrograms/", 
#    help='Path to the processed spectrogram files')

parser.add_argument('--make_full_preds', action='store_true', 
    help = 'Generate predictions for the full test spectrograms')
parser.add_argument('--full_stats', action='store_true',
    help = 'Compute statistics on the full test spectrograms')
parser.add_argument('--pred_calls', action='store_true', 
    help = 'Generate the predicted (start, end) calls for test spectrograms')
parser.add_argument('--model_id', type=str, default='16',
    help = 'ID of the model to test on')




device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
THRESHOLD = 0.5
predictions_path = '../Predictions'
spectrogram_path = '../elephant_dataset/New_Data/Spectrograms'

def loadModel(model_id):
    model = torch.load(parameters.MODEL_SAVE_PATH + parameters.DATASET + '_model_' + model_id + ".pt", map_location=device)
    print (model)
    return model

  
def call_recall(dloader, model):
    """
        Metric:
        Calculates of the true calls how many do we find
        at least to some degree, for example a percantage
        of the call is identified
    """

    total_calls = 0
    total_labeled = 0
    for inputs, labels in dloader:
        inputs = inputs.float()
        labels = labels.float()

        inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))
        # Forward pass
        outputs = model(inputs) # Shape - (batch_size, seq_len, 1)
        # Compute the sigmoid over our outputs
        compressed_out = outputs.squeeze()
        sig = nn.Sigmoid()
        predictions = sig(compressed_out)
        binary_preds = np.where(predictions > THRESHOLD, 1, 0)

        # Travesere the batch
        for i in range(len(inputs)):
            example = binary_preds[i]
            labeling = labels[i]

            num_matched = 0

            #for j in range(len(example)):


def call_identification(dloader, model):
    """
        Metric:
        Calculates accuracy based on how many elephant calls
        we correctly predict some frames of (i.e. identify part of).
        This metric should give a bit of a sense of how well the 
        model (lstm) is picking up on the fact that we have seen 
        an elephant call, ignoring the constraint that we have 
        to correctly label all of the call. 
    """  
    # Specifically:
    # Of all the calls that we predicted as a string of 1's
    # how many fall in an elephant call. Obviously if predict 
    # all ones we get 100%, but that is not our original objective
    # function so it should be interesting to look into this.
    total_labeled = 0
    in_calls = 0

    for inputs, labels in dloader:
        inputs = inputs.float()
        labels = labels.float()

        inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))
        # Forward pass
        outputs = model(inputs) # Shape - (batch_size, seq_len, 1)
        # Compute the sigmoid over our outputs
        compressed_out = outputs.squeeze()
        sig = nn.Sigmoid()
        predictions = sig(compressed_out)
        binary_preds = np.where(predictions > THRESHOLD, 1, 0)

        # Traverse the predictions vector for each data example
        # to see how many predictions overalap a true call
        for i in range(len(inputs)):
            example = binary_preds[i]
            labeling = labels[i]

            # Keeps track of whether we have already
            # overlapped a call for the given prediction
            matched = False
            predictLen = 0
            inPredict = False
            MIN_LENGTH = 3
            # Must predict for at least X time frames
            for j in range(example.shape[0]):
                if predictLen > MIN_LENGTH:
                    if not inPredict:
                        total_labeled += 1
                        inPredict = True

                    # Also check if we have matched
                    if matched:
                        in_calls += 1

                if example[j] == 1:
                    predictLen += 1
                    #inPredict = True
                    #total_labeled += 1
                elif example[j] == 0:
                    inPredict = False
                    matched = False

                # Check if we hit an elephant call
                # and we haven't hit one already. 
                # We do this to avoid double counting
                if inPredict > 0 and labeling[j] == 1 and not matched:
                    #in_calls += 1
                    matched = True


    call_identification_acc = float(in_calls) / float(total_labeled)
    print (call_identification_acc)


def trigger_word_accuracy(output, truth):
    num_triggered = 0
    num_calls = 0
    i = 0
    while i < len(truth):
        if truth[i] == 1:
            num_calls += 1
            call_detected_flag = False
            while i < len(truth) and truth[i] == 1:
                if not call_detected_flag:
                    if output[i] == 1:
                        call_detected_flag = True
                        num_triggered += 1
                i += 1
        i += 1
    if num_calls == 0:
        return 0.0
    return num_triggered / num_calls

def visual_time_series(spectrum, predicts, labels):
    # Do the timeseries with the most calls
    most_index = 0
    number_most = 0
    for i in range(20):
        num_calls = 0
        inCall = False
        for j in range(spectrum.shape[1]):
            if not inCall and labels[i, j] == 1:
                inCall = True
                num_calls += 1
            if labels[i, j] == 0:
                inCall = False

        if num_calls > number_most:
            most_index = i
            number_most = num_calls

    print (number_most)
    # Just visualize one time series
    spect = spectrum[most_index].detach().numpy()
    label = labels[most_index].detach().numpy()
    predict = torch.sigmoid(predicts[most_index]).detach().numpy()

    # get num chunks
    num_chunks = int(spect.shape[0] / 64)
    for j in range(num_chunks):
        chunk_start = j * 64
        chunk = spect[chunk_start: chunk_start + 64, :]
        lab = label[chunk_start: chunk_start + 64]
        pred = predict[chunk_start: chunk_start + 64]

        fig, (ax1, ax2, ax3) = plt.subplots(3,1)
        # new_features = np.flipud(10*np.log10(features).T)
        new_features = np.flipud(chunk.T)
        min_dbfs = new_features.flatten().mean()
        max_dbfs = new_features.flatten().mean()
        min_dbfs = np.maximum(new_features.flatten().min(),min_dbfs-2*new_features.flatten().std())
        max_dbfs = np.minimum(new_features.flatten().max(),max_dbfs+6*new_features.flatten().std())
        ax1.imshow(np.flipud(new_features), cmap="magma_r", vmin=min_dbfs, vmax=max_dbfs, interpolation='none', origin="lower", aspect="auto")
        ax2.plot(np.arange(pred.shape[0]), pred)
        # Plot the threshold
        ax2.axhline(y=THRESHOLD, color='r', linestyle='-')
        ax2.set_ylim([0,1])
        ax3.plot(np.arange(lab.shape[0]), lab)
        ax3.set_ylim([0, 1])
        mngr = plt.get_current_fig_manager()
        geom = mngr.window.geometry()
        #x,y,dx,dy = geom.getRect()
        #mngr.window.Position((400, 500))
        mngr.window.wm_geometry("+400+250")
        #plt.draw()
        #plt.pause(0.001)
        #plt.clf()
        plt.show()

def visual_full_test(spectrogram, predictions, labels, chunk_size=256):
    # get num chunks
    num_chunks = int(spectrogram.shape[0] / chunk_size)
    for j in range(num_chunks):
        chunk_start = j * chunk_size
        chunk = spectrogram[chunk_start: chunk_start + chunk_size, :]
        lab = labels[chunk_start: chunk_start + chunk_size]
        pred = predictions[chunk_start: chunk_start + chunk_size]

        fig, (ax1, ax2, ax3) = plt.subplots(3,1)
        new_features = chunk.T
        min_dbfs = new_features.flatten().mean()
        max_dbfs = new_features.flatten().mean()
        min_dbfs = np.maximum(new_features.flatten().min(),min_dbfs-2*new_features.flatten().std())
        max_dbfs = np.minimum(new_features.flatten().max(),max_dbfs+6*new_features.flatten().std())
        ax1.imshow(new_features, cmap="magma_r", vmin=min_dbfs, vmax=max_dbfs, interpolation='none', origin="lower", aspect="auto")
        ax2.plot(np.arange(pred.shape[0]), pred)
        # Plot the threshold
        ax2.axhline(y=THRESHOLD, color='r', linestyle='-')
        ax2.set_ylim([0,1])
        ax3.plot(np.arange(lab.shape[0]), lab)
        ax3.set_ylim([0, 1])
        mngr = plt.get_current_fig_manager()
        geom = mngr.window.geometry()
        mngr.window.wm_geometry("+400+250")
        plt.show()

def visual_full_recall(spectrogram, predictions, labels, binary_preds, chunk_size=256):
    """
        Step through the ground truth labels and visualize
        the models predictions around these calls
    """
    search = 0
    while True:
        begin = search
        # Look for the start of a predicted calls
        while begin < spectrogram.shape[0] and labels[begin] == 0:
            begin += 1

        if begin >= spectrogram.shape[0]:
            break

        end = begin + 1
        # Look for the end of the call
        while end < spectrogram.shape[0] and labels[end] == 1:
            end += 1

        call_length = end - begin

        if (call_length > chunk_size):
            visualize(spectrogram[begin: end], predictions[begin: end] ,labels[begin:end])
        else:
            # Let us position the call in the middle
            padding = (chunk_size - call_length) // 2
            window_start = max(begin - padding, 0)
            window_end = min(end + padding, spectrogram.shape[0])
            visualize(spectrogram[window_start: window_end], predictions[window_start: window_end], 
                labels[window_start: window_end], binary_preds[window_start: window_end], vert_lines=(begin - window_start, end - window_start))

        search = end + 1


def pcr(dloader, model, visualize=False):
    """
        Generate Precision Recall Plot as well as print the F1 score for 
        each binary class
    """
    predVals = np.ones(1)
    labelVals = np.ones(1)
    
    for inputs, labels in dloader:
        inputs = inputs.float()
        labels = labels.float()
        #print (inputs.shape)

        inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))

        # Forward pass
        outputs = model(inputs) # Shape - (batch_size, seq_len, 1)
        # Try to visualize
        if visualize:
            visual_time_series(inputs, outputs, labels)
        # Compress to 
        # Shape - (batch_size * seq_length)
        # Compute the sigmoid over our outputs
        compressed_out = outputs.view(-1, 1)
        compressed_out = compressed_out.squeeze()
        sig = nn.Sigmoid()
        predictions = sig(compressed_out)
        compressed_labels = labels.view(-1, 1)
        compressed_labels = compressed_labels.squeeze()

        predVals = np.concatenate((predVals, predictions.detach().numpy()))
        labelVals = np.concatenate((labelVals, compressed_labels.detach().numpy()))
        
    predVals = predVals[1:]
    labelVals = labelVals[1:]

    precision, recall, thresholds = precision_recall_curve(labelVals, predVals)
    plt.plot(precision,recall)
    plt.show()

    # Compute F1-score
    # Make predictions based on threshold
    binary_preds = np.where(predVals > THRESHOLD, 1, 0)
    f1 = f1_score(labelVals, binary_preds, labels=[0, 1], average=None)  
    print ("F1 score label 0 (no call): ", f1[0])
    print ("F1 score label 1 (call): ", f1[1])

    precision, recall, fbeta, _ = precision_recall_fscore_support(labelVals, binary_preds, labels=[0,1], average=None)
    print ("Precision label 0 (no call): ", precision[0])
    print ("Precision label 1 (call): ", precision[1])
    print ("Recall label 0 (no call): ", recall[0])
    print ("Recall label 1 (call): ", recall[1])

    # Calculate Accuracy
    correct = (binary_preds == labelVals).sum()
    accuracy = float(correct) / binary_preds.shape[0]
    #np.save('precision_Rnn.npy',precision)
    #np.save('recall_Rnn.npy',recall)

    print("Trigger word detection recall was {:4f}".format(trigger_word_accuracy(binary_preds, labelVals)))
    print("Trigger word detection precision was {:4f}".format(trigger_word_accuracy(labelVals, binary_preds)))
    print("Those two should be different overall. Problem if they're the same (?)")



##############################################################################
### Full Time series evaluation assuming we have access to the spectrogram ###
##############################################################################

def test_overlap(s1, e1, s2, e2, threshold=0.1, is_truth=False):
    """
        Test is the source call defined by [s1: e1 + 1] 
        overlaps (if is_truth = False) or is overlaped
        (if is_truth = True) by the call defined by
        [s2: e2 + 1] with given threshold.
    """
    print ("Checking overlap of s1 = {}, e1 = {}".format(s1, e1))
    print ("and s2 = {}, e2 = {}".format(s2, e2))
    len_source = e1 - s1 + 1
    test_call_length = e2 - s2 + 1
    # Overlap check
    if s2 < s1: # Test call starts before the source call
        # The call is larger than our source call
        if e2 > e1:
            if is_truth:
                return True
            # The call we are comparing to is larger then the source 
            # call, but we must make sure that the source covers at least
            # overlap xs this call. Kind of edge case should watch this
            elif len_source >= max(threshold * test_call_length, 1): 
                print ('Prediction = portion of GT')
                return True
        else:
            overlap = e2 - s1 + 1
            print ("Test call starts before source call")
            if is_truth and overlap >= max(threshold * len_source, 1): # Overlap x% of ourself
                return True
            elif not is_truth and overlap >= max(threshold * test_call_length, 1): # Overlap x% of found call
                return True
    elif e2 > e1: # Call ends after the source call
        overlap = e1 - s2 + 1
        print ('Test call ends after source')
        if is_truth and overlap >= max(threshold * len_source, 1): # Overlap x% of ourself
            return True
        elif not is_truth and overlap >= max(threshold * test_call_length, 1): # Overlap x% of found call
            return True
    else: # We are completely in the call
        print ('Test call completely in source call')
        if not is_truth:
            return True
        elif test_call_length >= max(threshold * len_source, 1):
            return True

    return False

def call_prec_recall(test, compare, threshold=0.1, is_truth=False, spectrogram=None):
    """
        Adapted from Peter's paper.
        Given calls defined by their start and end time (in sorted order by occurence)
        we want to compute the "precision" and "recall." If is_truth = False then 
        we are comparing the predicted elephant calls from the detector to the ground
        truth elephant calls ==> (True Pos. and False Pos.). If is_truth = True then we
        are comparing the ground truth to the predictions and looking for ==>
        (True Pos. and False Neg.).

        1) is_truth = True: try to find a call in compare that overlaps (by at least prop. threshold) for
        each call in test

        2) is_truth = False: for a call in test, try to find a call in compare that it
        overlaps (by at least prop. threshold).

        Note 2) if threshold = 0 then we just want any amount of threshold
    """
    index_compare = 0
    true_events = []
    false_events = []
    for call in test:
        # Loop through the calls compare checking if the start is before 
        # our end
        start = call[0]
        end = call[1]
        length = call[2]

        # Rewind index_compare. Although this leads
        # to potentially extra computation, it is necessary
        # to avoid issues with overlapping calls, etc.
        # Can occur if on call overlaps two calls
        # or if we have overlapping calls
        # We basically want to consider all the calls
        # that could possibly overlap us.
        while index_compare > 0:
            # Back it up if we are past the end
            if (index_compare >= len(compare)):
                index_compare -= 1
                continue

            compare_call = compare[index_compare]
            compare_start = compare_call[0]
            compare_end = compare_call[1]

            # May have gone to far so back up
            if (compare_end > start):
                index_compare -= 1
            else:
                break

        found = False
        while index_compare < len(compare):
            # Get the call we are on to compare
            compare_call = compare[index_compare]
            compare_start = compare_call[0]
            compare_end = compare_call[1]
            compare_len = compare_call[2]

            # Call ends before 
            if compare_end < start:
                index_compare += 1
                continue

            # Call start after. Thus
            # all further calls end after. 
            if compare_start > end:
                break

            # If we are here then we know
            # compare_end >= start and compare_start <= end
            # so the calls overlap.
            if test_overlap(start, end, compare_start, compare_end, threshold, is_truth):
                found = True
                break

            # Let us think about tricky case with overlapping calls!
            # May need to rewind!
            index_compare += 1

        if found:
            true_events.append(call)
        else:
            false_events.append(call)

    return true_events, false_events


def process_ground_truth(label_path, in_seconds=False, samplerate=8000, NFFT=4096., hop=800.): # Was 3208, 641
    """
        Given ground truth call data for a given day / spectrogram, 
        generate a list of elephant calls as (start, end, duration).

        If in_seconds = True we leave the values in seconds

        Otherwise convert to the corresponding spectrogram "slice"
    """
    labelFile = csv.DictReader(open(label_path,'rt'), delimiter='\t')
    calls = []

    for row in labelFile:
        # Use the file offset to determine the start of the call
        start_time = float(row['File Offset (s)'])
        call_length = float(row['End Time (s)']) - float(row['Begin Time (s)'])
        end_time = start_time + call_length
        
        if in_seconds:
            calls.append((start_time, end_time, call_length))
        else: 
            # Figure out which spectrogram slices we are on
            # to get columns that we want to span with the given
            # slice. This math transforms .wav indeces to spectrogram
            # indices
            start_spec = max(math.ceil((start_time * samplerate - NFFT / 2.) / hop), 0)
            end_spec = min(math.ceil((end_time * samplerate - NFFT / 2.) / hop), labelMatrix.shape[0])
            len_spec = end_spec - start_spec 
            calls.append((start_spec, end_spec, len_spec))

    return calls


def find_elephant_calls(binary_preds, min_length=10, in_seconds=False, samplerate=8000., NFFT=4096., hop=800.): # Was 3208, 641
    """
        Given a binary predictions vector, we now want
        to step through and locate all of the elephant
        calls. For each continuous stream of 1s, we define
        a found call by the start and end frame within 
        the spectrogram (if in_seconds = False), otherwise
        we convert to the true start and end time of the call
        from begin time 0.

        If min_length != 0, then we only keep calls of
        a given length. Note that min_length is in FRAMES!!

        Note: some reference frame lengths.
        - 20 frames = 2.4 seconds 
        - 15 frames = 1.9 seconds
        - 10 frames = 1.4 seconds
    """
    calls = []
    processed_preds = binary_preds.copy()
    search = 0
    while True:
        begin = search
        # Look for the start of a predicted calls
        while begin < binary_preds.shape[0] and binary_preds[begin] == 0:
            begin += 1

        if begin >= binary_preds.shape[0]:
            break

        end = begin + 1
        # Look for the end of the call
        while end < binary_preds.shape[0] and binary_preds[end] == 1:
            end += 1

        call_length = end - begin

        # Found a predicted call!
        if (call_length >= min_length):
            if not in_seconds:
                # Note we subtract -1 to get the last frame 
                # that has the actual call
                calls.append((begin, end - 1, call_length))
            else:
                # Frame k is centered around second
                # (NFFT / sr / 2) + (k) * (hop / sr)
                # Meaning the first time it captures is (k) * (hop / sr)
                # Remember we zero index!
                begin_s = (begin) * (hop / samplerate) #(NFFT / samplerate / 2.) + 
                # Here we subtract 1 because end goes one past last column
                # And the last time it captures is (k) * (hop / sr) + NFFT / sr
                end_s = (end - 1) * (hop / samplerate) + (NFFT / samplerate)
                # k frames spans ==> (NFFT / sr) + (k-1) * (hop / sr) seconds
                call_length_s = (NFFT / samplerate) + (call_length - 1) * (hop / samplerate)

                calls.append((begin_s, end_s, call_length_s))
        else: # zero out the too short predictions
            processed_preds[begin:end] = 0

        search = end + 1

    return calls, processed_preds


def get_binary_predictions(predictions, threshold=0.5, smooth=True, sigma=1):
    """
        Generate the binary 0/1 predictions and output the 
        predictions vector used to do so. Note this is only important
        when we smooth the predictions vector.
    """
    if (smooth):
        predictions = gaussian_filter1d(predictions, sigma)

    binary_preds = np.where(predictions > threshold, 1, 0)

    return binary_preds, predictions


def predict_spec_full(spectrogram, model):
    """
        Generate the frame level predictions for a full spectrogram.
        Here we assume that the model does not use convolutions across
        the time dimension and thus can be simply run over the entire
        audio sequence with lstm model. 

        Return:
        Sigmoid predictions for each timestep of the spectrogram 

        Note: Even a convolutional model could be run over the entire
        sequence, if the convolutions are only used to create feature
        representations of the spectrogram before running an LSTM model
        over these representations; however, for this case, because we
        cannot run "on the edge" or "real-time" (i.e. one spectrogram
        frame at a time) we likely want to do a sliding window model
    """
    # By default the spectrogram has the time axis second
    spectrogram = torch.from_numpy(spectrogram).float()
    # Add a batch dim for the model!
    spectrogram = torch.unsqueeze(spectrogram, 0) # Shape - (1, time, freq)
    print (spectrogram.shape)

    spectrogram = Variable(spectrogram.to(device))

    outputs = model(spectrogram) # Shape - (1, seq_len, 1)
    compressed_out = outputs.view(-1, 1)
    compressed_out = outputs.squeeze()

    sig = nn.Sigmoid()
    predictions = sig(compressed_out)
    # Generate the binary predictions
    predictions = predictions.detach().numpy()

    return predictions

def predict_spec_sliding_window(spectrogram, model, chunk_size=256, jump=128):
    """
        Generate the prediction sequence for a full audio sequence
        using a sliding window. Slide the window by one spectrogram frame
        and pass each window through the given model. Compute the average
        over overlapping window predictions to get the final prediction.
    """
    # Get the number of frames in the full audio clip
    predictions = np.zeros(spectrogram.shape[0])
    overlap_counts = np.zeros(spectrogram.shape[0])

    spectrogram = torch.from_numpy(spectrogram).float()
    # Add a batch dim for the model!
    spectrogram = torch.unsqueeze(spectrogram, 0) # Shape - (1, time, freq)

    # For the sliding window we slide the window by one spectrogram
    # frame, determined by the hop size.
    spect_idx = 0 # The frame idx of the beginning of the current window
    i = 0
    # How can I parralelize this shit??????
    while  spect_idx + chunk_size <= spectrogram.shape[1]:
        if (i % 10 == 0):
            print ("Chunk number " + str(i))

        spect_slice = spectrogram[:, spect_idx: spect_idx + chunk_size, :]
        spect_slice = Variable(spect_slice.to(parameters.device))

        outputs = model(spect_slice) # Shape - (1, chunk_size, 1)
        compressed_out = outputs.view(-1, 1)
        compressed_out = outputs.squeeze()

        overlap_counts[spect_idx: spect_idx + chunk_size] += 1
        predictions[spect_idx: spect_idx + chunk_size] += compressed_out.detach().numpy()

        spect_idx += jump
        i += 1

    # Do the last one if it was not covered
    if (spect_idx - jump + chunk_size != spectrogram.shape[1]):
        print ('One final chunk!')
        spect_slice = spectrogram[:, spect_idx: , :]
        spect_slice = Variable(spect_slice.to(parameters.device))

        outputs = model(spect_slice) # Shape - (1, chunk_size, 1)
        compressed_out = outputs.view(-1, 1)
        compressed_out = outputs.squeeze()

        overlap_counts[spect_idx: ] += 1
        predictions[spect_idx: ] += compressed_out.detach().numpy() 


    # Average the predictions on overlapping frames
    predictions = predictions / overlap_counts

    # Get squashed [0, 1] predictions
    predictions = sigmoid(predictions)

    return predictions

def generate_predictions_full_spectrograms(dataset, model, model_id, predictions_path, 
    sliding_window=True, chunk_size=256, jump=128):
    """
        For each full test spectrogram, run a trained model to get the model
        prediction and save these predictions to the predictions folder. Namely,
        for each model we create a sub-folder with predictions for that model. Note,
        in the future we will not only save models as there number but also save
        them based on the negative factor that they were trained on. This will
        come based on the negative factor being included in the model_id

        Status:
        - works without saving with negative factor
    """
    for data in dataset:
        spectrogram = data[0]
        gt_call_path = data[2]

        # Get the spec id
        tags = gt_call_path.split('/')
        tags = tags[-1].split('_')
        data_id = tags[0] + '_' + tags[1]
        print ("Generating Prediction for:", data_id)

        if sliding_window:
            predictions = predict_spec_sliding_window(spectrogram, model, chunk_size=chunk_size, jump=jump)
        else:
            predictions = predict_spec_full(spectrogram, model)

        # Save preditions
        # Save for now to a folder determined by the model id
        path = predictions_path + '/' + model_id
        if not os.path.isdir(path):
            os.mkdir(path)
        # The data id associates predictions with a particular spectrogram
        np.save(path + '/' + data_id  + '.npy', predictions)


def eval_full_spectrograms(dataset, model_id, predictions_path, pred_threshold=0.5, overlap_threshold=0.1, smooth=True, 
            in_seconds=False, use_call_bounds=False, min_call_lengh=15, visualize=False):
    """

        After saving predictions for the test set of full spectrograms, we
        now want to calculate evaluation metrics on each of these spectrograms. 
        First we load in the sigmoid (0, 1) predictions and use the defined
        threshold and smoothing flag to convert the predictions into binary
        0/1 predcitions for each time slice. Then, we convert time slice
        predictions to full call (start, end) time predictions so that
        we can calculate elephant call specific evaluation metrics. Bellow
        we discuss the metrics that are calculated individually for each
        full spectrogram, as well as accross all the test spectrograms
        
        Metrics:
        - Call Prediction True Positives
        - Call Prediction False Positives
        - Call Recall True Positives
        - Call Recall False Negatives
        - F-score
        - Accuracy
        - Old Call Precision --- To be implemented
        - Old Call Recall --- To be implemented

    """
    # Maps spectrogram ids to dictionary of results for each spect
    # Additionally includes a key "summary" that computes aggregated
    # statistics over the entire test set of spectrograms
    results = {} 
    results['summary'] = {'true_pos': 0,
                            'false_pos': 0,
                            'true_pos_recall': 0,
                            'false_neg': 0,
                            'f_score': 0,
                            'accuracy': 0
                            }
    # Used to track the number of total calls for averaging
    # aggregated statistics
    num_preds = 0
    num_gt = 0
    for data in dataset:
        spectrogram = data[0]
        labels = data[1]
        gt_call_path = data[2]

        # Get the spec id
        tags = gt_call_path.split('/')
        tags = tags[-1].split('_')
        data_id = tags[0] + '_' + tags[1]
        print ("Generating Prediction for:", data_id)
        
        predictions = np.load(predictions_path + '/' + model_id + "/" + data_id + '.npy')

        binary_preds, smoothed_predictions = get_binary_predictions(predictions, threshold=pred_threshold, smooth=smooth)

        # Process the predictions to get predicted elephant calls
        # Figure out better way to try different combinations of this
        # Note that processed_preds zeros out predictions that are not long
        # enough to be an elephant call
        predicted_calls, processed_preds = find_elephant_calls(binary_preds, in_seconds=in_seconds)
        print ("Num predicted calls", len(predicted_calls))

        # Use the calls as defined in the orginal hand labeled file.
        # This looks to avoid issues of overlapping calls seeming like
        # single very large calls in the gt labeling 
        if use_call_bounds:
            print ("Using CSV file with ground truth call start and end times")
            gt_calls = process_ground_truth(gt_call_path, in_seconds=in_seconds)
        else:
            print ("Using GT spectrogram labeling to generate GT calls")
            # We should never compute this in seconds
            # Also let us keep all the calls, i.e. set min_length = 0
            gt_calls, _ = find_elephant_calls(labels, min_length=0)

        print ("Number of ground truth calls", len(gt_calls))

        # Visualize the predictions around the gt calls
        if visualize: # This is not super important
            visual_full_recall(spectrogram, smoothed_predictions, labels, processed_preds)       
        
        # Look at precision metrics
        # Call Prediction True Positives
        # Call Prediction False Positives
        true_pos, false_pos = call_prec_recall(predicted_calls, gt_calls, threshold=overlap_threshold, is_truth=False)

        # Look at recall metrics
        # Call Recall True Positives
        # Call Recall False Negatives
        true_pos_recall, false_neg = call_prec_recall(gt_calls, predicted_calls, threshold=overlap_threshold, is_truth=True)

        f_score = get_f_score(binary_preds, labels) # just for the postive class
        accuracy = calc_accuracy(binary_preds, labels)

        results[data_id] = {'true_pos': true_pos,
                            'false_pos': false_pos,
                            'true_pos_recall': true_pos_recall,
                            'false_neg': false_neg,
                            'f_score': f_score,
                            'predictions': smoothed_predictions,
                            'binary_preds': processed_preds,
                            'accuracy': accuracy
                            }
        # Update summary stats
        results['summary']['true_pos'] += len(true_pos)
        results['summary']['false_pos'] += len(false_pos)
        results['summary']['true_pos_recall'] += len(true_pos_recall)
        results['summary']['false_neg'] += len(false_neg)
        results['summary']['f_score'] += f_score
        results['summary']['accuracy'] += accuracy

    # Calculate averaged statistics
    results['summary']['true_pos'] 
    results['summary']['false_pos']
    results['summary']['true_pos_recall'] 
    results['summary']['false_neg'] 
    results['summary']['f_score'] /= len(dataset)
    results['summary']['accuracy'] /= len(dataset)

    return results


def get_spectrogram_paths(test_files_path, spectrogram_path):
    """
        In the test set folder, there is a file that includes
        all of the recording files used for the test set. Based
        on these files we want to get the spectrograms and gt
        labeling files that correspond
    """
    # Holds the paths to the:
    # - spectrograms
    # - labels for each spectrogram slice
    # - gt (start, end) times for calls
    paths = {'specs': [],
            'labels': [],
            'gts': []}

    with open(test_files_path, 'r') as f:
        lines = f.readlines()

    files = [x.strip() for x in lines]

    for file in files:
        # Create the spectrogram path by concatenating
        # the test file with the path to the folder
        # containing the spectrogram files
        paths['specs'].append(spectrogram_path + '/' + file + '_spec.npy')
        paths['labels'].append(spectrogram_path + '/' + file + '_label.npy')
        paths['gts'].append(spectrogram_path + '/' + file + '_gt.npy')

    return paths
    

def sigmoid(x):                                        
    return 1 / (1 + np.exp(-x))

def get_f_score(binary_preds, labels):
    return sklearn.metrics.f1_score(labels, binary_preds)

def calc_accuracy(binary_preds, labels):
    accuracy = (binary_preds == labels).sum() / labels.shape[0]
    return accuracy


def main():
    
    args = parser.parse_args()
    
    model = loadModel(args.model_id)
    
    full_test_spect_paths = get_spectrogram_paths(args.test_files, args.spect_path)
    full_dataset = ElephantDatasetFull(full_test_spect_paths['specs'],
                 full_test_spect_paths['labels'], full_test_spect_paths['gts'])    

    if args.make_full_preds:
        generate_predictions_full_spectrograms(full_dataset, model, args.model_id, args.predictions_path,
             sliding_window=True, chunk_size=256, jump=128)         # Add in these arguments
    elif args.full_stats:
        # Now we have to decide what to do with these stats
        results = eval_full_spectrograms(full_dataset, args.model_id, args.predictions_path)

        # Display the output of results as peter did
        TP_truth = results['summary']['true_pos_recall']
        FN = results['summary']['false_neg']
        TP_test = results['summary']['true_pos']
        FP = results['summary']['false_pos']

        recall = TP_truth / (TP_truth + FN)
        precision = TP_test / (TP_test + FP)
        f1_call = 2 * (precision * recall) / (precision + recall)
        # Do false pos rate later!!!!
        total_duration = 24. * len(full_test_spect_paths['specs'])
        false_pos_per_hour = FP / total_duration

        print ("Summary results")
        print("Call precision:", precision)
        print("Call recall:", recall)
        print("f1 score calls:", f1_call)
        print("False positve rate (FP / hr):", false_pos_per_hour)
        print("Segmentation f1-score:", results['summary']['f_score'])
        print("Average accuracy:", results['summary']['accuracy'])


    '''
    assert(len(sys.argv) > 2)

    run_type = sys.argv[1]
    model_id = sys.argv[2]

    model = loadModel(model_id)

    # Load val dataset to test on metrics 
    validation_loader = get_loader("../elephant_dataset/Test_Atlas/" + parameters.DATASET + '_Label/', parameters.BATCH_SIZE, parameters.NORM, parameters.SCALE)

    if run_type == "prc":
        pcr(validation_loader, model)
    elif run_type == "identify":
        call_identification(validation_loader, model)
    elif run_type == "Full_Test":
        full_data_loader = get_loader("../elephant_dataset/Test_Atlas/" + parameters.DATASET + '_Full_test/', parameters.BATCH_SIZE, parameters.NORM, parameters.SCALE)
        pcr(full_data_loader, model, False)
    else:
        print ("Enter options: (prc, identify)")
    '''
    
    

    #visualize_predictions(results['nn10b_20180604']['false_neg'], spectrogram, results['nn10b_20180604']['binary_preds'], labels)
    #print (results)


################################################################
##################### TO BE DEVELOPED ##########################
################################################################
'''

def predict_full_audio_semi_real_time(raw_audio, model, spectrogram_info):
    """
        Generate the prediction sequence for a full audio sequence
        that may be very long (such as a 24 hours day). Because of
        the audio size it is inefficient to generate
        and store the entire spectrogram at once. Therefore, we 
        pass in chunks of the spectrogram at a time. Since the real
        time model does not use temporal convolutions, we can essentially
        feed in each spectrogram slice indepentently while not re-setting the
        hidden state. Here we will do a psuedo real time version where
        we given chunks of a certain size rather than one spectrogram slice at a time.
    """
    prediction = np.zeros(1)
    
    NFFT = spectrogram_info['NFFT']
    samplerate = spectrogram_info['samplerate']
    hop = spectrogram_info['hop']
    max_freq = spectrogram_info['max_freq']
    chunk_size = int(spectrogram_info['window'] * spectrogram_info['hop'] + spectrogram_info['NFFT']) - 1

    start_idx = 0 # Index within the 1d audio
    inital_hidden = None
    while start_idx < raw_audio.shape[0]:
        # Extract the spectrogram chunk
        chunk_end = start_idx + chunk_size
        [spectrum, freqs, t] = ml.specgram(raw_audio[start_idx: chunk_end], 
                NFFT=NFFT, Fs=samplerate, noverlap=(NFFT - hop), window=ml.window_hanning)

        spectrum = spectrum[(freqs <= max_freq)]

        # If the start index = 0 
        # zero out the hidden state of the model
        # DO HERE ****
        if (start_idx == 0):
            print("zero out initial hidden state")
        else:
            # Use the hidden state from the end of the
            # last chunk (i.e. when we call forward in the LSTM
            # block we need to save the value of the returned final
            # hidden state
            # In the forward function we should define the
            # ability to pass in a starting hidden state
            print ("run model on chunk with old hidden state")
            print ("save the last hidden state for next time")

        start_idx += chunk_size

def predict_full_audio_sliding_window(raw_audio, model, spectrogram_info):
    """
        Generate the prediction sequence for a full audio sequence
        using a sliding window. Slide the window by one spectrogram frame
        and pass each window through the given model. Compute the average
        over overlapping window predictions to get the final prediction.
    """
    # Get the number of frames in the full audio clip
    #raw_audio = raw_audio[:324596]
    len_audio = math.floor((raw_audio.shape[0] - spectrogram_info['NFFT']) / spectrogram_info['hop'] + 1)
    print (len_audio)
    prediction = np.zeros(len_audio)
    # Keep a count of how many terms to average buy
    # for each index
    overlap_counts = np.zeros(len_audio)

    NFFT = spectrogram_info['NFFT']
    samplerate = spectrogram_info['samplerate']
    hop = spectrogram_info['hop']
    max_freq = spectrogram_info['max_freq']
    window = spectrogram_info['window'] # In spectrogram frames
    chunk_size = (spectrogram_info['window'] - 1) * spectrogram_info['hop'] + spectrogram_info['NFFT'] # In raw audio frames

    # Create the first chunk
    spectrum, freqs, _ = ml.specgram(raw_audio[0: chunk_size], 
                NFFT=NFFT, Fs=samplerate, noverlap=(NFFT - hop), window=ml.window_hanning) 
    spectrum = spectrum[(freqs <= max_freq)]

    # For the sliding window we slide the window by one spectrogram
    # frame, determined by the hop size.
    raw_audio_idx = chunk_size - NFFT # The beginning of the last fft window computed
    spect_idx = 0 # The frame idx of the beginning of the current window
    while  True:
        # Make predictions on the current chunk ??? Check what the expected input shape
        # is!
        #visualize(spectrum, labels=np.arange(spect_idx, spect_idx + window))
        #temp_predictions = model(spectrum.T)
        # Look at shape!! Probably need to compress dimension / squeeze

        #predictions[spect_idx: window] += temp_predictions
        overlap_counts[spect_idx: spect_idx + window] += 1

        # Try generating the next slice
        raw_audio_idx += hop
        spect_idx += 1
        if raw_audio_idx + NFFT >= raw_audio.shape[0]:
            break

        # For end of the file pad with zeros if necessary!
        #if (raw_audio.shape[0] - raw_audio_idx < window):
            #print ('here')
            #raw_audio = np.concatenate((raw_audio, np.zeros(window - (raw_audio.shape[0] - raw_audio_idx + 1))))
        # Generate a single spectrogram slice
        new_slice, freqs, _ = ml.specgram(raw_audio[raw_audio_idx: min(raw_audio_idx + NFFT, raw_audio.shape[0])], 
                NFFT=NFFT, Fs=samplerate, noverlap=(NFFT - hop), window=ml.window_hanning) 
        new_slice = new_slice[(freqs <= max_freq)]
        # Check this shape, should be 1 dim in the time axis

        # Add the new time slice
        spectrum = np.concatenate((spectrum, new_slice), axis = 1)
        # Get rid of old slice
        spectrum = spectrum[:, 1:]

    print (spect_idx - 1)
    print (len_audio)

################################################################
######################### OLD CODE #############################
################################################################

def test_full_spectrograms(dataset, model, model_id, sliding_window=True, 
            pred_threshold=0.5, overlap_threshold=0.1, chunk_size=256, jump=128, smooth=True, 
            in_seconds=False, use_call_bounds=False, min_call_lengh=15,
            visualize=False):
    """
        Given a dataset of containing full spectrogram information
        (i.e. (spectogram, label_vec, ground truth start end time file))
        compute test time metrics on each full spectrogram. The test time
        metric is defined to be that of Peter's paper.

        Parameters:

    """
    results = {} # This will map spectrogram id to dictionary of results for each spect
    for data in dataset:
        spectrogram = data[0]
        labels = data[1]
        gt_call_path = data[2]

        # Get the spec id
        tags = gt_call_path.split('/')
        tags = tags[-1].split('_')
        data_id = tags[0] + '_' + tags[1]
        print (data_id)
        
        # Include something if we have predictions to just load!!
        """
        if sliding_window:
            predictions = predict_spec_sliding_window(spectrogram, model, chunk_size=chunk_size, jump=jump, threshold=pred_threshold)
        else:
            predictions = predict_spec_full(spectrogram, model, threshold=pred_threshold)

        # Save preditions
        if not os.path.isdir(predictions_path):
            os.mkdir(predictions_path)
        np.save(predictions_path + '/' + data_id + '_' + model_id + '.npy', predictions)
        """
        
        predictions = np.load(predictions_path + '/' + data_id + '_' + model_id + '.npy')

        binary_preds, smoothed_predictions = get_binary_predictions(predictions, threshold=pred_threshold, smooth=smooth)


        # Process the predictions to get predicted elephant calls
        # Figure out better way to try different combinations of this
        predicted_calls, processed_preds = find_elephant_calls(binary_preds, in_seconds=in_seconds)
        print ("Num predicted calls", len(predicted_calls))
        if use_call_bounds:
            print ("Using CSV file with ground truth call start and end times")
            gt_calls = process_ground_truth(gt_call_path, in_seconds=in_seconds)
        else:
            print ("Using GT spectrogram labeling to generate GT calls")
            # We should never compute this in seconds
            # Also let us keep all the calls, i.e. set min_length = 0
            gt_calls, _ = find_elephant_calls(labels, min_length=0)

        print (len(gt_calls))
        # Visualize the predictions around the gt calls
        if visualize:
            visual_full_recall(spectrogram, smoothed_predictions, labels, processed_preds)       
        
        # Look at precision metrics
        true_pos, false_pos = call_prec_recall(predicted_calls, gt_calls, threshold=overlap_threshold, is_truth=False)
        # Look at recall metrics
        # Note true_pos_r should be the same as true_pos
        true_pos_r, false_neg = call_prec_recall(gt_calls, predicted_calls, threshold=overlap_threshold, is_truth=True)

        f_score = get_f_score(binary_preds, labels)

        results[data_id] = {'true_pos': true_pos,
                            'false_pos': false_pos,
                            'true_pos_r': true_pos_r,
                            'false_neg': false_neg,
                            'f_score': f_score,
                            'predictions': smoothed_predictions,
                            'binary_preds': processed_preds}
    return results




# Old method, purely based on segmentation
def call_overlaps(start, end, source, compare, threshold=.1, is_truth=False):
    """
        Given a call in the "source" stream (start, end)
        see if there is a call in the "compare" stream that overlaps.
        The measure of overlap changes depending on the value of is_truth.

        1) is_truth == True: the source sequence is the ground truth
        labeling and we want to see if there is a call that overlaps at least
        10% of the source call

        2) is_truth == False: the source sequence is the prediction and
        we want to see if we overlap a ground truth call by at least 10%
    """
    # Find the bounds of overlaping calls in the compare stream
    start_comp = start
    while start_comp >= 0 and compare[start_comp] == 1:
        start_comp -= 1
    # Need to add one back
    # if we took steps to the left
    if start_comp != start:
        start_comp += 1

    end_comp = end - 1
    while end_comp < compare.shape[0] and compare[end_comp] == 1:
        end_comp += 1

    # If no call extends past the end 
    # need to reset end_comp
    if (end_comp == end - 1):
        end_comp = end

    print ("Begin of search window", start_comp)
    print ('End of search window', end_comp)
    # Now we have defined the new bounded region in compare 
    # where overlapping calls can exist. Now we want to look 
    # to see if these calls are either overlapped by 10% or 
    # overlapp by 10%
    len_call = end - start 

    start_call = -1
    in_call = False
    index = start_comp
    while index < end_comp:
        if compare[index] == 1 and not in_call:
            in_call = True
            start_call = index

        # Exiting call so check overlap
        if in_call and compare[index] == 0 or in_call and index == end_comp - 1:
            # Edge case on right end of the window
            if (compare[index] != 0):
                index += 1

            # Should make sure call is large enough!!!!!!!
            if test_overlap_1(start, end, start_call, index, threshold=threshold, is_truth=is_truth):
                return True
    
            in_call = False
            start_call = index

        index += 1

    print ("No overlaps")
    return False # No good overlaps found

# Old method to test overlap
def test_overlap_1(s1, e1, s2, e2, threshold=0.1, is_truth=False):
    """
        Test is the source call defined by [s1: e1] 
        overlaps (if is_truth = False) or is overlaped
        (if is_truth = True) by the call defined by
        [s2: e2] with given threshold.
    """
    print ("Checking overlap of s1 = {}, e1 = {}".format(s1, e1))
    print ("and s2 = {}, e2 = {}".format(s2, e2))
    len_source = e1 - s1
    test_call_length = e2 - s2
    # Overlap check
    if s2 < s1: # Call starts before the source call
        # The call is larger than our source call
        if e2 > e1:
            if is_truth:
                return True
            # The call we are comparing to is larger then the source 
            # call, but we must make sure that the source covers at least
            # 10 % of this call. Kind of edge case should watch this
            elif len_source >= max(threshold * test_call_length, 1): 
                print ('edge case')
                return True
        else:
            overlap = e2 - s1
            print ("begin")
            if is_truth and overlap >= max(threshold * len_source, 1): # Overlap 10% of ourself
                return True
            elif not is_truth and overlap >= max(threshold * test_call_length, 1): # Overlap 10% of found call
                return True
    elif e2 > e1: # Call ends after the source call
        overlap = e1 - s2
        print ('end')
        if is_truth and overlap >= max(threshold * len_source, 1): # Overlap 10% of ourself
            return True
        elif not is_truth and overlap >= max(threshold * test_call_length, 1): # Overlap 10% of found call
            return True
    else: # We are completely in the call
        print ('whole')
        if not is_truth:
            return True
        elif test_call_length >= max(threshold * len_source, 1):
            return True

    return False

# Old method for comparing pred to test
def test_event_metric(test, compare, threshold=0.1, is_truth=False, call_length=15, spectrogram=None):
    """
        Adapted from Peter's paper.
        We are comparing the calls in the test sequence to the compare sequence.
        The is_truth tells us if test is the ground truth set or not. From there
        we look to for each call in the test set see if it overlaps with a call in
        the compare set to a "certain" degree determined by if is_truth or not:

        1) is_truth: try to find a call in compare that overlaps 10% for
        each call in test

        2) not is_truth: try to find a call in compare that each call in test 
        overlaps at least 10 % of.


        Note) Only say call if length >= 2 seconds (calculate this based on the samplerate)
        which for NFFT = 3208, hop = 641, sr = 8000 is 20 frames / 15 frames is 1.5 seconds

        Note 2) if threshold = 0 then we just want any amount of overlap
    """
    #two_sec = 2
    search = 0
    true_events = 0
    false_events = 0
    while True:
        begin = search
        # Look for the start of a predicted calls
        while begin < test.shape[0] and test[begin] == 0:
            begin += 1

        if begin >= test.shape[0]:
            break

        end = begin + 1
        # Look for the end of the call
        while end < test.shape[0] and test[end] == 1:
            end += 1

        # Check to see if the call is of adequete length to be a predicted call
        # may not need this!
        if end - begin < call_length: 
            search = end + 1
            continue

        # Check if we overlap a call
        print ("checking call s = {} e = {}".format(begin, end))

        if call_overlaps(begin, end, test, compare, threshold=threshold, is_truth=is_truth):
            true_events += 1
            print ("Success")
        else:
            print ("Failure")
            false_events +=1

        if (spectrogram is not None):
            # Let us position the call in the middle
            padding = (256 - call_length) // 2
            window_start = max(begin - padding, 0)
            window_end = min(end + padding, spectrogram.shape[0])
            visualize(spectrogram[window_start: window_end], test[window_start: window_end], compare[window_start: window_end])


        search = end + 1
        print()

    return true_events, false_events
'''


if __name__ == '__main__':
    main()