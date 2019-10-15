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
from data import get_loader
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
from model import Model0, Model1, Model2, Model3, Model4, Model5, Model6, Model7, Model8, Model9, Model10, Model11

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
THRESHOLD = 0.5

def loadModel(model_id):
    model = torch.load(parameters.MODEL_SAVE_PATH + parameters.DATASET + '_model_' + model_id + ".pt", map_location=device)
    print (model)
    return model

  
#def call_matched(start_idx, ground_truth, )

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

def main():
    # Make sure we specify the metric to test and
    # the model to test on!
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


if __name__ == '__main__':
    main()