from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_curve, f1_score
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
from model import Model0, Model1, Model2, Model3, Model4, Model5, Model6, Model7, Model8, Model9, Model10

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
THRESHOLD = 0.5

def loadModel(model_id):
    model = torch.load(parameters.MODEL_SAVE_PATH + parameters.DATASET + '_model_' + model_id + ".pt", map_location=device)
    print (model)
    return model
  
  


def pcr(dloader, model):
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
    #np.save('precision_Rnn.npy',precision)
    #np.save('recall_Rnn.npy',recall)



def main():
    # Make sure we specify the metric to test and
    # the model to test on!
    assert(len(sys.argv) > 2)

    run_type = sys.argv[1]
    model_id = sys.argv[2]

    model = loadModel(model_id)

    # Load val dataset to test on metrics 
    validation_loader = get_loader("../elephant_dataset/Test/" + parameters.DATASET + '_Label/', parameters.BATCH_SIZE, parameters.NORM, parameters.SCALE)

    if run_type == "PRC":
        pcr(validation_loader, model)

if __name__ == '__main__':
    main()