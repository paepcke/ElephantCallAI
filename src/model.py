"""
To train a model: 

python model.py <model_number>

For visualization:

python model.py visualize <path_to_model>
"""

from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_curve
from data import get_loader
from torchsummary import summary
import time
from tensorboardX import SummaryWriter
import sklearn
import sys
import copy
import os
import torch.nn.functional as F

import parameters
#import Metrics
from visualization import visualize
import torchvision.models as models
from torchvision import transforms
import matplotlib
import matplotlib.pyplot as plt

np.random.seed(parameters.RANDOM_SEED)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_model(idx):
    if idx == 0:
        return Model0(parameters.INPUT_SIZE, parameters.OUTPUT_SIZE, parameters.LOSS, parameters.FOCAL_WEIGHT_INIT)
    elif idx == 1:
        return Model1(parameters.INPUT_SIZE, parameters.OUTPUT_SIZE, parameters.LOSS, parameters.FOCAL_WEIGHT_INIT)
    elif idx == 2:
        return Model2(parameters.INPUT_SIZE, parameters.OUTPUT_SIZE, parameters.LOSS, parameters.FOCAL_WEIGHT_INIT)
    elif idx == 3:
        return Model3(parameters.INPUT_SIZE, parameters.OUTPUT_SIZE, parameters.LOSS, parameters.FOCAL_WEIGHT_INIT)
    elif idx == 4:
        return Model4(parameters.INPUT_SIZE, parameters.OUTPUT_SIZE, parameters.LOSS, parameters.FOCAL_WEIGHT_INIT)
    elif idx == 5:
        return Model5(parameters.INPUT_SIZE, parameters.OUTPUT_SIZE, parameters.LOSS, parameters.FOCAL_WEIGHT_INIT)
    elif idx == 6:
        return Model6(parameters.INPUT_SIZE, parameters.OUTPUT_SIZE, parameters.LOSS, parameters.FOCAL_WEIGHT_INIT)
    elif idx == 7:
        return Model7(parameters.INPUT_SIZE, parameters.OUTPUT_SIZE, parameters.LOSS, parameters.FOCAL_WEIGHT_INIT)
    elif idx == 8:
        return Model8(parameters.INPUT_SIZE, parameters.OUTPUT_SIZE, parameters.LOSS, parameters.FOCAL_WEIGHT_INIT)
    elif idx == 9:
        return Model9(parameters.INPUT_SIZE, parameters.OUTPUT_SIZE, parameters.LOSS, parameters.FOCAL_WEIGHT_INIT)
    elif idx == 10:
        return Model10(parameters.INPUT_SIZE, parameters.OUTPUT_SIZE, parameters.LOSS, parameters.FOCAL_WEIGHT_INIT)
    elif idx == 11:
        return Model11(parameters.INPUT_SIZE, parameters.OUTPUT_SIZE, parameters.LOSS, parameters.FOCAL_WEIGHT_INIT)
    elif idx == 12:
        return Model12(parameters.INPUT_SIZE, parameters.OUTPUT_SIZE, parameters.LOSS, parameters.FOCAL_WEIGHT_INIT)
    elif idx == 13:
        return Model13(parameters.INPUT_SIZE, parameters.OUTPUT_SIZE, parameters.LOSS, parameters.FOCAL_WEIGHT_INIT)
    elif idx == 14:
        return Model14(parameters.INPUT_SIZE, parameters.OUTPUT_SIZE, parameters.LOSS, parameters.FOCAL_WEIGHT_INIT)
    elif idx == 15:
        return Model15(parameters.INPUT_SIZE, parameters.OUTPUT_SIZE, parameters.LOSS, parameters.FOCAL_WEIGHT_INIT)
    elif idx == 16:
        return Model16(parameters.INPUT_SIZE, parameters.OUTPUT_SIZE, parameters.LOSS, parameters.FOCAL_WEIGHT_INIT)
    elif idx == 17:
        return Model17(parameters.INPUT_SIZE, parameters.OUTPUT_SIZE, parameters.LOSS, parameters.FOCAL_WEIGHT_INIT)

"""
Basically what Brendan was doing
"""
class Model0(nn.Module):
    def __init__(self, input_size, output_size, loss="CE", weight_init=0.01):
        super(Model0, self).__init__()

        self.input_dim = input_size
        self.hidden_size = 128
        self.output_size = output_size

        self.hidden_state = nn.Parameter(torch.rand(1, 1, self.hidden_size), requires_grad=True).to(device)
        self.cell_state = nn.Parameter(torch.rand(1, 1, self.hidden_size), requires_grad=True).to(device)

        self.lstm = nn.LSTM(input_size, self.hidden_size, batch_first=True)
        self.hiddenToClass = nn.Linear(self.hidden_size, self.output_size)

        if loss.lower() == "focal":
            self.hiddenToClass.weight.data.fill_(-np.log((1 - weight_init) / weight_init))

    def forward(self, inputs):
        # input shape - (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(inputs, [self.hidden_state.repeat(1, inputs.shape[0], 1), 
                                         self.cell_state.repeat(1, inputs.shape[0], 1)])
        logits = self.hiddenToClass(lstm_out)
        return logits

"""
Now with a conv1d flavor
"""
class Model1(nn.Module):
    def __init__(self, input_size, output_size, loss="CE", weight_init=0.01):
        super(Model1, self).__init__()

        self.input_size = input_size
        self.hidden_size = 128
        self.num_filters = 25
        self.kernel_size = 5
        self.num_layers = 1
        self.padding = 2
        self.output_size = output_size
        self.conv_out = input_size - self.kernel_size + 2 * self.padding + 1 # Ouput of the feature vectors after the 1D conv
        self.lstm_input = self.conv_out * self.num_filters

        self.hidden_state = nn.Parameter(torch.rand(self.num_layers, 1, self.hidden_size), requires_grad=True).to(device)
        self.cell_state = nn.Parameter(torch.rand(self.num_layers, 1, self.hidden_size), requires_grad=True).to(device)

        # I think that we want input size to be 1
        self.convLayer = nn.Conv1d(1, self.num_filters, self.kernel_size, padding=self.padding) # keep same dimension
        self.lstm = nn.LSTM(self.lstm_input, self.hidden_size, self.num_layers, batch_first=True) # added 2 layer lstm capabilities
        self.hiddenToClass = nn.Linear(self.hidden_size, self.output_size)

        if loss.lower() == "focal":
            self.hiddenToClass.weight.data.fill_(-np.log((1 - weight_init) / weight_init))


    def forward(self, inputs):
        # input shape - (batch, seq_len, input_size)

        # Reshape the input before passing to the 1d
        reshaped_inputs = inputs.view(-1, 1, self.input_size)
        convFeatures = self.convLayer(reshaped_inputs)
        convFeatures = convFeatures.view(inputs.shape[0], inputs.shape[1], -1)
        lstm_out, _ = self.lstm(convFeatures, [self.hidden_state.repeat(1, inputs.shape[0], 1), 
                                                 self.cell_state.repeat(1, inputs.shape[0], 1)])
        logits = self.hiddenToClass(lstm_out)
        return logits

"""
With maxpool as well
"""
class Model2(nn.Module):
    def __init__(self, input_size, output_size, loss="CE", weight_init=0.01):
        super(Model2, self).__init__()

        self.input_size = input_size
        self.hidden_size = 128
        self.num_filters = input_size
        self.kernel_size = 19 # Roughly 77 / 4 
        self.num_layers = 2
        self.output_size = output_size
        self.conv_out = input_size - self.kernel_size + 1 # Ouput of the feature vectors after the 1D conv

        self.hidden_state = nn.Parameter(torch.rand(self.num_layers, 1, self.hidden_size), requires_grad=True).to(device)
        self.cell_state = nn.Parameter(torch.rand(self.num_layers, 1, self.hidden_size), requires_grad=True).to(device)

        # I think that we want input size to be 1
        self.convLayer = nn.Conv1d(1, self.num_filters, self.kernel_size) # Don't use padding
        self.maxpool = nn.MaxPool1d(self.conv_out) # Perform a max pool over the resulting 1d freq. conv.
        self.lstm = nn.LSTM(self.num_filters, self.hidden_size, self.num_layers, batch_first=True) 
        self.hiddenToClass = nn.Linear(self.hidden_size, self.output_size)

        if loss.lower() == "focal":
            self.hiddenToClass.weight.data.fill_(-np.log((1 - weight_init) / weight_init))

    def forward(self, inputs):
        # input shape - (batch, seq_len, input_size)

        # Reshape the input before passing to the 1d
        reshaped_inputs = inputs.view(-1, 1, self.input_size)
        convFeatures = self.convLayer(reshaped_inputs)
        pooledFeatures = self.maxpool(convFeatures)

        # Re-Shape to be - (batch, seq_len, num_filters)
        pooledFeatures = pooledFeatures.view(-1, inputs.shape[1], self.num_filters)

        lstm_out, _ = self.lstm(pooledFeatures, [self.hidden_state.repeat(1, inputs.shape[0], 1), 
                                                 self.cell_state.repeat(1, inputs.shape[0], 1)])
        logits = self.hiddenToClass(lstm_out)
        return logits

"""
CONV1D_BiLSTM_Maxpool
"""
class Model3(nn.Module):
    def __init__(self, input_size, output_size, loss="CE", weight_init=0.01):
        super(Model3, self).__init__()

        self.input_size = input_size
        self.hidden_size = 128
        self.num_filters = 25
        self.kernel_size = 5
        self.num_layers = 1
        self.output_size = output_size

        self.hidden_state = nn.Parameter(torch.rand(2*self.num_layers, 1, self.hidden_size), requires_grad=True).to(device) # allow for bi-direct
        self.cell_state = nn.Parameter(torch.rand(2*self.num_layers, 1, self.hidden_size), requires_grad=True).to(device)

        # I think that we want input size to be 1
        self.convLayer = nn.Conv1d(1, self.num_filters, self.kernel_size, padding=2) # keep same dimension
        self.maxpool = nn.MaxPool1d(self.input_size) # Perform a max pool over the resulting 1d freq. conv.
        self.lstm = nn.LSTM(self.num_filters, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)
        self.hiddenToClass = nn.Linear(self.hidden_size*2, self.output_size)

        if loss.lower() == "focal":
            self.hiddenToClass.weight.data.fill_(-np.log((1 - weight_init) / weight_init))

    def forward(self, inputs):
        # input shape - (batch, seq_len, input_size)

        # Reshape the input before passing to the 1d
        reshaped_inputs = inputs.view(-1, 1, self.input_size)
        convFeatures = self.convLayer(reshaped_inputs)
        pooledFeatures = self.maxpool(convFeatures)

        # Re-Shape to be - (batch, seq_len, num_filters)
        pooledFeatures = pooledFeatures.view(-1, inputs.shape[1], self.num_filters)

        lstm_out, _ = self.lstm(pooledFeatures, [self.hidden_state.repeat(1, inputs.shape[0], 1), 
                                                 self.cell_state.repeat(1, inputs.shape[0], 1)])
        logits = self.hiddenToClass(lstm_out)
        return logits

"""
CONV1D_BiLSTM_NO_POOL
"""
class Model4(nn.Module):
    def __init__(self, input_size, output_size, loss="CE", weight_init=0.01):
        super(Model4, self).__init__()

        self.input_size = input_size
        self.hidden_size = 128
        self.num_filters = 25
        self.kernel_size = 5
        self.num_layers = 1
        self.padding = 2
        self.output_size = output_size
        self.conv_out = input_size - self.kernel_size + 2*self.padding + 1 # Ouput of the feature vectors after the 1D conv
        self.lstm_input = self.conv_out * self.num_filters

        self.hidden_state = nn.Parameter(torch.rand(2*self.num_layers, 1, self.hidden_size), requires_grad=True).to(device) # allow for bi-direct
        self.cell_state = nn.Parameter(torch.rand(2*self.num_layers, 1, self.hidden_size), requires_grad=True).to(device)

        # I think that we want input size to be 1
        self.convLayer = nn.Conv1d(1, self.num_filters, self.kernel_size, padding=self.padding) # keep same dimension
        self.maxpool = nn.MaxPool1d(self.input_size) # Perform a max pool over the resulting 1d freq. conv.
        self.lstm = nn.LSTM(self.lstm_input, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)
        self.hiddenToClass = nn.Linear(self.hidden_size*2, self.output_size)

        if loss.lower() == "focal":
            self.hiddenToClass.weight.data.fill_(-np.log((1 - weight_init) / weight_init))

    def forward(self, inputs):
        # input shape - (batch, seq_len, input_size)

        # Reshape the input before passing to the 1d

        reshaped_inputs = inputs.view(-1, 1, self.input_size)
        convFeatures = self.convLayer(reshaped_inputs)
        convFeatures = convFeatures.view(inputs.shape[0], inputs.shape[1], -1)
        lstm_out, _ = self.lstm(convFeatures, [self.hidden_state.repeat(1, inputs.shape[0], 1), 
                                                 self.cell_state.repeat(1, inputs.shape[0], 1)])

        logits = self.hiddenToClass(lstm_out)
        return logits

"""
Adding a hidden layer to beginning of model0
"""
class Model5(nn.Module):
    def __init__(self, input_size, output_size, loss="CE", weight_init=0.01):
        super(Model5, self).__init__()

        self.input_dim = input_size
        self.hidden_size = 128
        self.output_size = output_size

        self.hidden_state = nn.Parameter(torch.rand(1, 1, self.hidden_size), requires_grad=True).to(device)
        self.cell_state = nn.Parameter(torch.rand(1, 1, self.hidden_size), requires_grad=True).to(device)

        self.linear = nn.Linear(input_size, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)
        self.hiddenToClass = nn.Linear(self.hidden_size, self.output_size)

        if loss.lower() == "focal":
            self.hiddenToClass.weight.data.fill_(-np.log((1 - weight_init) / weight_init))

    def forward(self, inputs):
        # input shape - (batch, seq_len, input_size)
        out = self.linear(inputs)
        lstm_out, _ = self.lstm(out, [self.hidden_state.repeat(1, inputs.shape[0], 1), 
                                         self.cell_state.repeat(1, inputs.shape[0], 1)])
        logits = self.hiddenToClass(lstm_out)
        return logits

"""
Adding two hidden layers to beginning of model0
"""
class Model6(nn.Module):
    def __init__(self, input_size, output_size, loss="CE", weight_init=0.01):
        super(Model6, self).__init__()

        self.input_dim = input_size
        self.hidden_size = 128
        self.output_size = output_size

        self.hidden_state = nn.Parameter(torch.rand(1, 1, self.hidden_size), requires_grad=True).to(device)
        self.cell_state = nn.Parameter(torch.rand(1, 1, self.hidden_size), requires_grad=True).to(device)

        self.linear = nn.Linear(input_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)
        self.hiddenToClass = nn.Linear(self.hidden_size, self.output_size)

        if loss.lower() == "focal":
            self.hiddenToClass.weight.data.fill_(-np.log((1 - weight_init) / weight_init))

    def forward(self, inputs):
        # input shape - (batch, seq_len, input_size)
        out = self.linear(inputs)
        out = self.linear2(out)
        lstm_out, _ = self.lstm(out, [self.hidden_state.repeat(1, inputs.shape[0], 1), 
                                         self.cell_state.repeat(1, inputs.shape[0], 1)])
        logits = self.hiddenToClass(lstm_out)
        return logits

"""
With linear layer pool after conv1d as well
"""
class Model7(nn.Module):
    def __init__(self, input_size, output_size, loss="CE", weight_init=0.01):
        super(Model7, self).__init__()

        self.input_size = input_size
        self.hidden_size = 128
        self.num_filters = input_size
        self.kernel_size = 19 # Roughly 77 / 4 
        self.num_layers = 2
        self.output_size = output_size
        self.conv_out = input_size - self.kernel_size + 1 # Ouput of the feature vectors after the 1D conv

        self.hidden_state = nn.Parameter(torch.rand(self.num_layers, 1, self.hidden_size), requires_grad=True).to(device)
        self.cell_state = nn.Parameter(torch.rand(self.num_layers, 1, self.hidden_size), requires_grad=True).to(device)

        # I think that we want input size to be 1
        self.convLayer = nn.Conv1d(1, self.num_filters, self.kernel_size) # Don't use padding
        self.linear_pool = nn.Linear(self.conv_out, 1) 
        self.lstm = nn.LSTM(self.num_filters, self.hidden_size, self.num_layers, batch_first=True) 
        self.hiddenToClass = nn.Linear(self.hidden_size, self.output_size)

        if loss.lower() == "focal":
            self.hiddenToClass.weight.data.fill_(-np.log((1 - weight_init) / weight_init))

    def forward(self, inputs):
        # input shape - (batch, seq_len, input_size)

        # Reshape the input before passing to the 1d
        reshaped_inputs = inputs.view(-1, 1, self.input_size)
        convFeatures = self.convLayer(reshaped_inputs)
        pooledFeatures = self.linear_pool(convFeatures)

        # Re-Shape to be - (batch, seq_len, num_filters)
        pooledFeatures = pooledFeatures.view(-1, inputs.shape[1], self.num_filters)

        lstm_out, _ = self.lstm(pooledFeatures, [self.hidden_state.repeat(1, inputs.shape[0], 1), 
                                                 self.cell_state.repeat(1, inputs.shape[0], 1)])
        logits = self.hiddenToClass(lstm_out)
        return logits

"""
With avg-pool after conv1d as well
"""
class Model8(nn.Module):
    def __init__(self, input_size, output_size, loss="CE", weight_init=0.01):
        super(Model8, self).__init__()

        self.input_size = input_size
        self.hidden_size = 128
        self.num_filters = input_size
        self.kernel_size = 19 # Roughly 77 / 4 
        self.num_layers = 2
        self.output_size = output_size
        self.conv_out = input_size - self.kernel_size + 1 # Ouput of the feature vectors after the 1D conv

        self.hidden_state = nn.Parameter(torch.rand(self.num_layers, 1, self.hidden_size), requires_grad=True).to(device)
        self.cell_state = nn.Parameter(torch.rand(self.num_layers, 1, self.hidden_size), requires_grad=True).to(device)

        # I think that we want input size to be 1
        self.convLayer = nn.Conv1d(1, self.num_filters, self.kernel_size) # Don't use padding
        self.avgpool = nn.AvgPool1d(self.conv_out) # Perform a max pool over the resulting 1d freq. conv. 
        self.lstm = nn.LSTM(self.num_filters, self.hidden_size, self.num_layers, batch_first=True) 
        self.hiddenToClass = nn.Linear(self.hidden_size, self.output_size)

        if loss.lower() == "focal":
            self.hiddenToClass.weight.data.fill_(-np.log((1 - weight_init) / weight_init))

    def forward(self, inputs):
        # input shape - (batch, seq_len, input_size)

        # Reshape the input before passing to the 1d
        reshaped_inputs = inputs.view(-1, 1, self.input_size)
        convFeatures = self.convLayer(reshaped_inputs)
        pooledFeatures = self.avgpool(convFeatures)

        # Re-Shape to be - (batch, seq_len, num_filters)
        pooledFeatures = pooledFeatures.view(-1, inputs.shape[1], self.num_filters)

        lstm_out, _ = self.lstm(pooledFeatures, [self.hidden_state.repeat(1, inputs.shape[0], 1), 
                                                 self.cell_state.repeat(1, inputs.shape[0], 1)])
        logits = self.hiddenToClass(lstm_out)
        return logits

"""
Model 4 but with a batchnorm first
"""
class Model9(nn.Module):
    def __init__(self, input_size, output_size, loss="CE", weight_init=0.01):
        super(Model9, self).__init__()

        self.input_size = input_size
        self.hidden_size = 128
        self.num_filters = 25
        self.kernel_size = 5
        self.num_layers = 1
        self.padding = 2
        self.output_size = output_size
        self.conv_out = input_size - self.kernel_size + 2*self.padding + 1 # Ouput of the feature vectors after the 1D conv
        self.lstm_input = self.conv_out * self.num_filters

        self.hidden_state = nn.Parameter(torch.rand(2*self.num_layers, 1, self.hidden_size), requires_grad=True).to(device) # allow for bi-direct
        self.cell_state = nn.Parameter(torch.rand(2*self.num_layers, 1, self.hidden_size), requires_grad=True).to(device)

        # I think that we want input size to be 1
        self.batchnorm = nn.BatchNorm1d(self.input_size)
        self.convLayer = nn.Conv1d(1, self.num_filters, self.kernel_size, padding=self.padding) # keep same dimension
        self.maxpool = nn.MaxPool1d(self.input_size) # Perform a max pool over the resulting 1d freq. conv.
        self.lstm = nn.LSTM(self.lstm_input, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)
        self.hiddenToClass = nn.Linear(self.hidden_size*2, self.output_size)

        if loss.lower() == "focal":
            self.hiddenToClass.weight.data.fill_(-np.log((1 - weight_init) / weight_init))

    def forward(self, inputs):
        # input shape - (batch, seq_len, input_size)

        # Reshape the input before passing to the 1d
        batch_norm_inputs = self.batchnorm(inputs.view(-1, self.input_size))

        reshaped_inputs = batch_norm_inputs.view(-1, 1, self.input_size)
        convFeatures = self.convLayer(reshaped_inputs)
        convFeatures = convFeatures.view(inputs.shape[0], inputs.shape[1], -1)
        lstm_out, _ = self.lstm(convFeatures, [self.hidden_state.repeat(1, inputs.shape[0], 1), 
                                                 self.cell_state.repeat(1, inputs.shape[0], 1)])

        logits = self.hiddenToClass(lstm_out)
        return logits

"""
Model0 but with batchnorm
"""
class Model10(nn.Module):
    def __init__(self, input_size, output_size, loss="CE", weight_init=0.01):
        super(Model10, self).__init__()

        self.input_size = input_size
        self.hidden_size = 128
        self.output_size = output_size

        self.hidden_state = nn.Parameter(torch.rand(1, 1, self.hidden_size), requires_grad=True).to(device)
        self.cell_state = nn.Parameter(torch.rand(1, 1, self.hidden_size), requires_grad=True).to(device)

        self.batchnorm = nn.BatchNorm1d(self.input_size)
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, batch_first=True)
        self.hiddenToClass = nn.Linear(self.hidden_size, self.output_size)

        if loss.lower() == "focal":
            self.hiddenToClass.weight.data.fill_(-np.log((1 - weight_init) / weight_init))

    def forward(self, inputs):
        # input shape - (batch, seq_len, input_size)
        batch_norm_inputs = self.batchnorm(inputs.view(-1, self.input_size))
        batch_norm_inputs = batch_norm_inputs.view(inputs.shape)
        lstm_out, _ = self.lstm(batch_norm_inputs, [self.hidden_state.repeat(1, inputs.shape[0], 1), 
                                         self.cell_state.repeat(1, inputs.shape[0], 1)])
        logits = self.hiddenToClass(lstm_out)
        return logits

"""
Model1 with batchnorm
"""
class Model11(nn.Module):
    def __init__(self, input_size, output_size, loss="CE", weight_init=0.01):
        super(Model11, self).__init__()

        self.input_size = input_size
        self.hidden_size = 128
        self.num_filters = 25
        self.kernel_size = 5
        self.num_layers = 1
        self.padding = 2
        self.output_size = output_size
        self.conv_out = input_size - self.kernel_size + 2 * self.padding + 1 # Ouput of the feature vectors after the 1D conv
        self.lstm_input = self.conv_out * self.num_filters

        self.hidden_state = nn.Parameter(torch.rand(self.num_layers, 1, self.hidden_size), requires_grad=True).to(device)
        self.cell_state = nn.Parameter(torch.rand(self.num_layers, 1, self.hidden_size), requires_grad=True).to(device)

        # I think that we want input size to be 1
        self.batchnorm = nn.BatchNorm1d(self.input_size)
        self.convLayer = nn.Conv1d(1, self.num_filters, self.kernel_size, padding=self.padding) # keep same dimension
        self.lstm = nn.LSTM(self.lstm_input, self.hidden_size, self.num_layers, batch_first=True) # added 2 layer lstm capabilities
        self.hiddenToClass = nn.Linear(self.hidden_size, self.output_size)

        if loss.lower() == "focal":
            self.hiddenToClass.weight.data.fill_(-np.log((1 - weight_init) / weight_init))

    def forward(self, inputs):
        # input shape - (batch, seq_len, input_size)

        batch_norm_inputs = self.batchnorm(inputs.view(-1, self.input_size))
        batch_norm_inputs = batch_norm_inputs.view(inputs.shape)
        # Reshape the input before passing to the 1d
        reshaped_inputs = batch_norm_inputs.view(-1, 1, self.input_size)
        convFeatures = self.convLayer(reshaped_inputs)
        convFeatures = convFeatures.view(inputs.shape[0], inputs.shape[1], -1)
        lstm_out, _ = self.lstm(convFeatures, [self.hidden_state.repeat(1, inputs.shape[0], 1), 
                                                 self.cell_state.repeat(1, inputs.shape[0], 1)])
        logits = self.hiddenToClass(lstm_out)
        return logits

"""
Basically what Brendan was doing
"""
class Model12(nn.Module):
    def __init__(self, input_size, output_size, loss="CE", weight_init=0.01):
        super(Model12, self).__init__()

        self.input_size = input_size
        self.hidden_size = 128
        self.output_size = output_size

        self.hidden_state = nn.Parameter(torch.rand(1, 1, self.hidden_size), requires_grad=True).to(device)
        self.cell_state = nn.Parameter(torch.rand(1, 1, self.hidden_size), requires_grad=True).to(device)

        self.batchnorm = nn.BatchNorm1d(self.input_size)
        self.linear = nn.Linear(self.input_size, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.hiddenToClass = nn.Linear(self.hidden_size, self.output_size)

        if loss.lower() == "focal":
            self.hiddenToClass.weight.data.fill_(-np.log((1 - weight_init) / weight_init))

    def forward(self, inputs):
        # input shape - (batch, seq_len, input_size)
        batch_norm_inputs = self.batchnorm(inputs.view(-1, self.input_size)).view(inputs.shape)
        out = self.linear(batch_norm_inputs)
        lstm_out, _ = self.lstm(out, [self.hidden_state.repeat(1, inputs.shape[0], 1), 
                                         self.cell_state.repeat(1, inputs.shape[0], 1)])
        out = self.linear2(lstm_out)
        logits = self.hiddenToClass(out)
        return logits

"""
Now with a conv1d flavor
"""
class Model13(nn.Module):
    def __init__(self, input_size, output_size, loss="CE", weight_init=0.01):
        super(Model13, self).__init__()

        self.input_size = input_size
        self.hidden_size = 128
        self.num_filters = 128
        self.kernel_size = input_size
        self.num_layers = 1
        self.padding = 0
        self.output_size = output_size
        self.conv_out = input_size - self.kernel_size + 2 * self.padding + 1 # Ouput of the feature vectors after the 1D conv
        self.lstm_input = self.conv_out * self.num_filters

        self.hidden_state = nn.Parameter(torch.rand(self.num_layers, 1, self.hidden_size), requires_grad=True).to(device)
        self.cell_state = nn.Parameter(torch.rand(self.num_layers, 1, self.hidden_size), requires_grad=True).to(device)

        # I think that we want input size to be 1
        self.batchnorm = nn.BatchNorm1d(self.input_size)
        self.convLayer = nn.Conv1d(1, self.num_filters, self.kernel_size, padding=self.padding) # keep same dimension
        self.lstm = nn.LSTM(self.lstm_input, self.hidden_size, self.num_layers, batch_first=True) # added 2 layer lstm capabilities
        self.hiddenToClass = nn.Linear(self.hidden_size, self.output_size)

        if loss.lower() == "focal":
            self.hiddenToClass.weight.data.fill_(-np.log((1 - weight_init) / weight_init))

    def forward(self, inputs):
        # input shape - (batch, seq_len, input_size)

        # Reshape the input before passing to the 1d
        batch_norm_inputs = self.batchnorm(inputs.view(-1, self.input_size)).view(inputs.shape)
        reshaped_inputs = inputs.view(-1, 1, self.input_size)
        convFeatures = self.convLayer(reshaped_inputs)
        convFeatures = convFeatures.view(inputs.shape[0], inputs.shape[1], -1)
        lstm_out, _ = self.lstm(convFeatures, [self.hidden_state.repeat(1, inputs.shape[0], 1), 
                                                 self.cell_state.repeat(1, inputs.shape[0], 1)])
        logits = self.hiddenToClass(lstm_out)
        return logits

"""
Basically what Brendan was doing
"""
class Model14(nn.Module):
    def __init__(self, input_size, output_size, loss="CE", weight_init=0.01):
        super(Model14, self).__init__()

        self.input_size = input_size
        self.lin_size = 64
        self.hidden_size = 128
        self.output_size = output_size

        self.hidden_state = nn.Parameter(torch.rand(1, 1, self.hidden_size), requires_grad=True).to(device)
        self.cell_state = nn.Parameter(torch.rand(1, 1, self.hidden_size), requires_grad=True).to(device)

        self.batchnorm = nn.BatchNorm1d(self.input_size)
        self.linear = nn.Linear(self.input_size, self.lin_size)
        self.linear2 = nn.Linear(self.lin_size, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)
        self.linear3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear4 = nn.Linear(self.hidden_size, self.lin_size)
        self.hiddenToClass = nn.Linear(self.lin_size, self.output_size)

        if loss.lower() == "focal":
            self.hiddenToClass.weight.data.fill_(-np.log((1 - weight_init) / weight_init))

    def forward(self, inputs):
        # input shape - (batch, seq_len, input_size)
        batch_norm_inputs = self.batchnorm(inputs.view(-1, self.input_size)).view(inputs.shape)
        out = self.linear(batch_norm_inputs)
        out = nn.ReLU()(out)
        out = self.linear2(out)
        lstm_out, _ = self.lstm(out, [self.hidden_state.repeat(1, inputs.shape[0], 1), 
                                         self.cell_state.repeat(1, inputs.shape[0], 1)])
        out = self.linear3(lstm_out)
        out = nn.ReLU()(out)
        out = self.linear4(out)
        logits = self.hiddenToClass(out)
        return logits

"""
Try new convolutional model (not based on 1D convolutions)
"""
class Model15(nn.Module):
    def __init__(self, input_size, output_size, loss="CE", weight_init=0.01):
        super(Model15, self).__init__()

        self.input_size = input_size

        self.pool_sizes = [5, 2, 2]
        self.filter_size = [(5, 5), (5, 5), (5, 5)]
        self.num_filters = [128, 128, 128]

        self.hidden_size = 128
        self.output_size = output_size

        # Make the conv layers
        cnn_layers = []
        in_channels = 1
        for i in range(len(self.pool_sizes)):
            # We should set the padding later as another hyper param!
            conv2d = nn.Conv2d(in_channels, self.num_filters[i], kernel_size=self.filter_size[i], padding=2)
            # Gotta figure out the shapes here so skip the batch norm for now
            #layers += [conv2d, nn.BatchNorm1d(self.)]
            #layers +=  [conv2d, nn.ReLU(inplace=True)]

            # Now we need to do the max pooling!

            in_channels = self.num_filters[i]


        self.hidden_state = nn.Parameter(torch.rand(1, 1, self.hidden_size), requires_grad=True).to(device)
        self.cell_state = nn.Parameter(torch.rand(1, 1, self.hidden_size), requires_grad=True).to(device)

        self.batchnorm = nn.BatchNorm1d(self.input_size)
        self.linear = nn.Linear(self.input_size, self.lin_size)
        self.linear2 = nn.Linear(self.lin_size, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)
        self.linear3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear4 = nn.Linear(self.hidden_size, self.lin_size)
        self.hiddenToClass = nn.Linear(self.lin_size, self.output_size)

        if loss.lower() == "focal":
            self.hiddenToClass.weight.data.fill_(-np.log((1 - weight_init) / weight_init))


    def make_layers(cfg, batch_norm=False):
        layers = []
        in_channels = 1 # The input spectogram is always 
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, inputs):
        # input shape - (batch, seq_len, input_size)
        batch_norm_inputs = self.batchnorm(inputs.view(-1, self.input_size)).view(inputs.shape)
        out = self.linear(batch_norm_inputs)
        out = self.linear2(out)
        lstm_out, _ = self.lstm(out, [self.hidden_state.repeat(1, inputs.shape[0], 1), 
                                         self.cell_state.repeat(1, inputs.shape[0], 1)])
        out = self.linear3(lstm_out)
        out = self.linear4(out)
        logits = self.hiddenToClas


"""
Go bigger with lstm
"""
class Model16(nn.Module):
    def __init__(self, input_size, output_size, loss="CE", weight_init=0.01):
        super(Model16, self).__init__()

        self.input_size = input_size
        self.lin_size = 64
        self.hidden_size = 128
        self.num_layers = 2 # lstm
        self.output_size = output_size

        self.hidden_state = nn.Parameter(torch.rand(2 * self.num_layers, 1, self.hidden_size), requires_grad=True).to(device)
        self.cell_state = nn.Parameter(torch.rand(2 * self.num_layers, 1, self.hidden_size), requires_grad=True).to(device)

        self.batchnorm = nn.BatchNorm1d(self.input_size)
        self.linear = nn.Linear(self.input_size, self.lin_size)
        self.linear2 = nn.Linear(self.lin_size, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=True)
        self.linear3 = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.linear4 = nn.Linear(self.hidden_size, self.lin_size)
        self.hiddenToClass = nn.Linear(self.lin_size, self.output_size)

        if loss.lower() == "focal":
            self.hiddenToClass.weight.data.fill_(-np.log((1 - weight_init) / weight_init))

    def forward(self, inputs):
        # input shape - (batch, seq_len, input_size)
        batch_norm_inputs = self.batchnorm(inputs.view(-1, self.input_size)).view(inputs.shape)
        out = self.linear(batch_norm_inputs)
        out = nn.ReLU()(out)
        out = self.linear2(out)
        lstm_out, _ = self.lstm(out, [self.hidden_state.repeat(1, inputs.shape[0], 1), 
                                         self.cell_state.repeat(1, inputs.shape[0], 1)])
        out = self.linear3(lstm_out)
        out = nn.ReLU()(out)
        out = self.linear4(out)
        logits = self.hiddenToClass(out)
        return logits


class Model17(nn.Module):
    def __init__(self, input_size, output_size, loss="CE", weight_init=0.01):
        super(Model17, self).__init__()

        self.input_size = input_size

        self.model = models.resnet18()
        self.model.fc = nn.Sequential(
           nn.Linear(512, 128),
           nn.ReLU(inplace=True),
           nn.Linear(128, 256)) # This is hard coded to the size of the training windows

        if loss.lower() == "focal":
            self.model.fc[2].weight.data.fill_(-np.log((1 - weight_init) / weight_init))


    def forward(self, inputs):
        inputs = inputs.unsqueeze(1)
        inputs = inputs.repeat(1, 3, 1, 1)
        out = self.model(inputs)
        return out


# For the focal loss we just want to initialize the bias of the final layer to be
# log((1-pi)/pi) where pi = 0.01 is good
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
        # Calculate the standard BCE loss and then 
        # re-weight it by the term (a_t (1 - p_t)^gamma)
        # Let us see how the weighting term would actually apply here later!
        # Let us compare how this works an added alpha term
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        # Get the actual values of pt = e ^ (log(pt)) from bce loss where we have -log(pt)
        pt = torch.exp(-bce_loss)
        #print (pt[targets==1])
        #print(targets)
        # Value of alpha for class 1 and 1 - alpha for class 0
        alpha = torch.tensor([1 - self.alpha, self.alpha]).to(device)
        # Select the appropriate alpha based on label y
        alpha_t = alpha[targets.data.view(-1).long()].view_as(targets)
        
        focal_loss = alpha_t * (1 - pt)**self.gamma * bce_loss

        if self.reduce:
            return torch.mean(focal_loss)

        return focal_loss



##### END OF MODELS

def num_non_zero(logits, labels, threshold=0.5):
    sig = nn.Sigmoid()
    with torch.no_grad():
        pred = sig(logits)
        binary_preds = pred > threshold
        # Cast to proper type!
        binary_preds = binary_preds.float()
        num_non_zero = binary_preds.sum()

    return num_non_zero

def num_correct(logits, labels, threshold=0.5):
    sig = nn.Sigmoid()
    with torch.no_grad():
        pred = sig(logits)
        binary_preds = pred > threshold
        # Cast to proper type!
        binary_preds = binary_preds.float()
        num_correct = (binary_preds == labels).sum()

    return num_correct


def get_f_score(logits, labels, threshold=0.5):
    sig = nn.Sigmoid()
    with torch.no_grad():
        pred = sig(logits)
        binary_preds = pred > threshold
        # Cast to proper type!
        binary_preds = binary_preds.float()
        f_score = sklearn.metrics.f1_score(labels.data.cpu().numpy(), binary_preds.data.cpu().numpy())

    return f_score


def train_model(dataloders, model, criterion, optimizer, scheduler, writer, num_epochs):
    since = time.time()

    dataset_sizes = {'train': len(dataloders['train'].dataset), 
                     'valid': len(dataloders['valid'].dataset)}

    best_valid_acc = 0.0
    best_valid_fscore = 0.0
    best_model_wts = None

    try:
        for epoch in range(num_epochs):
            for phase in ['train', 'valid']:
                if phase == 'train':
                    model.train(True)
                else:
                    model.train(False)

                running_loss = 0.0
                running_corrects = 0
                running_samples = 0
                running_non_zero = 0
                running_fscore = 0.0
                iterations = 0

                running_trig_word_recall = 0.0
                running_trig_word_precision = 0.0
                running_trig_word_count = 0

                i = 0
                print ("Num batches:", len(dataloders[phase]))
                for inputs, labels, _ in dataloders[phase]:
                    i += 1
                    if (i % 10 == 0):
                        print ("Batch number {} of {}".format(i, len(dataloders[phase])))
                    # Cast the variables to the correct type
                    inputs = inputs.float()
                    
                    labels = labels.float()

                    inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))

                    optimizer.zero_grad()

                    # Forward pass
                    logits = model(inputs) # Shape - (batch_size, seq_len, 1)
                    # Are we zeroing out the hidden state in the model??

                    # Flatten it for criterion and num_correct
                    logits = logits.view(-1, 1)
                    labels = labels.view(-1, 1)

                    logits = logits.squeeze()
                    labels = labels.squeeze()

                    loss = criterion(logits, labels)

                    # Backward pass
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item()
                    running_corrects += num_correct(logits, labels)
                    running_non_zero += num_non_zero(logits, labels)
                    running_samples += logits.shape[0]
                    running_fscore += get_f_score(logits, labels)
                    iterations += 1

                    # Bad style sorry don't care its late
                    # output = nn.Sigmoid()(logits)
                    # output = np.where(output.cpu().detach().numpy() > 0.5, 1, 0)
                    # running_trig_word_recall += Metrics.trigger_word_accuracy(output, labels)
                    # running_trig_word_precision += Metrics.trigger_word_accuracy(labels, output)
                    # running_trig_word_count += 1
                
                if phase == 'train':
                    train_epoch_loss = running_loss / iterations
                    train_epoch_acc = float(running_corrects) / running_samples
                    train_non_zero = running_non_zero
                else:
                    valid_epoch_loss = running_loss / iterations
                    valid_epoch_acc = float(running_corrects) / running_samples
                    valid_epoch_fscore = running_fscore / iterations
                    valid_epoch_trig_recall = 0
                    valid_epoch_trig_prec = 0
                    valid_non_zero = running_non_zero
                    
                if phase == 'valid' and valid_epoch_acc > best_valid_acc:
                    best_valid_acc = valid_epoch_acc
                    best_valid_fscore = valid_epoch_fscore
                    best_model_wts = model.state_dict()

            print ('Epoch [{}/{}] Train Non-Zero: {} Val Non-Zero: {}'.format(epoch, num_epochs - 1, train_non_zero, valid_non_zero))
            print('Epoch [{}/{}] Training loss: {:.6f} acc: {:.4f} ' 
                  'Validation loss: {:.6f} acc: {:.4f} f-score: {:.4f} trig recall: {:.4f} trig precision: {:.4f} time: {:.4f}'.format(
                    epoch, num_epochs - 1,
                    train_epoch_loss, train_epoch_acc, 
                    valid_epoch_loss, valid_epoch_acc, 
                    valid_epoch_fscore, valid_epoch_trig_recall, valid_epoch_trig_prec, (time.time()-since)/60))

            ## Write important metrics to tensorboard
            writer.add_scalar('train_epoch_loss', train_epoch_loss, epoch)
            writer.add_scalar('train_epoch_acc', train_epoch_acc, epoch)
            writer.add_scalar('valid_epoch_loss', valid_epoch_loss, epoch)
            writer.add_scalar('valid_epoch_acc', valid_epoch_acc, epoch)
            writer.add_scalar('learning_rate', scheduler.get_lr(), epoch)

            scheduler.step()

    except KeyboardInterrupt:
        print("Early stopping due to keyboard intervention")

    print('Best val Acc: {:4f}'.format(best_valid_acc))
    print('Best val F-score: {:4f}'.format(best_valid_fscore))
    return best_model_wts

def adversarial_discovery(dataloader, model, threshold=0.5, min_length=0):
    """
        Data loop for self supervised adversarial negative sample discovery. 
        Run the data through the model, and for negative samples (i.e.) that 
        have no elephant calls in them, flag those in which we identify a false
        positive call. For each data chunk, run through the model to get the model
        predictions. To check for a false positive example that we want to add
        to the training loop see if we make a prediction in a chunk without any
        ground truth calls. Note, this will identify necessary chunks because
        the false positives in chunks with elephant calls in them will already
        exist in the dataset.
    """
    # Note there may be edge cases where an adversarial example exists right
    # near an elephant call and is not included in the training dataset because
    # of the way that the chunks are created for training. i.e. the chunks in 
    # the training dataset may not have included the adversarial examples, but
    # when creating chunks for the 24hrs the chunks may be aligned differently
    adversarial_examples = []
    # This dataset includes chunks from the full 24 hours
    for inputs, labels, data_files in dataloader:
        inputs = inputs.float()
                    
        labels = labels.cpu().detach().numpy()

        inputs = Variable(inputs.to(device))

        # Forward pass
        logits = model(inputs) # Shape - (batch_size, seq_len, 1)
        predictions = torch.sigmoid(logits).cpu().detach().numpy()

        # Now for each chunk we want to see whether it should be flagged as 
        # a hard negative sample. Look over number in batch.
        # Pre-compute the number of pos. slices in each chunk
        gt_counts = np.sum(labels, axis=1) # Shape - (batch_size)
        # Threshold the predictions - May add guassian blur
        binary_preds = np.where(predictions > threshold, 1, 0)
        pred_counts = np.sum(binary_preds, axis=1).squeeze() # Shape - (batch_size)
        for example in range(gt_counts.shape[0]):
            # Flag chunks with false pos in empy chunks.
            if gt_counts[example] == 0 and pred_counts[example] > min_length:
                print ("found an adversarial examples")
                adversarial_examples.append(data_files[example])
                # visualize it
                if VERBOSE:
                    features = inputs[example].detach().numpy()
                    output = predictions[example]
                    label = labels[example]

                    visualize(features, output, label)


    return adversarial_examples


def calc_num_chunks_calls(data_loader):
    num_chunks_calls = 0
    num_chunks_total = 0

    for inputs, labels, data_files in data_loader:

        num_chunks_total += inputs.shape[0]

        # Calculate how many of the chunks have actual elephant calls
        labels = labels.cpu().detach().numpy()
        gt_counts = np.sum(labels, axis=1)
        gt_counts = np.where(gt_counts > 0, 1, 0)
        num_chunks_calls += np.sum(gt_counts)

    print ("Number of chunks with calls:", num_chunks_calls)
    print ("Total chunks:", num_chunks_total)
    print ("Ratio (call chunk) / chunks seen:", float(num_chunks_calls) / float(num_chunks_total))


def main():
    ## Build Dataset
    # "/home/jgs8/ElephantCallAI/elephant_dataset/Train_nouab/Neg_Samples_x" + str(parameters.NEG_SAMPLES) + "/"

    train_loader = get_loader("../elephant_dataset/Train/Full_24_hrs", parameters.BATCH_SIZE, parameters.NORM, parameters.SCALE)
    # Quatro
    #train_loader = get_loader("/home/data/elephants/processed_data/Train_nouab/Full_24_hrs/", parameters.BATCH_SIZE, parameters.NORM, parameters.SCALE)
    # The validation loader should be the full 24hr trainind data use for adversarial discovery
    #validation_loader 
    test_loader = get_loader("../elephant_dataset/Test/Neg_Samples_x" + str(parameters.NEG_SAMPLES) + "/", parameters.BATCH_SIZE, parameters.NORM, parameters.SCALE)
    # Quatro
    #test_loader = get_loader("/home/data/elephants/processed_data/Test_nouab/Neg_Samples_x" + str(parameters.NEG_SAMPLES) + "/", parameters.BATCH_SIZE, parameters.NORM, parameters.SCALE)
    # Quatro
    #train_loader = get_loader("/tmp/jgs8_data/Train/Neg_Samples_x2/", parameters.BATCH_SIZE, parameters.NORM, parameters.SCALE)
    #validation_loader = get_loader("/tmp/jgs8_data/Test/Neg_Samples_x2/", parameters.BATCH_SIZE, parameters.NORM, parameters.SCALE)

    dloaders = {'train':train_loader, 'valid':test_loader}

    if sys.argv[1].lower() == 'help':
        print ("Run types are as follows:")
        print ("1) model.py visualize model_path - for visualizing pre trained model predictions")
        print ("2) model.py adversarial model_path - for testing a pre trained models adversarial discovery")
        print ("3) model.py model_id - train a given model")
    elif len(sys.argv) > 1 and sys.argv[1]  == 'visualize':
        ## Data Visualization
        # model = torch.load(parameters.MODEL_SAVE_PATH + parameters.DATASET + '_model_' + sys.argv[2] + ".pt", map_location=device)
        model = torch.load(sys.argv[2], map_location=device)
        print(model)

        for inputs, labels in dloaders['valid']:
            inputs = inputs.float()
            labels = labels.float()

            inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))

            # Forward pass
            outputs = model(inputs) # Shape - (batch_size, seq_len, 1)

            print('Accuracy on the test set for this batch is {:4f}'.format(float(num_correct(outputs.view(-1, 1), labels.view(-1, 1))) / outputs.view(-1, 1).shape[0]))

            for i in range(len(inputs)):
                features = inputs[i].detach().numpy()
                output = torch.sigmoid(outputs[i]).detach().numpy()
                label = labels[i].detach().numpy()

                visualize(features, output, label)

    elif len(sys.argv) > 1 and sys.argv[1] == 'adversarial':
        # Load a model that was already trained and run through adversarial 
        # discovery
        model = torch.load(sys.argv[2], map_location=device)

        adversarial_examples = adversarial_discovery(test_loader, model, threshold=0.5, min_length=0)
        print (adversarial_examples)
        print (len(adversarial_examples))

    else:
        ## Training
        model_id = int(sys.argv[1])

        save_path = parameters.SAVE_PATH + parameters.DATASET + '_model_' + str(model_id) + "_" + parameters.NORM + "_Negx" + str(parameters.NEG_SAMPLES) + "_Loss_" + parameters.LOSS + "_" + str(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))

        model = get_model(model_id)

        model.to(device)

        print(model)

        writer = SummaryWriter(save_path)
        writer.add_scalar('batch_size', parameters.BATCH_SIZE)
        writer.add_scalar('weight_decay', parameters.HYPERPARAMETERS[model_id]['l2_reg'])

        #criterion = torch.nn.BCEWithLogitsLoss()
        # Try focal loss
        if parameters.LOSS.upper() == "CE":
            criterion = torch.nn.BCEWithLogitsLoss()
            print ("Using Binary Cross Entropy Loss")
        elif parameters.LOSS.upper() == "FOCAL":
            criterion = FocalLoss(alpha=parameters.FOCAL_ALPHA, gamma=parameters.FOCAL_GAMMA)
            print ("Using Focal Loss with parameters alpha: {}, gamma: {}, pi: {}".format(parameters.FOCAL_ALPHA, parameters.FOCAL_GAMMA, parameters.FOCAL_WEIGHT_INIT))
        else:
            print("Unknown Loss")
            return

        optimizer = torch.optim.Adam(model.parameters(), lr=parameters.HYPERPARAMETERS[model_id]['lr'], weight_decay=parameters.HYPERPARAMETERS[model_id]['l2_reg'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, parameters.HYPERPARAMETERS[model_id]['lr_decay_step'], gamma=parameters.HYPERPARAMETERS[model_id]['lr_decay'])

        start_time = time.time()
        model_wts = None

        model_wts = train_model(dloaders, model, criterion, optimizer, scheduler, writer, parameters.NUM_EPOCHS)

        if model_wts:
            model.load_state_dict(model_wts)
            save_path = save_path + "/" + "model.pt"
            if not os.path.exists(parameters.SAVE_PATH):
                os.makedirs(parameters.SAVE_PATH)
            torch.save(model, save_path)
            print('Saved best val acc model to path {}'.format(save_path))
        else:
            print('For some reason I don\'t have a model to save')

        print('Training time: {:10f} minutes'.format((time.time()-start_time)/60))

        writer.close()

if __name__ == '__main__':
    main()
