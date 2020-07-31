import numpy as np
import torch
import torch.nn as nn
#from torch import optim
#from torch.autograd import Variable
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.metrics import precision_recall_curve
from data import get_loader, get_loader_fuzzy
from torchsummary import summary
import time
from tensorboardX import SummaryWriter
#import sklearn
#import sys
#import copy
import os
import torch.nn.functional as F

import parameters
from visualization import visualize
import torchvision.models as models
import pdb
#from torchvision import transforms
#import matplotlib
#import matplotlib.pyplot as plt
#from collections import deque

from utils import set_seed


def get_model(model_id):
    # Make sure to set the numpy and cuda seeds
    # for the model
    print ("Model Seed:", parameters.MODEL_SEED)
    set_seed(parameters.MODEL_SEED)

    if model_id == 0:
        return Model0(parameters.INPUT_SIZE, parameters.OUTPUT_SIZE, parameters.LOSS, parameters.FOCAL_WEIGHT_INIT)
    elif model_id == 1:
        return Model1(parameters.INPUT_SIZE, parameters.OUTPUT_SIZE, parameters.LOSS, parameters.FOCAL_WEIGHT_INIT)
    elif model_id == 2:
        return Model2(parameters.INPUT_SIZE, parameters.OUTPUT_SIZE, parameters.LOSS, parameters.FOCAL_WEIGHT_INIT)
    elif model_id == 3:
        return Model3(parameters.INPUT_SIZE, parameters.OUTPUT_SIZE, parameters.LOSS, parameters.FOCAL_WEIGHT_INIT)
    elif model_id == 4:
        return Model4(parameters.INPUT_SIZE, parameters.OUTPUT_SIZE, parameters.LOSS, parameters.FOCAL_WEIGHT_INIT)
    elif model_id == 5:
        return Model5(parameters.INPUT_SIZE, parameters.OUTPUT_SIZE, parameters.LOSS, parameters.FOCAL_WEIGHT_INIT)
    elif model_id == 6:
        return Model6(parameters.INPUT_SIZE, parameters.OUTPUT_SIZE, parameters.LOSS, parameters.FOCAL_WEIGHT_INIT)
    elif model_id == 7:
        return Model7(parameters.INPUT_SIZE, parameters.OUTPUT_SIZE, parameters.LOSS, parameters.FOCAL_WEIGHT_INIT)
    elif model_id == 8:
        return Model8(parameters.INPUT_SIZE, parameters.OUTPUT_SIZE, parameters.LOSS, parameters.FOCAL_WEIGHT_INIT)
    elif model_id == 9:
        return Model9(parameters.INPUT_SIZE, parameters.OUTPUT_SIZE, parameters.LOSS, parameters.FOCAL_WEIGHT_INIT)
    elif model_id == 10:
        return Model10(parameters.INPUT_SIZE, parameters.OUTPUT_SIZE, parameters.LOSS, parameters.FOCAL_WEIGHT_INIT)
    elif model_id == 11:
        return Model11(parameters.INPUT_SIZE, parameters.OUTPUT_SIZE, parameters.LOSS, parameters.FOCAL_WEIGHT_INIT)
    elif model_id == 12:
        return Model12(parameters.INPUT_SIZE, parameters.OUTPUT_SIZE, parameters.LOSS, parameters.FOCAL_WEIGHT_INIT)
    elif model_id == 13:
        return Model13(parameters.INPUT_SIZE, parameters.OUTPUT_SIZE, parameters.LOSS, parameters.FOCAL_WEIGHT_INIT)
    elif model_id == 14:
        return Model14(parameters.INPUT_SIZE, parameters.OUTPUT_SIZE, parameters.LOSS, parameters.FOCAL_WEIGHT_INIT)
    elif model_id == 15:
        return Model15(parameters.INPUT_SIZE, parameters.OUTPUT_SIZE, parameters.LOSS, parameters.FOCAL_WEIGHT_INIT)
    elif model_id == 16:
        return Model16(parameters.INPUT_SIZE, parameters.OUTPUT_SIZE, parameters.LOSS, parameters.FOCAL_WEIGHT_INIT)
    elif model_id == 17:
        return Model17(parameters.INPUT_SIZE, parameters.OUTPUT_SIZE, parameters.LOSS, parameters.FOCAL_WEIGHT_INIT)
    elif model_id == 18:
        return Model18(parameters.INPUT_SIZE, parameters.OUTPUT_SIZE, parameters.LOSS, parameters.FOCAL_WEIGHT_INIT)
    elif model_id == 19:
        return Model19(parameters.INPUT_SIZE, parameters.OUTPUT_SIZE, parameters.LOSS, parameters.FOCAL_WEIGHT_INIT)
    elif model_id == 20:
        return Model20(parameters.INPUT_SIZE, parameters.OUTPUT_SIZE, parameters.LOSS, parameters.FOCAL_WEIGHT_INIT)
    elif model_id == 21:
        return Model21(parameters.INPUT_SIZE, parameters.OUTPUT_SIZE, parameters.LOSS, parameters.FOCAL_WEIGHT_INIT)
    elif model_id == 22:
        return Model22(parameters.INPUT_SIZE, parameters.OUTPUT_SIZE, parameters.LOSS, parameters.FOCAL_WEIGHT_INIT)
    elif model_id == 23:
        return Model23(parameters.INPUT_SIZE, parameters.OUTPUT_SIZE, parameters.LOSS, parameters.FOCAL_WEIGHT_INIT)

"""
Basically what Brendan was doing
"""
class Model0(nn.Module):
    def __init__(self, input_size, output_size, loss="CE", weight_init=0.01):
        super(Model0, self).__init__()

        self.input_dim = input_size
        self.hidden_size = 128
        self.output_size = output_size

        self.hidden_state = nn.Parameter(torch.rand(1, 1, self.hidden_size), requires_grad=True).to(parameters.device)
        self.cell_state = nn.Parameter(torch.rand(1, 1, self.hidden_size), requires_grad=True).to(parameters.device)

        self.lstm = nn.LSTM(input_size, self.hidden_size, batch_first=True)
        self.hiddenToClass = nn.Linear(self.hidden_size, self.output_size)

        if loss.lower() == "focal":  # NEED to think about this later! Not on top of priority!
            self.hiddenToClass.bias.data.fill_(-np.log10((1 - weight_init) / weight_init))

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

        # Allow us to learn the initialization of these
        self.hidden_state = nn.Parameter(torch.rand(self.num_layers, 1, self.hidden_size), requires_grad=True).to(parameters.device)
        self.cell_state = nn.Parameter(torch.rand(self.num_layers, 1, self.hidden_size), requires_grad=True).to(parameters.device)

        # I think that we want input size to be 1
        self.convLayer = nn.Conv1d(1, self.num_filters, self.kernel_size, padding=self.padding) # keep same dimension
        self.lstm = nn.LSTM(self.lstm_input, self.hidden_size, self.num_layers, batch_first=True) # added 2 layer lstm capabilities
        self.hiddenToClass = nn.Linear(self.hidden_size, self.output_size)

        if loss.lower() == "focal":
            self.hiddenToClass.bias.data.fill_(-np.log10((1 - weight_init) / weight_init))


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

        self.hidden_state = nn.Parameter(torch.rand(self.num_layers, 1, self.hidden_size), requires_grad=True).to(parameters.device)
        self.cell_state = nn.Parameter(torch.rand(self.num_layers, 1, self.hidden_size), requires_grad=True).to(parameters.device)

        # I think that we want input size to be 1
        self.convLayer = nn.Conv1d(1, self.num_filters, self.kernel_size) # Don't use padding
        self.maxpool = nn.MaxPool1d(self.conv_out) # Perform a max pool over the resulting 1d freq. conv.
        self.lstm = nn.LSTM(self.num_filters, self.hidden_size, self.num_layers, batch_first=True) 
        self.hiddenToClass = nn.Linear(self.hidden_size, self.output_size)

        if loss.lower() == "focal":
            self.hiddenToClass.bias.data.fill_(-np.log10((1 - weight_init) / weight_init))

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

        self.hidden_state = nn.Parameter(torch.rand(2*self.num_layers, 1, self.hidden_size), requires_grad=True).to(parameters.device) # allow for bi-direct
        self.cell_state = nn.Parameter(torch.rand(2*self.num_layers, 1, self.hidden_size), requires_grad=True).to(parameters.device)

        # I think that we want input size to be 1
        self.convLayer = nn.Conv1d(1, self.num_filters, self.kernel_size, padding=2) # keep same dimension
        self.maxpool = nn.MaxPool1d(self.input_size) # Perform a max pool over the resulting 1d freq. conv.
        self.lstm = nn.LSTM(self.num_filters, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)
        self.hiddenToClass = nn.Linear(self.hidden_size*2, self.output_size)

        if loss.lower() == "focal":
            self.hiddenToClass.bias.data.fill_(-np.log10((1 - weight_init) / weight_init))

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

        self.hidden_state = nn.Parameter(torch.rand(2*self.num_layers, 1, self.hidden_size), requires_grad=True).to(parameters.device) # allow for bi-direct
        self.cell_state = nn.Parameter(torch.rand(2*self.num_layers, 1, self.hidden_size), requires_grad=True).to(parameters.device)

        # I think that we want input size to be 1
        self.convLayer = nn.Conv1d(1, self.num_filters, self.kernel_size, padding=self.padding) # keep same dimension
        self.maxpool = nn.MaxPool1d(self.input_size) # Perform a max pool over the resulting 1d freq. conv.
        self.lstm = nn.LSTM(self.lstm_input, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)
        self.hiddenToClass = nn.Linear(self.hidden_size*2, self.output_size)

        if loss.lower() == "focal":
            self.hiddenToClass.bias.data.fill_(-np.log10((1 - weight_init) / weight_init))

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

        self.hidden_state = nn.Parameter(torch.rand(1, 1, self.hidden_size), requires_grad=True).to(parameters.device)
        self.cell_state = nn.Parameter(torch.rand(1, 1, self.hidden_size), requires_grad=True).to(parameters.device)

        self.linear = nn.Linear(input_size, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)
        self.hiddenToClass = nn.Linear(self.hidden_size, self.output_size)

        if loss.lower() == "focal":
            self.hiddenToClass.bias.data.fill_(-np.log10((1 - weight_init) / weight_init))

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

        self.hidden_state = nn.Parameter(torch.rand(1, 1, self.hidden_size), requires_grad=True).to(parameters.device)
        self.cell_state = nn.Parameter(torch.rand(1, 1, self.hidden_size), requires_grad=True).to(parameters.device)

        self.linear = nn.Linear(input_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)
        self.hiddenToClass = nn.Linear(self.hidden_size, self.output_size)

        if loss.lower() == "focal":
            self.hiddenToClass.bias.data.fill_(-np.log10((1 - weight_init) / weight_init))

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

        self.hidden_state = nn.Parameter(torch.rand(self.num_layers, 1, self.hidden_size), requires_grad=True).to(parameters.device)
        self.cell_state = nn.Parameter(torch.rand(self.num_layers, 1, self.hidden_size), requires_grad=True).to(parameters.device)

        # I think that we want input size to be 1
        self.convLayer = nn.Conv1d(1, self.num_filters, self.kernel_size) # Don't use padding
        self.linear_pool = nn.Linear(self.conv_out, 1) 
        self.lstm = nn.LSTM(self.num_filters, self.hidden_size, self.num_layers, batch_first=True) 
        self.hiddenToClass = nn.Linear(self.hidden_size, self.output_size)

        if loss.lower() == "focal":
            self.hiddenToClass.bias.data.fill_(-np.log10((1 - weight_init) / weight_init))

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

        self.hidden_state = nn.Parameter(torch.rand(self.num_layers, 1, self.hidden_size), requires_grad=True).to(parameters.device)
        self.cell_state = nn.Parameter(torch.rand(self.num_layers, 1, self.hidden_size), requires_grad=True).to(parameters.device)

        # I think that we want input size to be 1
        self.convLayer = nn.Conv1d(1, self.num_filters, self.kernel_size) # Don't use padding
        self.avgpool = nn.AvgPool1d(self.conv_out) # Perform a max pool over the resulting 1d freq. conv. 
        self.lstm = nn.LSTM(self.num_filters, self.hidden_size, self.num_layers, batch_first=True) 
        self.hiddenToClass = nn.Linear(self.hidden_size, self.output_size)

        if loss.lower() == "focal":
            self.hiddenToClass.bias.data.fill_(-np.log10((1 - weight_init) / weight_init))

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

        self.hidden_state = nn.Parameter(torch.rand(2*self.num_layers, 1, self.hidden_size), requires_grad=True).to(parameters.device) # allow for bi-direct
        self.cell_state = nn.Parameter(torch.rand(2*self.num_layers, 1, self.hidden_size), requires_grad=True).to(parameters.device)

        # I think that we want input size to be 1
        self.batchnorm = nn.BatchNorm1d(self.input_size)
        self.convLayer = nn.Conv1d(1, self.num_filters, self.kernel_size, padding=self.padding) # keep same dimension
        self.maxpool = nn.MaxPool1d(self.input_size) # Perform a max pool over the resulting 1d freq. conv.
        self.lstm = nn.LSTM(self.lstm_input, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)
        self.hiddenToClass = nn.Linear(self.hidden_size*2, self.output_size)

        if loss.lower() == "focal":
            self.hiddenToClass.bias.data.fill_(-np.log10((1 - weight_init) / weight_init))

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

        self.hidden_state = nn.Parameter(torch.rand(1, 1, self.hidden_size), requires_grad=True).to(parameters.device)
        self.cell_state = nn.Parameter(torch.rand(1, 1, self.hidden_size), requires_grad=True).to(parameters.device)

        self.batchnorm = nn.BatchNorm1d(self.input_size)
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, batch_first=True)
        self.hiddenToClass = nn.Linear(self.hidden_size, self.output_size)

        if loss.lower() == "focal":
            self.hiddenToClass.bias.data.fill_(-np.log10((1 - weight_init) / weight_init))

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

        self.hidden_state = nn.Parameter(torch.rand(self.num_layers, 1, self.hidden_size), requires_grad=True).to(parameters.device)
        self.cell_state = nn.Parameter(torch.rand(self.num_layers, 1, self.hidden_size), requires_grad=True).to(parameters.device)

        # I think that we want input size to be 1
        self.batchnorm = nn.BatchNorm1d(self.input_size)
        self.convLayer = nn.Conv1d(1, self.num_filters, self.kernel_size, padding=self.padding) # keep same dimension
        self.lstm = nn.LSTM(self.lstm_input, self.hidden_size, self.num_layers, batch_first=True) # added 2 layer lstm capabilities
        self.hiddenToClass = nn.Linear(self.hidden_size, self.output_size)

        if loss.lower() == "focal":
            self.hiddenToClass.bias.data.fill_(-np.log10((1 - weight_init) / weight_init))

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

        self.hidden_state = nn.Parameter(torch.rand(1, 1, self.hidden_size), requires_grad=True).to(parameters.device)
        self.cell_state = nn.Parameter(torch.rand(1, 1, self.hidden_size), requires_grad=True).to(parameters.device)

        self.batchnorm = nn.BatchNorm1d(self.input_size)
        self.linear = nn.Linear(self.input_size, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.hiddenToClass = nn.Linear(self.hidden_size, self.output_size)

        if loss.lower() == "focal":
            self.hiddenToClass.bias.data.fill_(-np.log10((1 - weight_init) / weight_init))

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

        self.hidden_state = nn.Parameter(torch.rand(self.num_layers, 1, self.hidden_size), requires_grad=True).to(parameters.device)
        self.cell_state = nn.Parameter(torch.rand(self.num_layers, 1, self.hidden_size), requires_grad=True).to(parameters.device)

        # I think that we want input size to be 1
        self.batchnorm = nn.BatchNorm1d(self.input_size)
        self.convLayer = nn.Conv1d(1, self.num_filters, self.kernel_size, padding=self.padding) # keep same dimension
        self.lstm = nn.LSTM(self.lstm_input, self.hidden_size, self.num_layers, batch_first=True) # added 2 layer lstm capabilities
        self.hiddenToClass = nn.Linear(self.hidden_size, self.output_size)

        if loss.lower() == "focal":
            self.hiddenToClass.bias.data.fill_(-np.log10((1 - weight_init) / weight_init))

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

        self.hidden_state = nn.Parameter(torch.rand(1, 1, self.hidden_size), requires_grad=True).to(parameters.device)
        self.cell_state = nn.Parameter(torch.rand(1, 1, self.hidden_size), requires_grad=True).to(parameters.device)

        self.batchnorm = nn.BatchNorm1d(self.input_size)
        self.linear = nn.Linear(self.input_size, self.lin_size)
        self.linear2 = nn.Linear(self.lin_size, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)
        self.linear3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear4 = nn.Linear(self.hidden_size, self.lin_size)
        self.hiddenToClass = nn.Linear(self.lin_size, self.output_size)

        if loss.lower() == "focal":
            self.hiddenToClass.bias.data.fill_(-np.log10((1 - weight_init) / weight_init))

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
Try new convolutional model (not based on 1D convolutions)!!
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


        self.hidden_state = nn.Parameter(torch.rand(1, 1, self.hidden_size), requires_grad=True).to(parameters.device)
        self.cell_state = nn.Parameter(torch.rand(1, 1, self.hidden_size), requires_grad=True).to(parameters.device)

        self.batchnorm = nn.BatchNorm1d(self.input_size)
        self.linear = nn.Linear(self.input_size, self.lin_size)
        self.linear2 = nn.Linear(self.lin_size, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)
        self.linear3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear4 = nn.Linear(self.hidden_size, self.lin_size)
        self.hiddenToClass = nn.Linear(self.lin_size, self.output_size)

        if loss.lower() == "focal":
            self.hiddenToClass.bias.data.fill_(-np.log10((1 - weight_init) / weight_init))


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

        self.hidden_state = nn.Parameter(torch.rand(2 * self.num_layers, 1, self.hidden_size), requires_grad=True).to(parameters.device)
        self.cell_state = nn.Parameter(torch.rand(2 * self.num_layers, 1, self.hidden_size), requires_grad=True).to(parameters.device)

        self.batchnorm = nn.BatchNorm1d(self.input_size)
        self.linear = nn.Linear(self.input_size, self.lin_size)
        self.linear2 = nn.Linear(self.lin_size, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=True)
        self.linear3 = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.linear4 = nn.Linear(self.hidden_size, self.lin_size)
        self.hiddenToClass = nn.Linear(self.lin_size, self.output_size)

        if loss.lower() == "focal":
            print("USING FOCAL LOSS INITIALIZATION")
            self.hiddenToClass.bias.data.fill_(-np.log10((1 - weight_init) / weight_init))

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
ResNet-18
"""
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
            print("USING FOCAL LOSS INITIALIZATION")
            print ("Init:", -np.log10((1 - weight_init) / weight_init))
            self.model.fc[2].bias.data.fill_(-np.log10((1 - weight_init) / weight_init))


    def forward(self, inputs):
        inputs = inputs.unsqueeze(1)
        inputs = inputs.repeat(1, 3, 1, 1)
        out = self.model(inputs)
        return out


"""
ResNet-18 for entire window classification!
"""
# Consider a deeper resnet
class Model18(nn.Module):
    def __init__(self, input_size, output_size, loss="CE", weight_init=0.01):
        """
            Note output size is not actually used here!
        """
        super(Model18, self).__init__()

        self.input_size = input_size # Number of frequency bins

        self.model = models.resnet18()
        self.model.fc = nn.Sequential(
           nn.Linear(512, 128),
           nn.ReLU(inplace=True),
           nn.Linear(128, 1)) # Single window output!


        if loss.lower() == "focal":
            print("USING FOCAL LOSS INITIALIZATION")
            print ("Init:", -np.log10((1 - weight_init) / weight_init))
            self.model.fc[2].bias.data.fill_(-np.log10((1 - weight_init) / weight_init))
            #print (self.model.fc[2].bias)


    def forward(self, inputs):
        inputs = inputs.unsqueeze(1)
        inputs = inputs.repeat(1, 3, 1, 1)
        out = self.model(inputs)
        return out

"""
ResNet-50! 
"""
class Model19(nn.Module):
    def __init__(self, input_size, output_size, loss="CE", weight_init=0.01):
        """
            Note output size is not actually used here!
        """
        super(Model19, self).__init__()

        self.input_size = input_size # Number of frequency bins

        self.model = models.resnet50()
        # Need to figure out what the sizes will be here!!
        self.model.fc = nn.Sequential(
           nn.Linear(2048, 512),
           nn.ReLU(inplace=True),
           nn.Linear(512, 256)) # Single window output!


        if loss.lower() == "focal":
            print("USING FOCAL LOSS INITIALIZATION")
            print ("Init:", -np.log10((1 - weight_init) / weight_init))
            self.model.fc[2].bias.data.fill_(-np.log10((1 - weight_init) / weight_init))
            #print (self.model.fc[2].bias)


    def forward(self, inputs):
        inputs = inputs.unsqueeze(1)
        inputs = inputs.repeat(1, 3, 1, 1)
        out = self.model(inputs)
        return out


"""
Go bigger with lstm for window classification
"""
class Model20(nn.Module):
    def __init__(self, input_size, output_size, loss="CE", weight_init=0.01):
        super(Model20, self).__init__()

        self.input_size = input_size
        self.lin_size = 64
        self.hidden_size = 128
        self.num_layers = 2 # lstm
        self.output_size = output_size
        self.dropout_val = .4

        self.hidden_state = nn.Parameter(torch.rand(2 * self.num_layers, 1, self.hidden_size), requires_grad=True).to(parameters.device)
        self.cell_state = nn.Parameter(torch.rand(2 * self.num_layers, 1, self.hidden_size), requires_grad=True).to(parameters.device)

        self.batchnorm = nn.BatchNorm1d(self.input_size)
        self.linear = nn.Linear(self.input_size, self.lin_size)
        self.linear2 = nn.Linear(self.lin_size, self.hidden_size)
        # Consider some dropout??
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=self.num_layers, 
                                batch_first=True, bidirectional=True, dropout=self.dropout_val)
        
        self.linear3 = nn.Linear(self.hidden_size * 2, self.lin_size)
        #self.linear4 = nn.Linear(self.hidden_size, self.lin_size)
        self.hiddenToClass = nn.Linear(self.lin_size, self.output_size)

        # Create Dropout Layer that can be used on the
        # output of layers where we want to add dropout 
        self.dropout = nn.Dropout(self.dropout_val)

        if loss.lower() == "focal":
            print("USING FOCAL LOSS INITIALIZATION")
            self.hiddenToClass.bias.data.fill_(-np.log10((1 - weight_init) / weight_init))

    def forward(self, inputs):
        # input shape - (batch, seq_len, input_size)
        batch_norm_inputs = self.batchnorm(inputs.view(-1, self.input_size)).view(inputs.shape)
        out = self.linear(batch_norm_inputs)
        out = nn.ReLU()(out)
        out = self.linear2(out)
        out = nn.ReLU()(out)
        # Get the final hidden states (h_t for t = seq_len) for the forward and backwards directions!
        # Shape - [num_layers * num_directions, batch, hidden_size]
        _, (final_hiddens, final_cells) = self.lstm(out, [self.hidden_state.repeat(1, inputs.shape[0], 1), 
                                         self.cell_state.repeat(1, inputs.shape[0], 1)])

        # For classification, we take the last hidden units for the forward 
        # and backward directions, concatenate them and predict
        # Reshape to - [num_layers, directions, batch, hidden_size]
        final_hiddens = final_hiddens.view(self.num_layers, 2, final_hiddens.shape[1], self.hidden_size)
        linear_input = torch.cat((final_hiddens[-1, 0, :, :], final_hiddens[-1, 1, : :]), dim=1)

        # Let us see
        out = self.linear3(self.dropout(linear_input))
        out = nn.ReLU()(out)
        logits = self.hiddenToClass(out)
        
        return logits

# For segmentation!
class Model21(nn.Module):
    def __init__(self, input_size, output_size, loss="CE", weight_init=0.01,
                cnn_nb_filt=64, cnn_pool_size=[5, 2, 2]):#, filter_size=[3, 3, 3]):
        super(Model21, self).__init__()

        # In the future should have a map that maps kernal size
        # to padding size!
        self.kernal_size = [3, 3, 3]
        #self.padding = []
        self.lin_size = 32
        # For the GRU
        self.hidden_size = 32 # made it small maybe to avoid overfitting
        self.num_layers = 1 # GRU
        self.dropout_val = 0

        # Create the layers
        # For now just have 1 input chanell
        self.conv_layers = []
        feature_dim = input_size
        for layer in range(len(cnn_pool_size)):
            in_channels = cnn_nb_filt
            if layer == 0:
                # Maybe try 
                in_channels = 1

            # Keep dim the same!
            conv2d = nn.Conv2d(in_channels=in_channels, out_channels=cnn_nb_filt, kernel_size=3, padding=1, bias=False)
            # Should try either BN or LayerNorm
            #batchnorm = nn.BatchNorm2d()
            # Curious to try layer norm which should normalize just over each individual frequency
            layer_norm = nn.LayerNorm(feature_dim)
            # Max pool - consier doing just conv to downsample!
            max_pool = nn.MaxPool2d(kernel_size=[1, cnn_pool_size[layer]])
            feature_dim = feature_dim // cnn_pool_size[layer]

            self.conv_layers += [conv2d, layer_norm, nn.ReLU(inplace=True), max_pool, nn.Dropout(self.dropout_val)]

        self.conv_layers = nn.Sequential(*self.conv_layers)

        # This reflects our new cnn extracted feature dim
        feature_dim = cnn_nb_filt * feature_dim
        self.gru = nn.GRU(feature_dim, self.hidden_size, num_layers=self.num_layers, 
                                batch_first=True, bidirectional=True, dropout=self.dropout_val)
        # For bi-directional add *2
        self.linear_1 = nn.Linear(self.hidden_size * 2, self.lin_size)
        self.out = nn.Linear(self.lin_size, output_size)

    def forward(self, inputs):
        inputs = inputs.unsqueeze(1)
        # Shape - [batch, channels, seq_len, conv_features]
        conv_features = self.conv_layers(inputs) 

        # Stack all of the conv-layers to form new sequential
        # features - [batch, seq_len, channels * conv_features]
        conv_features = conv_features.permute(0, 2, 1, 3).contiguous()
        conv_features = conv_features.view(conv_features.shape[0], conv_features.shape[1], -1)

        # Feed through the LSTM
        # Get the final hidden states (h_t for t = seq_len) for the forward and backwards directions!
        # Shape - [num_layers * num_directions, batch, hidden_size]
        gru_out, _ = self.gru(conv_features)
        # Final fully connected layers
        linear_out = self.linear_1(gru_out)
        linear_out = nn.ReLU()(linear_out)
        logits = self.out(linear_out)

        return logits



# CRNN withough maxpool and directly downsample with convolution!
class Model22(nn.Module):
    def __init__(self, input_size, output_size, loss="CE", weight_init=0.01,
                cnn_nb_filt=64, compress_factors=[5, 2, 2]):#, filter_size=[3, 3, 3]):
        super(Model22, self).__init__()

        self.lin_size = 32
        # For the GRU
        self.hidden_size = 32 # made it small maybe to avoid overfitting
        self.num_layers = 1 # GRU
        self.dropout_val = 0

        # Create the layers
        # For now just have 1 input chanell
        self.conv_layers = []
        feature_dim = input_size
        for layer in range(len(compress_factors)):
            in_channels = cnn_nb_filt
            if layer == 0:
                # Maybe try 
                in_channels = 1

            # Use the stride to compress in the frequency dimension!
            conv2d = nn.Conv2d(in_channels=in_channels, out_channels=cnn_nb_filt, kernel_size=3, 
                                        stride=(1, compress_factors[layer]), padding=1, bias=False)
            # Should try either BN or LayerNorm
            #batchnorm = nn.BatchNorm2d()
            # Curious to try layer norm which should normalize just over each individual frequency
            # Compute as: (feature_dim + 2(padding) - (kernal_size - 1) - 1) / stride + 1
            feature_dim = int((feature_dim + 2 - 2 - 1) / compress_factors[layer] + 1)
            layer_norm = nn.LayerNorm(feature_dim)

            self.conv_layers += [conv2d, layer_norm, nn.ReLU(inplace=True), nn.Dropout(self.dropout_val)]

        self.conv_layers = nn.Sequential(*self.conv_layers)

        # This reflects our new cnn extracted feature dim
        feature_dim = cnn_nb_filt * feature_dim 
        # Use GRU to maybe control overfitting
        self.gru = nn.GRU(feature_dim, self.hidden_size, num_layers=self.num_layers, 
                                batch_first=True, bidirectional=True, dropout=self.dropout_val)
        # For bi-directional add *2
        self.linear_1 = nn.Linear(self.hidden_size * 2, self.lin_size)
        self.out = nn.Linear(self.lin_size, output_size)

    def forward(self, inputs):
        inputs = inputs.unsqueeze(1)
        # Shape - [batch, channels, seq_len, conv_features]
        conv_features = self.conv_layers(inputs) 

        # Stack all of the conv-layers to form new sequential
        # features - [batch, seq_len, channels * conv_features]
        conv_features = conv_features.permute(0, 2, 1, 3).contiguous()
        conv_features = conv_features.view(conv_features.shape[0], conv_features.shape[1], -1)

        # Feed through the LSTM
        # Get the final hidden states (h_t for t = seq_len) for the forward and backwards directions!
        # Shape - [num_layers * num_directions, batch, hidden_size]
        gru_out, _ = self.gru(conv_features)
        # Final fully connected layers
        linear_out = self.linear_1(gru_out)
        linear_out = nn.ReLU()(linear_out)
        logits = self.out(linear_out)

        return logits


# CRNN withough maxpool and using convolutional "blocks" to downsample
# the feature dimension. Question in res-net when they downsample in a
# block they use e.g. stride = 2 in conv1 not conv2. 
class Model23(nn.Module):
    def __init__(self, input_size, output_size, loss="CE", weight_init=0.01,
                cnn_nb_filt=64, compress_factors=[5, 2, 2]):#, filter_size=[3, 3, 3]):
        super(Model23, self).__init__()

        self.lin_size = 32
        # For the GRU
        self.hidden_size = 32 # made it small maybe to avoid overfitting
        self.num_layers = 1 # GRU
        self.dropout_val = 0

        # Create the layers
        # For now just have 1 input chanell
        self.conv_layers = []
        feature_dim = input_size
        for layer in range(len(compress_factors)):
            in_channels = cnn_nb_filt
            if layer == 0:
                # Maybe try 
                in_channels = 1

            # Use the stride to compress in the frequency dimension!
            conv1 = nn.Conv2d(in_channels=in_channels, out_channels=cnn_nb_filt, kernel_size=3, 
                                        stride=(1, compress_factors[layer]), padding=1, bias=False)
            conv2 = nn.Conv2d(cnn_nb_filt, cnn_nb_filt, kernel_size=3, padding=1, bias=False)
            # Should try either BN or LayerNorm
            #batchnorm = nn.BatchNorm2d()
            # Compute as: (feature_dim + 2(padding) - (kernal_size - 1) - 1) / stride + 1
            feature_dim = int((feature_dim + 2 - 2 - 1) / compress_factors[layer] + 1)
            layer_norm = nn.LayerNorm(feature_dim)

            self.conv_layers += [conv1, layer_norm, nn.ReLU(inplace=True), nn.Dropout(self.dropout_val),
                                conv2, layer_norm, nn.ReLU(inplace=True), nn.Dropout(self.dropout_val)]

        self.conv_layers = nn.Sequential(*self.conv_layers)

        # This reflects our new cnn extracted feature dim
        feature_dim = cnn_nb_filt * feature_dim 
        # Use GRU to maybe control overfitting
        self.gru = nn.GRU(feature_dim, self.hidden_size, num_layers=self.num_layers, 
                                batch_first=True, bidirectional=True, dropout=self.dropout_val)
        # For bi-directional add *2
        self.linear_1 = nn.Linear(self.hidden_size * 2, self.lin_size)
        self.out = nn.Linear(self.lin_size, output_size)

    def forward(self, inputs):
        inputs = inputs.unsqueeze(1)
        # Shape - [batch, channels, seq_len, conv_features]
        conv_features = self.conv_layers(inputs) 

        # Stack all of the conv-layers to form new sequential
        # features - [batch, seq_len, channels * conv_features]
        conv_features = conv_features.permute(0, 2, 1, 3).contiguous()
        conv_features = conv_features.view(conv_features.shape[0], conv_features.shape[1], -1)

        # Feed through the LSTM
        # Get the final hidden states (h_t for t = seq_len) for the forward and backwards directions!
        # Shape - [num_layers * num_directions, batch, hidden_size]
        gru_out, _ = self.gru(conv_features)
        # Final fully connected layers
        linear_out = self.linear_1(gru_out)
        linear_out = nn.ReLU()(linear_out)
        logits = self.out(linear_out)

        return logits

