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
#from torchvision import transforms
#import matplotlib
#import matplotlib.pyplot as plt
#from collections import deque

def set_seed():
    """
        Set the seed across all different necessary platforms
        to allow for comparrsison of different model seeding. 
        Additionally, in the adversarial discovery, we want to
        initialize and train each of the models with the same
        seed.
    """
    torch.manual_seed(parameters.RANDOM_SEED)
    torch.cuda.manual_seed_all(parameters.RANDOM_SEED)
    # Not totally sure what these two do!
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(parameters.RANDOM_SEED)
    os.environ['PYTHONHASHSEED'] = str(parameters.RANDOM_SEED)


def get_model(model_id):
    # Make sure to set the numpy and cuda seeds
    # for the model
    set_seed()

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
ResNet-18 for entire chunk classification!
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
ResNet-50! for entire chunk classification!
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
           nn.Linear(512, 1)) # Single window output!


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




