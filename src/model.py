"""
Also here is the command for an interactive shell:



srun -p aida -w c0019 -n2 --ntasks-per-core 1 --mem 40G --gres=gpu:1 --time=1800 --pty /bin/bash



And here is a sample for a script to submit a job 



the command would be: sbatch your_script_name.sh



and your_script_name.sh would have this at the top:

#!/bin/bash
#SBATCH -p aida --time=3600 --gres=gpu:1 -n 8



python experiment_script.py 
"""

from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_curve
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
import Metrics

np.random.seed(parameters.RANDOM_SEED)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_model(idx):
    if idx == 0:
        return Model0(parameters.INPUT_SIZE, parameters.OUTPUT_SIZE)
    elif idx == 1:
        return Model1(parameters.INPUT_SIZE, parameters.OUTPUT_SIZE)
    elif idx == 2:
        return Model2(parameters.INPUT_SIZE, parameters.OUTPUT_SIZE)
    elif idx == 3:
        return Model3(parameters.INPUT_SIZE, parameters.OUTPUT_SIZE)
    elif idx == 4:
        return Model4(parameters.INPUT_SIZE, parameters.OUTPUT_SIZE)
    elif idx == 5:
        return Model5(parameters.INPUT_SIZE, parameters.OUTPUT_SIZE)
    elif idx == 6:
        return Model6(parameters.INPUT_SIZE, parameters.OUTPUT_SIZE)
    elif idx == 7:
        return Model7(parameters.INPUT_SIZE, parameters.OUTPUT_SIZE)
    elif idx == 8:
        return Model8(parameters.INPUT_SIZE, parameters.OUTPUT_SIZE)
    elif idx == 9:
        return Model9(parameters.INPUT_SIZE, parameters.OUTPUT_SIZE)
    elif idx == 10:
        return Model10(parameters.INPUT_SIZE, parameters.OUTPUT_SIZE)
    elif idx == 11:
        return Model11(parameters.INPUT_SIZE, parameters.OUTPUT_SIZE)

"""
Basically what Brendan was doing
"""
class Model0(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model0, self).__init__()

        self.input_dim = input_size
        self.hidden_size = 128
        self.output_size = output_size

        self.hidden_state = nn.Parameter(torch.rand(1, 1, self.hidden_size), requires_grad=True).to(device)
        self.cell_state = nn.Parameter(torch.rand(1, 1, self.hidden_size), requires_grad=True).to(device)

        self.lstm = nn.LSTM(input_size, self.hidden_size, batch_first=True)
        self.hiddenToClass = nn.Linear(self.hidden_size, self.output_size)

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
    def __init__(self, input_size, output_size):
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
    def __init__(self, input_size, output_size):
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
    def __init__(self, input_size, output_size):
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
    def __init__(self, input_size, output_size):
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
    def __init__(self, input_size, output_size):
        super(Model5, self).__init__()

        self.input_dim = input_size
        self.hidden_size = 128
        self.output_size = output_size

        self.hidden_state = nn.Parameter(torch.rand(1, 1, self.hidden_size), requires_grad=True).to(device)
        self.cell_state = nn.Parameter(torch.rand(1, 1, self.hidden_size), requires_grad=True).to(device)

        self.linear = nn.Linear(input_size, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)
        self.hiddenToClass = nn.Linear(self.hidden_size, self.output_size)

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
    def __init__(self, input_size, output_size):
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
    def __init__(self, input_size, output_size):
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
    def __init__(self, input_size, output_size):
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
    def __init__(self, input_size, output_size):
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
    def __init__(self, input_size, output_size):
        super(Model10, self).__init__()

        self.input_size = input_size
        self.hidden_size = 128
        self.output_size = output_size

        self.hidden_state = nn.Parameter(torch.rand(1, 1, self.hidden_size), requires_grad=True).to(device)
        self.cell_state = nn.Parameter(torch.rand(1, 1, self.hidden_size), requires_grad=True).to(device)

        self.batchnorm = nn.BatchNorm1d(self.input_size)
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, batch_first=True)
        self.hiddenToClass = nn.Linear(self.hidden_size, self.output_size)

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
    def __init__(self, input_size, output_size):
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


def num_correct(logits, labels, threshold=0.5):
    sig = nn.Sigmoid()
    with torch.no_grad():
        pred = sig(logits)
        binary_preds = pred > threshold
        # Cast to proper type!
        binary_preds = binary_preds.float()
        num_correct = (binary_preds == labels).sum()

    return num_correct

def train_model(dataloders, model, criterion, optimizer, scheduler, writer, num_epochs):
    since = time.time()

    dataset_sizes = {'train': len(dataloders['train'].dataset), 
                     'valid': len(dataloders['valid'].dataset)}

    best_valid_acc = 0.0
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

                running_trig_word_recall = 0.0
                running_trig_word_precision = 0.0
                running_trig_word_count = 0


                for inputs, labels in dataloders[phase]:
                    # Cast the variables to the correct type
                    inputs = inputs.float()
                    
                    labels = labels.float()

                    inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))

                    optimizer.zero_grad()

                    # Forward pass
                    logits = model(inputs) # Shape - (batch_size, seq_len, 1)

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
                    running_samples += logits.shape[0]

                    # Bad style sorry don't care its late
                    output = nn.Sigmoid()(logits)
                    output = np.where(output.cpu().detach().numpy() > 0.5, 1, 0)
                    running_trig_word_recall += Metrics.trigger_word_accuracy(output, labels)
                    running_trig_word_precision += Metrics.trigger_word_accuracy(labels, output)
                    running_trig_word_count += 1
                
                if phase == 'train':
                    train_epoch_loss = running_loss / running_samples
                    train_epoch_acc = float(running_corrects) / running_samples
                else:
                    valid_epoch_loss = running_loss / running_samples
                    valid_epoch_acc = float(running_corrects) / running_samples
                    valid_epoch_trig_recall = running_trig_word_recall / running_trig_word_count
                    valid_epoch_trig_prec = running_trig_word_precision / running_trig_word_count
                    
                if phase == 'valid' and valid_epoch_acc > best_valid_acc:
                    best_valid_acc = valid_epoch_acc
                    best_model_wts = model.state_dict()

            print('Epoch [{}/{}] train loss: {:.6f} acc: {:.4f} ' 
                  'valid loss: {:.6f} acc: {:.4f} trig recall: {:.4f} trig precision: {:.4f} time: {:.4f}'.format(
                    epoch, num_epochs - 1,
                    train_epoch_loss, train_epoch_acc, 
                    valid_epoch_loss, valid_epoch_acc, 
                    valid_epoch_trig_recall, valid_epoch_trig_prec, (time.time()-since)/60))

            ## Write important metrics to tensorboard
            writer.add_scalar('train_epoch_loss', train_epoch_loss, epoch)
            writer.add_scalar('train_epoch_acc', train_epoch_acc, epoch)
            writer.add_scalar('valid_epoch_loss', valid_epoch_loss, epoch)
            writer.add_scalar('valid_epoch_acc', valid_epoch_acc, epoch)
            writer.add_scalar('learning_rate', scheduler.get_lr(), epoch)

            scheduler.step()
    finally:
        print('Best val Acc: {:4f}'.format(best_valid_acc))
        return best_model_wts

def main():
    ## Build Dataset
    train_loader = get_loader("../elephant_dataset/Train/" + parameters.DATASET + '_Label/', parameters.BATCH_SIZE, parameters.NORM, parameters.SCALE)
    validation_loader = get_loader("../elephant_dataset/Test/" + parameters.DATASET + '_Label/', parameters.BATCH_SIZE, parameters.NORM, parameters.SCALE)


    dloaders = {'train':train_loader, 'valid':validation_loader}

    if len(sys.argv) > 1 and sys.argv[1]  == 'visualize':
        ## Data Visualization
        model = torch.load(parameters.MODEL_SAVE_PATH + parameters.DATASET + '_model_' + sys.argv[2] + ".pt", map_location=device)
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

                fig, (ax1, ax2, ax3) = plt.subplots(3,1)
                # new_features = np.flipud(10*np.log10(features).T)
                new_features = np.flipud(features.T)
                min_dbfs = new_features.flatten().mean()
                max_dbfs = new_features.flatten().mean()
                min_dbfs = np.maximum(new_features.flatten().min(),min_dbfs-2*new_features.flatten().std())
                max_dbfs = np.minimum(new_features.flatten().max(),max_dbfs+6*new_features.flatten().std())
                ax1.imshow(np.flipud(new_features), cmap="magma_r", vmin=min_dbfs, vmax=max_dbfs, interpolation='none', origin="lower", aspect="auto")
                ax2.plot(np.arange(output.shape[0]), output)
                ax3.plot(np.arange(label.shape[0]), label)
                plt.show()

    else:
        ## Training
        model_id = int(sys.argv[1])
        model = get_model(model_id)

        model.to(device)

        print(model)

        writer = SummaryWriter(parameters.LOGS_SAVE_PATH + parameters.DATASET + '_model_' + str(model_id) + "_" + parameters.NORM + "_" + str(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())))
        writer.add_scalar('batch_size', parameters.BATCH_SIZE)
        writer.add_scalar('weight_decay', parameters.HYPERPARAMETERS[model_id]['l2_reg'])

        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=parameters.HYPERPARAMETERS[model_id]['lr'], weight_decay=parameters.HYPERPARAMETERS[model_id]['l2_reg'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, parameters.HYPERPARAMETERS[model_id]['lr_decay_step'], gamma=parameters.HYPERPARAMETERS[model_id]['lr_decay'])

        start_time = time.time()
        model_wts = None
        try:
            model_wts = train_model(dloaders, model, criterion, optimizer, scheduler, writer, parameters.NUM_EPOCHS)
        finally:
            if model_wts:
                model.load_state_dict(model_wts)
                save_path = parameters.MODEL_SAVE_PATH + parameters.DATASET + '_model_' + str(model_id) + ".pt"
                torch.save(model, save_path)
                print('Saved best val acc model to path {}'.format(save_path))
            else:
                print('For some reason I don\'t have a model to save')

        print('Training time: {:10f} minutes'.format((time.time()-start_time)/60))

        writer.close()

if __name__ == '__main__':
    main()
