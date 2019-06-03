"""
Also here is the command for an interactive shell:



srun -p aida -w c0020 -n2 --ntasks-per-core 1 --mem 40G --gres=gpu:1 --time=1800 --pty /bin/bash



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
from data import get_test_loader, get_train_valid_loader
from torchsummary import summary
import time
from tensorboardX import SummaryWriter
import sys

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.cm
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.ticker as plticker
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

RANDOM_SEED = 42

LEARNING_RATE = 1e-3
LEARNING_RATE_DECAY_STEP = 4
LEARNING_RATE_DECAY = 0.8
WEIGHT_DECAY = 1e-5

BATCH_SIZE = 32
NUM_EPOCHS = 1000
MODEL_SAVE_PATH = '../weights/model.pt'

np.random.seed(RANDOM_SEED)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()

        self.input_dim = input_size
        self.hidden_dim = hidden_size

        self.hidden_state = nn.Parameter(torch.rand(1, 1, self.hidden_dim), requires_grad=True).to(device)
        self.cell_state = nn.Parameter(torch.rand(1, 1, self.hidden_dim), requires_grad=True).to(device)

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.hiddenToClass = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        # input shape - (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(inputs, [self.hidden_state.repeat(1, inputs.shape[0], 1), 
                                         self.cell_state.repeat(1, inputs.shape[0], 1)])
        logits = self.hiddenToClass(lstm_out)
        return logits

class CONV1D_NO_POOL_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_filters=25, kernel_size=5, num_layers=1):
        super(CONV1D_NO_POOL_LSTM, self).__init__()

        self.num_filters = num_filters
        self.kernel_size = kernel_size

        self.hidden_dim = hidden_size
        self.input_size = input_size

        self.hidden_state = nn.Parameter(torch.rand(num_layers, 1, self.hidden_dim), requires_grad=True).to(device)
        self.cell_state = nn.Parameter(torch.rand(num_layers, 1, self.hidden_dim), requires_grad=True).to(device)

        # I think that we want input size to be 1
        self.convLayer = nn.Conv1d(1, num_filters, kernel_size, padding=2) # keep same dimension
        self.lstm = nn.LSTM(1925, hidden_size, num_layers, batch_first=True) # added 2 layer lstm capabilities
        self.hiddenToClass = nn.Linear(hidden_size, output_size)

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


class CONV1D_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_filters=25, kernel_size=5, num_layers=1):
        super(CONV1D_LSTM, self).__init__()

        self.num_filters = num_filters
        self.kernel_size = kernel_size

        self.hidden_dim = hidden_size
        self.input_size = input_size

        self.hidden_state = nn.Parameter(torch.rand(num_layers, 1, self.hidden_dim), requires_grad=True).to(device)
        self.cell_state = nn.Parameter(torch.rand(num_layers, 1, self.hidden_dim), requires_grad=True).to(device)

        # I think that we want input size to be 1
        self.convLayer = nn.Conv1d(1, num_filters, kernel_size, padding=2) # keep same dimension
        self.maxpool = nn.MaxPool1d(self.input_size) # Perform a max pool over the resulting 1d freq. conv.
        self.lstm = nn.LSTM(num_filters, hidden_size, num_layers, batch_first=True) # added 2 layer lstm capabilities
        self.hiddenToClass = nn.Linear(hidden_size, output_size)

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

class CONV1D_BiLSTM_Maxpool(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_filters=25, kernel_size=5, num_layers=1):
        super(CONV1D_BiLSTM_Maxpool, self).__init__()

        self.num_filters = num_filters # we should probably expand this!!
        self.kernel_size = kernel_size

        self.hidden_dim = hidden_size
        self.input_size = input_size

        self.hidden_state = nn.Parameter(torch.rand(2*num_layers, 1, self.hidden_dim), requires_grad=True).to(device) # allow for bi-direct
        self.cell_state = nn.Parameter(torch.rand(2*num_layers, 1, self.hidden_dim), requires_grad=True).to(device)

        # I think that we want input size to be 1
        self.convLayer = nn.Conv1d(1, num_filters, kernel_size, padding=2) # keep same dimension
        self.maxpool = nn.MaxPool1d(self.input_size) # Perform a max pool over the resulting 1d freq. conv.
        self.lstm = nn.LSTM(num_filters, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.hiddenToClass = nn.Linear(hidden_size*2, output_size)

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

class CONV1D_BiLSTM_NO_POOL(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_filters=25, kernel_size=5, num_layers=1):
        super(CONV1D_BiLSTM_NO_POOL, self).__init__()

        self.num_filters = num_filters # we should probably expand this!!
        self.kernel_size = kernel_size

        self.hidden_dim = hidden_size
        self.input_size = input_size

        self.hidden_state = nn.Parameter(torch.rand(2*num_layers, 1, self.hidden_dim), requires_grad=True).to(device) # allow for bi-direct
        self.cell_state = nn.Parameter(torch.rand(2*num_layers, 1, self.hidden_dim), requires_grad=True).to(device)

        # I think that we want input size to be 1
        self.convLayer = nn.Conv1d(1, num_filters, kernel_size, padding=2) # keep same dimension
        self.maxpool = nn.MaxPool1d(self.input_size) # Perform a max pool over the resulting 1d freq. conv.
        self.lstm = nn.LSTM(1925, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.hiddenToClass = nn.Linear(hidden_size*2, output_size)

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


def num_correct(logits, labels, threshold=0.5):
    sig = nn.Sigmoid()
    with torch.no_grad():
        pred = sig(logits)
        binary_preds = pred > threshold
        # Cast to proper type!
        binary_preds = binary_preds.float()
        num_correct = (binary_preds == labels).sum()

    return num_correct

def train_model(dataloders, model, criterion, optimizer, scheduler, writer, num_epochs, model_save_path):
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
                
                if phase == 'train':
                    train_epoch_loss = running_loss / running_samples
                    train_epoch_acc = float(running_corrects) / running_samples
                else:
                    valid_epoch_loss = running_loss / running_samples
                    valid_epoch_acc = float(running_corrects) / running_samples
                    
                if phase == 'valid' and valid_epoch_acc > best_valid_acc:
                    best_valid_acc = valid_epoch_acc
                    best_model_wts = model.state_dict()

            print('Epoch [{}/{}] train loss: {:.6f} acc: {:.4f} ' 
                  'valid loss: {:.6f} acc: {:.4f} time: {:.4f}'.format(
                    epoch, num_epochs - 1,
                    train_epoch_loss, train_epoch_acc, 
                    valid_epoch_loss, valid_epoch_acc, (time.time()-since)/60))

            ## Write important metrics to tensorboard
            writer.add_scalar('train_epoch_loss', train_epoch_loss, epoch)
            writer.add_scalar('train_epoch_acc', train_epoch_acc, epoch)
            writer.add_scalar('valid_epoch_loss', valid_epoch_loss, epoch)
            writer.add_scalar('valid_epoch_acc', valid_epoch_acc, epoch)
            writer.add_scalar('learning_rate', scheduler.get_lr(), epoch)

            scheduler.step()

    finally:
        if best_model_wts:
            torch.save(best_model_wts, model_save_path)
            print('Saved model from valid accuracy: ', best_valid_acc)
        else:
            print('For some reason I don\'t have a model to save')
    
    
    print('Best val Acc: {:4f}'.format(best_valid_acc))

    model.load_state_dict(best_model_wts)
    return model

def main():
    ## Build Dataset
    train_loader, validation_loader = get_train_valid_loader("../elephant_dataset/Train/Call_Label/",
                                                             BATCH_SIZE,
                                                             RANDOM_SEED)

    dloaders = {'train':train_loader, 'valid':validation_loader}

    ## Build Model
    input_size = 77 # Num of frequency bands in the spectogram
    hidden_size = 128

    # model = LSTM(input_size, hidden_size, 1)
    model = CONV1D_NO_POOL_LSTM(input_size, hidden_size, 1)
    # model = CONV1D_LSTM(input_size, hidden_size, 1)

    model.to(device)

    print(model)

    if len(sys.argv) > 1 and sys.argv[1]  == 'visualize':
        ## Data Visualization

        model.load_state_dict(torch.load(MODEL_SAVE_PATH))

        for inputs, labels in dloaders['valid']:
            inputs = inputs.float()
            labels = labels.float()

            inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))

            # Forward pass
            outputs = model(inputs) # Shape - (batch_size, seq_len, 1)

            for i in range(len(inputs)):
                features = inputs[i].detach().numpy()
                output = torch.sigmoid(outputs[i]).detach().numpy()
                label = labels[i].detach().numpy()

                fig, (ax1, ax2, ax3) = plt.subplots(3,1)
                # new_features = np.flipud(10*np.log10(features).T)
                new_features = features.T
                min_dbfs = new_features.flatten().mean()
                max_dbfs = new_features.flatten().mean()
                min_dbfs = np.maximum(new_features.flatten().min(),min_dbfs-2*new_features.flatten().std())
                max_dbfs = np.minimum(new_features.flatten().max(),max_dbfs+6*new_features.flatten().std())
                ax1.imshow(np.flipud(new_features), cmap="magma_r", vmin=min_dbfs, vmax=max_dbfs, interpolation='none', origin="lower", aspect="auto")
                ax2.plot(np.arange(output.shape[0]), output)
                ax3.plot(np.arange(label.shape[0]), label)
                plt.show()

# model = LSTM(input_size, hidden_size, 1)
model = CONV1D_NO_POOL_LSTM(input_size, hidden_size, 1)
# model = CONV1D_BiLSTM_NO_POOL(input_size, hidden_size, 1, num_layers=2)
# model = CONV1D_LSTM(input_size, hidden_size, 1)

    else:
        ## Training
        writer = SummaryWriter()
        writer.add_scalar('batch_size', BATCH_SIZE)

        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, LEARNING_RATE_DECAY_STEP, gamma=LEARNING_RATE_DECAY)

        start_time = time.time()
        model = train_model(dloaders, model, criterion, optimizer, scheduler, writer, NUM_EPOCHS, MODEL_SAVE_PATH)

        print('Training time: {:10f} minutes'.format((time.time()-start_time)/60))

        writer.close()

if __name__ == '__main__':
    main()
