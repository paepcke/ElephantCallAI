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
import time

RANDOM_SEED = 42

BATCH_SIZE = 1
VALIDATION_SPLIT = .2

NUM_EPOCHS = 10

np.random.seed(RANDOM_SEED)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_size
        self.batch_size = batch_size

        self.hidden = self.init_hidden()
        self.input_size = input_size

        self.lstm = nn.LSTM(input_size, hidden_size)
        self.hiddenToClass = nn.Linear(hidden_size, output_size)
        #self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.ones(1, BATCH_SIZE, self.hidden_dim),
                torch.ones(1, BATCH_SIZE, self.hidden_dim))


    def forward(self, inputs):
        # Think we need this here
        self.hidden = self.init_hidden()

        # Re-Shape the input to be - (seq_len, batch, input_size)
        inputs = inputs.view(-1, self.batch_size, self.input_size)
        lstm_out, self.hidden = self.lstm(inputs, self.hidden)
        logits = self.hiddenToClass(lstm_out)
        #pred = self.softmax(classVal)
        #pred = self.sigmoid(classVal)
        return logits

class CONV1D_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size, num_filters=25, kernel_size=5):
        super(CONV1D_LSTM, self).__init__()

        self.num_filters = num_filters
        self.kernel_size = kernel_size

        self.hidden_dim = hidden_size
        self.batch_size = batch_size
        self.input_size = input_size

        self.hidden = self.init_hidden()

        # I think that we want input size to be 1
        self.convLayer = nn.Conv1d(1, num_filters, kernel_size, padding=2) # keep same dimension
        self.maxpool = nn.MaxPool1d(self.input_size) # Perform a max pool over the resulting 1d freq. conv.
        self.lstm = nn.LSTM(num_filters, hidden_size)
        self.hiddenToClass = nn.Linear(hidden_size, output_size)
        #self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

    # TODO: We should figure out how to init hidden
    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.ones(1, BATCH_SIZE, self.hidden_dim),
                torch.ones(1, BATCH_SIZE, self.hidden_dim))

    def forward(self, inputs):
        # Think we need this here
        self.hidden = self.init_hidden()

        # Reshape the input before passing to the 1d
        inputs = inputs.view(-1, 1, self.input_size)
        #print (inputs.shape)
        convFeatures = self.convLayer(inputs)
        #print (convFeatures.shape)
        # TODO: Flatten here or Maxpool
        # Try MaxPool for examples
        pooledFeatures = self.maxpool(convFeatures)
        #print (pooledFeatures.shape)

        # Re-Shape to be - (seq_len, batch, num_filters)
        pooledFeatures = pooledFeatures.view(-1, self.batch_size, self.num_filters)
        #print (pooledFeatures.shape)
        lstm_out, self.hidden = self.lstm(pooledFeatures, self.hidden)
        logits = self.hiddenToClass(lstm_out)
        #pred = self.softmax(classVal)
        #pred = self.sigmoid(classVal)
        return logits


def num_correct(logits, labels, threshold=0.5):
    sig = nn.Sigmoid()
    with torch.no_grad():
        pred = sig(logits)
        binary_preds = pred > threshold
        print (binary_preds)
        # Cast to proper type!
        binary_preds = binary_preds.float()
        num_correct = (binary_preds == labels).sum()

    return num_correct

def train_model(dataloders, model, criterion, optimizer, num_epochs=25):
    since = time.time()
    use_gpu = torch.cuda.is_available()

    dataset_sizes = {'train': len(dataloders['train'].dataset), 
                     'valid': len(dataloders['valid'].dataset)}

    best_valid_acc = 0.0
    best_model_wts = None

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
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()

                # Forward pass
                # Definitely need to re-init hidden. Maybe do this in the forward pass
                logits = model(inputs) # Shape - (seq_len, 1, 1)

                logits = logits.squeeze()
                labels = labels.squeeze()
                # TODO: Perhaps need to check the outputs value here

                loss = criterion(logits, labels)

                # Backward pass
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()
                # Check this!!
                # Maybe we have to do like with no grad or something??
                running_corrects += num_correct(logits, labels)
                running_samples += logits.shape[0]
                #running_corrects += torch.sum(preds == labels.data) #TODO: This may need to be double checked
            
            if phase == 'train':
                train_epoch_loss = running_loss / dataset_sizes[phase]
                #train_epoch_acc = running_corrects / dataset_sizes[phase]
                train_epoch_acc = float(running_corrects) / running_samples
            else:
                valid_epoch_loss = running_loss / dataset_sizes[phase]
                #valid_epoch_acc = running_corrects / dataset_sizes[phase]
                valid_epoch_acc = float(running_corrects) / running_samples
                
            if phase == 'valid' and valid_epoch_acc > best_valid_acc:
                best_valid_acc = valid_epoch_acc
                best_model_wts = model.state_dict()

        print('Epoch [{}/{}] train loss: {:.4f} acc: {:.4f} ' 
              'valid loss: {:.4f} acc: {:.4f} time: {:.4f}'.format(
                epoch, num_epochs - 1,
                train_epoch_loss, train_epoch_acc, 
                valid_epoch_loss, valid_epoch_acc, (time.time()-since)/60))
            
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model


### Get DataLoaders running
# from torch.utils.data.sampler import SubsetRandomSampler

# dataset = WhaleDataset("./data/whale_calls/data/train.csv", "./data/whale_calls/data/train/", (224, 224))
# batch_size = 16
# validation_split = .2
# shuffle_dataset = True
# random_seed= 42

## Build Dataset
'''
dataset = ElephantDataset()

dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(VALIDATION_SPLIT * dataset_size))
np.random.shuffle(indices)

train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=valid_sampler)
'''
train_loader, validation_loader = get_train_valid_loader("../elephant_dataset/Train/Activate_Label/",
                           BATCH_SIZE,
                           RANDOM_SEED)

dloaders = {'train':train_loader, 'valid':validation_loader}

## Build Model
input_size = 77 # Num of frequency bands in the spectogram
hidden_size = 128
#model = LSTM(input_size, hidden_size, 1, BATCH_SIZE)
model = CONV1D_LSTM(input_size, hidden_size, 1, BATCH_SIZE)

use_gpu = torch.cuda.is_available()
if use_gpu:
    model = model.cuda()

criterion = torch.nn.BCEWithLogitsLoss()
# Ramp this up for sure
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

start_time = time.time()
model = train_model(dloaders, model, criterion, optimizer, num_epochs=NUM_EPOCHS)

print('Training time: {:10f} minutes'.format((time.time()-start_time)/60))

# TODO: Save model to something



