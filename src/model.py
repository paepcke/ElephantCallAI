
RANDOM_SEED = 42

BATCH_SIZE = 1
VALIDATION_SPLIT = .2

NUM_EPOCHS = 2

np.random.seed(RANDOM_SEED)

class LSTM(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(RNN, self).__init__()
		self.hidden_dim = hidden_size

		self.hidden = self.init_hidden()

		self.lstm = nn.LSTM(input_size, hidden_size)
		self.hiddenToClass = nn.Linear(hidden_size, output_size)
		self.softmax = nn.Softmax()

	def init_hidden(self):
		# Before we've done anything, we dont have any hidden state.
		# Refer to the Pytorch documentation to see exactly
		# why they have this dimensionality.
		# The axes semantics are (num_layers, minibatch_size, hidden_dim)
		return (torch.ones(1, BATCH_SIZE, self.hidden_dim),
				torch.ones(1, BATCH_SIZE, self.hidden_dim))


	def forward(self, input):
		lstm_out, self.hidden = self.lstm(input, self.hidden)
		classVal = self.hiddenToClass(lstm_out)
		pred = self.softmax(classVal)
		return pred

class CONV1D_LSTM(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(RNN, self).__init__()

		num_filters = 25
		kernel_size = 5

		self.hidden_dim = hidden_size

		self.hidden = self.init_hidden()

		self.convLayer = nn.Conv1d(input_size, num_filters, kernel_size, padding=2)
		self.lstm = nn.LSTM(num_filters, hidden_size)
		self.hiddenToClass = nn.Linear(hidden_size, output_size)
		self.softmax = nn.Softmax()

	# TODO: We should figure out how to init hidden
	def init_hidden(self):
		# Before we've done anything, we dont have any hidden state.
		# Refer to the Pytorch documentation to see exactly
		# why they have this dimensionality.
		# The axes semantics are (num_layers, minibatch_size, hidden_dim)
		return (torch.ones(1, BATCH_SIZE, self.hidden_dim),
				torch.ones(1, BATCH_SIZE, self.hidden_dim))

	def forward(self, input):
		convFeatures = self.convLayer(input)
		# TODO: Flatten here or Maxpool
		lstm_out, self.hidden = self.lstm(convFeatures, self.hidden)
		classVal = self.hiddenToClass(lstm_out)
		pred = self.softmax(classVal)
		return pred


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

			for inputs, labels in dataloders[phase]:
				if use_gpu:
					inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
				else:
					inputs, labels = Variable(inputs), Variable(labels)

				optimizer.zero_grad()

				# Forward pass
				outputs = model(inputs)

				# TODO: Perhaps need to check the outputs value here

				loss = criterion(outputs, labels)

				# Backward pass
				if phase == 'train':
					loss.backward()
					optimizer.step()

				running_loss += loss.item()
				running_corrects += torch.sum(preds == labels.data) #TODO: This may need to be double checked
			
			if phase == 'train':
				train_epoch_loss = running_loss / dataset_sizes[phase]
				train_epoch_acc = running_corrects / dataset_sizes[phase]
			else:
				valid_epoch_loss = running_loss / dataset_sizes[phase]
				valid_epoch_acc = running_corrects / dataset_sizes[phase]
				
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

dloaders = {'train':train_loader, 'valid':validation_loader}

## Build Model
model = LSTM(input_size, hidden_size, NUM_CLASSES)

use_gpu = torch.cuda.is_available()
if use_gpu:
	model = model.cuda()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

start_time = time.time()
model = train_model(dloaders, model, criterion, optimizer, num_epochs=NUM_EPOCHS)

print('Training time: {:10f} minutes'.format((time.time()-start_time)/60))

# TODO: Save model to something



