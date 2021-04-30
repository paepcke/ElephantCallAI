import parameters
from utils import set_seed
from gun_data.GunshotDataset import GunshotDataset
import torch
import numpy as np
from models import Model17
from torch import nn


def get_loader(data_dir,
               batch_size,
               random_seed=8,
               shuffle=True,
               num_workers=16,
               pin_memory=False):
    # TODO: add just-in-time transform/data augmentation options
    """
    Utility function for loading and returning train and valid
    multi-process iterators.
    """
    print("DataLoader Seed:", parameters.DATA_LOADER_SEED)
    set_seed(parameters.DATA_LOADER_SEED)

    dataset = GunshotDataset(data_dir)

    print('Size of dataset at {} is {} samples'.format(data_dir, len(dataset)))

    # Set the data_loader random seed for reproducibility.
    def _init_fn(worker_id):
        # Assign each worker its own seed
        np.random.seed(int(random_seed) + worker_id)
        # Is this bad??
        # This seems bad as each epoch will be the same order of data!
        # torch.manual_seed(int(random_seed) + worker_id)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory,
                                              worker_init_fn=_init_fn)

    return data_loader

def get_model(input_size) -> nn.Module:
    """
    This function is temporary, just to test out the end-to-end workflow here
    """
    model = Model17(input_size, 2340)
    model.model.fc = nn.Sequential(
           nn.Linear(512, 128),
           nn.ReLU(inplace=True),
           nn.Linear(128, 3))  # This is hard coded to the size of the training windows
    return model

def compute_acc(labels: np.ndarray, logits: np.ndarray):
    predictions = np.argmax(logits, axis=1)
    agreement = np.where(labels == predictions, 1, 0)
    return np.mean(agreement) * 100

if __name__ == "__main__":
    # TODO: flesh out this code and move it to a separate file. Add tensorboard visualizations, model abstraction, model saving/loading, early stopping, etc.
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    model = get_model((95, 525))
    model = model.to(device)
    train_dataloader = get_loader("/home/deschwa2/gun_data/processed/datasets/train", 4)
    # val_dataloader = get_loader("/home/deschwa2/gun_data/processed/datasets/val", 4)
    num_epochs = 40

    optimizer = torch.optim.Adam(model.parameters())
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        num_batches = 0
        running_loss = 0
        running_acc = 0
        for batch in train_dataloader:
            optimizer.zero_grad()
            data, labels = batch[0], batch[1]
            data = data.to(device)
            labels = labels.to(device)

            logits = model.forward(data)

            loss = loss_func(logits, labels)

            running_loss += loss.item()
            running_acc += compute_acc(labels.detach().cpu().numpy(), logits.detach().cpu().numpy())

            loss.backward()
            optimizer.step()

            num_batches += 1
        avg_loss = running_loss/num_batches
        avg_acc = running_acc/num_batches
        print(f"Epoch {epoch} complete, avg loss: {avg_loss}, avg acc: {avg_acc}%")



