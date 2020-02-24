from data import get_loader
import model as model_file
import parameters
from tensorboardX import SummaryWriter
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import sys
import time
import os

np.random.seed(parameters.RANDOM_SEED)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def initialize_training(model_id, save_path):
    model = model_file.get_model(model_id).to(device)
    print(model)
    writer = SummaryWriter(save_path)
    writer.add_scalar('batch_size', parameters.BATCH_SIZE)
    writer.add_scalar('weight_decay', parameters.HYPERPARAMETERS[model_id]['l2_reg'])

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=parameters.HYPERPARAMETERS[model_id]['lr'], weight_decay=parameters.HYPERPARAMETERS[model_id]['l2_reg'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, parameters.HYPERPARAMETERS[model_id]['lr_decay_step'], gamma=parameters.HYPERPARAMETERS[model_id]['lr_decay'])

    return model, criterion, optimizer, scheduler, writer


def outerLoop(model_id):

    train_loader = get_loader("/home/data/elephants/processed_data/Train_nouab/Neg_Samples_x2/", parameters.BATCH_SIZE, parameters.NORM, parameters.SCALE)
    validation_loader = get_loader("/home/data/elephants/processed_data/Test_nouab/Neg_Samples_x2/", parameters.BATCH_SIZE, parameters.NORM, parameters.SCALE)
    full_train_loader = get_loader("/home/data/elephants/processed_data/Train_nouab/Full_24_hrs/", parameters.BATCH_SIZE, parameters.NORM, parameters.SCALE)

    for outer_iteration in range(10):
        dloaders = {'train':train_loader, 'valid':validation_loader}

        save_path = parameters.SAVE_PATH + parameters.DATASET + '_model_' + str(model_id) + "_" + parameters.NORM + "_Negx" + str(parameters.NEG_SAMPLES) + "_" + str(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()) + "adversarial_iteration_" + str(outer_iteration))

        start_time = time.time()
        model, criterion, optimizer, scheduler, writer = initialize_training(model_id, save_path)
        model_wts = model_file.train_model(dloaders, model, criterion, optimizer, scheduler, writer, parameters.NUM_EPOCHS)

        assert model_wts != None

        model.load_state_dict(model_wts)
        model_save_path = save_path + "/" + "model.pt"
        if not os.path.exists(parameters.SAVE_PATH):
            os.makedirs(parameters.SAVE_PATH)
        torch.save(model, model_save_path)
        print('Saved best val acc model to path {}'.format(model_save_path))

        print('Training time: {:10f} minutes'.format((time.time()-start_time)/60))
        writer.close()

        # Evaluate on entire dataset
        adversarial_files = model_file.adversarial_discovery(full_train_loader, model, len(train_loader.dataset) * 0.25)
        np.save(save_path + "/" + "adversarial_examples_" + str(outer_iteration) + ".txt" , adversarial_files)

        # Select randomly the same number as current num calls
        # adversarial_files = np.random.choice()
        

        # Update training dataset with adversarial files
        train_loader.dataset.features += adversarial_files
        train_loader.dataset.initialize_labels()

        assert len(train_loader.dataset.features) == len(train_loader.dataset.labels)
        print("Length of features is now {} ".format(len(train_loader.dataset.features)))


if __name__ == "__main__":
    outerLoop(int(sys.argv[1]))


