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

# THis should now be set in model.py
#np.random.seed(parameters.RANDOM_SEED)
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def initialize_training(model_id, save_path):
    # The get_model method is in charge of 
    # setting the same seed for each loaded model.
    # Thus, for each inner loop we train the same initialized model
    model = model_file.get_model(model_id).to(parameters.device)
    print(model)
    writer = SummaryWriter(save_path)
    writer.add_scalar('batch_size', parameters.BATCH_SIZE)
    writer.add_scalar('weight_decay', parameters.HYPERPARAMETERS[model_id]['l2_reg'])

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=parameters.HYPERPARAMETERS[model_id]['lr'], weight_decay=parameters.HYPERPARAMETERS[model_id]['l2_reg'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, parameters.HYPERPARAMETERS[model_id]['lr_decay_step'], gamma=parameters.HYPERPARAMETERS[model_id]['lr_decay'])

    return model, criterion, optimizer, scheduler, writer


def outerLoop(model, train_loader, validation_loader, full_train_loader, save_path):
    # get_loader now handles setting random seed for reproducability
    # Add flexability to start decide what intial neg_sample ratio to 
    # start. Later!!! Include what random seed dataset to sample from
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for outer_iteration in range(parameters.ADVERSARIAL_LOOPS):
        dloaders = {'train':train_loader, 'valid':validation_loader}

        iteration_save_path = save_path + '/' + parameters.DATASET + '_model_' + str(model_id) + "_" + parameters.NORM + \
                                      "_Negx" + str(parameters.NEG_SAMPLES) + "_Seed_" + str(parameters.RANDOM_SEED) + \
                                        "_adversarial_iteration_" + str(outer_iteration)+ "_" + str(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
        #iteration_save_path = save_path + '/' + "Adversarial_iteration_" + str(outer_iteration)
        if not os.path.exists(iteration_save_path):
            os.makedirs(iteration_save_path)

        start_time = time.time()
        model, criterion, optimizer, scheduler, writer = initialize_training(model_id, iteration_save_path)
        model_wts = model_file.train_model(dloaders, model, criterion, optimizer, scheduler, writer, parameters.NUM_EPOCHS)

        assert model_wts != None

        model.load_state_dict(model_wts)
        model_save_path = iteration_save_path + '/' + "model.pt"
        torch.save(model, model_save_path)
        print('Saved best val acc model to path {}'.format(model_save_path))

        print('Training time: {:10f} minutes'.format((time.time()-start_time)/60))
        writer.close()

        # Evaluate on entire dataset
        # Add 1/4 * (num_data_points) to the training data
        # We should maybe include a scaling term to the loss function!
        num_adversarial = len(train_loader.dataset) * parameters.ADVERSARIAL_SAMPLES if parameters.ADVERSARIAL_SAMPLES != -1 else -1
        adversarial_files = model_file.adversarial_discovery(full_train_loader, model,
                                                         num_files_to_return=num_adversarial, min_length=parameters.ADVERSARIAL_THRESHOLD)
        adversarial_save_path = iteration_save_path + "/" + "adversarial_examples_" + str(outer_iteration) + ".txt"
        with open(adversarial_save_path, 'w') as f:
            for file in adversarial_files:
                f.write('{}\n'.format(file))

        #np.save(save_path + "/" + "adversarial_examples_" + str(outer_iteration) + ".txt" , adversarial_files)

        # Select randomly the same number as current num calls
        # This is now done locally
        # adversarial_files = np.random.choice()
        

        # Update training dataset with adversarial files
        train_loader.dataset.features += adversarial_files
        train_loader.dataset.initialize_labels()

        assert len(train_loader.dataset.features) == len(train_loader.dataset.labels)
        # Should keep track of ratio of neg-to-pos
        print("Length of features is now {} ".format(len(train_loader.dataset.features)))


if __name__ == "__main__":
    model = int(sys.argv[1])

    train_loader = get_loader("/home/data/elephants/processed_data/Train_nouab/Neg_Samples_x" + str(parameters.NEG_SAMPLES) + "/", parameters.BATCH_SIZE, random_seed=parameters.DATA_LOADER_SEED, norm=parameters.NORM, scale=parameters.SCALE)
    validation_loader = get_loader("/home/data/elephants/processed_data/Test_nouab/Neg_Samples_x" + str(parameters.NEG_SAMPLES) + "/", parameters.BATCH_SIZE, random_seed=parameters.DATA_LOADER_SEED, norm=parameters.NORM, scale=parameters.SCALE)
    full_train_loader = get_loader("/home/data/elephants/processed_data/Train_nouab/Full_24_hrs/", parameters.BATCH_SIZE, random_seed=parameters.DATA_LOADER_SEED, norm=parameters.NORM, scale=parameters.SCALE)

    save_path = parameters.SAVE_PATH + "Adversarial_training_" + parameters.DATASET + '_model_' + str(model_id) + "_" + parameters.NORM + "_Negx" + str(parameters.NEG_SAMPLES) + "_Seed_" + str(parameters.RANDOM_SEED) + "_" + str(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))

    outerLoop(model, train_loader, validation_loader, full_train_loader, save_path)


