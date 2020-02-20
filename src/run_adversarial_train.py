from data import get_loader
import parameters


def initialize_training(model_id):
    save_path = parameters.SAVE_PATH + parameters.DATASET + '_model_adversarial_' + str(model_id) + "_" + parameters.NORM + "_Negx" + str(parameters.NEG_SAMPLES) + "_" + str(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))

    model = get_model(model_id).to(device)
    print(model)
    writer = SummaryWriter(save_path)
    writer.add_scalar('batch_size', parameters.BATCH_SIZE)
    writer.add_scalar('weight_decay', parameters.HYPERPARAMETERS[model_id]['l2_reg'])

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=parameters.HYPERPARAMETERS[model_id]['lr'], weight_decay=parameters.HYPERPARAMETERS[model_id]['l2_reg'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, parameters.HYPERPARAMETERS[model_id]['lr_decay_step'], gamma=parameters.HYPERPARAMETERS[model_id]['lr_decay'])

    return model, criterion, optimizer, scheduler, writer


def evaluate_model_on_dataset(model, data_loader):
    model.eval()

    difficult_files = []

    i = 0
    print ("Num batches:", len(data_loader))
    for inputs, labels, feature_file_paths in data_loader:
        i += 1
        if (i % 1000 == 0):
            print ("Batch number {} of {}".format(i, len(dataloders[phase])))  

        inputs = inputs.float()
                    
        labels = labels.float()

        inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))

        logits = model(inputs)

        # TODO: Figure out the rest here

    ### ????



def outerLoop(model_id):

    train_loader = get_loader("/home/data/elephants/processed_data/Train_nouab/Neg_Samples_x2/", parameters.BATCH_SIZE, parameters.NORM, parameters.SCALE)
    validation_loader = get_loader("/home/data/elephants/processed_data/Test_nouab/Neg_Samples_x2/", parameters.BATCH_SIZE, parameters.NORM, parameters.SCALE)
    full_train_loader = get_loader("/home/data/elephants/processed_data/Train_nouab/Full_24_hrs/", parameters.BATCH_SIZE, parameters.NORM, parameters.SCALE)

    for outer_iteration in range(10):
        dloaders = {'train':train_loader, 'valid':validation_loader}

        save_path = parameters.SAVE_PATH + parameters.DATASET + '_model_' + str(model_id) + "_" + parameters.NORM + "_Negx" + str(parameters.NEG_SAMPLES) + "_" + str(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))

        start_time = time.time()
        model, criterion, optimizer, scheduler, writer = initialize_training(model_id)
        model_wts = train_model(dloaders, model, criterion, optimizer, scheduler, writer, parameters.NUM_EPOCHS)

        assert model_wts != None

        model.load_state_dict(model_wts)
        save_path = save_path + "/" + "model.pt"
        if not os.path.exists(parameters.SAVE_PATH):
            os.makedirs(parameters.SAVE_PATH)
        torch.save(model, save_path)
        print('Saved best val acc model to path {}'.format(save_path))


        print('Training time: {:10f} minutes'.format((time.time()-start_time)/60))
        writer.close()

        # Evaluate on entire dataset

        # Update training dataset with 

