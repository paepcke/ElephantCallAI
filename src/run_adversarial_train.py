from data import get_loader
import parameters
# def outerLoop(model):

train_loader = get_loader("/home/data/elephants/processed_data/Train_nouab/Neg_Samples_x2/", parameters.BATCH_SIZE, parameters.NORM, parameters.SCALE)
validation_loader = get_loader("/home/jgs8/ElephantCallAI/elephant_dataset/Test_nouab/Neg_Samples_x2", parameters.BATCH_SIZE, parameters.NORM, parameters.SCALE)

print(train_loader.dataset)

    # for outer_iteration in range(10):


# def main():
#     ## Build Dataset
#     # "/home/jgs8/ElephantCallAI/elephant_dataset/Train_nouab/Neg_Samples_x" + str(parameters.NEG_SAMPLES) + "/"



#     ## Training
#     model_id = int(sys.argv[1])
#     save_path = parameters.SAVE_PATH + parameters.DATASET + '_model_' + str(model_id) + "_" + parameters.NORM + "_Negx" + str(parameters.NEG_SAMPLES) + "_" + str(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))

#     model = get_model(model_id)

#     model.to(device)

#     print(model)

#     writer = SummaryWriter(save_path)
#     writer.add_scalar('batch_size', parameters.BATCH_SIZE)
#     writer.add_scalar('weight_decay', parameters.HYPERPARAMETERS[model_id]['l2_reg'])

#     criterion = torch.nn.BCEWithLogitsLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=parameters.HYPERPARAMETERS[model_id]['lr'], weight_decay=parameters.HYPERPARAMETERS[model_id]['l2_reg'])
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, parameters.HYPERPARAMETERS[model_id]['lr_decay_step'], gamma=parameters.HYPERPARAMETERS[model_id]['lr_decay'])

#     start_time = time.time()
#     model_wts = None

#     model_wts = train_model(dloaders, model, criterion, optimizer, scheduler, writer, parameters.NUM_EPOCHS)

#     if model_wts:
#         model.load_state_dict(model_wts)
#         save_path = save_path + "/" + "model.pt"
#         if not os.path.exists(parameters.SAVE_PATH):
#             os.makedirs(parameters.SAVE_PATH)
#         torch.save(model, save_path)
#         print('Saved best val acc model to path {}'.format(save_path))
#     else:
#         print('For some reason I don\'t have a model to save')

#     print('Training time: {:10f} minutes'.format((time.time()-start_time)/60))

#     writer.close()

# if __name__ == '__main__':
#     main()