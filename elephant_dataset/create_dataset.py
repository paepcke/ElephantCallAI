'''
Idea is to iterate through data, find call, get the length of the call,
then make a single training chunk as a 1/2 time before call call then 1/2 time
after call. Thus dataset will be equal
'''
'''
Try for now to make all of the calls the same length
'''

import os,math
import numpy as np
from sklearn.model_selection import train_test_split


data_directory = './Processed_data/'
output_directory = './Features_Labels'
test_directory = './Test'
train_directory = './Train'

feature_set = []
label_set = []

seed = 8
TEST_SIZE = 0.2

# Determines the size of the chunks that we are creating around the elephant
# Call. This refers to number of columns in the spectogram. Based on the 
# Creation of the spectogram this equates to 25.5 second windows
# Definitely should do some padding then on both sides and do a random crop
# so the call is not always centered! 
FRAME_LENGTH = 64

# Define whether we label the call itself
# or label when the call ends. If True labels
# when the call ends
USE_POST_CALL_LABEL = False
# Number of time steps to add the 1
ACTIVATE_TIME = 5 if USE_POST_CALL_LABEL else 0


def makeChunk(start_index,feat_mat,label_mat):
    # 1. Determine length of call in number of indices
    length_of_call = 0

    for j in range(start_index,label_mat.shape[0]):
        if label_mat[j] != 1:
            break
        length_of_call += 1
        # If we are adding the labels at the end
        # zero out the actual call labels
        if (ACTIVATE_TIME != 0):
            label_mat[j] = 0

    if (start_index + length_of_call + ACTIVATE_TIME >= label_mat.shape[0]):
        return -1, None, None

    # Figure out how much to go before and after. We do not want to got .5 
    # because then the chunks are of different sizes
    # Randomly place the call in the window for now? Should use data_augmentation later
    # We want the whole call to be in there plus the labeling of the "activate"
    padding_frame = FRAME_LENGTH - length_of_call - ACTIVATE_TIME
    # if padding_frame is neg skip call
    # but still want to go to the next!!
    if padding_frame < 0:
        print ("skipping")
        return start_index + length_of_call + 1, None, None


    # Label the end of the call with 
    # ACTIVATE_TIME number of frames
    for i in range(ACTIVATE_TIME):
        label_mat[j + i] = 1

    
    # Randomly split the pad to before and after
    split = np.random.randint(0, padding_frame + 1)

    # Do some stuff to avoid the front and end!
    chunk_start_index = start_index - split
    chunk_end_index  = start_index + length_of_call + ACTIVATE_TIME + padding_frame - split
    # Do some quick voodo - assume cant have issue where 
    # the window of 64 frames is lareger than the sound file!
    if (chunk_start_index < 0):
        # Amount to transfer to end
        transfer_end = 0 - chunk_start_index
        chunk_start_index = 0
        chunk_end_index += transfer_end
    if (chunk_end_index >= label_mat.shape[0]):
        transfer_front = chunk_end_index - label_mat.shape[0] + 1
        chunk_end_index = label_mat.shape[0] - 1
        chunk_start_index -= transfer_front

    return_features = []
    return_labels = []
    # + 1 here?
    for j in range(chunk_start_index,chunk_end_index):
        return_features.append(feat_mat[j,:])
        return_labels.append(label_mat[j])

    # Note make sure that we skip over the call and the activate labels
    return start_index + length_of_call + ACTIVATE_TIME + 1, np.array(return_features),np.array(return_labels)



def makeDataSet(featFile,labFile):
    # 1. Open both feature file and label file as np arrays
    feature_file = np.genfromtxt(featFile,delimiter=',').transpose()
    label_file = np.genfromtxt(labFile,delimiter=',')

    # 2. Now iterate through label file, when find a call, pass to make chunk,
    # which will return new starting index, the feature chunk and label chunk
    skip_to_index = False
    for i in range(label_file.shape[0]):
        if skip_to_index:
            skip_to_index = False if i == skip else True
            continue
        if label_file[i] == 1:
            skip,feature_chunk, label_chunk = makeChunk(i,feature_file,label_file)
            # Skip this call because we are at the end of the file
            if (feature_chunk is not None):  
                feature_set.append(feature_chunk)
                label_set.append(label_chunk)
            skip_to_index = True

    return feature_set, label_set



def main():
    # 1. Iterate through all files in output
    for i,fileName in enumerate(os.listdir(data_directory)):
        print(fileName,i)
        if fileName[0:4] == 'Data':
            label_file = 'Label'+fileName[4:]
            feature_set, label_set = makeDataSet(data_directory+fileName,data_directory+label_file)

    feature_set = np.stack(feature_set)
    label_set = np.stack(label_set)
    
    print(feature_set.shape,label_set.shape)

    X_train, X_test, y_train, y_test = train_test_split(feature_set, label_set, test_size=TEST_SIZE,random_state=seed)

    print (X_train.shape)
    print (X_test.shape)

    label_type = '/Activate_Label'
    if (not USE_POST_CALL_LABEL):
        label_type = "/Call_Label"

    np.save(train_directory + label_type + '/features.npy', X_train)
    np.save(train_directory + label_type + '/labels.npy', y_train)
    np.save(test_directory + label_type + '/features.npy', X_test)
    np.save(test_directory + label_type + '/labels.npy', y_test)

    # Save the individual training files for visualization etc.
    for i in range(X_train.shape[0]):
        np.save(train_directory + label_type + '/features_{}'.format(i+1), X_train[i])
        np.save(train_directory + label_type + '/labels_{}'.format(i+1), y_train[i])

    for i in range(X_test.shape[0]):
        np.save(test_directory + label_type + '/features_{}'.format(i+1), X_test[i])
        np.save(test_directory + label_type + '/labels_{}'.format(i+1), y_test[i])


if __name__ == '__main__':
    main()



