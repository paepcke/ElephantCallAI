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
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.cm
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.ticker as plticker
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
import random
from random import shuffle 
import math


MFCC_Data = './Processed_data_MFCC/'
Spect_Data = './Processed_data/'
full_call_directory = 'call/'
activate_directory = 'activate/'

output_directory = './Features_Labels'
test_directory = './Test'
train_directory = './Train'

#feature_set = []
#label_set = []

seed = 8
random.seed(seed)
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

USE_MFCC_FEATURES = False

VERBOSE = False


def makeChunk(start_index,feat_mat,label_mat):
    # 1. Determine length of call in number of indices
    length_of_call = 0

    for j in range(start_index,label_mat.shape[0]):
        if label_mat[j] != 1:
            break
        length_of_call += 1

    # Figure out how much to go before and after. We do not want to got .5 
    # because then the chunks are of different sizes
    # Randomly place the call in the window for now? Should use data_augmentation later
    # We want the whole call to be in there plus the labeling of the "activate"
    padding_frame = FRAME_LENGTH - length_of_call #- ACTIVATE_TIME
    # if padding_frame is neg skip call
    # but still want to go to the next!!
    if padding_frame < 0:
        print ("skipping")
        return start_index + length_of_call + 1, None, None

    
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

    return_features = feat_mat[chunk_start_index: chunk_end_index, :]
    return_labels = label_mat[chunk_start_index: chunk_end_index]

    if VERBOSE:
        display_call(return_features, return_labels)

    # Note make sure that we skip over the call and the activate labels
    return start_index + length_of_call + 1, return_features, return_labels


def makeDataSet(featFile,labFile):
    # 1. Open both feature file and label file as np arrays
    feature_file = np.genfromtxt(featFile,delimiter=',').transpose()
    label_file = np.genfromtxt(labFile,delimiter=',')

    feature_set = []
    label_set = []
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

# Assumes that we are making chunks going backwards
def makeChunkActivate(activate_index,feat_mat,label_mat):
    # 1. Determine length of call in number of indices
    start_index = label_mat[activate_index, 1]

    # Determine where the call actually ends
    # To ultimately get the length of the call
    i = activate_index
    while i >= 0 and label_mat[i, 1] == start_index:
        i -= 1

    call_end = i
    length_of_call = call_end - start_index

    # Skip a call at the very end if the activate labels can't fit
    if (start_index + length_of_call + ACTIVATE_TIME >= label_mat.shape[0]):
        return start_index - 1, None, None

    # Figure out how much to go before and after. We do not want to got .5 
    # because then the chunks are of different sizes
    # Randomly place the call in the window for now? Should use data_augmentation later
    # We want the whole call to be in there plus the labeling of the "activate"
    padding_frame = FRAME_LENGTH - length_of_call - ACTIVATE_TIME
    # if padding_frame is neg skip call
    # but still want to go to the next!!
    if padding_frame < 0:
        print ("skipping")
        return start_index - 1, None, None
    
    # Randomly split the pad to before and after
    split = np.random.randint(0, padding_frame + 1)

    # Do some stuff to avoid the front and end!
    chunk_start_index = start_index - split
    chunk_end_index  = start_index + length_of_call + ACTIVATE_TIME + (padding_frame - split)
    # Edge case if window is near the beginning or end of the file
    # the window of 64 frames is larger than the sound file!
    if (chunk_start_index < 0):
        # Make the window start at 0
        chunk_start_index = 0
        chunk_end_index = FRAME_LENGTH - 1
    if (chunk_end_index >= label_mat.shape[0]):
        chunk_end_index = label_mat.shape[0]
        chunk_start_index = label_mat.shape[0] - FRAME_LENGTH

    chunk_start_index = int(chunk_start_index); chunk_end_index = int(chunk_end_index)
    
    return_features = feat_mat[chunk_start_index: chunk_end_index, :]
    return_labels = label_mat[chunk_start_index: chunk_end_index, 0]

    if VERBOSE:
        display_call(return_features, return_labels)

    # Note make sure that we skip over the call and the activate labels
    return start_index - 1, return_features, return_labels


def display_call(features, labels):
    """
        Assumes features is of shape (time, freq)
    """
    fig, (ax1, ax2) = plt.subplots(2,1)
    new_features = np.flipud(10*np.log10(features).T)
    min_dbfs = new_features.flatten().mean()
    max_dbfs = new_features.flatten().mean()
    min_dbfs = np.maximum(new_features.flatten().min(),min_dbfs-2*new_features.flatten().std())
    max_dbfs = np.minimum(new_features.flatten().max(),max_dbfs+6*new_features.flatten().std())
    ax1.imshow(np.flipud(new_features), cmap="magma_r", vmin=min_dbfs, vmax=max_dbfs, interpolation='none', origin="lower", aspect="auto")
    print (labels)
    ax2.plot(np.arange(labels.shape[0]), labels)
    plt.show()
    

def extend_activate_label(labFile):
    """
        Make the activate label for each call have 
    """

    for i in reversed(range(labFile.shape[0])):
        # Extend the file label!
        if (labFile[i, 0] == 1):
            # Copy the start index 
            # So we know where the call 
            # started that is ending for
            # creating chunks
            start = labFile[i, 1]
            for j in range(ACTIVATE_TIME):
                if (i + j >= labFile.shape[0]):
                    break
                labFile[i + j, 0] = 1
                labFile[i + j, 1] = start

def makeDataSetActivate(featFile,labFile):
    # 1. Open both feature file and label file as np arrays
    feature_file = np.genfromtxt(featFile,delimiter=',').transpose()
    label_file = np.genfromtxt(labFile,delimiter=',')

    feature_set = []
    label_set = []
    # Extend our labels to have the right number of activate labels
    extend_activate_label(label_file)
    # 2. Now iterate through label file, when find a call, pass to make chunk,
    # which will return new starting index, the feature chunk and label chunk
    skip_to_index = False
    for i in reversed(range(label_file.shape[0])):
        if skip_to_index:
            skip_to_index = False if i == skip else True
            continue
        if label_file[i, 0] == 1:
            skip,feature_chunk, label_chunk = makeChunkActivate(i,feature_file,label_file)
            # Skip this call because we are at the end of the file
            if (feature_chunk is not None):  
                feature_set.append(feature_chunk)
                label_set.append(label_chunk)
            skip_to_index = True

    return feature_set, label_set



def main():
    # 1. Iterate through all files in output
    data_directory = MFCC_Data if USE_MFCC_FEATURES else Spect_Data
    data_directory += activate_directory if USE_POST_CALL_LABEL else full_call_directory

    datafiles = []
    for i,fileName in enumerate(os.listdir(data_directory)):
        if fileName[0:4] == 'Data':
            datafiles.append(fileName)

    # Shuffle the files before train test split
    shuffle(datafiles)
    
    split_index = math.floor(len(datafiles) * (1 - TEST_SIZE))
    train_data_files = datafiles[:split_index]
    test_data_files = datafiles[split_index:]
    
    train_feature_set = []
    train_label_set = []
    print ("Making Train Set")
    print ("Size: ", len(train_data_files))
    # Make the training dataset
    index = 0
    for file in train_data_files:
        print(file, index)
        label_file = 'Label'+file[4:]
        if (ACTIVATE_TIME == 0):
            feature_set, label_set = makeDataSet(data_directory+file,data_directory+label_file)
        else:
            feature_set, label_set = makeDataSetActivate(data_directory + file, data_directory + label_file)
        
        train_feature_set.extend(feature_set)
        train_label_set.extend(label_set) 
        index += 1      

    print (train_data_files)
    X_train = np.stack(train_feature_set)
    y_train = np.stack(train_label_set)

    print ("Making Test Set")
    print ("Size: ", len(test_data_files))
    # Make the test dataset
    test_feature_set = []
    test_label_set = []
    # Make the training dataset
    index = 0
    for file in test_data_files:
        print(file, index)
        label_file = 'Label'+file[4:]
        if (ACTIVATE_TIME == 0):
            feature_set, label_set = makeDataSet(data_directory+file,data_directory+label_file)
        else:
            feature_set, label_set = makeDataSetActivate(data_directory + file, data_directory + label_file)
        
        test_feature_set.extend(feature_set)
        test_label_set.extend(label_set)   
        index += 1    

    X_test = np.stack(test_feature_set)
    y_test = np.stack(test_label_set)
    

    print (X_train.shape, X_test.shape)
    print (y_train.shape, y_test.shape)

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



