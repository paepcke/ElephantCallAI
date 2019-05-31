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


data_directory = './Processed_data/'

output_directory = './Features_Labels'


feature_set = []
label_set = []

# Determines the size of the chunks that we are creating around the elephant
# Call. This refers to number of columns in the spectogram. Based on the 
# Creation of the spectogram this equates to 25.5 second windows
# Definitely should do some padding then on both sides and do a random crop
# so the call is not always centered! 
FRAME_LENGTH = 64

# Number of time steps to add the 1
ACTIVATE_TIME = 5

def makeChunk(start_index,feat_mat,label_mat):
    # 1. Determine length of call in number of indices
    length_of_call = 0
    # Now we also want to zero out this! because we are gonna
    # label just the end of the call
    for j in range(start_index,label_mat.shape[0]):
        if label_mat[j] != 1:
            break
        length_of_call += 1
        label_mat[j] = 0

    if (start_index + length_of_call + ACTIVATE_TIME >= label_mat.shape[0]):
        return -1, None, None
    # Label the end of the call with 
    # ACTIVATE_TIME number of frames
    for i in range(ACTIVATE_TIME):
        label_mat[j + i] = 1

    # Figure out how much to go before and after. We do not want to got .5 
    # because then the chunks are of different sizes
    # Randomly place the call in the window for now? Should use data_augmentation later
    # We want the whole call to be in there plus the labeling of the "activate"
    padding_frame = FRAME_LENGTH - length_of_call - ACTIVATE_TIME
    # if padding_frame is neg skip call
    assert(padding_frame >= 0)
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

    '''
    # 2. Now go back floor(.5) before call, and ceiling(.5) after call as chunk
    # and create corresponding label
    # 2a set start and end indices, considering might be at beginining/end of
    # file
    chunk_start_index = start_index - math.floor(length_of_call/2)
    chunk_start_index = max(chunk_start_index,0)
    chunk_end_index = start_index + length_of_call + math.floor(length_of_call/2)
    chunk_end_index = min(chunk_end_index,label_mat.shape[0])
    '''
    return_features = []
    return_labels = []
    # + 1 here?
    for j in range(chunk_start_index,chunk_end_index):
        return_features.append(feat_mat[j,:])
        return_labels.append(label_mat[j])

    #print (np.array(return_features).shape)

    return start_index + length_of_call + ACTIVATE_TIME + 1,np.array(return_features),np.array(return_labels)



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
            if (skip == -1):
                break
            feature_set.append(feature_chunk)
            print(feature_chunk.shape)
            label_set.append(label_chunk)
            skip_to_index = True
    return feature_set, label_set



# 1. Iterate through all files in output
for i,fileName in enumerate(os.listdir(data_directory)):
    print(fileName,i)
    if fileName[0:4] == 'Data':
        label_file = 'Label'+fileName[4:]
        feature_set, label_set = makeDataSet(data_directory+fileName,data_directory+label_file)

#print (len(feature_set))
feature_set = np.stack(feature_set)
#print (feature_set.shape)
label_set = np.stack(label_set)
print(feature_set.shape,label_set.shape)
np.save(output_directory + '/features.npy',feature_set)
np.save(output_directory + '/labels.npy',label_set)



