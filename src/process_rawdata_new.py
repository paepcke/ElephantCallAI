from matplotlib import pyplot as plt
from matplotlib import mlab as ml
import numpy as np
import csv
import os
import time
import multiprocessing
from scipy.io import wavfile
from visualization import visualize
import math
import argparse
from random import shuffle 


parser = argparse.ArgumentParser()
parser.add_argument('--data', dest='dataDir', default='../elephant_dataset/New_Data/Truth_Logs/', 
    type=str, help='The top level directory with the data (e.g. Truth_Logs)')

# For use on quatro
#parser.add_argument('--data', dest='dataDir', default='/home/data/elephants/rawdata/raw_2018', 
#    type=str, help='The top level directory with the data (e.g. Truth_Logs)')

parser.add_argument('--out', dest='outputDir', default='../elephant_dataset/Processed_data_new/',
     help='The output directory')
parser.add_argument('--train_dir', default='../elephant_dataset/Train_New', 
     help='Directory for the training chunks for the new data')
parser.add_argument('--test_dir', default='../elephant_dataset/Test_New',
     help='Directory for the test chunks for the new data')
parser.add_argument('--NFFT', type=int, default=3208, help='Window size used for creating spectrograms')
parser.add_argument('--hop', type=int, default=641, help='Hop size used for creating spectrograms')
parser.add_argument('--window', type=int, default=256, 
    help='Deterimes the window size in frames of the resulting spectrogram') # Default corresponds to 21s - should just check this! 
parser.add_argument('--max_f', dest='max_freq', type=int, default=150, help='Deterimes the maximum frequency band')
parser.add_argument('--pad', dest='pad_to', type=int, default=4096, 
    help='Deterimes the padded window size that we want to give a particular grid spacing (i.e. 1.95hz')
parser.add_argument('--test_size', type=float, default=0.1, help='Determines the relative size of the test set')
parser.add_argument('--neg_fact', type=int, default=1, 
    help="Determines number of negative samples to sample as neg_fact x (pos samples)")


np.random.seed(8)
VERBOSE = False

def generate_labels(labels, spectrogram_info, len_wav):
    '''
        Given ground truth label file 'label' create the full 
        segmentation labeling for a .wav file. Namely, return
        a vector containing a 0/1 labeling for each time slice
        corresponding to the .wav file transformed into a spectrogram.
        The key challenge here is that we want the labeling to match
        up with the corresponding spectrogram without actually creating
        the spectrogram 
    '''
    labelFile = csv.DictReader(open(labels,'rt'), delimiter='\t')
    # Formula is: frames = floor((wav - overlap) / hop)
    len_labels = math.floor((len_wav - (spectrogram_info['NFFT'] - spectrogram_info['hop'])) / spectrogram_info['hop'])
    labelMatrix = np.zeros(shape=(len_labels),dtype=int)

    samplerate = spectrogram_info['samplerate']
    # Iterates through labels and marks the segments with elephant calls
    for row in labelFile:
        # Use the file offset to determine the start of the call
        start_time = float(row['File Offset (s)'])
        call_length = float(row['End Time (s)']) - float(row['Begin Time (s)'])
        end_time = start_time + call_length
        
        # Figure out which spectrogram slices we are on
        # to get columns that we want to span with the given
        # slice. This math transforms .wav indeces to spectrogram
        # indices
        start_spec = max(math.ceil((start_time * samplerate - spectrogram_info['NFFT'] / 2.) / spectrogram_info['hop']), 0)
        end_spec = min(math.ceil((end_time * samplerate - spectrogram_info['NFFT'] / 2.) / spectrogram_info['hop']), labelMatrix.shape[0])
        labelMatrix[start_spec : end_spec] = 1

    return labelMatrix

def generate_empty_chunks(n, raw_audio, label_vec, spectrogram_info):
    """
        Generate n empty data chunks by uniformally sampling 
        time sections with no elephant calls present
    """
    # Step through the labels vector and collect the indeces from
    # which we can define a window with now elephant call
    # i.e. all start indeces such that the window (start, start + window_sz)
    # does not contain an elephant call
    valid_starts = []
    window_size = spectrogram_info['window']
    # Step backwards and keep track of how far away the
    # last elephant call was
    last_elephant = 0  # For now is the size of the window
    for i in range(label_vec.shape[0] - 1, -1, -1):
        last_elephant += 1

        # Check if we encounter an elephant call
        if (label_vec[i] == 1):
            last_elephant = 0

        # If we haven't seen an elephant call
        # for a chunk size than record this index
        if (last_elephant >= window_size):
            valid_starts.append(i)

    # Generate num_empty uniformally random 
    # empty chunks
    empty_features = []
    empty_labels = []
    NFFT = spectrogram_info['NFFT']
    samplerate = spectrogram_info['samplerate']
    hop = spectrogram_info['hop']
    max_freq = spectrogram_info['max_freq']
    pad_to = spectrogram_info['pad_to']

    for i in range(n):
        # Generate a valid empty start chunk
        # index by randomly sampling from our
        # ground truth labels
        start = np.random.choice(valid_starts)

        # Now we have to do a litle back conversion to get 
        # the raw audio index in raw audio frames
        # Number of hops in we are marks the first raw audio frame to use
        chunk_start = start * spectrogram_info['hop']
        chunk_size = (spectrogram_info['window'] - 1) * spectrogram_info['hop'] + spectrogram_info['NFFT'] 
        chunk_end = int(chunk_start + chunk_size)

        # Get the spectrogram chunk
        # Extract the spectogram
        [spectrum, freqs, t] = ml.specgram(raw_audio[chunk_start: chunk_end], 
                    NFFT=NFFT, Fs=samplerate, noverlap=(NFFT - hop), window=ml.window_hanning, pad_to=pad_to)
        
        # Cutout the high frequencies that are not of interest
        spectrum = spectrum[(freqs <= max_freq)]
        assert(spectrum.shape[1] == spectrogram_info['window'])

        data_labels = label_vec[start : start + spectrum.shape[1]]
        # Make sure that no call exists in the chunk
        assert(np.sum(data_labels) == 0)

        if VERBOSE:
            visualize(spectrum, labels=data_labels)

        # We want spectrograms to be time x freq
        spectrum = spectrum.T
        empty_features.append(spectrum)
        empty_labels.append(data_labels)

    return empty_features, empty_labels



def generate_chunk(start_time, end_time, raw_audio, truth_labels, spectrogram_info):
    '''
        Generate a data chunk around a given elephant call. The data
        chunk is of size "chunk_length" seconds and has the call
        of interest randomly placed inside the window

        Parameters:
        - start_time and end_time in seconds
    '''
    # Convert the times to .wav frames to help ensure
    # robustness of approach
    start_frame = int(math.floor(start_time * spectrogram_info['samplerate']))
    end_frame = int(math.ceil(end_time * spectrogram_info['samplerate']))
    # Convert from window size in spectrogram frames to raw audio size
    # Note we use the -1 term to force the correct number of frames
    # wav = frames * hop - hop + window ==> wav = frames * hop + overlap
    chunk_size = (spectrogram_info['window'] - 1) * spectrogram_info['hop'] + spectrogram_info['NFFT'] 

    # Padding to call
    call_length = end_frame - start_frame # In .wav frames
    padding_length = chunk_size - call_length
    # if padding_frame is neg skip call
    # but still want to go to the next!!
    if padding_length < 0:
        print ("skipping too long of call") # Maybe don't need this let us se
        return None, None
    
    # Randomly split the pad to before and after
    pad_front = np.random.randint(0, padding_length + 1)

    # Do some stuff to avoid the front and end!
    chunk_start = start_frame - pad_front
    chunk_end  = start_frame + call_length + (padding_length - pad_front)
    
    # Do some quick voodo - assume cant have issue where 
    # the window of 64 frames is lareger than the sound file!
    if (chunk_start < 0):
        # Amount to transfer to end
        chunk_start = 0
        chunk_end = chunk_size
    # See if we have passed the end of the sound file.
    # Note divide by sr to get sound file length in seconds
    if (chunk_end >= raw_audio.shape[0]):
        chunk_end = raw_audio.shape[0]
        chunk_start = sound_file.shape[0] - chunk_size

    assert(chunk_end - chunk_start == chunk_size)
    # Make sure the call is fully in the region
    assert(chunk_start <= start_frame and chunk_end >= end_frame)

    NFFT = spectrogram_info['NFFT']
    samplerate = spectrogram_info['samplerate']
    hop = spectrogram_info['hop']
    max_freq = spectrogram_info['max_freq']
    pad_to = spectrogram_info['pad_to']
    # Extract the spectogram
    [spectrum, freqs, t] = ml.specgram(raw_audio[chunk_start: chunk_end], 
                NFFT=NFFT, Fs=samplerate, noverlap=(NFFT - hop), window=ml.window_hanning, pad_to=pad_to)

    # Check our math
    assert(spectrum.shape[1] == spectrogram_info['window'])
    
    # Cutout the high frequencies that are not of interest
    spectrum = spectrum[(freqs <= max_freq)]
    # Get the corresponding labels
    # Calculate the relative start time w/r 
    # to the entire spectogram for the given chunk 
    start_spec = max(math.ceil((chunk_start - spectrogram_info['NFFT'] / 2.) / spectrogram_info['hop']), 0)
    end_spec = start_spec + spectrum.shape[1] 
    
    data_labels = truth_labels[start_spec: end_spec]

    if VERBOSE:
        visualize(spectrum, labels=data_labels)

    # We want spectrograms to be time x freq
    spectrum = spectrum.T
    return spectrum, data_labels

def generate_elephant_chunks(raw_audio, labels, label_vec, spectrogram_info):
    """ 
        Generate the data chunks for each elephant call in a given 
        audio recording, where a data chunk is defined as spectrogram
        of a given window size with the given call randomly placed within

        Parameters
        - raw_audio: Vector of raw 1D audio
        - labels: Dictionary representaiton of the truth labels csv
        - label_vec: Vector with 0/1 label for each spectrogram column
        - spectrogram_info: Spectrogram params
    """
    feature_set = []
    label_set = []
    # 2. Now iterate through label file, when find a call, pass to make chunk,
    for call in labels:
        start_time = float(call['File Offset (s)'])
        call_length = float(call['End Time (s)']) - float(call['Begin Time (s)'])
        end_time = start_time + call_length

        feature_chunk, label_chunk = generate_chunk(start_time, end_time, raw_audio, label_vec, spectrogram_info)
        
        if (feature_chunk is not None):  
            feature_set.append(feature_chunk)
            label_set.append(label_chunk)

    return feature_set, label_set


def extract_data_chunks(audio_file, label_file, spectrogram_info):
    """
        Extract an equal number of empty and elephant call
        data chunks from a raw audio file. Each data chunk
        is of a fixed size, and for data chunks with an 
        elephant call we make a chunk around a given reference
        call and then randomly place that call within the fram
    """
    samplerate, raw_audio = wavfile.read(audio_file)
    labels = csv.DictReader(open(label_file,'rt'), delimiter='\t')
    # Generate the spectrogram index labelings
    spectrogram_info['samplerate'] = samplerate
    label_vec = generate_labels(label_file, spectrogram_info, raw_audio.shape[0])

    feature_set, label_set = generate_elephant_chunks(raw_audio, labels, label_vec, spectrogram_info)

    empty_features, empty_labels = generate_empty_chunks(len(feature_set) * spectrogram_info['neg_fact'],
            raw_audio, label_vec, spectrogram_info)
    feature_set.extend(empty_features)
    label_set.extend(empty_labels)

    return feature_set, label_set

if __name__ == '__main__':
    args = parser.parse_args()
    dataDir = args.dataDir
    outputDir = args.outputDir
    test_size = args.test_size
    train_dir = args.train_dir
    test_dir = args.test_dir
    spectrogram_info = {'NFFT': args.NFFT,
                        'hop': args.hop,
                        'max_freq': args.max_freq,
                        'window': args.window, 
                        'pad_to': args.pad_to,
                        'neg_fact': args.neg_fact}

    print(args)

    # Create Train / Test split
    # Really should also include Val set, but
    # we will worry about this later
    # Collect the wav/txt file pairs that
    # we will then split
    # Iterate through all data directories
    allDirs = [];
    # Get the directories that contain the data files
    for (dirpath, dirnames, filenames) in os.walk(dataDir):
        allDirs.extend(dirnames);
        break

    # Iterate through all files with in data directories
    for dirName in allDirs:
        #Iterate through each dir and get files within
        currentDir = dataDir + '/' + dirName;
        for(dirpath, dirnames, filenames) in os.walk(dataDir+'/'+dirName):
            # Iterate through the files to create data/label 
            # pairs (i.e. (.wav, .txt))
            data_pairs = {}
            for eachFile in filenames:
                # Strip off the location and time tags
                tags = eachFile.split('_')
                data_id = tags[0] + '_' + tags[1]
                file_type = eachFile.split('.')[1]

                if (file_type not in ['wav', 'txt']):
                    continue

                # Insert the file name into the dictionary
                # with the file type tag for a given id
                if not data_id in data_pairs:
                    data_pairs[data_id] = {}

                data_pairs[data_id][file_type] = eachFile
                data_pairs[data_id]['id'] = data_id

    # data_pairs now contains all of the wav/txt data file pairs.
    # Let us create a list of these pairs to randomly split into
    # train/test or train/val/test or whatever we want

    file_pairs = [(pair['wav'], pair['txt'], pair['id']) for _, pair in data_pairs.items()]
    # Shuffle the files before train test split
    shuffle(file_pairs)

    # Some inherant issues even with this!
    split_index = math.ceil(len(file_pairs) * (1 - test_size))
    train_data_files = file_pairs[:split_index]
    test_data_files = file_pairs[split_index:]

    train_feature_set = []
    train_label_set = []
    print ("Making Train Set")
    print ("Size: ", len(train_data_files))

    def wrapper_processData(data_pair):
        audio_file = data_pair[0]
        label_file = data_pair[1]
        data_id = data_pair[2]

        #generate_whole_spectogram(currentDir + '/' + audio_file, outputDir, spectrogram_info)
        #quit()
        feature_set, label_set = extract_data_chunks(currentDir + '/' + audio_file, 
                                        currentDir + '/' + label_file, spectrogram_info)
        return feature_set, label_set

    if not VERBOSE:
        pool = multiprocessing.Pool()
        print('Multiprocessing on {} CPU cores'.format(os.cpu_count()))
        start_time = time.time()
        output = pool.map(wrapper_processData, train_data_files)
        print('Multiprocessed took {}'.format(time.time()-start_time))
        pool.close()
        for feature, label in output:
            train_feature_set.extend(feature)
            train_label_set.extend(label)
        print('Multiprocessed took {}'.format(time.time()-start_time))
    else:
        for pair in train_data_files:
            feature, label = wrapper_processData(pair)
            train_feature_set.extend(feature)
            train_label_set.extend(label)
    
    X_train = np.stack(train_feature_set)
    y_train = np.stack(train_label_set)

    test_feature_set = []
    test_label_set = []
    print ("Making Test Set")
    print ("Size: ", len(test_data_files))

    if not VERBOSE:
        pool = multiprocessing.Pool()
        print('Multiprocessing on {} CPU cores'.format(os.cpu_count()))
        start_time = time.time()
        output = pool.map(wrapper_processData, test_data_files)
        pool.close()
        for feature, label in output:
            test_feature_set.extend(feature)
            test_label_set.extend(label)
        print('Multiprocessed took {}'.format(time.time()-start_time))
    else:
        for file in test_data_files:
            feature_set, label_set = wrapper_processData(file)
            test_feature_set.extend(feature_set)
            test_label_set.extend(label_set)

    X_test = np.stack(test_feature_set)
    y_test = np.stack(test_label_set)

    print (X_train.shape, X_test.shape)
    print (y_train.shape, y_test.shape)

    train_dir += '/Neg_Samples_x' + str(arg.neg_fact)
    test_dir += '/Neg_Samples_x' + str(arg.neg_fact)

    if not os.path.isdir(train_dir):
        os.mkdir(train_dir)

    if not os.path.isdir(test_dir):
        os.mkdir(test_dir)

    np.save(train_dir + '/features.npy', X_train)
    np.save(train_dir + '/labels.npy', y_train)
    np.save(test_dir + '/features.npy', X_test)
    np.save(test_dir + '/labels.npy', y_test)


    '''
    # Iterate through all data directories
    allDirs = [];
    # Get the directories that contain the data files
    for (dirpath, dirnames, filenames) in os.walk(dataDir):
        allDirs.extend(dirnames);
        break

    feature_set = []
    label_set = []
    # Iterate through all files with in data directories
    for dirName in allDirs:
        #Iterate through each dir and get files within
        currentDir = dataDir + '/' + dirName;
        for(dirpath, dirnames, filenames) in os.walk(dataDir+'/'+dirName):
            # Iterate through the files to create data/label 
            # pairs (i.e. (.wav, .txt))
            data_pairs = {}
            for eachFile in filenames:
                # Strip off the location and time tags
                tags = eachFile.split('_')
                data_id = tags[0] + '_' + tags[1]
                file_type = eachFile.split('.')[1]

                if (file_type not in ['wav', 'txt']):
                    continue

                # Insert the file name into the dictionary
                # with the file type tag for a given id
                if not data_id in data_pairs:
                    data_pairs[data_id] = {}

                data_pairs[data_id][file_type] = eachFile
                data_pairs[data_id]['id'] = data_id
                
            # Create a list of (wav_file, label_file, id) tuples to be processed
            file_pairs = [(pair['wav'], pair['txt'], pair['id']) for _, pair in data_pairs.items()]
            # For each .flac file call processData()
            def wrapper_processData(data_pair):
                audio_file = data_pair[0]
                label_file = data_pair[1]
                data_id = data_pair[2]

                #generate_whole_spectogram(currentDir + '/' + audio_file, outputDir, spectrogram_info)
                #quit()
                feature_set, label_set = extract_data_chunks(currentDir + '/' + audio_file, 
                                                currentDir + '/' + label_file, spectrogram_info)
                return feature_set, label_set


            if not VERBOSE:
                pool = multiprocessing.Pool()
                print('Multiprocessing on {} CPU cores'.format(os.cpu_count()))
                start_time = time.time()
                output = pool.map(wrapper_processData, file_pairs)
                print('Multiprocessed took {}'.format(time.time()-start_time))
                pool.close()
                for feature, label in output:
                    feature_set.extend(feature)
                    label_set.extend(label)
                print('Multiprocessed took {}'.format(time.time()-start_time))
            else:
                for pair in file_pairs:
                    feature, label = wrapper_processData(pair)
                    feature_set.extend(feature)
                    label_set.extend(label)
                    

    # Last thing is to do train/val/test splits
    X_train = np.stack(feature_set)
    y_train = np.stack(label_set)
    print (X_train.shape)
    print (y_train.shape)
    np.save(outputDir + '/features.npy', X_train)
    np.save(outputDir + '/labels.npy', y_train)

    # Save the individual training files for visualization etc.
    for i in range(X_train.shape[0]):
        np.save(outputDir + '/features_{}'.format(i+1), X_train[i])
        np.save(outputDir + '/labels_{}'.format(i+1), y_train[i])
    '''


