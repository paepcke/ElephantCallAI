from matplotlib import pyplot as plt
from matplotlib import mlab as ml
import numpy as np
import csv
import os
import librosa
import time
import multiprocessing
from scipy.io import wavfile
from visualization import visualize
import math

seed = 8
np.random.seed(seed)

# Inputs
dataDir = '../elephant_dataset/New_Data/Truth_Logs/' # Dir containing .wav and corresponding label files
outputDir = '../elephant_dataset/Processed_data_new/' # Dir that will contain all of the output data

numFFT = 4096 # was 3208 # We want a frequency resolution of 1.95 Hz
hop_length = 641
FREQ_MAX = 100.

CHUNK_LENGTH = 21 # Will make this a parameter to pass in! - Use power of two frames
VERBOSE = True
num_empty = 48

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
    len_labels = math.ceil((len_wav - spectrogram_info['NFFT']) / spectrogram_info['hop'])
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
        #end_spec = start_spec + math.ceil((call_length * samplerate - (spectrogram_info['NFFT']))/ spectrogram_info['hop'])
        end_spec = min(math.ceil((end_time * samplerate - spectrogram_info['NFFT'] / 2.) / spectrogram_info['hop']), labelMatrix.shape[0])
        #print ("Believed Start:", start_spec, "Test end:", end_spec)
        labelMatrix[start_spec : end_spec] = 1

    return labelMatrix

def generate_empty_chunks(raw_audio, label_file, spectrogram_info):
    
    samplerate, raw_audio = wavfile.read(raw_audio)
    spectrogram_info['samplerate'] = samplerate
    labels = csv.DictReader(open(label_file,'rt'), delimiter='\t')

    # Step through the labels vector and collect the indeces from
    # which we can define a window with now elephant call
    # i.e. all start indeces such that the window (start, start + window_sz)
    # does not contain an elephant call
    label_vec = generate_labels(label_file, spectrogram_info, raw_audio.shape[0])

    valid_starts = []
    # Step backwards and keep track of the last
    # elephant call seen
    last_elephant = 245  # For now is the size of the window
    for i in range(label_vec.shape[0] - 1, -1, -1):
        last_elephant += 1
        # If we haven't seen an elephant call
        # for a chunk size than record this index
        if (last_elephant >= 245):
            valid_starts.append(i)
            last_elephant = 0

    # Generate num_empty uniformally random 
    # empty chunks
    empty_chunks = []
    for i in range(num_empty):
        # Generate a valid empty start chunk
        # index by randomly sampling from our
        # ground truth labels
        start = np.random.choice(valid_starts)

        # Now we have to do a litle back conversion to get 
        # the raw audio index in raw audio frames
        chunk_start = int(start * spectrogram_info['hop'] + spectrogram_info['NFFT'] / 2.)
        chunk_end = int(chunk_start + CHUNK_LENGTH * spectrogram_info['samplerate'])

        # Get the spectrogram chunk
        NFFT = spectrogram_info['NFFT']
        samplerate = spectrogram_info['samplerate']
        hop = spectrogram_info['hop']
        max_freq = spectrogram_info['max_freq']
        # Extract the spectogram
        [spectrum, freqs, t] = ml.specgram(raw_audio[chunk_start: chunk_end], 
                    NFFT=NFFT, Fs=samplerate, noverlap=(NFFT - hop), window=ml.window_hanning)
        
        # Cutout the high frequencies that are not of interest
        spectrum = spectrum[(freqs <= max_freq)]
        print (spectrum.shape)

        data_labels = label_vec[start : start + spectrum.shape[1]]

        if VERBOSE:
            visualize(spectrum, labels=data_labels)

        empty_chunks.append((spectrum, data_labels))

    return empty_chunks



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
    chunk_size = int(CHUNK_LENGTH * spectrogram_info['samplerate'])

    # Padding to call
    call_length = end_frame - start_frame # In .wav frames
    # Figure out how much to go before and after in seconds. 
    padding_length = chunk_size - call_length
    # if padding_frame is neg skip call
    # but still want to go to the next!!
    if padding_length < 0:
        print ("skipping too long of call") # Maybe don't need this let us se
    
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

    if (chunk_end - chunk_start != chunk_size):
        print ("fuck")
        quit()

    NFFT = spectrogram_info['NFFT']
    samplerate = spectrogram_info['samplerate']
    hop = spectrogram_info['hop']
    max_freq = spectrogram_info['max_freq']
    # Extract the spectogram
    [spectrum, freqs, t] = ml.specgram(raw_audio[chunk_start: chunk_end], 
                NFFT=NFFT, Fs=samplerate, noverlap=(NFFT - hop), window=ml.window_hanning)

    # Cutout the high frequencies that are not of interest
    spectrum = spectrum[(freqs <= max_freq)]
    print(spectrum.shape)
    # Get the corresponding labels
    # Calculate the relative start time w/r 
    # to the entire spectogram for the given chunk 
    start_spec = max(math.ceil((chunk_start - spectrogram_info['NFFT'] / 2.) / spectrogram_info['hop']), 0)
    end_spec = start_spec + spectrum.shape[1] 
    
    data_labels = truth_labels[start_spec: end_spec]

    if VERBOSE:
        visualize(spectrum, labels=data_labels)

    # Note we want axis 0 to be sound and axis 1 to be freq?
    return spectrum.T, data_labels

def extract_data_chunks(audio_file, label_file, spectrogram_info):
    samplerate, raw_audio = wavfile.read(audio_file)
    labels = csv.DictReader(open(label_file,'rt'), delimiter='\t')
    # Generate the spectrogram index labelings
    spectrogram_info['samplerate'] = samplerate
    label_vec = generate_labels(label_file, spectrogram_info, raw_audio.shape[0])

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

if __name__ == '__main__':
    # Processing parameters, later we should be using 
    # arg parser
    spectrogram_info = {'NFFT': numFFT,
                        'hop': hop_length,
                        'max_freq': FREQ_MAX}
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
                
            # Create a list of (wav_file, label_file, id) tuples to be processed
            file_pairs = [(pair['wav'], pair['txt'], pair['id']) for _, pair in data_pairs.items()]
            print (file_pairs)
            # For each .flac file call processData()
            def wrapper_processData(data_pair):
                audio_file = data_pair[0]
                label_file = data_pair[1]
                data_id = data_pair[2]

                #empty = generate_empty_chunks(currentDir + '/' + audio_file, 
                                                #currentDir + '/' + label_file, spectrogram_info)
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
                    train_feature_set.extend(feature)
                    train_label_set.extend(label)
                print('Multiprocessed took {}'.format(time.time()-start_time))
            else:
                for pair in file_pairs:
                    feature_set, label_set = wrapper_processData(pair)
                    #train_feature_set.extend(feature_set)
                    #train_label_set.extend(label_set)
                


