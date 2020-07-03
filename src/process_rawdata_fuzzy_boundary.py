from matplotlib import pyplot as plt
from matplotlib import mlab as ml
import numpy as np
import csv
import os
import time
import multiprocessing
from multiprocessing import Value
from scipy.io import wavfile
from visualization import visualize
import math
import argparse
from random import shuffle
import random
from functools import partial
import generate_spectrograms

"""
Example Runs Locally:

python process_rawdata_fuzzy_boundary.py --data_dirs ../elephant_dataset/New_Data/Train_Data_file/wav_files --out ../elephant_dataset/Train

"""

parser = argparse.ArgumentParser()

parser.add_argument('--data_dirs', dest='data_dirs', nargs='+', type=str,
    help='Provide the data_dirs with the files that you want to be processed')

# For use on quatro
#parser.add_argument('--data', dest='dataDir', default='/home/data/elephants/rawdata/raw_2018', 
#   type=str, help='The top level directory with the data (e.g. Truth_Logs)')

parser.add_argument('--out', dest='out_dir', default='../elephant_dataset/Train',
     help='The output directory for the processed files. We need to specify this!')
parser.add_argument('--NFFT', type=int, default=4096, help='Window size used for creating spectrograms') # was 3208!!
parser.add_argument('--hop', type=int, default=800, help='Hop size used for creating spectrograms') # was 641!!
parser.add_argument('--window', type=int, default=256, 
    help='Deterimes the window size in frames of the resulting spectrogram') # Default corresponds to 21s - should just check this! 
parser.add_argument('--max_f', dest='max_freq', type=int, default=150, help='Deterimes the maximum frequency band')
parser.add_argument('--pad', dest='pad_to', type=int, default=4096, 
    help='Deterimes the padded window size that we want to give a particular grid spacing (i.e. 1.95hz')
parser.add_argument('--neg_fact', type=int, default=1, 
    help="Determines number of negative samples to sample as neg_fact x (pos samples)")
parser.add_argument('--full_24_hr', action='store_true', 
    help="Determines whether to create all chunks or just ones corresponding to labels and negative samples")
parser.add_argument('--seed', type=int, default=8,
    help="Set the random seed used for creating the datasets. This is primarily important for determining the negative samples")
parser.add_argument('--call_repeats', type=int, default=10,
    help="For each call, determine the number of random positions that we sample for that call within its given chunk. Serves as a form of oversampling")
parser.add_argument('--fudge_factor', type=int, default=0, 
    help="Determines how lenient for the boarders of the elephant calls." \
    "We assume that the boarders are somewhat random and thus do not need to be exact."\
    "Default 0 means no boarder masks generated.")
parser.add_argument('--overlap_boundaries', action='store_true', 
    help="Flag indicating that we calculate 'fuzzy' call boundaries based on 0/1 boundaries rather than for each individual call." +\
    "This allows us to potentially deal with strange issues of overlapping calls.")

'''
parser.add_argument('--val_size', type=float, default=0.1, help='Determines the relative size of the val set if we are creating one')
parser.add_argument('--train_val', action='store_true', 
    help = 'Generate a train and validation split based on the training data')
parser.add_argument('--val_dir', dest='val_dir', default='../elephant_dataset/Val/',
     help='The output directory for the validation files. Only need this if creating validation files')
'''

VERBOSE = False

def generate_labels_fuzzy(labels, spectrogram_info, len_wav):
    '''
        Given ground truth label file 'label' create the full 
        segmentation labeling for a .wav file. Namely, return
        a vector containing a 0/1 labeling for each time slice
        corresponding to the .wav file transformed into a spectrogram.
        Also create a mask that contains the boarder regions around
        each call. These boarder regions we want to treat as "fudge"
        regions where we are less strict on the classifier. The
        variable fudge_factor specifies how many slices to "fudge"
        on each side of a call boarder.
    '''
    # Formula is: frames = floor((wav - overlap) / hop)
    len_labels = math.floor((len_wav - (spectrogram_info['NFFT'] - spectrogram_info['hop'])) / spectrogram_info['hop'])
    labelMatrix = np.zeros(shape=(len_labels),dtype=int)
    boarder_mask = np.zeros_like(labelMatrix)
    fudge_factor = spectrogram_info['fudge_factor']

    # In the case where there is actually no GT label file because no elephant
    # calls occred in the recording, we simply output all zeros
    if labels is None:
        return labelMatrix, boarder_mask

    labelFile = csv.DictReader(open(labels,'rt'), delimiter='\t')
    
    samplerate = spectrogram_info['samplerate']
    # Iterates through labels and marks the segments with elephant calls
    for row in labelFile:
        # Use the file offset to determine the start of the call
        start_time = float(row['File Offset (s)'])
        call_length = float(row['End Time (s)']) - float(row['Begin Time (s)'])
        end_time = start_time + call_length
        
        # This math transforms .wav indeces to spectrogram indices
        start_spec = max(math.ceil((start_time * samplerate - spectrogram_info['NFFT'] / 2.) / spectrogram_info['hop']), 0)
        end_spec = min(math.ceil((end_time * samplerate - spectrogram_info['NFFT'] / 2.) / spectrogram_info['hop']), labelMatrix.shape[0])
        labelMatrix[start_spec : end_spec] = 1

        length = end_spec - start_spec
        # Calculate boarder boundaries for case where we 
        # are doing per call boarders
        if spectrogram_info['individual_boarders']: 
            # Fudge_factor should be chosen such that it is << len call.
            assert(length > fudge_factor)
            # Make sure to not go over the ends of the file
            # Add 1 to the right fudge idx.
            fudge_start_left = max(0, start_spec - fudge_factor)
            fudge_start_right = start_spec + fudge_factor
            fudge_end_left = end_spec - fudge_factor
            fudge_end_right = min(boarder_mask.shape[0], end_spec + fudge_factor)

            boarder_mask[fudge_start_left: fudge_start_right] = 1
            boarder_mask[fudge_end_left: fudge_end_right] = 1

    # Calculate call boundaries for the case where we merge
    # overlapping calls
    if not spectrogram_info['individual_boarders']:
        in_call = False
        for i in range(labelMatrix.shape[0]):
            if not in_call and labelMatrix[i] == 1:
                in_call = True
                fudge_start_left = max(0, i - fudge_factor)
                fudge_start_right = i + fudge_factor
                boarder_mask[fudge_start_left: fudge_start_right] = 1
            elif in_call and labelMatrix[i] == 0:
                in_call = False
                # Note we back the index up by 1
                fudge_end_left = (i-1) - fudge_factor
                fudge_end_right = min(boarder_mask.shape[0], (i-1) + fudge_factor)
                boarder_mask[fudge_end_left: fudge_end_right] = 1


    return labelMatrix, boarder_mask



def generate_empty_chunks(n, raw_audio, label_vec, boundary_mask_vec, spectrogram_info):
    """
        Generate n empty data chunks by uniformally sampling 
        time sections with no elephant calls present
    """
    # Step through the labels vector and collect the indeces from
    # which we can define a window with no elephant call
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
    empty_boundary_masks = []
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
        boundary_mask = boundary_mask_vec[start : start + spectrum.shape[1]]
        assert(np.sum(boundary_mask) == 0)

        if VERBOSE:
            new_features = 10*np.log10(spectrum)
            visualize(new_features.T, labels=data_labels)

        # We want spectrograms to be time x freq
        spectrum = spectrum.T
        empty_features.append(spectrum)
        empty_labels.append(data_labels)
        empty_boundary_masks.append(boundary_mask)

    return empty_features, empty_labels, empty_boundary_masks



def generate_chunk(start_time, end_time, raw_audio, truth_labels, boundary_mask_vec, spectrogram_info):
    '''
        Generate a data chunk around a given elephant call. The data
        chunk is of size "chunk_length" seconds and has the call
        of interest randomly placed inside the window

        Parameters:
        - start_time and end_time in seconds
        - truth_labels: Gives the ground truth elephant call labelings
        - boundary_mask_vec: Gives the location of the "fuzzy" boundary regions around each call
        where we want to allow for flexability in prediction
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
        chunk_start = raw_audio.shape[0] - chunk_size

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
    boundary_mask = boundary_mask_vec[start_spec: end_spec]

    if VERBOSE:
        new_features = 10*np.log10(spectrum)
        visualize(new_features.T, labels=data_labels, boundaries=boundary_mask)

    # We want spectrograms to be time x freq
    spectrum = spectrum.T
    return spectrum, data_labels, boundary_mask

def generate_elephant_chunks(raw_audio, labels, label_vec, boundary_mask_vec, spectrogram_info):
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
    boundary_mask_set = []
    # 2. Now iterate through label file, when find a call, pass to make chunk,
    for call in labels:
        start_time = float(call['File Offset (s)'])
        call_length = float(call['End Time (s)']) - float(call['Begin Time (s)'])
        end_time = start_time + call_length

        for _ in range(spectrogram_info['num_random_call_positions']):
            feature_chunk, label_chunk, boundary_mask = generate_chunk(start_time, end_time, raw_audio, label_vec, boundary_mask_vec, spectrogram_info)
            
            if (feature_chunk is not None):  
                feature_set.append(feature_chunk)
                label_set.append(label_chunk)
                boundary_mask_set.append(boundary_mask)

    return feature_set, label_set, boundary_mask_set

def generate_data_chunks(audio_file, label_file, spectrogram_info, num_neg=0):
    """
        Extract either elephant calls from the audio file or
        negative examples. If num_neg = 0 then extract the 
        elephant calls, otherwise extract num_neg negative
        samples from the audio file.

        Return:
        feature_set - a list of the physical spectogram chunks
        label_set - a list of the accompanying gt labels
        boundary_mask_set - a list of the "fuzzy" boundary masks for each chunk 
    """
    # Just for the bai elephants
    # Need to loook into this
    try:
        samplerate, raw_audio = wavfile.read(audio_file)
        print ("File size", raw_audio.shape)
    except:
        print("FILE Failed", audio_file)
    # No calls to extract
    if label_file is None and num_neg == 0:
        return [], [], []
    elif label_file is not None and num_neg == 0: # We have a calls to extract
        labels = csv.DictReader(open(label_file,'rt'), delimiter='\t')

    # Generate the spectrogram index labelings
    spectrogram_info['samplerate'] = samplerate
    label_vec, boundary_mask_vec = generate_labels_fuzzy(label_file, spectrogram_info, raw_audio.shape[0])

    if num_neg == 0:
        print ("Extracting elephant calls from", audio_file)
        feature_set, label_set, boundary_mask_set = generate_elephant_chunks(raw_audio, labels, label_vec, boundary_mask_vec, spectrogram_info)
    else:
        print ("Extracting negative chuncks from", audio_file)
        # Note the boundary_mask_set here is just np.zeros!
        feature_set, label_set, boundary_mask_set = generate_empty_chunks(num_neg, raw_audio, label_vec, boundary_mask_vec, spectrogram_info)

    return feature_set, label_set, boundary_mask_set



if __name__ == '__main__':
    args = parser.parse_args()
    data_dirs = args.data_dirs
    out_dir = args.out_dir
    spectrogram_info = {'NFFT': args.NFFT,
                        'hop': args.hop,
                        'max_freq': args.max_freq,
                        'window': args.window, 
                        'pad_to': args.pad_to,
                        'neg_fact': args.neg_fact,
                        'fudge_factor': args.fudge_factor,
                        'num_random_call_positions': args.call_repeats,
                        'individual_boarders': not args.overlap_boundaries}

    print(args)

    np.random.seed(args.seed)
    random.seed(args.seed) 

    # Collect the wav/txt file pairs 
    # Iterate through all files with in data directories
    data_pairs = {}
    for currentDir in data_dirs:
        #Iterate through each dir and get files within
        for(dirpath, dirnames, filenames) in os.walk(currentDir):
            # Create data/label 
            # pairs (i.e. (.wav, .txt))
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
                    data_pairs[data_id]['id'] = data_id
                    data_pairs[data_id]['dir'] = currentDir

                data_pairs[data_id][file_type] = eachFile

    # data_pairs now contains all of the wav/txt data file pairs.
    # Let us create a list of these pairs to randomly split into
    # train/val if the flag is set
   
    # Some wav files do not have a gt because there are no calls!
    file_pairs = []
    for _, pair in data_pairs.items():
        # We must be careful in the case where there is no
        # label file basically there are no calls!
        if 'wav' in pair:
            gt_labels = None if 'txt' not in pair else pair['txt']
            file_pairs.append(((pair['wav'], gt_labels, pair['id'], pair['dir'])))



    if args.full_24_hr:
        print("Processing all 24 hours and saving chunks from them")

        spectrogram_info['samplerate'] = 8000

        def wrapper_processFull24Hours(directory, data_pair):
            """
            This worker function is called on every data sample
            """
            audio_file = data_pair[0]
            label_file = data_pair[1]
            data_id = data_pair[2]
            curren_dir = data_pair[3]

            # Catch case where no calls exist so the gt file does not
            label_path = curren_dir + '/' + label_file if label_file is not None else None
            full_24_hr_spectogram = generate_spectrograms.generate_whole_spectogram(curren_dir + '/' + audio_file, spectrogram_info, "-1")
            labels = generate_spectrograms.generate_labels(label_path, spectrogram_info, full_24_hr_spectogram.shape[1])
            print("Shapes of full spectrograms and labels")
            print(full_24_hr_spectogram.shape)
            print(labels.shape)

            # Save the individual files seperately for each location!
            num_chunks = 0
            for i in range(math.floor(full_24_hr_spectogram.shape[1] / 256)):
                feature = full_24_hr_spectogram[:, i * 256:(i + 1) * 256]
                label = labels[i * 256:(i + 1) * 256]

                if feature.shape[1] != 256:
                    print("MAJOR PROBLEMSSSS WHY DOESNT MULTIPROCESSING SURFACE ERRORS")
                assert feature.shape[1] == 256
                assert label.shape[0] == 256

                np.save(directory + '/' + data_id + "_features_" + str(i), feature)
                np.save(directory + '/' + data_id + "_labels_" + str(i), label)
                call_counter.value += 1
                num_chunks += 1
            
            print("Saved successfully {} chunks.".format(num_chunks))

        out_dir += '/Full_24_hrs'
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)


        # Process Given Data pairs.
        print ("Processing full 24 hours examples")
        print ("Num Files: ", len(file_pairs))

        call_counter = Value("i", 0) # Shared thread variable to count the number of postive call examples
        pool = multiprocessing.Pool(20)
        print('Multiprocessing')
        start_time = time.time()
        pool.map(partial(wrapper_processFull24Hours, out_dir), file_pairs)
        print('Multiprocessed took {}'.format(time.time()-start_time))
        pool.close()
        print('Multiprocessed took {}'.format(time.time()-start_time))

        quit()

    def wrapper_processPos(directory, data_pair):
        """
        This worker function is called on every data sample
        """
        audio_file = data_pair[0]
        label_file = data_pair[1]
        data_id = data_pair[2]
        curren_dir = data_pair[3]

        # Catch case where no calls exist so the gt file does not
        label_path = curren_dir + '/' + label_file if label_file is not None else None
        feature_set, label_set, boundary_mask_set = generate_data_chunks(curren_dir + '/' + audio_file, label_path, spectrogram_info)
        call_counter.value += len(feature_set)

        # Save the individual files seperately for each location!
        for i in range(len(feature_set)):
            np.save(directory + '/' + data_id + "_features_" + str(i), feature_set[i])
            np.save(directory + '/' + data_id + "_labels_" + str(i), label_set[i])
            # Only save boundary files if flag is set
            if spectrogram_info['fudge_factor'] > 0:
                np.save(directory + '/' + data_id + "_boundary-masks_" + str(i), boundary_mask_set[i])


    # Add the random seed, the fudge factor, and the number of repeated calls
    # test this out
    out_dir += '/Neg_Samples_x' + str(args.neg_fact) + '_Seed_' + str(args.seed) + '_CallRepeats_' + str(args.call_repeats)
    if args.fudge_factor > 0:
        out_dir += '_FudgeFact_' + str(args.fudge_factor) + '_Individual-Boarders_' + str(spectrogram_info['individual_boarders'])
    
    #out_dir += '/Neg_Samples_x' + str(args.neg_fact) + '_Seed_' \
    #            + str(args.seed) + '_FudgeFact_' + str(args.fudge_factor) + '_CallRepeats_' + str(args.call_repeats)

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    # Process Given Data pairs.
    print ("Processing Positive examples")
    print ("Num Files: ", len(file_pairs))

    
    call_counter = Value("i", 0) # Shared thread variable to count the number of postive call examples
    pool = multiprocessing.Pool()
    print('Multiprocessing on {} CPU cores'.format(os.cpu_count()))
    start_time = time.time()
    pool.map(partial(wrapper_processPos, out_dir), file_pairs)
    print('Multiprocessed took {}'.format(time.time()-start_time))
    pool.close()
    print('Multiprocessed took {}'.format(time.time()-start_time))
    
    # For visualizing don't multi process
    # Let us not multip process this!!!
    #for file_pair in file_pairs:
    #    wrapper_processPos(out_dir, file_pair)
    

    num_calls = call_counter.value
    num_sound_files = len(file_pairs)

    num_neg_samples = spectrogram_info['neg_fact'] * num_calls
    samples_per_file = int(math.ceil(num_neg_samples / float(num_sound_files)))
    print ("Num calls:", num_calls)
    print("Num neg:", num_neg_samples)
    print("samples per:", samples_per_file)

    def wrapper_processNeg(directory, num_negative, data_pair):
        """
        This worker function is called on every data sample
        """
        audio_file = data_pair[0]
        label_file = data_pair[1]
        data_id = data_pair[2]
        curren_dir = data_pair[3]

        # Catch case where no calls exist so the gt file does not
        label_path = curren_dir + '/' + label_file if label_file is not None else None
        feature_set, label_set, boundary_mask_set = generate_data_chunks(curren_dir + '/' + audio_file, 
                                        label_path, spectrogram_info, num_negative)

        # Save the individual files seperately for each location!
        for i in range(len(feature_set)):
            np.save(directory + '/' + data_id + "_neg-features_" + str(i), feature_set[i])
            np.save(directory + '/' + data_id + "_neg-labels_" + str(i), label_set[i])
            if spectrogram_info['fudge_factor'] > 0:
                np.save(directory + '/' + data_id + "_neg-boundary-masks_" + str(i), boundary_mask_set[i])

    # Generate num_neg_samples negative examples where we randomly 
    # sample "samples_per_file" examples from each file
    print ("Processing Positive examples")
    print ("Size: ", len(file_pairs))
    pool = multiprocessing.Pool()
    print('Multiprocessing on {} CPU cores'.format(os.cpu_count()))
    start_time = time.time()
    pool.map(partial(wrapper_processNeg, out_dir, samples_per_file), file_pairs)
    print('Multiprocessed took {}'.format(time.time()-start_time))
    pool.close()
    print('Multiprocessed took {}'.format(time.time()-start_time))
    
    # Save which files were used for main data files
    with open(out_dir + '/files.txt', 'w') as f:
        for i in range(len(file_pairs)):
            file = file_pairs[i]
            # write the id of each file pair (wav/txt)
            # along with which time period it came from
            # e.g. jan/id
            dirs = file[3].split('/')
            time_tag = dirs[-1]

            path = time_tag + '/' + file[2]
            if i != len(file_pairs) - 1:
                path += '\n'

            f.write(path)

