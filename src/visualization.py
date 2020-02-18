"""
Methods for visualizing numpy arrays of the spectograms, outputs, and labels

Can run as a standalone function to visualize individual wav files.
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import csv
from matplotlib import mlab as ml
import math


parser = argparse.ArgumentParser()
parser.add_argument('wav', help='name of wav file to visualize') # may not want the -
parser.add_argument('labelWav', help='The label file for the corresponding wav')
parser.add_argument('--NFFT', type=int, default=3208, help='Window size used for creating spectrograms')
parser.add_argument('--hop', type=int, default=641, help='Hop size used for creating spectrograms')
parser.add_argument('--window', type=int, default=10, help='Deterimes the window size in seconds of the resulting spectrogram')


def visualize(features, outputs=None, labels=None, binary_preds=None, title=None, vert_lines=None):
    """
    Visualizes the spectogram and associated predictions/labels. 
    features is the entire spectrogram that will be visualized

    For now this just has placeholder plots for outputs and labels
    when they're not passed in. 

    Inputs are numpy arrays
    """
    if binary_preds is not None: # Add a forth plot that shows the binarized predictions
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1)
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(3,1)
        
    #new_features = np.flipud(10*np.log10(features).T)
    # TODO: Delete above line?
    new_features = features.T
    min_dbfs = new_features.flatten().mean()
    max_dbfs = new_features.flatten().mean()
    min_dbfs = np.maximum(new_features.flatten().min(),min_dbfs-2*new_features.flatten().std())
    max_dbfs = np.minimum(new_features.flatten().max(),max_dbfs+6*new_features.flatten().std())

    ax1.imshow(new_features, cmap="magma_r", vmin=min_dbfs, vmax=max_dbfs, interpolation='none', origin="lower", aspect="auto")
    
    if outputs is not None:
        ax2.plot(np.arange(outputs.shape[0]), outputs)
        ax2.set_ylim([0,1])
        ax2.axhline(y=0.5, color='r', linestyle='-')
        # Include vertical lines if we want to show what
        # call we are focusing on
        if vert_lines is not None:
            ax2.axvline(x=vert_lines[0], color='r', linestyle=':')
            ax2.axvline(x=vert_lines[1], color='r', linestyle=':')

    if binary_preds is not None:
        ax3.plot(np.arange(binary_preds.shape[0]), binary_preds)
        ax3.set_ylim([0,1])
        ax3.axhline(y=0.5, color='r', linestyle='-')
        # Include vertical lines if we want to show what
        # call we are focusing on
        if vert_lines is not None:
            ax3.axvline(x=vert_lines[0], color='r', linestyle=':')
            ax3.axvline(x=vert_lines[1], color='r', linestyle=':')

    if labels is not None:
        gt_ax = ax3 if binary_preds is None else ax4
        gt_ax.plot(np.arange(labels.shape[0]), labels)
        if vert_lines is not None:
            gt_ax.axvline(x=vert_lines[0], color='r', linestyle=':')
            gt_ax.axvline(x=vert_lines[1], color='r', linestyle=':')
    
    # Make the plot appear in a specified location on the screen
    mngr = plt.get_current_fig_manager()
    geom = mngr.window.geometry()  
    mngr.window.wm_geometry("+400+150")

    if title is not None:
        ax1.set_title(title)

    plt.show()


def visualize_predictions(calls, spectrogram, prediction_labels, gt_labels, chunk_size=256, label='True_Pos'):
    '''
        Visualize the predicted labels and gt labels for the calls provided.
        This is used to visualize the results of predictions on for example
        the full spectrogram. 

        Parameters:
        - calls: Assumed to be list of tuples of (start, end, len) where start
        and end are for now in spect frames
        - label: gives what the calls represent (i.e. true_pos, false pos, false_neg) 


        Down the road maybe we should include the % overlap
    '''
    for call in calls:
        start = call[0]
        end = call[1]
        length = call[2]

        # Let us position the call in the middle
        # Then visualize
        padding = (chunk_size - length) // 2
        window_start = max(start - padding, 0)
        window_end = min(end + padding, spectrogram.shape[0])
        visualize(spectrogram[window_start: window_end], 
            prediction_labels[window_start: window_end], gt_labels[window_start: window_end],
            title=label, vert_lines=(start - window_start, end - window_start))



def visualize_wav(wav, labels, spectrogram_info):
    """
        Given a wav file and corresponding label file.
        visualize the calls.
    """
    samplerate, raw_audio = wavfile.read(wav)
    labelFile = csv.DictReader(open(labels,'rt'), delimiter='\t')
    
    # Test length of the audio in hours
    len_audio = raw_audio.shape[0] / samplerate / 60. / 60.

    labelMatrix = generate_labels(labels, spectrogram_info, raw_audio.shape[0], samplerate)
    # Visualize each call:
    # Create a spectogram around each call that
    # Is twice as wide as the call
    NFFT = spectrogram_info['NFFT']
    hop = spectrogram_info['hop']
    window = spectrogram_info['window']
    for call in labelFile:
        start_time = float(call['File Offset (s)'])
        call_length = float(call['End Time (s)']) - float(call['Begin Time (s)'])
        end_time = start_time + call_length

        # Padding to call
        padding = max((window - call_length) / 2, 0)
        start_spec = max(start_time - padding, 0)
        end_spec = min(end_time + padding, raw_audio.shape[0] / samplerate) # Note we divide by sample rate to get into seconds

        print (end_spec - start_spec)
    
        # Convert the start and end times to the corresponding audio slicing
        start_spec *= samplerate
        end_spec *= samplerate

        # Extract the spectogram
        [spectrum, freqs, t] = ml.specgram(raw_audio[int(start_spec): int(end_spec)], 
                    NFFT=NFFT, Fs=samplerate, noverlap=(NFFT - hop), window=ml.window_hanning)

        # Cutout the high frequencies that are not of interest
        spectrum = spectrum[(freqs < 400)]
        print (spectrum.shape)
        # Get the corresponding labels
        # Calculate the relative start time w/r 
        # to the entire spectogram for the given chunk 
        start_spec = max(math.ceil((start_spec - spectrogram_info['NFFT'] / 2.) / spectrogram_info['hop']), 0)
        end_spec = start_spec + spectrum.shape[1] #min(math.ceil((end_spec - spectrogram_info['NFFT'] / 2.) / spectrogram_info['hop']), labelMatrix.shape[0])
        temp_labels = labelMatrix[start_spec: end_spec]
        
        new_features = 10*np.log10(spectrum.T) # Probably wrong
        visualize(new_features, labels=temp_labels)

def generate_labels(labels, spectrogram_info, len_wav, samplerate):
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

    # Iterate through labels, and changes labelMatrix, should an elephant call be present
    for row in labelFile:
        # Use the file offset to determine the start of the call
        start_time = float(row['File Offset (s)'])
        call_length = float(row['End Time (s)']) - float(row['Begin Time (s)'])
        end_time = start_time + call_length
        
        # Figure out which spectogram slices we are on
        # to get columns that we want to span with the given
        # slice. This math transforms .wav indeces to spectrogram
        # indices
        start_spec = max(math.ceil((start_time * samplerate - spectrogram_info['NFFT'] / 2.) / spectrogram_info['hop']), 0)
        end_spec = min(math.ceil((end_time * samplerate - spectrogram_info['NFFT'] / 2.) / spectrogram_info['hop']), labelMatrix.shape[0])
        labelMatrix[start_spec : end_spec] = 1

    return labelMatrix

def test_generate_labels(wav, labels, spectrogram_info):
    # Let us create a small spectogram and then test how
    # Create spectogram from first 40 seconds
    samplerate, raw_audio = wavfile.read(wav)
    labelFile = csv.DictReader(open(labels,'rt'), delimiter='\t')
    
    start_spec = 0
    end_spec = 40 * samplerate
    NFFT = spectrogram_info['NFFT']
    hop = spectrogram_info['hop']
    window = spectrogram_info['window']

    [spectrum, freqs, t] = ml.specgram(raw_audio[int(start_spec): int(end_spec)], 
                    NFFT=NFFT, Fs=samplerate, noverlap=(NFFT - hop), window=ml.window_hanning)
    print(spectrum.shape)

    # Iterate through labels, and changes labelMatrix, should an elephant call be present
    test_label_1 = np.zeros(shape=(spectrum.shape[1]),dtype=int)
    for row in labelFile:
        # Use the file offset to determine the start of the call
        start_time = float(row['File Offset (s)'])
        call_length = float(row['End Time (s)']) - float(row['Begin Time (s)'])
        end_time = start_time + call_length
        
        start = False
        end = False

        for i in range(len(t)):
            times = t[i]
            if (times >= start_time) and not start:
                print ('Start bucket', i, "time", times)
                start = True
            elif times > end_time and start:
                print ('End bucket', i, "time", times)
                break


        test_label_1[(t >= start_time) & (end_time > t)] = 1

    test_label_2 = generate_labels(labels, spectrogram_info, end_spec, samplerate)

    print (np.sum(test_label_1 - test_label_2))


def main():
    args = parser.parse_args()

    wavfile = args.wav
    labelWav = args.labelWav
    spectrogram_info = {'NFFT': args.NFFT,
                        'hop': args.hop, 
                        'window': args.window}

    visualize_wav(wavfile, labelWav, spectrogram_info)
    #test_generate_labels(wavfile, labelWav, spectrogram_info)



if __name__ == '__main__':
    main()