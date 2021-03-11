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
parser.add_argument('--NFFT', type=int, default=4096, help='Window size used for creating spectrograms')
parser.add_argument('--hop', type=int, default=800, help='Hop size used for creating spectrograms')
parser.add_argument('--window', type=int, default=22    , help='Deterimes the window size in seconds of the resulting spectrogram')


def visualize(features, model_predictions, ground_truth, title=None, vert_lines=None, times=None):
    """
        Provides a visualization of a given spectrogram slice and corresponding
        model predictions for that time slice. The bottom plot will always represent
        the ground truth label predictions. The plots then in-between can represent
        either binary vs. smoothed model predictions or different model predictions
        such as model_0 vs. model_1 in the hierarchical model.

        Inputs:
        featuers - The spectrogram window
        model_predictions - a list of model predictions of different types for the given window
        ground_truth - the labels corresponding to the GT call
        vert_lines - vertical lines representing what we want to highlight
        times - the time in seconds that correspond with the spectrogram window
    """
    fig, axes = plt.subplots(2 + len(model_predictions), 1)

    new_features = features.T
    min_dbfs = new_features.flatten().mean()
    max_dbfs = new_features.flatten().mean()
    min_dbfs = np.maximum(new_features.flatten().min(),min_dbfs-2*new_features.flatten().std())
    max_dbfs = np.minimum(new_features.flatten().max(),max_dbfs+6*new_features.flatten().std())

    if times is not None:
        # Adjust the x and y axis ticks to have time on the x axis and freq on the y axis (Note we have freq up to 150)!
        axes[0].imshow(new_features, cmap="magma_r", vmin=min_dbfs, vmax=max_dbfs, 
                interpolation='none', origin="lower", aspect="auto", extent=[times[0], times[times.shape[0] - 1], 0, 150])
    else:
        axes[0].imshow(new_features, cmap="magma_r", vmin=min_dbfs, vmax=max_dbfs, 
                interpolation='none', origin="lower", aspect="auto")

    # Show the predictions / GT labels
    for i in range(1, len(model_predictions) + 2):
        # Get the needed variables
        ax = axes[i]
        data = model_predictions[i - 1] if i < len(model_predictions) + 1 else ground_truth
        # Do some plotting
        if times is not None:
            ax.plot(times, data)
        else:
            ax.plot(np.arange(data.shape[0]), data)

        ax.set_ylim([0,1])
        # Toss them all in for now
        ax.axhline(y=0.5, color='r', linestyle='-')
        # Include vertical lines if we want to show what
        # call we are focusing on
        if vert_lines is not None:
            if times is not None:
                ax.axvline(x=times[vert_lines[0]], color='r', linestyle=':')
                ax.axvline(x=times[vert_lines[1]], color='r', linestyle=':')
            else:
                ax.axvline(x=vert_lines[0], color='r', linestyle=':')
                ax.axvline(x=vert_lines[1], color='r', linestyle=':')

    # Make the plot appear in a specified location on the screen
    if plt.get_backend() == "TkAgg":
        mngr = plt.get_current_fig_manager()
        geom = mngr.window.geometry()  
        mngr.window.wm_geometry("+400+150")

    if title is not None:
        axes[0].set_title(title)

    plt.show()



def spect_frame_to_time(frame, NFFT=4096, hop=800, samplerate=8000):
    """
        Given the index of a spectrogram frame compute 
        the corresponding time in seconds for the middle of the 
        window.
    """
    middle_s = (float(NFFT) / samplerate / 2.)
    frame_s = middle_s + frame * (float(hop) / samplerate)
    
    return frame_s
    
    
def spect_call_to_time(call, NFFT=4096, hop=800):
    """
        Given a elephant call prediction of the form
        (spect start, spect end, length), convert
        the spectrogram frames to time in seconds
    """
    begin, end, length = call
    begin_s = spect_frame_to_time(begin)
    end_s = spect_frame_to_time(end)
    length_s = begin_s - end_s

    return (begin_s, end_s, length_s)

"""
def visualize_predictions(calls, spectrogram, prediction_labels_binary, prediction_labels_smoothed, 
                                gt_labels, model_0_predictions=None, chunk_size=256, label='True_Pos', times=None):
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
        start, end, length = call
        start_s, end_s, _ = spect_call_to_time(call)
        print ("Visualizing Call Prediction ({}, {})".format(start_s, end_s))

        # Let us position the call in the middle
        # Then visualize
        padding = (chunk_size - length) // 2
        window_start = max(start - padding, 0)
        window_end = min(end + padding, spectrogram.shape[0])
        print (spectrogram.shape)
        # Only include times if provided
        if times is not None:
            window_time = times[window_start:window_end]

        # Only include model_0 predictions for hierarchical model
        if model_0_predictions is not None:
            model_0_window = model_0_predictions[window_start: window_end]

        visualize(spectrogram[window_start: window_end], outputs=prediction_labels_smoothed[window_start:window_end], 
            labels=gt_labels[window_start: window_end], binary_preds=prediction_labels_binary[window_start: window_end],
            model_0_predictions=model_0_window, title=label, 
            vert_lines=(start - window_start, end - window_start), times=window_time)
"""

def visualize_predictions(calls, spectrogram, model_predictions, gt_labels, chunk_size=256, label='True_Pos', times=None):
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
        start, end, length = call
        start_s, end_s, _ = spect_call_to_time(call)
        print ("Visualizing Call Prediction ({}, {})".format(start_s, end_s))

        # Let us position the call in the middle
        # Then visualize
        padding = (chunk_size - length) // 2
        window_start = max(start - padding, 0)
        window_end = min(end + padding, spectrogram.shape[0])
        print (spectrogram.shape)
        # Only include times if provided
        window_time = None
        if times is not None:
            window_time = times[window_start:window_end]

        # Get just the correct time windows!
        model_prediction_windows = []
        for i in range(len(model_predictions)):
            model_prediction_windows.append(model_predictions[i][window_start: window_end])

        visualize(spectrogram[window_start: window_end], model_prediction_windows, gt_labels[window_start: window_end],
                    title=label, vert_lines=(start - window_start, end - window_start), times=window_time)

        


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
    window = spectrogram_info['window'] * 2
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
        spectrum = spectrum[(freqs < 150)]
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


def main(args=None):
    if args == None:
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