from matplotlib import pyplot as plt
from matplotlib import mlab as ml
import numpy as np
import csv
import os
import librosa
import time
import multiprocessing
from scipy.io import wavfile

# Inputs
dataDir = './New_Data/Truth_Logs/' # Dir containing .wav and corresponding label files
outputDir = './Processed_data_new/' # Dir that will contain all of the output data

numFFT = 3208 #3208
hop_length = 1604 #641
FREQ_MAX = 150.


def processData(currentDir, outputDir, audioFileName, labelFileName, outputDataName, outputLabelName):
    # Read the .wav file
    samplerate, raw_audio = wavfile.read(currentDir + '/' + audioFileName)
    # Corresponding label file
    labelFile = csv.DictReader(open(currentDir+'/'+labelFileName,'rt'), delimiter='\t')
   
    # Setup time limits
    timePerFrame = 1. / samplerate
    print ('Time Per Frame:', timePerFrame)
    print (raw_audio.shape)
    
    [spectrum, freqs, t] = ml.specgram(raw_audio, NFFT=numFFT, Fs=samplerate, noverlap=(numFFT - hop_length), window=ml.window_hanning)
    print ('Spectrogram Shape:', spectrum.shape)
    #print (spectrum)
    #print (spectrum[freqs < FREQ_MAX, :].shape)
    #print (10*np.log10(spectrum[freqs<FREQ_MAX,:]))
    #quit()
    # Cut down the frequency to only contain less than Freq_max
    # The data above FREQ_MAX Hertz is irrelephant so get rid of it.
    # This does not seem quite right so we will re-do this
    spectrum = spectrum[(freqs < FREQ_MAX)]
    freqs = freqs[(freqs < FREQ_MAX)]
    
    # Determing time spacing of columns in Data Matrix
    # amount of time per colum
    # Not actually used
    timeSpacing = numFFT * timePerFrame
    #print (timeSpacing)

    
    labelMatrix = np.zeros(shape=(spectrum.shape[1]),dtype=int);
    
    # Iterate through labels, and changes labelMatrix, should an elephant call be present
    for row in labelFile:
        # Use the file offset to determine the start of the call
        start_time = float(row['File Offset (s)'])
        call_length = float(row['End Time (s)']) - float(row['Begin Time (s)'])
        end_time = start_time + call_length
        # Mark the end of the call with a single 1 
        # (Note when creating the dataset we can extrend how many 1s!
        # Additionally, in at labelMat[end, 1] mark how far back the call starts
         
        labelMatrix[(t >= start_time) & (end_time > t)] = 1

    # Save output files, spectrum (contains all frequency data) and labelMatrix
    # For now save as .csv so we can inspect to make sure it is right! Later consider using .npy
    np.savetxt(outputDir + outputDataName+'.csv', spectrum, delimiter=",")
    np.savetxt(outputDir + outputLabelName+'.csv', labelMatrix, delimiter=",")


if __name__ == '__main__':
    #    main()
    #def main():
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

                processData(currentDir, outputDir, audio_file, label_file,
                    'Data_' + data_id, 'Label_' + data_id)

                
            pool = multiprocessing.Pool()
            print('Multiprocessing on {} CPU cores'.format(os.cpu_count()))
            start_time = time.time()
            pool.map(wrapper_processData, file_pairs)
            print('Multiprocessed took {}'.format(time.time()-start_time))
            pool.close()


