from matplotlib import pyplot as plt
from matplotlib import mlab as ml
import numpy as np
import soundfile as sf
import csv
import os
import librosa
import time
import multiprocessing

# Inputs
dataDir = './Data/' # Dir containing all raw data in seperate files like 'ceb1_XXXXXXX'
outputSpect = './Processed_data/' # Dir that will contain all of the output data
outputMel = './Processed_data_MFcc/'
activateLabel = 'activate/'
callLabel = 'call/'
#outputDirMel = './Processed_data_MFCC/'
#outputDirActivate = './Processed_data_activate/'
numFFT        = 512 # Number of points to do FFT over
hop_length    = 128 * 3 # Number of points between successive FFT 
timeStart     = 0.0 # Start time of file to being data generation
timeStop      = 0.0 # End time to look at, enter '0' to look at entire file
N_MELS        = 77  # Dimension of the features in the mel spectogram. Want to equal normal spect
FREQ_MAX      = 150.
USE_MEL       = False
ACTIVATE_LABELS = True


## Function that given .flac file, lable file and starting hour will generate spec and label data
def processData(dataDir,currentDir,outputDir,audioFileName,labelFileName,outputDataName,outputLabelName,startHour):
    ''' Can use to splice the file
    flac_file = sf.SoundFile(dataDir+'/'+currentDir+'/'+audioFileName)
    rate = flac_file.samplerate
    '''
    raw_audio, samplerate = sf.read(dataDir+'/'+currentDir+'/'+audioFileName) # Can use start / stop
    numFrames = raw_audio.shape[0]
    
    labelFile = csv.DictReader(open(dataDir+'/'+currentDir+'/'+labelFileName,'rt'))
   
    # Setup time limits
    #print (samplerate)
    timePerFrame = 1. / samplerate
    #print (timePerFrame)
    if not USE_MEL:
        [spectrum, freqs, t] = ml.specgram(raw_audio, NFFT=numFFT, Fs=samplerate)
        #print (spectrum.shape)
        #print (spectrum)
        #print (spectrum[freqs < FREQ_MAX, :].shape)
        #print (10*np.log10(spectrum[freqs<FREQ_MAX,:]))
        #quit()
        # The data above FREQ_MAX Hertz is irrelephant so get rid of it.
        # This does not seem quite right so we will re-do this
        spectrum = spectrum[(freqs < FREQ_MAX)]
        freqs = freqs[(freqs < FREQ_MAX)]
    else:
        # This is hacky but we want to get the times centered for each
        # spectogram frame
        spectrum, freqs, t = ml.specgram(raw_audio, NFFT=numFFT, Fs=samplerate)
        mel_spect = librosa.feature.melspectrogram(S=spectrum, n_mels=N_MELS)

        # Converst to db scale
        mel_db = (librosa.power_to_db(mel_spect, ref=np.max)) #+ 40)/40

        spectrum = mel_db
    
    # Determing time spacing of columns in Data Matrix
    # amount of time per colum
    # Not actually used
    timeSpacing = numFFT * timePerFrame
    #print (timeSpacing)

    # Initialize LabelVector
    if ACTIVATE_LABELS:
        # Add the extra dimension to help determine where the call starts
        labelMatrix = np.zeros((spectrum.shape[1], 2), dtype=int)
    else:
        labelMatrix = np.zeros(shape=(spectrum.shape[1]),dtype=int);
    
    # Iterate through labels, and changes labelMatrix, should an elephant call be present
    for row in labelFile:
        # Check whether the label File is pertaining to the correct data file
        if (int(row['begin_hour']) == startHour):
            # Change input times to be relative to file, by subtracting start hour
            # begin_time (s) - begin_hour * 3600 sec/hour
            relStartTime = float(row['begin_time']) - float(row['begin_hour'])*3600
            relEndTime = float(row['end_time']) - float(row    ['begin_hour'])*3600


            # Mark the end of the call with a single 1 
            # (Note when creating the dataset we can extrend how many 1s!
            # Additionally, in at labelMat[end, 1] mark how far back the call starts
            if ACTIVATE_LABELS: 
                # Find the column where the call ends
                # Each time column is spaced apart by .384s
                # = hop_size / sr
                #mask = (t >= relEndTime) & (t < relEndTime + 0.384)
                inCall = False
                start = -1
                for i in range(len(t)):
                    if not inCall and t[i] >= relStartTime:
                        inCall = True
                        start = i
                    elif inCall and t[i] > relEndTime:
                        labelMatrix[i, 0] = 1
                        # Keep track of the start of the call!
                        labelMatrix[i, 1] = start
                        break
            else: # Mark the location where the call is 
                labelMatrix[(t >= relStartTime) & (relEndTime > t)] = 1

    # Save output files, spectrum (contains all frequency data) and labelMatrix
    # For now save as .csv so we can inspect to make sure it is right! Later consider using .npy
    np.savetxt(outputDir + outputDataName+'.csv', spectrum, delimiter=",")
    np.savetxt(outputDir + outputLabelName+'.csv', labelMatrix, delimiter=",")

# Iterate through all data directories
allDirs = [];
# Get the directories that contain the data files
for (dirpath, dirnames, filenames) in os.walk(dataDir):
    allDirs.extend(dirnames);
    break

out_path = outputMel if USE_MEL else outputSpect
out_path += activateLabel if ACTIVATE_LABELS else callLabel
# Iterate through all files with in data directories
for dirName in allDirs:
    #Iterate through each dir and get files within
    tempCurrentDir = dirName;
    for(dirpath, dirnames, filenames) in os.walk(dataDir+'/'+dirName):
        #Iterate through each file within
        for eachFile in filenames:
            print(eachFile)
            #Get the label file, and the special ID for the data
            if eachFile.split('.')[1] == 'csv':
                tempLabelFile = eachFile;
                tempID = eachFile.split('_')[1]

        # For each .flac file call processData()
        def wrapper_processData(eachFile):
            if eachFile.split('.')[1] == 'flac':
                tempAudioFile = eachFile
                tempStartHour = int(eachFile.split('_')[2][0:2])

                processData(dataDir,tempCurrentDir,out_path,tempAudioFile,tempLabelFile,'Data_'+tempID+'_Hour'+str(tempStartHour),'Label_'+tempID+'_Hour'+str(tempStartHour),tempStartHour)
        pool = multiprocessing.Pool()
        print('Multiprocessing on {} CPU cores'.format(os.cpu_count()))
        start_time = time.time()
        pool.map(wrapper_processData, filenames)
        print('Multiprocessed took {}'.format(time.time()-start_time))
        pool.close()
            

                
