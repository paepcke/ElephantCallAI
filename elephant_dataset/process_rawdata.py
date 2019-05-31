from matplotlib import pyplot as plt
from matplotlib import mlab as ml
import numpy as np
import soundfile as sf
import csv
import os

# Inputs
dataDir = './Data/' # Dir containing all raw data in seperate files like 'ceb1_XXXXXXX'
outputDir = './Processed_data/' # Dir that will contain all of the output data
numFFT        = 512 # Number of points to do FFT over
timeStart     = 0.0 # Start time of file to being data generation
timeStop      = 0.0 # End time to look at, enter '0' to look at entire file
FREQ_MAX = 150.


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

    [spectrum, freqs, t] = ml.specgram(raw_audio, NFFT=numFFT, Fs=samplerate)

    #print (spectrum.shape)
    # The data above 100 Hertz is irrelephant so get rid of it.
    # This does not seem quite right so we will re-do this
    spectrum = spectrum[(freqs < FREQ_MAX)]
    freqs = freqs[(freqs < FREQ_MAX)]
    
    # Determing time spacing of columns in Data Matrix
    # amount of time per colum
    # Not actually used
    timeSpacing = numFFT * timePerFrame
    #print (timeSpacing)

    # Initialize LabelVector
    labelMatrix = np.zeros(shape=(len(t)),dtype=int);
    
    # Iterate through labels, and changes labelMatrix, should an elephant call be present
    for row in labelFile:
        # Check whether the label File is pertaining to the correct data file
        if (int(row['begin_hour']) == startHour):
            # Change input times to be relative to file, by subtracting start hour
            # begin_time (s) - begin_hour * 3600 sec/hour
            relStartTime = float(row['begin_time']) - float(row['begin_hour'])*3600
            relEndTime = float(row['end_time']) - float(row    ['begin_hour'])*3600

            # Mark the location where the call is
            labelMatrix[(t > relStartTime) & (relEndTime > t)] = 1


    # Save output files, spectrum (contains all frequency data) and labelMatrix
    # For now save as .csv so we can inspect to make sure it is right! Later consider using .npy
    np.savetxt(outputDir+'/'+outputDataName+'.csv', spectrum, delimiter=",")
    np.savetxt(outputDir+'/'+outputLabelName+'.csv', labelMatrix, delimiter=",")


# Iterate through all data directories
allDirs = [];
# Get the directories that contain the data files
for (dirpath, dirnames, filenames) in os.walk(dataDir):
    allDirs.extend(dirnames);
    break
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
        for eachFile in filenames:
            if eachFile.split('.')[1] == 'flac':
                tempAudioFile = eachFile;
                tempStartHour = int(eachFile.split('_')[2][0:2]);
                processData(dataDir,tempCurrentDir,outputDir,tempAudioFile,tempLabelFile,'Data_'+tempID+'_Hour'+str(tempStartHour),'Label_'+tempID+'_Hour'+str(tempStartHour),tempStartHour)























#
