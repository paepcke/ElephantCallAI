import csv
from scipy.io import wavfile
from matplotlib import mlab as ml


labelFile = csv.DictReader(open('New_Data/Truth_Logs/nn_201801_jan/nn10b_20180604_hele_00-24hr_enb_ok.txt','rt'), delimiter='\t')

samplerate, raw_audio = wavfile.read('New_Data/Truth_Logs/nn_201801_jan/nn10b_20180604_000000.wav')
# Corresponding label file

# Setup time limits
timePerFrame = 1. / samplerate
print ('Time Per Frame:', timePerFrame)
print (raw_audio.shape)

numFFT = 512 * 4
hop_length = (128 * 3) * 4

[spectrum, freqs, t] = ml.specgram(raw_audio, NFFT=numFFT, Fs=samplerate, noverlap=(numFFT - hop_length), window=ml.window_hanning)
print ('done?')