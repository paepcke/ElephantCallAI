import matplotlib.pyplot as plt
import numpy as np
import torchaudio
import torch
#import scikits.audiolab
import aifc


#print(torch.backends.mkl.is_available())

data, sample_rate = torchaudio.load("./data/whale_calls/data/train/train9.aiff")
print (sample_rate)
#data, sample_rate, enc = aiffread("./whale-detection-challenge/data/train/train1.aiff")
fft_data = torch.rfft(data, 1)

print (fft_data.shape)

filename = "./whale-detection-challenge/data/train/train893.aiff"

s = aifc.open(filename)

nframes = s.getnframes()
strsig = s.readframes(nframes)

y = np.frombuffer(strsig, np.short).byteswap()
print (y.shape)

L = y.shape[0] / sample_rate
Pxx, freqs, bins, im = plt.specgram(y , NFFT=256, Fs=4000, noverlap=240, cmap=plt.cm.gray_r)
f, ax = plt.subplots(figsize=(4.8, 2.4))
Pxx = np.abs(Pxx)
print (Pxx)
Pxx = Pxx[[freqs<400.]]
#Pxx = 20 * np.log10(Pxx / np.max(Pxx))
ax.imshow(Pxx, origin='lower', cmap='viridis',
          extent=(0, L, 0, sample_rate / 2 / 1000))
ax.axis('tight')
ax.set_ylabel('Frequency [kHz]')
ax.set_xlabel('Time [s]')

print (Pxx.shape)
#Pxx = Pxx[[freqs<250.]] # Right-whales call occur under 250Hz
print(Pxx.shape)
#plt.axis('off')
plt.show()
#print Pxx.shape
#y = y / np.mean(y)
#print (y[:100])
#print (data[:100])
print(data.shape)

from scipy import signal
print("Sample rate", sample_rate)
freqs, times, Sx = signal.spectrogram(y, fs=sample_rate, window='hanning',
                                      nperseg=256, noverlap=230,
                                      detrend=False, scaling='spectrum')

f, ax = plt.subplots(figsize=(4.8, 2.4))
print (Sx.shape)
print (freqs.shape)
ax.pcolormesh(times, freqs / 1000, 10 * np.log10(Sx), cmap='inferno_r')
ax.set_ylabel('Frequency [kHz]')
ax.set_xlabel('Time [s]');
plt.show()


Sx = imresize(np.log10(Sx),(224,224), interp= 'lanczos').astype('float32')
