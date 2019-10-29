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

samplerate, raw_audio = wavfile.read(audio_file)
spectrum, freqs, _ = ml.specgram(raw_audio, 
                NFFT=4096, Fs=samplerate, noverlap=(4096 - 641), window=ml.window_hanning) 

print (spectrum.shape)