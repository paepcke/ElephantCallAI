from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import csv
from scipy import signal
import aifc
import numpy as np
from scipy.misc import imresize
import matplotlib.pyplot as plt

X = []
Y = []

filenames = []

with open("./data/whale_calls/data/train.csv") as csvfile:
	spamreader = csv.reader(csvfile)
	for idx, row in enumerate(spamreader):
		if idx == 0:
			continue

		filenames.append(row[0])
		Y.append(row[1])

		if idx >= 20000:
			break


print(len(filenames))


for filename in filenames:
	full_filename = "./data/whale_calls/data/train/" + filename


	s = aifc.open(full_filename)
	nframes = s.getnframes()
	strsig = s.readframes(nframes)
	y = np.frombuffer(strsig, np.short).byteswap()

	freqs, times, Sx = signal.spectrogram(y, fs=2000, window='hanning',
                                      nperseg=256, noverlap=236,
                                      detrend=False, scaling='spectrum')

	
	# Sx = imresize(np.log10(Sx),(224,224), interp= 'lanczos').astype('float32')

	X.append(Sx)

X = np.array(X)
X = X.reshape(X.shape[0], -1)
Y = np.array(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# clf = RandomForestClassifier()
# clf = SVC(verbose=True)
clf = LogisticRegression()

print(X_train.shape)
print(y_train.shape)

clf.fit(X_train, y_train)
print("Accuracy: ", clf.score(X_test, y_test))




