from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import numpy as np


predictions = np.zeros(100)

# Make some calls
predictions[25: 75] = 1

predictions = gaussian_filter1d(predictions, sigma=3)
print (predictions)

num_num_zero = 0
for i in range(25):
	if predictions[i] > 0:
		num_num_zero += 1 

print ("Num non-zero:", num_num_zero)

fig, ax = fig, axes = plt.subplots(1, 1)

ax.plot(np.arange(predictions.shape[0]), predictions)

ax.set_ylim([0,1])
# Toss them all in for now
ax.axhline(y=0.5, color='r', linestyle='-')

plt.show()