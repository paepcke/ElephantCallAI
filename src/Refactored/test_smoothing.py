from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import numpy as np
import torch


predictions = np.zeros(100)
gt = np.zeros(100)

gt[28:79] = 1
gt_smooth = gaussian_filter1d(gt,sigma=3)

# Make some calls
predictions[25: 75] = 1

predictions_smooth = gaussian_filter1d(predictions, sigma=3)

loss = torch.nn.BCEWithLogitsLoss()
print ("LOSS:", loss(torch.tensor(predictions_smooth), torch.tensor(gt_smooth)))
print ("LOSS UN-SMOOTHED:", loss(torch.tensor(predictions), torch.tensor(gt)))

num_num_zero = 0
for i in range(25):
	if predictions_smooth[i] > 0:
		num_num_zero += 1 

print ("Num non-zero:", num_num_zero)

fig, ((ax, ax2), (ax3, ax4)) = plt.subplots(2, 2)

ax.plot(np.arange(predictions_smooth.shape[0]), predictions_smooth)

ax.set_ylim([0,1])
# Toss them all in for now
ax.axhline(y=0.5, color='r', linestyle='-')

ax3.plot(np.arange(gt_smooth.shape[0]), gt_smooth)

ax3.set_ylim([0,1])
# Toss them all in for now
ax3.axhline(y=0.5, color='r', linestyle='-')

ax2.plot(np.arange(predictions.shape[0]), predictions)

ax2.set_ylim([0,1])
# Toss them all in for now
ax2.axhline(y=0.5, color='r', linestyle='-')

ax4.plot(np.arange(gt.shape[0]), gt)

ax4.set_ylim([0,1])
# Toss them all in for now
ax4.axhline(y=0.5, color='r', linestyle='-')

plt.show()