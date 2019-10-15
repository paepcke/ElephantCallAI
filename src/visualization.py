"""
Methods for visualizing numpy arrays of the spectograms, outputs, and labels

Can run as a standalone function to visualize individual wav files. TODO!!!
"""
import matplotlib.pyplot as plt
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('filename', help='name of wav file to visualize')


def visualize(features, outputs=None, labels=None):
	"""
	Visualizes the spectogram and associated predictions/labels

	For now this just has placeholder plots for outputs and labels
	when they're not passed in. In the future maybe we should create
	a different plot or something.

	Inputs are numpy arrays
	"""
	fig, (ax1, ax2, ax3) = plt.subplots(3,1)
	# new_features = np.flipud(10*np.log10(features).T)
	# TODO: Delete above line?

	new_features = np.flipud(features.T)
	min_dbfs = new_features.flatten().mean()
	max_dbfs = new_features.flatten().mean()
	min_dbfs = np.maximum(new_features.flatten().min(),min_dbfs-2*new_features.flatten().std())
	max_dbfs = np.minimum(new_features.flatten().max(),max_dbfs+6*new_features.flatten().std())

	ax1.imshow(np.flipud(new_features), cmap="magma_r", vmin=min_dbfs, vmax=max_dbfs, interpolation='none', origin="lower", aspect="auto")
	
	if outputs is not None:
		ax2.plot(np.arange(outputs.shape[0]), outputs)
		ax2.set_ylim([0,1])
		ax2.axhline(y=0.5, color='r', linestyle='-')

	if labels is not None:
		ax3.plot(np.arange(labels.shape[0]), labels)
	
	plt.show()


def main():
	args = parser.parse_args()

	# TODO: access the files and visualize the array


if name == '__main__':
	main()