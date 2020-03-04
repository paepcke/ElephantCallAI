import numpy as np
import argparse


"""
	Compare different experiments in adversarial discovery.

	Supported comparisons:
	1) Compare the number of shared adversarial files discovered 
	across the provided adversarial example files

"""

parser = argparse.ArgumentParser()
parser.add_argument('files', type=str, nargs='+', 
    help='A list of adversarial examples discovered by different models that we want to compare')


def count_same_examples(file_discoveries):
	"""
		Count the number of examples that were flagged by all
		of the different models
	"""
	# Since we are checking same across all
	# it is sufficient to look through the files
	# discovered by one model
	files_shared = 0
	model_0 = file_discoveries[0]
	for file in model_0:
		# Check if it is found by the others
		found = True
		for model_x_files in file_discoveries:
			if not file in model_x_files:
				found = False
				break

		if found:
			files_shared += 1

	return files_shared


def main():
	args = parser.parse_args()

	files = args.files

	file_discoveries = []
	for file in files:
		with open(file, 'w') as f:
			lines = f.readlines()
			lines = set([x.strip() for x in lines])

			file_discoveries.append(lines)

	num_files_shared = count_same_examples(file_discoveries)
	i = 0
	for model_x_files in file_discoveries:
		print ("Model {} found: {} files".format(i, len(model_x_files)))
		i += 1

	print ("Number files shared across all models:", num_files_shared)


if __name__ == '__main__':
	main()