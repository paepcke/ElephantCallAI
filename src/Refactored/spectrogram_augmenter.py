#!/usr/bin/env python
from scipy.io import wavfile
from random import randint
import numpy as np
import argparse
import os
import csv
import sys
from matplotlib import mlab as ml
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_utils import FileFamily
from data_utils import DATAUtils
from data_utils import AudioType
from visualization import visualize

class SpectrogramAugmenter(object):
	ELEPHANT_CALL_LENGTH = 255*800 + 4096
	def __init__(self, 
				 infiles,
				 ratio=1,
				 outdir=None):
		self.infiles = infiles # list of tuple of label_file, wav_file
		self.outdir = outdir
		self.min_freq=0       # Hz 
		self.max_freq=150      # Hz
		self.nfft=4096
		self.pad_to=4096
		self.hop=800
		self.framerate=8000
		self.ratio=ratio

		elephant_calls, call_indices = self.get_elephant_calls(infiles)
		print(f"Got {len(elephant_calls)} elephant calls")
		print(f"Max elephant call length: {max([len(call) for call in elephant_calls])}")
		self.negs_per_wav_file = int(round(len(elephant_calls)*ratio/len(infiles)))
		print(f"Getting {self.negs_per_wav_file} negatives")
		non_call_segments = self.get_non_call_segments(infiles, call_indices)
		combined_calls, combined_call_indices = self.combine_segments(elephant_calls, non_call_segments, ratio)
		print("Combined segments")
		self.make_spectrogram(combined_calls, combined_call_indices, elephant_calls, outdir)
		print(f"Saved spectrograms to {outdir}")

	def get_elephant_calls(self, infiles):
		elephant_calls = []
		call_indices = {} # maps wav file to array of indices we can sample from
		
		for label_file, wav_file in infiles:
			sr, samples = wavfile.read(wav_file)
			fd = open(label_file, 'r')
			reader = csv.DictReader(fd, delimiter='\t')
			#start_end_times = {} # maps (start, end)
			start_end_times = []
			# get portion of sample where label_mask = 1
			file_offset_key = 'File Offset (s)'
			begin_time_key = 'Begin Time (s)'
			end_time_key   = 'End Time (s)'
			# preprocess to allow for overlapping calls
			for label_dict in reader:
				try:
					begin_time = float(label_dict[file_offset_key])
					call_length = float(label_dict[end_time_key]) - float(label_dict[begin_time_key])
					end_time = begin_time + call_length
					start_end_times.append((begin_time, end_time))
					'''set_of_curr_times = set(range(int(begin_time), int(end_time)))
					intersections = [set_of_curr_times.intersection(set(time_range)) for time_range in start_end_times]

					if intersections == [] or all(len(intersection) == 0 for intersection in intersections):
						start_end_times[tuple(set_of_curr_times)] = (begin_time, end_time)
					else:
						# get the index of intersection
						intersection_idx = [idx for idx in range(len(intersections)) if len(intersections[idx]) > 0]
						for idx in intersection_idx:
							new_key = set_of_curr_times.union(set(list(start_end_times.keys())[idx]))
							new_value = (min(new_key), max(new_key))
							start_end_times[tuple(new_key)] = new_value
							# remove old keys
							start_end_times.pop(list(start_end_times.keys())[idx])'''

				except KeyError:
					raise IOError(f"Raven label file {label_txt_file} does not contain one "
								  f"or both of keys '{begin_time_key}', {end_time_key}'")
			# Get each el call time range spec in the labels:
			#for (begin_time, end_time) in start_end_times.values():
			for (begin_time, end_time) in start_end_times:
				begin_index = int(begin_time * sr)
				end_index = int(end_time * sr)
				if wav_file in call_indices:
					call_indices[wav_file] += [index for index in range(begin_index - SpectrogramAugmenter.ELEPHANT_CALL_LENGTH, end_index)]
				else:
					call_indices[wav_file] = [index for index in range(begin_index - SpectrogramAugmenter.ELEPHANT_CALL_LENGTH, end_index)]
				if call_length > SpectrogramAugmenter.ELEPHANT_CALL_LENGTH: # TODO fix this
					continue
				elephant_calls.append(samples[begin_index:end_index])
			print("finished a wav file")
		return elephant_calls, call_indices

	def get_non_call_segments(self, infiles, call_indices):
		non_call_segments = []
		for label_file, wav_file in infiles:
			sr, samples = wavfile.read(wav_file)
			call_index = call_indices[wav_file]
			for i in range(self.negs_per_wav_file):
				# randomly sample outside of call_indices
				found_index = False
				while found_index == False:
					start_index = randint(0, len(samples) - SpectrogramAugmenter.ELEPHANT_CALL_LENGTH)
					if start_index not in call_index:
						found_index = True
				non_call_segments.append(samples[start_index:start_index + SpectrogramAugmenter.ELEPHANT_CALL_LENGTH])
		np.random.shuffle(non_call_segments)
		print(f"Got {len(non_call_segments)} non call segments")
		return non_call_segments

	def combine_segments(self, elephant_calls, non_call_segments, ratio):
		non_call_counter = 0
		combined_calls = []
		combined_call_indices = [] 
		for call in elephant_calls:
			padded_call = np.zeros_like(non_call_segments[non_call_counter])
			rand_start_ind = randint(0, padded_call.shape[0] - call.shape[0])


			padded_call[rand_start_ind: rand_start_ind + call.shape[0]] = call * 0.5
			for i in range(ratio):
				overlap_call = non_call_segments[non_call_counter]
				overlap_call = np.add(overlap_call, padded_call)
				
				start_time = rand_start_ind/8000
				end_time = start_time + call.shape[0]/8000
				overlap_call[int(start_time):int(end_time)] = overlap_call[int(start_time):int(end_time)] * 0.5
				combined_call_indices.append((start_time, end_time))
				combined_calls.append(overlap_call)
				non_call_counter += 1
			
		return combined_calls, combined_call_indices


	def make_spectrogram(self, combined_calls, combined_call_indices, elephant_calls, outdir):
		for num, call in enumerate(combined_calls):
			raw_audio = call
			# visualize elephant call
			elephant_call = elephant_calls[int(num/self.ratio)]
			[spectrum, freqs, t] = ml.specgram(elephant_call, 
					NFFT=self.nfft, Fs=self.framerate, noverlap=(self.nfft - self.hop), 
					window=ml.window_hanning, pad_to=self.pad_to)
			spectrum = spectrum[(freqs <= self.max_freq)]
			spectrum = 10 * np.log10(spectrum)
			pad = np.zeros((77, 50))
			spectrum = np.concatenate((pad, spectrum, pad), axis=1)
			#visualize(spectrum.T, [np.zeros(spectrum.shape[1])], np.zeros(spectrum.shape[1]))

			#visualize and save combined call
			[spectrum, freqs, t] = ml.specgram(raw_audio, 
					NFFT=self.nfft, Fs=self.framerate, noverlap=(self.nfft - self.hop), 
					window=ml.window_hanning, pad_to=self.pad_to)
			spectrum = spectrum[(freqs <= self.max_freq)]
			if spectrum.shape[1] is not 256:
				raise ValueError("Spectrum is of the wrong shape!!")
			spectro_outfile = os.path.join(self.outdir, f"call_{str(num)}_spectro")
			print(f"Saving spectrogram to {spectro_outfile}")
			np.save(spectro_outfile, spectrum)
			spectrum = 10 * np.log10(spectrum)

			# save labels (a list of length 256 with 0s and 1s)
			begin_time, end_time = combined_call_indices[num]
			pre_begin_indices = np.nonzero(t < begin_time)[0] 
			if len(pre_begin_indices) == 0:
				start_bin_idx = 0
			else:
				# Make the bounds a bit tighter by adding one to the last
				# index with the time < begin_time
				start_bin_idx = pre_begin_indices[-1] + 1
			
			# Similarly with end time:
			post_end_indices = np.nonzero(t > end_time)[0]
			if len(post_end_indices) == 0:
				# Label end time is beyond recording. Just 
				# go up to the end:
				end_bin_idx = len(t)
			else:
				# Similar, make bounds a bit tighter 
				end_bin_idx = post_end_indices[0] - 1

			labels = np.zeros(256)
			labels[start_bin_idx:end_bin_idx] = 1
			label_outfile = os.path.join(self.outdir, f"call_{str(num)}_labels")
			print(f"Saving labels to {label_outfile}")
			np.save(label_outfile, labels)

			#visualize(spectrum.T, [labels], labels)

if __name__ == '__main__':
	# we pass in tuple of label, wav files
	parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
									 formatter_class=argparse.RawTextHelpFormatter,
									 description="Spectrogram augmentation"
									 )
	parser.add_argument('infiles',
						nargs='+',
						help="Directory for wav and label files"
						)
	parser.add_argument('-o', '--outdir', 
						default=None, 
						help='Outfile for any created files'
						)
	parser.add_argument('-r', '--ratio', 
						default=None, 
						help='Ratio for non-calls to calls',type=int
						)
	args = parser.parse_args();
	label_wav_pairs = []
	args.infiles = args.infiles[0]
	for(dirpath, dirnames, filenames) in os.walk(args.infiles):
		for file in filenames:
			# Remember to add the directory!
			full_file = os.path.join(dirpath, file)
			if file.endswith('.wav'):
				file_family = FileFamily(full_file)

				# Append tuple with (label file, wav file
				label_wav_pairs.append((file_family.fullpath(AudioType.LABEL), full_file))

	SpectrogramAugmenter(label_wav_pairs, args.ratio, args.outdir)




