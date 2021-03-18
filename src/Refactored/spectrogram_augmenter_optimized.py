#!/usr/bin/env python
from scipy.io import wavfile
from random import randint
import numpy as np
import argparse
import os
import csv
import math
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
                 outdir=None, remove_marginals=True):
        self.infiles = infiles # list of tuple of label_file, wav_file
        self.outdir = outdir
        self.min_freq=0       # Hz 
        self.max_freq=150      # Hz
        self.nfft=4096
        self.pad_to=4096
        self.hop=800
        self.framerate=8000
        self.ratio=ratio
        self.remove_marginals = remove_marginals

        call_indices = self.get_elephant_calls(infiles)
        self.num_calls = sum([len(value) for key, value in call_indices.items()])
        print(f"Got {self.num_calls} elephant calls")
        self.negs_per_wav_file = math.ceil(self.num_calls*ratio*1.0/len(infiles))
        print(f"Getting {self.negs_per_wav_file} negatives per file, upwards of {self.num_calls*ratio} negatives total")
        non_call_segments = self.get_non_call_segments(infiles, call_indices)
        self.combine_segments(non_call_segments, call_indices, ratio, infiles, outdir)
        print(f"Saved spectrograms to {outdir}")

    def get_elephant_calls(self, infiles):
        call_indices = {} # maps wav file to array of indices we can sample from
        counter = 0
        
        for label_file, wav_file in infiles:
            sr, samples = wavfile.read(wav_file)
            if not os.path.exists(label_file):
                continue
            fd = open(label_file, 'r')
            reader = csv.DictReader(fd, delimiter='\t')
            #start_end_times = {} # maps (start, end)
            start_end_times = []
            # get portion of sample where label_mask = 1
            file_offset_key = 'File Offset (s)'
            begin_time_key = 'Begin Time (s)'
            end_time_key   = 'End Time (s)'
            marginal_key = 'Marginal'
            num_marginal_calls = 0
            # preprocess to allow for overlapping calls
            for label_dict in reader:
                try:
                    if self.remove_marginals and label_dict[marginal_key] == "yes":
                        print ("Skipping marginal call number:", num_marginal_calls)
                        num_marginal_calls += 1
                        continue
                    begin_time = float(label_dict[file_offset_key])
                    call_length = float(label_dict[end_time_key]) - float(label_dict[begin_time_key])
                    end_time = begin_time + call_length
                    start_end_times.append((begin_time, end_time))
                    begin_index = int(begin_time * sr)
                    end_index = int(end_time * sr)
                    if wav_file not in call_indices:
                        call_indices[wav_file] = [(begin_index, end_index)]
                    call_indices[wav_file].append((begin_index, end_index))

                except KeyError:
                    raise IOError(f"Raven label file {label_txt_file} does not contain one "
                                  f"or both of keys '{begin_time_key}', {end_time_key}'")
                
            print(f"finished wav file #{counter}")
            counter += 1
            fd.close()
        self.valid_wav_files = counter
        return call_indices

    def get_non_call_segments(self, infiles, call_indices):
        #non_call_segments = np.zeros(0)
        non_call_segments = np.zeros(self.negs_per_wav_file*self.valid_wav_files*SpectrogramAugmenter.ELEPHANT_CALL_LENGTH)
        counter = 0
        for label_file, wav_file in infiles:
            if not os.path.exists(label_file):
                continue
            sr, samples = wavfile.read(wav_file)
            if wav_file not in call_indices:
                call_index = []
            else:
                call_index = call_indices[wav_file]
            for i in range(self.negs_per_wav_file):
                # randomly sample outside of call_indices
                found_index = False
                while found_index == False:
                    start_index = randint(0, len(samples) - SpectrogramAugmenter.ELEPHANT_CALL_LENGTH)
                    valid_index = True
                    for begin, end in call_index:
                        if start_index > begin and start_index < end:
                            valid_index = False
                            break
                    found_index = valid_index
                non_call_segments[counter*SpectrogramAugmenter.ELEPHANT_CALL_LENGTH:(counter+1)*SpectrogramAugmenter.ELEPHANT_CALL_LENGTH] = samples[start_index:start_index + SpectrogramAugmenter.ELEPHANT_CALL_LENGTH]
                #non_call_segments = np.concatenate([non_call_segments, list(samples[start_index:start_index + SpectrogramAugmenter.ELEPHANT_CALL_LENGTH])])
                counter += 1
            print(f"finished wav file #{counter}")
            
            del samples

        print(f"Got {len(non_call_segments)/SpectrogramAugmenter.ELEPHANT_CALL_LENGTH} non call segments")
        return non_call_segments

    def combine_segments(self, non_call_segments, call_indices, ratio, infiles, outdir):
        non_call_counter = 0
        # generate random list of indices to index into non_call_segments with
        non_call_indices = np.random.randint(0, len(non_call_segments)/SpectrogramAugmenter.ELEPHANT_CALL_LENGTH, self.num_calls*ratio)
        for label_file, wav_file in infiles:
            if not os.path.exists(label_file):
                continue
            sr, samples = wavfile.read(wav_file)
            if wav_file not in call_index:
                continue
            for call_index in call_indices[wav_file]:
                call = samples[call_index[0]:call_index[1]]
                
                for i in range(ratio):
                    padded_call = np.zeros_like(non_call_segments[0:SpectrogramAugmenter.ELEPHANT_CALL_LENGTH])
                    rand_start_ind = randint(0, padded_call.shape[0] - call.shape[0])
                    padded_call[rand_start_ind: rand_start_ind + call.shape[0]] = call * 0.5

                    curr_index = non_call_indices[non_call_counter]
                    overlap_call = non_call_segments[curr_index*SpectrogramAugmenter.ELEPHANT_CALL_LENGTH:curr_index*SpectrogramAugmenter.ELEPHANT_CALL_LENGTH+SpectrogramAugmenter.ELEPHANT_CALL_LENGTH]
                    #overlap_call = non_call_segments[non_call_counter]
                    overlap_call = np.add(overlap_call, padded_call)
                    
                    start_time = rand_start_ind/8000
                    end_time = start_time + call.shape[0]/8000
                    overlap_call[int(start_time):int(end_time)] = overlap_call[int(start_time):int(end_time)] * 0.5
                    #combined_call_indices.append((start_time, end_time))
                    #combined_calls.append(overlap_call)
                    
                    raw_audio = overlap_call
                    #visualize and save combined call
                    [spectrum, freqs, t] = ml.specgram(raw_audio, 
                            NFFT=self.nfft, Fs=self.framerate, noverlap=(self.nfft - self.hop), 
                            window=ml.window_hanning, pad_to=self.pad_to)
                    spectrum = spectrum[(freqs <= self.max_freq)]
                    if spectrum.shape[1] is not 256:
                        raise ValueError("Spectrum is of the wrong shape!!")
                    spectro_outfile = os.path.join(self.outdir, f"call_{str(non_call_counter)}_spectro")
                    print(f"Saving spectrogram to {spectro_outfile}")
                    np.save(spectro_outfile, spectrum)
                    spectrum = 10 * np.log10(spectrum)

                    # save labels (a list of length 256 with 0s and 1s)
                    pre_begin_indices = np.nonzero(t < start_time)[0] 
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
                    label_outfile = os.path.join(self.outdir, f"call_{str(non_call_counter)}_labels")
                    print(f"Saving labels to {label_outfile}")
                    np.save(label_outfile, labels)
                    non_call_counter += 1

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
    parser.add_argument('-m', '--remove_marginals', 
                        help='Set to false if we dont want to use marginals',action='store_true'
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
                if args.remove_marginals:
                    label_wav_pairs.append((file_family.fullpath(AudioType.MARGINAL_LABEL), full_file))
                else:
                # Append tuple with (label file, wav file
                    label_wav_pairs.append((file_family.fullpath(AudioType.LABEL), full_file))
    print(label_wav_pairs)
    SpectrogramAugmenter(label_wav_pairs, args.ratio, args.outdir, args.remove_marginals)




