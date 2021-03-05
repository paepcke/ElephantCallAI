import generate_spectrograms
import argparse
import numpy as np
from scipy.io import wavfile
from os import path
import os



"""
Update this file!!
This file is just temporary to generate spectrograms without labels for testing on 
Peter's new data! Assumes we only want to process .wav files.

Let us also create a file at the end with all of the specs createdddd
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # For use on quatro
    parser.add_argument('--data_dirs', dest='data_dirs', nargs='+', type=str,
        help='Provide the data_dirs with the files that you want to be processed')
    parser.add_argument('--out', dest='outputDir', default='/home/data/elephants/rawdata/Spectrograms/',
         help='The output directory')

    parser.add_argument('--NFFT', type=int, default=4096, help='Window size used for creating spectrograms') 
    parser.add_argument('--hop', type=int, default=800, help='Hop size used for creating spectrograms')
    parser.add_argument('--window', type=int, default=256, 
        help='Deterimes the window size in frames of the resulting spectrogram') # Default corresponds to 21s
    parser.add_argument('--max_f', dest='max_freq', type=int, default=150, help='Deterimes the maximum frequency band')
    parser.add_argument('--pad', dest='pad_to', type=int, default=4096, 
        help='Deterimes the padded window size that we want to give a particular grid spacing (i.e. 1.95hz')


    args = parser.parse_args()
    data_dirs = args.data_dirs
    outputDir = args.outputDir
    # Make sure this exists
    if not os.path.exists(outputDir):
        os.mkdir(outputDir)

    spectrogram_info = {'NFFT': args.NFFT,
                        'hop': args.hop,
                        'max_freq': args.max_freq,
                        'window': args.window,
                        'pad_to': args.pad_to}

    # Loop through the directory and process the .wav files
    for currentDir in data_dirs:
        # Get the final name of the directory with the spect files
        files_dirs = currentDir.split('/')
        file_dir_name = files_dirs[-2] if files_dirs[-1] == '' else files_dirs[-1]
        # Create the output directory
        spect_dir = path.join(outputDir, file_dir_name)
        if not path.exists(spect_dir):
            os.mkdir(spect_dir)

        # Create and save a file of the names of each of the spectrograms produced
        spect_files = []
        for(dirpath, dirnames, filenames) in os.walk(currentDir):
            # Iterate through the .wav spectrogram files to generate them!
            for audio_file in filenames:
                # Strip off the location and time tags
                tags = audio_file.split('_')
                data_id = tags[0] + '_' + tags[1]
                file_type = audio_file.split('.')[1]

                if (file_type not in ['wav']):
                    continue

                # We need to read the audio file so that we can use generate_spectrogram
                audio_path = path.join(dirpath, audio_file)
                try:
                    samplerate, raw_audio = wavfile.read(audio_path)
                    if (samplerate < 4000):
                        print ("Sample Rate Unexpectadly low!", samplerate)
                    print ("File size", raw_audio.shape)
                except:
                    print("FILE Failed", audio_file)
                    # Let us try this for now to see if it stops the failing
                    continue

                spectrogram_info['samplerate'] = samplerate
                spectrogram = generate_spectrograms.generate_spectogram(raw_audio, spectrogram_info, data_id)

                # Want to save the corresponding label_file with the spectrogram!!
                np.save(path.join(spect_dir, data_id + "_spec.npy"), spectrogram)
                print ("processed " + data_id)
                spect_files.append(data_id)

        # Save the spect ids
        with open(path.join(spect_dir, 'spects.txt'), 'w') as f:
            for spect_id in spect_files:
                f.write(spect_id + "\n")
                

