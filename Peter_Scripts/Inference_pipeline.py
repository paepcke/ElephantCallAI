#!/usr/bin/env python

"""
Script used for running the inference prediction pipeline.
The prediction pipeline consists of three main components:
    1) Data processing. Given a directory with .wav files,
    generate the corresponding spectrogram representations
    2) 2-stage model prediction. Predict the 0/1 segmentation
    given the 2-stage model.
    3) Elephant call prediction. Output in csv format the model's
    predictions for start / end times of calls.

This script allows for complete and partial runs of the 3 steps above.
Namely, the flag '--process_data' indicates that we want to process
the '.wav' files (step 1), and the flag '--make_predictions' means we want
to generate elephant call predictions (step 2 + 3). Other arguments 
are then provided to properly run the auxilliary scripts.

Example run:

Help:
python Inference_pipeline.py --help

------------------------------------

Full Pipeline:
python Inference_pipeline.py --process_data --data_dir <data directory> --spect_out <output directory> --make_predictions --model_0 2_Stage_Model/first_stage.pt --model_1 2_Stage_Model/second_stage.pt

------------------------------------

Data Generation:
python Inference_pipeline.py --process_data --data_dir <data directory> --spect_out <output directory> 

------------------------------------

Model Predictions:
python Inference_pipeline.py --make_predictions --model_0 2_Stage_Model/first_stage.pt --model_1 2_Stage_Model/second_stage.pt --spect_path <directory with processed spectrograms>

"""

import temp_spectrogramer
import argparse
import subprocess
import sys
import os

parser = argparse.ArgumentParser()



class Inferance_Pipeline(object):
    """docstring for Inferance_Pipeline"""
    def __init__(self, data_flag, spect_dir, spect_out, make_predictions, model_0, model_1, spect_path=None):
        """
        @param data_flag: Boolean indicating whether to process the data
        @param spect_dir: The directory with the .wav files we want to convert to spectrograms
        if the data_flag is one
        @param spect_out: The data directory that we would ouput the generated spectrograms
        @param make_predictions: Boolean flag indicating that we want to make predictions
        on the spectrograms. This assumes spectrograms have been processed!
        @param model_0: Path to the stage_1 model
        @param model_1: Path to the stage_2 model
        @param spect_path: The path to the directory with the processed spectrograms. If 
        None, we assume that the spectrograms are located in the directory 'spect_out/spect_dir'

        """
        super(Inferance_Pipeline, self).__init__()
        
        # First we should write a method that checks the necessary flag configuration!

        # Process the spectograms
        if data_flag:
            print ("#####################################")
            print ("###### Processing Spectrograms ######")
            print ("#####################################")
            subprocess.run(["python", "temp_spectrogramer.py", "--data_dirs", spect_dir, "--out", spect_out])

        # Here we assume the data has been created and the path to the spectrograms
        # is the 'spect_out' folder. 
        if spect_path is None:
            # We need to strip off the final directory that gets created for the spectrograms
            dirs = spect_dir.split("/")
            # Watch the case where the dir that we want is protected as stuff/dir/
            final_dir = dirs[-2] if dirs[-1] == "" else dirs[-1]
            spect_path = os.path.join(spect_out, final_dir)

        if make_predictions:
            # 1) First run the prediction model. Output predictions to a new folder './Predictions.'
            # Access the spect files to predict on based on "spects.txt" in the generated Spectrogram folder 
            print ("#############################################")
            print ("####### Generating Model Segmentation #######")
            print ("#############################################")
            subprocess.run(["python", "hierarchical_eval.py", "--model_0", model_0, 
                            "--model_1", model_1, "--preds_path", "./Predictions",
                            "--call_preds_path", "./Call_Predictions",
                            "--spect_path", spect_path, "--only_predictions",
                            "--test_files", os.path.join(spect_path, "spects.txt"),
                            "--make_full_preds"])

            print ("####################################################")
            print ("####### Generating Elephant Call Predictions #######")
            print ("####################################################")
            # 2) Generate the elephant call predictions based on the model per time step
            # segmentation. Save results to './Call_Predictions'
            subprocess.run(["python", "hierarchical_eval.py", "--model_0", model_0, 
                            "--model_1", model_1, "--preds_path", "./Predictions",
                            "--call_preds_path", "./Call_Predictions",
                            "--spect_path", spect_path, "--only_predictions",
                            "--test_files", os.path.join(spect_path, "spects.txt"),
                            "--save_calls"])





# ---------------- Main -------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Create noise gated wav files from input wav files."
                                     )

    # Data processing parameters!
    parser.add_argument('--data_dir', dest='data_dir', type=str,
        help='Provide the data_dir with the .wav files that you want to be processed for predictions')
    parser.add_argument('--spect_out', dest='spect_out', default='./Spectrograms/',
         help='The output data processing directory. Where the processed spectrograms will be located.' \
         + 'Note: the processed spectrograms will be outputed to a sub-directory based on the directory' \
         + '"data_dir" they came from.')
    parser.add_argument('--process_data', action='store_true',
        help = 'Flag indicating we first need to process the raw .wav files. NOTE: when this flag is set' \
        + ' make sure to provide proper arugments for "--data_dirs" and "--spect_out"')

    # Model Flags - These you do not need if you have the
    # provided models in the same folder as this script
    parser.add_argument('--make_predictions', action='store_true',
        help='Generate predictions for the data!')
    parser.add_argument('--model_0', type=str, default="./Model/first_stage.pt",
        help='Path to the model provided called "first_stage.pt"')
    parser.add_argument('--model_1', type=str, default="./Model/second_stage.pt",
        help='Path to the model provided called "second_stage.pt"')
    parser.add_argument('--spect_path', type=str, default=None,
        help='The directory where the spectrogram .npy files exist. This will be the same directory as' \
        + ' spect_out/data_dir" if the data was generated with this script. If this is None than we' \
        + ' assume that this directory is "spect_out/data_dir"')

    args = parser.parse_args();
    
    Inferance_Pipeline(
                        data_flag=args.process_data,
                        spect_dir=args.data_dir,
                        spect_out=args.spect_out,
                        make_predictions= args.make_predictions,
                        model_0=args.model_0,
                        model_1=args.model_1,
                        spect_path=args.spect_path
                        )

        
