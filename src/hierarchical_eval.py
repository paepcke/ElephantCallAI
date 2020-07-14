import numpy as np
import torch
import torch.nn as nn
import os
import argparse
from scipy.ndimage import gaussian_filter1d

import parameters
from data import get_loader, ElephantDatasetFull
from visualization import visualize, visualize_predictions
from utils import sigmoid, calc_accuracy, get_f_score

parser = argparse.ArgumentParser()
parser.add_argument('--preds_path', type=str, dest='predictions_path', default='../Predictions',
    help = 'Path to the folder where we output the full test predictions')

parser.add_argument('--test_files', type=str, default='../elephant_dataset/Test/Neg_Samples_x1/files.txt')
# For quatro
#parser.add_argument('--test_files', type=str, default='../elephant_dataset/Test/files.txt')

parser.add_argument('--spect_path', type=str, default="../elephant_dataset/New_Data/Spectrograms", 
    help='Path to the processed spectrogram files')
# For quatro
#parser.add_argument('--spect_path', type=str, default="/home/data/elephants/rawdata/Spectrograms/", 
#    help='Path to the processed spectrogram files')

parser.add_argument('--make_full_preds', action='store_true', 
    help = 'Generate predictions for the full test spectrograms')
parser.add_argument('--full_stats', action='store_true',
    help = 'Compute statistics on the full test spectrograms')
#parser.add_argument('--pred_calls', action='store_true', 
#    help = 'Generate the predicted (start, end) calls for test spectrograms')
parser.add_argument('--visualize', action='store_true',
    help='Visualize full spectrogram results')

parser.add_argument('--model', type=str,
    help = 'Path to the model to test on') # Now is path


'''
Example runs

# Make predictions 
# To customize change the model flag!
python eval.py --test_files /home/data/elephants/processed_data/Test_nouab/Neg_Samples_x1/files.txt --spect_path /home/data/elephants/rawdata/Spectrograms/nouabale\ ele\ general\ test\ sounds/ --model /home/data/elephants/models/selected_runs/Adversarial_training_17_nouab_and_bai_0.25_sampling_one_model/Call_model_17_norm_Negx1_Seed_8_2020-04-28_01:58:26/model_adversarial_iteration_9_.pt --make_full_pred

# Calculate Stats 
python eval.py --test_files /home/data/elephants/processed_data/Test_nouab/Neg_Samples_x1/files.txt --spect_path /home/data/elephants/rawdata/Spectrograms/nouabale\ ele\ general\ test\ sounds/ --model /home/data/elephants/models/selected_runs/Adversarial_training_17_nouab_and_bai_0.25_sampling_one_model/Call_model_17_norm_Negx1_Seed_8_2020-04-28_01:58:26/model_adversarial_iteration_9_.pt --full_stats
'''


def loadModel(model_path, is_hierarchical=True):
    model = torch.load(model_path, map_location=parameters.device)
    print (model)
    # Get the model name from the path
    tokens = model_path.split('/')
    # For now since the hierarchical models are the same
    if (is_hierarchical):
        model_id = tokens[-3]
    else:
        model_id = tokens[-2]
    return model, model_id


def predict_spec_sliding_window(spectrogram, model, chunk_size=256, jump=128, hierarchical_model=None, hierarchy_threshold=10):
    """
        Generate the prediction sequence for a full audio sequence
        using a sliding window. Slide the window by one spectrogram frame
        and pass each window through the given model. Compute the average
        over overlapping window predictions to get the final prediction.
    """
    # Get the number of frames in the full audio clip
    predictions = np.zeros(spectrogram.shape[0])
    overlap_counts = np.zeros(spectrogram.shape[0])

    # This is a bit janky but we will manually transform
    # each spectrogram chunk
    #spectrogram = torch.from_numpy(spectrogram).float()
    # Add a batch dim for the model!
    #spectrogram = torch.unsqueeze(spectrogram, 0) # Shape - (1, time, freq)

    # Added!
    spectrogram = np.expand_dims(spectrogram,axis=0)

    # For the sliding window we slide the window by one spectrogram
    # frame, determined by the hop size.
    spect_idx = 0 # The frame idx of the beginning of the current window
    i = 0
    # How can I parralelize this shit??????
    while  spect_idx + chunk_size <= spectrogram.shape[1]:
        #if (i % 1000 == 0):
        #    print ("Chunk number " + str(i))

        spect_slice = spectrogram[:, spect_idx: spect_idx + chunk_size, :]
        # Transform the slice - this is definitely sketchy!!!! 
        spect_slice = (spect_slice - np.mean(spect_slice)) / np.std(spect_slice)
        spect_slice = torch.from_numpy(spect_slice).float()
        spect_slice = spect_slice.to(parameters.device)

        outputs = model(spect_slice) # Shape - (1, chunk_size, 1)
        compressed_out = outputs.view(-1, 1).squeeze()

        # Now check if we are running the hierarchical model
        if hierarchical_model is not None:
            chunk_preds = torch.sigmoid(compressed_out)
            binary_preds = torch.where(chunk_preds > parameters.THRESHOLD, torch.tensor(1.0).to(parameters.device), torch.tensor(0.0).to(parameters.device))
            pred_counts = torch.sum(binary_preds)
            # Run second model
            if pred_counts.item() > hierarchy_threshold:
                print ("Doing hierarchy! With number predicted:", pred_counts.item())
                outputs = hierarchical_model(spect_slice)
                compressed_out = outputs.view(-1, 1).squeeze()

        overlap_counts[spect_idx: spect_idx + chunk_size] += 1
        predictions[spect_idx: spect_idx + chunk_size] += compressed_out.cpu().detach().numpy()

        spect_idx += jump
        i += 1

    # Do the last one if it was not covered
    if (spect_idx - jump + chunk_size != spectrogram.shape[1]):
        #print ('One final chunk!')
        spect_slice = spectrogram[:, spect_idx: , :]
        # Transform the slice 
        # Should use the function from the dataset!!
        spect_slice = (spect_slice - np.mean(spect_slice)) / np.std(spect_slice)
        spect_slice = torch.from_numpy(spect_slice).float()
        spect_slice = spect_slice.to(parameters.device)

        outputs = model(spect_slice) # Shape - (1, chunk_size, 1)
        # In the case of ResNet the output is forced to the chunk size
        compressed_out = outputs.view(-1, 1).squeeze()[:predictions[spect_idx: ].shape[0]]

        # Now check if we are running the hierarchical model
        if hierarchical_model is not None:
            chunk_preds = torch.sigmoid(compressed_out)
            binary_preds = torch.where(chunk_preds > parameters.THRESHOLD, torch.tensor(1.0).to(parameters.device), torch.tensor(0.0).to(parameters.device))
            pred_counts = torch.sum(binary_preds)
            # Run second model
            if pred_counts.item() > hierarchy_threshold:
                outputs = hierarchical_model(spect_slice)
                compressed_out = outputs.view(-1, 1).squeeze()[:predictions[spect_idx: ].shape[0]]


        overlap_counts[spect_idx: ] += 1
        predictions[spect_idx: ] += compressed_out.cpu().detach().numpy()


    # Average the predictions on overlapping frames
    predictions = predictions / overlap_counts

    # Get squashed [0, 1] predictions
    predictions = sigmoid(predictions)

    return predictions


def generate_predictions_full_spectrograms(dataset, model, model_id, predictions_path, 
    sliding_window=True, chunk_size=256, jump=128, hierarchical_model=None):
    """
        For each full test spectrogram, run a trained model to get the model
        prediction and save these predictions to the predictions folder. Namely,
        for each model we create a sub-folder with predictions for that model. Note,
        in the future we will not only save models as there number but also save
        them based on the negative factor that they were trained on. This will
        come based on the negative factor being included in the model_id

        Status:
        - works without saving with negative factor
    """
    for data in dataset:
        spectrogram = data[0]
        gt_call_path = data[2]

        # Get the spec id
        tags = gt_call_path.split('/')
        tags = tags[-1].split('_')
        data_id = tags[0] + '_' + tags[1]
        print ("Generating Prediction for:", data_id)

        if sliding_window:
            # May want to play around with the threhold for which we use the second model!
            # For the true predicitions we may also want to actually see if there is a contiguous segment
            # long enough!! Let us try!
            predictions = predict_spec_sliding_window(spectrogram, model, 
                                        chunk_size=chunk_size, jump=jump, 
                                        hierarchical_model=hierarchical_model, 
                                        hierarchy_threshold=parameters.FALSE_NEGATIVE_THRESHOLD)
        else:
            # Just leave out for now!
            predictions = predict_spec_full(spectrogram, model)

        # Save preditions
        # Save for now to a folder determined by the model id
        path = predictions_path + '/' + model_id
        if not os.path.isdir(path):
            os.mkdir(path)
        # The data id associates predictions with a particular spectrogram
        np.save(path + '/' + data_id  + '.npy', predictions)

def test_overlap(s1, e1, s2, e2, threshold=0.1, is_truth=False):
    """
        Test is the source call defined by [s1: e1 + 1] 
        overlaps (if is_truth = False) or is overlaped
        (if is_truth = True) by the call defined by
        [s2: e2 + 1] with given threshold.
    """
    #print ("Checking overlap of s1 = {}, e1 = {}".format(s1, e1))
    #print ("and s2 = {}, e2 = {}".format(s2, e2))
    len_source = e1 - s1 + 1
    #print ("Len Test Call: {}".format(len_source))
    test_call_length = e2 - s2 + 1
    #print ("Len Compare Call: {}".format(test_call_length))
    # Overlap check
    if s2 < s1: # Test call starts before the source call
        # The call is larger than our source call
        if e2 > e1:
            if is_truth:
                return True
            # The call we are comparing to is larger then the source 
            # call, but we must make sure that the source covers at least
            # overlap xs this call. Kind of edge case should watch this
            elif len_source >= max(threshold * test_call_length, 1): 
                #print ('Prediction = portion of GT')
                return True
            else:
                # NOTE this is an edge case and should be considered. We probably out of consistancy should not
                # Include this as a good predction because it is too small!
                #print ("Prediction = Smaller than threshold portion of GT") 
                return True
        else:
            overlap = e2 - s1 + 1
            #print ("Test call starts before source call")
            if is_truth and overlap >= max(threshold * len_source, 1): # Overlap x% of ourself
                return True
            elif not is_truth and overlap >= max(threshold * test_call_length, 1): # Overlap x% of found call
                return True
    elif e2 > e1: # Call ends after the source call
        overlap = e1 - s2 + 1
        #print ('Test call ends after source')
        if is_truth and overlap >= max(threshold * len_source, 1): # Overlap x% of ourself
            return True
        elif not is_truth and overlap >= max(threshold * test_call_length, 1): # Overlap x% of found call
            return True
    else: # We are completely in the call
        #print ('Test call completely in source call')
        if not is_truth:
            return True
        elif test_call_length >= max(threshold * len_source, 1):
            return True

    return False

def call_prec_recall(test, compare, threshold=0.1, is_truth=False, spectrogram=None, preds=None, gt_labels=None):
    """
        Adapted from Peter's paper.
        Given calls defined by their start and end time (in sorted order by occurence)
        we want to compute the "precision" and "recall." If is_truth = False then 
        we are comparing the predicted elephant calls from the detector to the ground
        truth elephant calls ==> (True Pos. and False Pos.). If is_truth = True then we
        are comparing the ground truth to the predictions and looking for ==>
        (True Pos. and False Neg.).

        1) is_truth = True: try to find a call in compare that overlaps (by at least prop. threshold) for
        each call in test

        2) is_truth = False: for a call in test, try to find a call in compare that it
        overlaps (by at least prop. threshold).

        Note 2) if threshold = 0 then we just want any amount of threshold
    """
    index_compare = 0
    true_events = []
    false_events = []
    for call in test:
        # Loop through the calls compare checking if the start is before 
        # our end
        start = call[0]
        end = call[1]
        length = call[2]

        # Rewind index_compare. Although this leads
        # to potentially extra computation, it is necessary
        # to avoid issues with overlapping calls, etc.
        # Can occur if on call overlaps two calls
        # or if we have overlapping calls
        # We basically want to consider all the calls
        # that could possibly overlap us.
        while index_compare > 0:
            # Back it up if we are past the end
            if (index_compare >= len(compare)):
                index_compare -= 1
                continue

            compare_call = compare[index_compare]
            compare_start = compare_call[0]
            compare_end = compare_call[1]

            # May have gone to far so back up
            if (compare_end > start):
                index_compare -= 1
            else:
                break

        found = False
        while index_compare < len(compare):
            # Get the call we are on to compare
            compare_call = compare[index_compare]
            compare_start = compare_call[0]
            compare_end = compare_call[1]
            compare_len = compare_call[2]

            # Call ends before 
            if compare_end < start:
                index_compare += 1
                continue

            # Call start after. Thus
            # all further calls end after. 
            if compare_start > end:
                break

            # If we are here then we know
            # compare_end >= start and compare_start <= end
            # so the calls overlap.
            if test_overlap(start, end, compare_start, compare_end, threshold, is_truth):
                found = True
                break

            # Let us think about tricky case with overlapping calls!
            # May need to rewind!
            index_compare += 1


        if found:
            true_events.append(call)
        else:
            false_events.append(call)
            #visualize_predictions([call], spectrogram, preds, gt_labels, label="False Pos")

    return true_events, false_events


def process_ground_truth(label_path, in_seconds=False, samplerate=8000, NFFT=4096., hop=800.): # Was 3208, 641
    """
        Given ground truth call data for a given day / spectrogram, 
        generate a list of elephant calls as (start, end, duration).

        If in_seconds = True we leave the values in seconds

        Otherwise convert to the corresponding spectrogram "slice"
    """
    labelFile = csv.DictReader(open(label_path,'rt'), delimiter='\t')
    calls = []

    for row in labelFile:
        # Use the file offset to determine the start of the call
        start_time = float(row['File Offset (s)'])
        call_length = float(row['End Time (s)']) - float(row['Begin Time (s)'])
        end_time = start_time + call_length
        
        if in_seconds:
            calls.append((start_time, end_time, call_length))
        else: 
            # Figure out which spectrogram slices we are on
            # to get columns that we want to span with the given
            # slice. This math transforms .wav indeces to spectrogram
            # indices
            start_spec = max(math.ceil((start_time * samplerate - NFFT / 2.) / hop), 0)
            end_spec = min(math.ceil((end_time * samplerate - NFFT / 2.) / hop), labelMatrix.shape[0])
            len_spec = end_spec - start_spec 
            calls.append((start_spec, end_spec, len_spec))

    return calls


def find_elephant_calls(binary_preds, min_length=10, in_seconds=False, samplerate=8000., NFFT=4096., hop=800.): # Was 3208, 641
    """
        Given a binary predictions vector, we now want
        to step through and locate all of the elephant
        calls. For each continuous stream of 1s, we define
        a found call by the start and end frame within 
        the spectrogram (if in_seconds = False), otherwise
        we convert to the true start and end time of the call
        from begin time 0.

        If min_length != 0, then we only keep calls of
        a given length. Note that min_length is in FRAMES!!

        Note: some reference frame lengths.
        - 20 frames = 2.4 seconds 
        - 15 frames = 1.9 seconds
        - 10 frames = 1.4 seconds
    """
    calls = []
    processed_preds = binary_preds.copy()
    search = 0
    while True:
        begin = search
        # Look for the start of a predicted calls
        while begin < binary_preds.shape[0] and binary_preds[begin] == 0:
            begin += 1

        if begin >= binary_preds.shape[0]:
            break

        end = begin + 1
        # Look for the end of the call
        while end < binary_preds.shape[0] and binary_preds[end] == 1:
            end += 1

        call_length = end - begin

        # Found a predicted call!
        if (call_length >= min_length):
            if not in_seconds:
                # Note we subtract -1 to get the last frame 
                # that has the actual call
                calls.append((begin, end - 1, call_length))
            else:
                # Frame k is centered around second
                # (NFFT / sr / 2) + (k) * (hop / sr)
                # Meaning the first time it captures is (k) * (hop / sr)
                # Remember we zero index!
                begin_s = (begin) * (hop / samplerate) #(NFFT / samplerate / 2.) + 
                # Here we subtract 1 because end goes one past last column
                # And the last time it captures is (k) * (hop / sr) + NFFT / sr
                end_s = (end - 1) * (hop / samplerate) + (NFFT / samplerate)
                # k frames spans ==> (NFFT / sr) + (k-1) * (hop / sr) seconds
                call_length_s = (NFFT / samplerate) + (call_length - 1) * (hop / samplerate)

                calls.append((begin_s, end_s, call_length_s))
        else: # zero out the too short predictions
            processed_preds[begin:end] = 0

        search = end + 1

    return calls, processed_preds


def get_binary_predictions(predictions, threshold=0.5, smooth=True, sigma=1):
    """
        Generate the binary 0/1 predictions and output the 
        predictions vector used to do so. Note this is only important
        when we smooth the predictions vector.
    """
    if (smooth):
        predictions = gaussian_filter1d(predictions, sigma)

    binary_preds = np.where(predictions > threshold, 1, 0)

    return binary_preds, predictions

def eval_full_spectrograms(dataset, model_id, predictions_path, pred_threshold=0.5, overlap_threshold=0.1, smooth=True, 
            in_seconds=False, use_call_bounds=False, min_call_lengh=15, visualize=False):
    """

        After saving predictions for the test set of full spectrograms, we
        now want to calculate evaluation metrics on each of these spectrograms. 
        First we load in the sigmoid (0, 1) predictions and use the defined
        threshold and smoothing flag to convert the predictions into binary
        0/1 predcitions for each time slice. Then, we convert time slice
        predictions to full call (start, end) time predictions so that
        we can calculate elephant call specific evaluation metrics. Bellow
        we discuss the metrics that are calculated individually for each
        full spectrogram, as well as accross all the test spectrograms
        
        Metrics:
        - Call Prediction True Positives
        - Call Prediction False Positives
        - Call Recall True Positives
        - Call Recall False Negatives
        - F-score
        - Accuracy
        - Old Call Precision --- To be implemented
        - Old Call Recall --- To be implemented

    """
    # Maps spectrogram ids to dictionary of results for each spect
    # Additionally includes a key "summary" that computes aggregated
    # statistics over the entire test set of spectrograms
    results = {} 
    results['summary'] = {'true_pos': 0,
                            'false_pos': 0,
                            'true_pos_recall': 0,
                            'false_neg': 0,
                            'f_score': 0,
                            'accuracy': 0
                            }
    # Used to track the number of total calls for averaging
    # aggregated statistics
    num_preds = 0
    num_gt = 0
    for data in dataset:
        spectrogram = data[0]
        labels = data[1]
        gt_call_path = data[2]

        # Get the spec id
        tags = gt_call_path.split('/')
        tags = tags[-1].split('_')
        data_id = tags[0] + '_' + tags[1]
        print ("Generating Prediction for:", data_id)
        
        predictions = np.load(predictions_path + '/' + model_id + "/" + data_id + '.npy')

        binary_preds, smoothed_predictions = get_binary_predictions(predictions, threshold=pred_threshold, smooth=smooth)

        # Process the predictions to get predicted elephant calls
        # Figure out better way to try different combinations of this
        # Note that processed_preds zeros out predictions that are not long
        # enough to be an elephant call
        predicted_calls, processed_preds = find_elephant_calls(binary_preds, in_seconds=in_seconds)
        print ("Num predicted calls", len(predicted_calls))

        # Use the calls as defined in the orginal hand labeled file.
        # This looks to avoid issues of overlapping calls seeming like
        # single very large calls in the gt labeling 
        if use_call_bounds:
            print ("Using CSV file with ground truth call start and end times")
            gt_calls = process_ground_truth(gt_call_path, in_seconds=in_seconds)
        else:
            print ("Using spectrogram labeling to generate GT calls")
            # We should never compute this in seconds
            # Also let us keep all the calls, i.e. set min_length = 0
            gt_calls, _ = find_elephant_calls(labels, min_length=0)

        print ("Number of ground truth calls", len(gt_calls))

        # Visualize the predictions around the gt calls
        if visualize: # This is not super important
            visual_full_recall(spectrogram, smoothed_predictions, labels, processed_preds)       
        
        # Look at precision metrics
        # Call Prediction True Positives
        # Call Prediction False Positives
        true_pos, false_pos = call_prec_recall(predicted_calls, gt_calls, threshold=overlap_threshold, is_truth=False,
                                                spectrogram=spectrogram, preds=binary_preds, gt_labels=labels)

        # Look at recall metrics
        # Call Recall True Positives
        # Call Recall False Negatives
        true_pos_recall, false_neg = call_prec_recall(gt_calls, predicted_calls, threshold=overlap_threshold, is_truth=True)

        f_score = get_f_score(binary_preds, labels) # just for the postive class
        accuracy = calc_accuracy(binary_preds, labels)

        results[data_id] = {'true_pos': true_pos,
                            'false_pos': false_pos,
                            'true_pos_recall': true_pos_recall,
                            'false_neg': false_neg,
                            'f_score': f_score,
                            'predictions': smoothed_predictions,
                            'binary_preds': processed_preds,
                            'accuracy': accuracy
                            }
        # Update summary stats
        results['summary']['true_pos'] += len(true_pos)
        results['summary']['false_pos'] += len(false_pos)
        results['summary']['true_pos_recall'] += len(true_pos_recall)
        results['summary']['false_neg'] += len(false_neg)
        results['summary']['f_score'] += f_score
        results['summary']['accuracy'] += accuracy

    # Calculate averaged statistics
    results['summary']['f_score'] /= len(dataset)
    results['summary']['accuracy'] /= len(dataset)

    return results

def get_spectrogram_paths(test_files_path, spectrogram_path):
    """
        In the test set folder, there is a file that includes
        all of the recording files used for the test set. Based
        on these files we want to get the spectrograms and gt
        labeling files that correspond
    """
    # Holds the paths to the:
    # - spectrograms
    # - labels for each spectrogram slice
    # - gt (start, end) times for calls
    paths = {'specs': [],
            'labels': [],
            'gts': []}

    with open(test_files_path, 'r') as f:
        lines = f.readlines()

    files = [x.strip() for x in lines]

    for file in files:
        # Create the spectrogram path by concatenating
        # the test file with the path to the folder
        # containing the spectrogram files
        paths['specs'].append(spectrogram_path + '/' + file + '_spec.npy')
        paths['labels'].append(spectrogram_path + '/' + file + '_label.npy')
        paths['gts'].append(spectrogram_path + '/' + file + '_gt.txt')

    return paths

def main(args):
    """
    Example runs:

    """
    # Load Model_0 and Model_1 of the hierarchical models
    hierarchical_model_path = args.model
    model_0_path = os.path.join(hierarchical_model_path, "Model_0/model.pt")
    model_1_path = os.path.join(hierarchical_model_path, "Model_1/model.pt")

    model_0, model_id = loadModel(model_0_path)
    model_1, _ = loadModel(model_1_path)
    # Put in eval mode!
    model_0.eval()
    model_1.eval()
    print (model_id)
    
    full_test_spect_paths = get_spectrogram_paths(args.test_files, args.spect_path)
    full_dataset = ElephantDatasetFull(full_test_spect_paths['specs'],
                 full_test_spect_paths['labels'], full_test_spect_paths['gts'])    

    if args.make_full_preds:
        generate_predictions_full_spectrograms(full_dataset, model_0, model_id, args.predictions_path,
             sliding_window=True, chunk_size=256, jump=128, hierarchical_model=model_1)         # Add in these arguments
    elif args.full_stats:
        # Now we have to decide what to do with these stats
        results = eval_full_spectrograms(full_dataset, model_id, args.predictions_path)

        if args.visualize: # Visualize the metric results
            test_elephant_call_metric(full_dataset, results)

        # Display the output of results as peter did
        TP_truth = results['summary']['true_pos_recall']
        FN = results['summary']['false_neg']
        TP_test = results['summary']['true_pos']
        FP = results['summary']['false_pos']

        recall = TP_truth / (TP_truth + FN)
        precision = 0 if TP_test + FP == 0 else TP_test / (TP_test + FP) # For edge 0 case
        f1_call = 0 if precision + recall == 0 else 2 * (precision * recall) / (precision + recall)
        # Do false pos rate later!!!!
        total_duration = 24. * len(full_test_spect_paths['specs'])
        false_pos_per_hour = FP / total_duration

        print ("Summary results")
        print("Call precision:", precision)
        print("Call recall:", recall)
        print("f1 score calls:", f1_call)
        print("False positve rate (FP / hr):", false_pos_per_hour)
        print("Segmentation f1-score:", results['summary']['f_score'])
        print("Average accuracy:", results['summary']['accuracy'])


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

