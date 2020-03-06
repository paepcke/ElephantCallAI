'''
Created on Mar 2, 2020

@author: paepcke
'''
import csv
import sys

from scipy.io import wavfile
import numpy as np
from collections import deque

class PrecRecComputer(object):
    '''
    Given a noise-gated .wav file, find all its non-zero
    places. Load a Raven label file, and compute precision/recall
    based on which manually labeled audio segments have at least
    10% overlap with a .wav sequence of non-zero voltages. The
    overlap is a parameter.
    '''

    SAMPLE_RATE_FOR_TESTING = 4000

    def __init__(self, wavefile, labelfile, print_res=True, plot=False, testing=False):
        '''
        Constructor
        '''

        # Read the samples:
        if not testing:
            try:
                self.log.info("Reading .wav file...")        
                (self.framerate, samples) = wavfile.read(wavefile)
                self.log.info("Done reading .wav file.")        
            except Exception as e:
                print(f"Cannot read .wav file: {repr(e)}")
                sys.exit(1)

            # Read the label file:
            try:
                with open(labelfile, 'r') as label_fd:
                    label_reader = csv.reader(label_fd, delimiter='\t')
            except Exception as e:
                print(f"Could not match waves and labels: {repr(e)}")
                sys.exit(1)
            finally:
                pass
    
            performance_result = self.compute_performance(samples, label_reader)
            
            if plot:
                self.plot(performance_result)
                
            if print_res:
                performance_result.print()
                
        else: # Unittesting:
            # Just assume some framerate.
            self.framerate = self.SAMPLE_RATE_FOR_TESTING

    #------------------------------------
    # compute_performance
    #-------------------
    
    def compute_performance(self, samples, label_csv_reader, overlap_perc_requirement):
        '''
        Workhorse. Takes audio sample array, and a csv reader
        pointed at a label file. The percentage number is the
        overlap between audio-derived events and the corresponding
        labeled event that is minimally required to cound an
        audio-detected event as a true event.
        
        @param samples: the voltages
        @type samples: np.array({float | int})
        @param label_csv_reader: csv.reader ready to read lines from 
            labels file.
        @type label_csv_reader: csv.reader
        @param overlap_perc_requirement: amount of overlap required between
            audio event and labeled event for the audio event to be counted
            correct.
        @type overlap_perc_requirement: {float | int}
        @return: a PerformanceResult object containing all results.
        @rtype: PerformanceResult
        '''
        
        # Get array of ElephantEvent instances as a deque
        # (double-ended queue):
        
        elephant_events = self.label_file_reader(label_csv_reader)
        audio_events    = self.collect_audio_events(samples)
        num_el_events   = len(elephant_events)
        num_aud_events  = len(audio_events)
        
        # Number of samples labeled as elephant calls:
        # For simplicity copy the deques into np arrays
        # b/c we have ready ops there:
        elephant_events_np = np.array(elephant_events.copy())
        
        # Get two vectors of 1s/0s. Each has length samples.size.
        # Each has a 1 if an elephant presence is predicted:
        (samples_mask, elephant_label_mask) = self.get_el_and_aud_masks(samples,elephant_events_np)
        
        # Easy to do the various audio false positive/negative counts
        # at the sample level using the two 1/0 masks:
        num_true_positive_samples  = np.sum(np.logical_and(elephant_label_mask, samples_mask))
        num_true_negative_samples  = np.sum(np.logical_not(np.logical_or(elephant_label_mask, samples_mask)))
        
        # More roundabout for false pos/neg: Use the np.ma.masked_array
        # facility: in samples_mask, blend out elements that are greater/less
        # then their corresponding label mask elements. Then count those
        # 'bad' elements. Note notation ambiguity: we have the above 
        # sample and elephant-label masks: [1,0,0,1,...]. The np.ma facility
        # speaks of arrays with a built-in mask. Think of the latter mask
        # as 'blending out' elements of our sample/elephant-label masks:
         
        # False positive: blend out all the samples_mask elements
        #   that are *greater* than their corresponding label mask.
        masked_as_false_pos = np.ma.masked_greater(samples_mask, elephant_label_mask)
        # Then grab only the blended elements, and see how many there are:
        num_false_positive_samples = masked_as_false_pos[masked_as_false_pos.mask].size
        
        # False negatives: blend out all in samples_mask elements
        #   that are *less* than their corresponding labels_mask:
        masked_as_false_neg = np.ma.masked_less(samples_mask, elephant_label_mask)
        # Grab only the blended elements, and see how many there are:
        num_false_negative_samples = masked_as_false_neg[masked_as_false_neg.mask].size

        # Remember number of ele events before
        # we start popping them off the queue:
        num_labeled_samples   = self.get_total_labeled_samples(elephant_events)
        
        # But: have not found a method for using just the masks
        # to compute percent overlap of each labeled burst. Use
        # loop over just the label-based ElephantEvent instances.
        # There aren't that many; maybe 2000 per 24 hours:
        
        # Remember that the audio and el_evnt objects
        # are in deque structures, not arrays. So we
        # can use as queues. Init the current audio/label events:

        curr_audio_evnt = audio_events.popleft()
        curr_el_evnt    = elephant_events.popleft()

        # Counters:
        num_false_pos_events       = 0
        num_false_neg_events       = 0
        
        # Keep all overlap percentages:
        overlap_perc    = np.array([])
        
        # All this in samples, not events: 
        done = False
        while not done:
            try:
                el_start = int(self.framerate * curr_el_evnt.begin_time)
                el_end   = int(self.framerate * curr_el_evnt.end_time)
                
                aud_start = curr_audio_evnt.begin_time
                aud_end   = curr_audio_evnt.end_time

                # Is audio event earlier than the el ev?
                if aud_end < el_start:
                    # False positive
                    num_false_pos_events  += 1
                    curr_audio_evnt = audio_events.popleft()
                    continue
                
                # Is audio event after the el ev?
                if aud_start > el_end:
                    # False negative:
                    num_false_neg_events  += 1
                    # Missed it; next elephant label:
                    curr_el_evnt = elephant_events.popleft()
                    continue
                
                # Amount of audio inside label range:
                overlap_percentage = self.compute_overlap_percentage(aud_start, aud_end, el_start, el_end)
                overlap_perc = np.append(overlap_perc, overlap_percentage)

                curr_audio_evnt = audio_events.popleft()
                curr_el_evnt    = elephant_events.popleft()
                continue
                
            except IndexError:
                # Either audio or el labels are all exhausted.
                # So we have all the overlap percentages now:
                done = True

            # Results at event level:
            num_true_found_events     = overlap_perc[overlap_perc >= overlap_perc_requirement].size
            # Count the events with insufficient overlap as false negatives:
            num_false_neg_events += num_el_events - num_true_found_events
            num_false_pos_events += overlap_perc.size - num_true_found_events
            
            recall_events     = num_true_found_events / num_el_events
            precision_events  = num_true_found_events / num_aud_events
            f1score_events    = 2 * precision_events * recall_events / (precision_events + recall_events)

            # Results at sample level:
            recall_samples    = num_true_positive_samples / num_labeled_samples
            precision_samples = num_true_positive_samples / (num_true_positive_samples + num_false_positive_samples)
            f1score_samples   = 2 * precision_samples * recall_samples/ (precision_samples+ recall_samples)            
            
            # Use num_true_negative_samples as the true negative
            # at the event level: think of each 0-sample as 
            # one negative event (as well as one negative sample):
            confusion_matrix_events = np.array([num_true_found_events,    num_false_pos_events,
                                               num_false_neg_events,    num_true_negative_samples 
                                               ]).reshape(2,2)
            
            confusion_matrix_samples = np.array([num_true_positive_samples, num_false_positive_samples,
                                                num_false_negative_samples, num_true_negative_samples
                                                ]).reshape(2,2)
            res = PerformanceResult(recall_events,
                                    precision_events,
                                    f1score_events,
                                    recall_samples,
                                    precision_samples,
                                    f1score_samples,
                                    overlap_perc,
                                    confusion_matrix_events,
                                    confusion_matrix_samples)

            return res

    #------------------------------------
    # compute_overlap_percentage
    #-------------------
    
    def compute_overlap_percentage(self, 
                                   aud_begin,
                                   aud_end,
                                   el_begin,
                                   el_end):
        
        # Is there overlap at all? If so, 
        # none of the two regions must be
        # entirely on the left, or on the 
        # right of the other:
        
        if (aud_end < el_begin) or (el_end < aud_begin):
            # No overlap at all:
            return 0
        
        el_length = el_end - el_begin
        if el_begin <= aud_begin:
            # Audio starts with label, or is partially
            # shifted right (audio extends beyond label):
            return 100 * (el_end - aud_begin) / el_length
        else:
            #  Audio is shifted left (starts before label)
            return 100 * (aud_end - el_begin) / el_length

    
    #------------------------------------
    # get_total_labeled_samples
    #-------------------
    
    def get_total_labeled_samples(self, el_event_objects_np):
        '''
        Given an array of labeled-event objects,
        go through each object, and compute the number
        of samples contained between start/end markers.
        
        Assumption: self.framerate contains framerate
        
        @param el_event_objects_np: ElephantEvent objects, each
            containing start and end times in seconds of the
            labeled burst.
        @type el_event_objects_np: ElephantEvent
        @return: number of audio samples contained in 
            each burst
        @rtype: int
        '''
        
        # We need to use a loop, but OK, since there
        # won't be *that* many labeled events:
        
        labeled_samples = 0
        for label_event in el_event_objects_np:
            begin_time = np.floor(label_event.begin_time)
            end_time   = np.ceil(label_event.end_time)            
            labeled_samples += self.framerate * (end_time - begin_time)

        return labeled_samples

    #------------------------------------
    # get_el_and_aud_masks
    #-------------------
    
    def get_el_and_aud_masks(self, samples, elephant_events_np):
        '''
        Given a samples np.array, and an np array of 
        ElephantEvent instances that each represent a label
        with its start/end times. Return two np.arrays of 1s 
        and 0s, indicating where samples show elephant call
        audio, and labels indicate an el event, respectively.
        
        I.e. return two masks, one for audio samples, and one
        for labeled data. The masks 
        
        @param samples: audio samples
        @type samples: np.array({float | int})
        @param elephant_events_np: np array of ElephantEvent instances
        @type elephant_events_np: np.array(ElephantEvent)
        '''
        
        # Audio samples are easy: activity is where
        # voltage > 0:
        
        # All zeros to start with:
        samples_mask = np.zeros(samples.size, dtype=int)
        
        # Get array of indices into samples where
        # value is non-zero:
        indx_to_nonzero = np.where(samples > 0)
        
        # Modify mask to have 1s where samples are non-zero:
        samples_mask[indx_to_nonzero] = 1
        
        # For the labels: start with a mask of all 0,
        # length same as sample mask:
        
        el_mask = np.zeros(samples.size, dtype=int)
        
        # For each labeled event, compute where
        # in the mask the event starts and ends,
        # and fill that mask portion with ones.
        
        for label_event in elephant_events_np:
            start_sample = self.framerate * np.floor(label_event.begin_time)
            end_sample   = self.framerate * np.ceil(label_event.end_time)
            # Ensure we honor the length of the mask:
            end_sample = min(end_sample, el_mask.size)
            el_mask[int(start_sample) : int(end_sample)] = 1
        
        return (samples_mask, el_mask) 

    #------------------------------------
    # collect_audio_events
    #-------------------    

    def collect_audio_events(self, samples):
        '''
        Returns an array of AudioEvent objects
        found in the samples array
        
        @param samples: array of audio samples
        @type samples: numpy.array
        @return: deque of AudioEvent objects
        @rtype: deque
        '''

        # Double-ended queue:
        audio_events = deque()
        
        # Get array of indexes into samples array
        # where sample is not zero; nonzero() returns
        # a one-tuple, therefore the [0]:

        non_zeros = np.nonzero(samples)[0]
        if non_zeros.size == 0:
            # No audio bursts at all:
            return audio_events
        
        curr_burst_start = non_zeros[0]
        curr_burst_end   = non_zeros[0]
        
        for non_zero_sample_pt in non_zeros[1:]:
            if non_zero_sample_pt == curr_burst_end + 1:
                # Still in a contiguous set of non-zeros:
                curr_burst_end = non_zero_sample_pt
            else:
                # End of a burst: 1 beyond the last non-zero
                # measurement:
                audio_events.append(AudioEvent(curr_burst_start, curr_burst_end + 1))
                curr_burst_start = non_zero_sample_pt
                curr_burst_end   = non_zero_sample_pt

        # If burst ended with the end of the sample file,
        # add that closing burst:
        audio_events.append(AudioEvent(curr_burst_start, curr_burst_end + 1)) 
        return audio_events

    #------------------------------------
    # label_file_reader
    #-------------------
    
    def label_file_reader(self, csv_reader):
        '''
        Given a CSV reader obj, return an array of 
        ElephantEvent instances.
        
        @param csv_reader:
        @type csv_reader:
        @returns instances of elephant events, one for each label entry 
            in the label table. Events are in a deque, so can be treated
            as a queue.
        @rtype: deque(ElepantEvent)
        '''
        
        # For result:
        events = deque()
        
        # Discard col header:
        _header = next(csv_reader)
        
        COL_SELECTION_INDEX = 0
        COL_BEGIN_TIME_INDEX = 3
        COL_END_TIME_INDEX = 4
        # Each line is an array of 18 fields:
        for line in csv_reader:
            # Watch for (usually closing...) empty line(s):
            if len(line) == 0:
                continue
            if csv_reader.dialect.delimiter != '\t':
                # Unless the csv reader knows that elephant
                # label files are tab-delimited, lines come 
                # in as an arr of len 1: a string with embedded 
                # '\t' chars. In that case: create an array:
                line = line[0].split('\t')
                
            selection_index = int(line[COL_SELECTION_INDEX])
            begin_time = float(line[COL_BEGIN_TIME_INDEX])
            end_time = float(line[COL_END_TIME_INDEX])
            events.append(ElephantEvent(selection_index, begin_time, end_time))
            
        return events

# -------------------------------- ElephantEvent -------------

class ElephantEvent(object):
    '''
    Instances hold information from the label files,
    i.e. the Raven selection tables.
    '''
    
    def __init__(self, selection_index, begin_time, end_time):
        self._selection_index = selection_index
        self._begin_time = begin_time
        self._end_time = end_time
        
    #------------------------------------
    # Properties
    #-------------------    

    @property
    def selection_index(self):
        return self._selection_index
    
    @property
    def begin_time(self):
        return self._begin_time

    @property
    def end_time(self):
        return self._end_time
    
# ------------------------------------ Audio Event -------------------------

class AudioEvent(object):
    '''
    Contains information about one continuous run
    of non-zero samples.
    '''
    
    def __init__(self, begin_time, end_time):
    
        self._begin_time = begin_time
        self._end_time = end_time
        
    @property
    def begin_time(self):
        return self._begin_time
    
    @property
    def end_time(self):
        return self._end_time

# --------------------------------------- PerformanceResult ------------------------

class PerformanceResult(object):
    '''
    Instances hold results from performance
    computations. Quantities ending in '_events'
    refer to performance at the level of 'burst'.
    For labels this means all labeled sets
    of samples. For samples it means clusteres of
    non-zero sample values.
    
    '''
    
    def __init__(self,
                 recall_events,
                 precision_events,
                 f1score_events,
                 recall_samples,
                 precision_samples,
                 f1score_samples,
                 overlap_percentages,
                 confusion_matrix_events,
                 confusion_matrix_samples
                 ):
        self.__recall_events = recall_events
        self.__precision_events = precision_events
        self.__f1score_events = f1score_events
        self.__recall_samples = recall_samples
        self.__precision_samples = precision_samples
        self.__f1score_samples = f1score_samples
        self.__overlap_percentages = overlap_percentages
        self.__confusion_matrix_events = confusion_matrix_events
        self.__confusion_matrix_samples = confusion_matrix_samples

    @property
    def recall_events(self):
        return self.__recall_events

    @property
    def precision_events(self):
        return self.__precision_events

    @property
    def f1score_events(self):
        return self.__f1score_events

    @property
    def recall_samples(self):
        return self.__recall_samples

    @property
    def precision_samples(self):
        return self.__precision_samples

    @property
    def f1score_samples(self):
        return self.__f1score_samples

    @property
    def confusion_matrix_events(self):
        return self.__confusion_matrix_events

    @property
    def confusion_matrix_samples(self):
        return self.__confusion_matrix_samples

    @property
    def overlap_percentages(self):
        return self.__overlap_percentages
    
    #------------------------------------
    # print_res
    #-------------------    
    
    def print_res(self, fd=sys.stdout):
        fd.write(f"Recall (event level):\t{self.recall_events}\n")
        fd.write(f"Precision (event level):\t{self.precision_events}\n")
        fd.write(f"F1 (event level):\t{self.f1score_events}\n")
        fd.write(f"Recall (sample level):\t{self.recall_samples}\n")
        fd.write(f"Precision (sample level):\t{self.precision_samples}\n")
        fd.write(f"F1 (sample level):\t{self.f1score_samples}\n")
        fd.write(f"Confusion matrix (event level):\t{self.confusion_matrix_events}\n")
        fd.write(f"Confusion matrix (event level):\t{self.confusion_matrix_samples}\n")
        
        
        
        
        
        