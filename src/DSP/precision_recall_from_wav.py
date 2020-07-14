'''
Created on Mar 2, 2020

@author: paepcke
'''

from collections import OrderedDict
import csv
import os
import re
import sys

from scipy.io import wavfile
from dsp_utils import SignalTreatmentDescriptor

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from elephant_utils.logging_service import LoggingService
import numpy as np
import numpy.ma as ma
from plotting.plotter import PlotterTasks

class PrecRecComputer(object):
    '''
    Given a noise-gated .wav file and a Raven elephant
    labels file, compute precision/recall, and confusion matrix. 
    These computations are performed separately for both 'events'
    (clusters of non-zero wav file signals), and individual samples.
    
    Confusion matrices are of the form:
                
                            Labels
                         true     false
                   true   2         0
            Audio false   1         6
    
    Clients of this class can specify how much overlap between the
    audio non-zero samples and the corresponding elephant labels are
    is required to count the audio as having discovered the burst.

    '''

    SAMPLE_RATE_FOR_TESTING = 4000
    logfile = '/tmp/precrec_computer.log'

    def __init__(self,
                 signal_treatment,
                 wavefile, 
                 labelfile, 
                 overlap_percentages=10, 
                 print_res=True, 
                 testing=False):
        '''
        Constructor
        '''
        PrecRecComputer.log = LoggingService(logfile=self.logfile)
        
        #*********
        import logging
        self.log.logging_level = logging.ERROR
        #*********

        
        self.log = PrecRecComputer.log

        if not isinstance(overlap_percentages, list):
            overlap_percentages = [overlap_percentages]
            
        self.print_res = print_res
        # Read the samples:
        if not testing:
            try:
                self.log.info("Reading .wav file...")        
                (self.framerate, samples) = wavfile.read(wavefile)
                self.log.info("Done reading .wav file.")        
            except Exception as e:
                print(f"Cannot read .wav file {wavefile}: {repr(e)}")
                sys.exit(1)

            self.performance_results = []
            for overlap_perc_requirement in overlap_percentages:
                signal_treatment.add_overlap(overlap_perc_requirement)
                performance_result = self.compute_performance(signal_treatment, 
                                                              samples, 
                                                              labelfile, 
                                                              overlap_perc_requirement)
                self.performance_results.append(performance_result)
            
                if PlotterTasks.has_task('performance_results'):
                    self.plot(performance_result)
                    
                if self.print_res:
                    performance_result.print()
                
        else: # Unittesting:
            # Just assume some framerate.
            self.framerate = self.SAMPLE_RATE_FOR_TESTING

    #------------------------------------
    # compute_performance
    #-------------------
    
    def compute_performance(self, 
                            signal_treatment, 
                            samples, 
                            label_file_path, 
                            overlap_perc_requirement):
        '''
        Workhorse. Takes audio sample array, and a csv path. 
        The percentage number is the overlap between audio-derived 
        events and the corresponding labeled event that is minimally 
        required to count an audio-detected event as a true event.
    
        @param signal_treatment: a string that identifies the 
            changes made to the original audio signal. Ex: '-30dB_300Hz'
        @type signal_treatment
        @param samples: the voltages
        @type samples: np.array({float | int})
        @param label_file_path: path to Raven label file 
        @type label_file_path: str
        @param overlap_perc_requirement: amount of overlap required between
            audio event and labeled event for the audio event to be counted
            correct.
        @type overlap_perc_requirement: {float | int}
        @return: a PerformanceResult object containing all results.
        @rtype: PerformanceResult
        '''
        
        # Phase 1: which *samples* are part of a true,
        #          labeled call?
        
        # Put start/ends of labeled events into the form:
        #     [[start_first_label_samples, end_first_label_samples],
        #      [start_second_label_samples, end_second_label_samples],
        #      ...
        #     ]
 
        (elephant_burst_indices, el_samples_mask) = self.label_file_reader(label_file_path)
 
        # For the audio bursts: get 
        #     [[start_first_burst, end_first_burst],
        #      [start_second_burst, end_second_burst],
        #      ...
        #     ]
        (audio_burst_indices, aud_samples_mask)  = self.collect_audio_events(samples)
        
        # Get the 'non-events' for both the labels and
        # the audio guesses. A non-event is the distance
        # between two event intervals. Using above ex.:
        #   [[end_first_burst, start_second_burst],
        #    [end_second_burst, start_third_burst],
        #    ...
        #   ]
        
        elephant_burst_indices_shifted = np.roll(elephant_burst_indices, -1, axis=0)
        last_sample_index = el_samples_mask.size # mele.size
        elephant_burst_indices_shifted[-1] = np.array([last_sample_index, -1])
        starts_and_ends = elephant_burst_indices[:,1], elephant_burst_indices_shifted[:,0]
        elephant_non_burst_indices = np.column_stack(starts_and_ends)
        # The first entry should be 0 to start of first interval.
        # Add that entry:
        elephant_non_burst_indices = np.vstack((np.array([0,elephant_burst_indices[0,0]]), elephant_non_burst_indices))

        # Same for audio bursts:
        audio_burst_indices_shifted = np.roll(audio_burst_indices, -1, axis=0)
        last_sample_index = aud_samples_mask.size # mele.size
        audio_burst_indices_shifted[-1] = np.array([last_sample_index, -1])
        starts_and_ends = audio_burst_indices[:,1], audio_burst_indices_shifted[:,0]
        audio_non_burst_indices = np.column_stack(starts_and_ends)
        # The first entry should be 0 to start of first interval.
        # Add that entry:
        audio_non_burst_indices = np.vstack((np.array([0,audio_burst_indices[0,0]]), audio_non_burst_indices))
        
        # Pad the shorter of the two masks with zeros to make
        # them the same lengths:
        mask_size_diff = aud_samples_mask.size - el_samples_mask.size
        if mask_size_diff > 0:
            el_samples_mask = np.pad(el_samples_mask, (0,mask_size_diff), 'constant')
        elif mask_size_diff < 0:
            aud_samples_mask = np.pad(aud_samples_mask, (0, -mask_size_diff), 'constant')
        
        # Easy to do the various audio false positive/negative counts
        # at the sample level using the two 1/0 masks:
        self.log.info('Computing true-pos at sample granularity...')
        num_true_positive_samples  = np.sum(np.logical_and(el_samples_mask, aud_samples_mask))
        self.log.info('Done computing true-pos at sample granularity.')
        
        self.log.info('Computing true-neg at sample granularity...')
        num_true_negative_samples  = np.sum(np.logical_not(np.logical_or(el_samples_mask, aud_samples_mask)))
        self.log.info('Done computing true-neg at sample granularity.')
        
        self.log.info('Computing false-neg at sample granularity...')
        num_false_negative_samples  = np.where(np.logical_and(el_samples_mask == 1, 
                                                              aud_samples_mask == 0))[0].size
        self.log.info('Done computing false-neg at sample granularity.')
        
        self.log.info('Computing false-pos at sample granularity...')
        num_false_positive_samples  = np.where(np.logical_and(el_samples_mask == 0, 
                                                              aud_samples_mask == 1))[0].size
        self.log.info('Done computing false-pos at sample granularity.')

        # Recall: samples recognized by audio as part of a call,
        #         over samples labeled as par of a call:
        self.log.info('Computing recall/precision/F1 at sample granularity...')
        try:
            recall_samples    = num_true_positive_samples / np.sum(el_samples_mask)
        except ZeroDivisionError:
            recall_samples = np.inf
            
        # Precision: samples recognized by audio as part of a call,
        #         over all the samples:
        
        try:
            precision_samples = num_true_positive_samples / (num_true_positive_samples + num_false_positive_samples)
        except ZeroDivisionError:
            precision_samples = np.inf 
        
        try:
            f_score_samples  = 2 * precision_samples * recall_samples / (precision_samples + recall_samples)
        except ZeroDivisionError:
            f_score_samples = np.inf
            
        self.log.info('Done computing recall/precision/F1 at sample granularity.')

        # Phase 2: At event level
        
        (percent_overlaps, 
         matches_aud, 
         num_true_pos_detected_events,
         num_false_neg_detected_events,
         num_false_pos_detected_events
         ) = self.compute_overlap_percentage(audio_burst_indices, 
                                             elephant_burst_indices).values()
        (percent_overlaps_non_bursts, 
         matches_aud_non_bursts,
         num_true_detected_non_events,           # Found non-events
         num_false_neg_detected_non_events,      # Was actually an event
         num_false_pos_detected_non_events       # Was actually not an event
         ) = self.compute_overlap_percentage(audio_non_burst_indices, 
                                             elephant_non_burst_indices).values()
        # Detected audio bursts, including the false ones,
        # and including the multiple separate aud events that 
        # lie within an individual ele event:
        num_detected_events     = audio_burst_indices[:,0].size
        num_elephant_events     = elephant_burst_indices[:,0].size

        self.log.info('Computing recall/precision/F1 at event granularity; no overlap constraints...')
        # First, compute prec/recall without considering
        # the constraint of overlap:
        recall_events_any_overlap = num_true_pos_detected_events / num_elephant_events
        try:
            precision_events_any_overlap = num_true_pos_detected_events / \
                               (num_true_pos_detected_events + num_false_pos_detected_events)
        except ZeroDivisionError:
            precision_events_any_overlap = np.inf
        try:
            f_score_events_any_overlap  = 2 * precision_events_any_overlap * recall_events_any_overlap / \
                            (precision_events_any_overlap + recall_events_any_overlap)
        except ZeroDivisionError:
            f_score_events_any_overlap = np.inf
        self.log.info('Done computing recall/precision/F1 at event granularity; no overlap constraints.')
                           
                                             
        # Keep only the audio non-burst events that overlap sufficiently:
        verified_audio_events = \
            matches_aud[np.nonzero(percent_overlaps[percent_overlaps >= overlap_perc_requirement])]
        num_verified_audio_events = verified_audio_events[:,0].size
        
        # Keep only the audio non-burst events that overlap sufficiently:
        verified_audio_non_events = \
            matches_aud_non_bursts[np.nonzero(percent_overlaps_non_bursts[percent_overlaps_non_bursts >= overlap_perc_requirement])]
        num_verified_audio_non_events = verified_audio_non_events[:,0].size
        
        self.log.info('Computing true/false-pos/neg at event granularity...')
        #num_true_pos_events = elephant_burst_indices[:,0].size
        #num_true_neg_events = elephant_non_burst_indices[:,0].size
        
        num_true_neg_detected_events = num_verified_audio_non_events
        
        self.log.info('Done computing true/false-pos/neg at event granularity.')

        # Recall: samples recognized by audio as part of a call,
        #         over events labeled as part of a call:
        self.log.info('Computing recall/precision/F1 at event granularity...')
        try:
            recall_events = num_verified_audio_events  / num_elephant_events
        except ZeroDivisionError:
            recall_events = np.inf
        
        # Precision: samples recognized by audio as part of a call,
        #         over all its predictions:
        try:
            precision_events = num_verified_audio_events / (num_verified_audio_events + num_false_pos_detected_events)
        except ZeroDivisionError:
            precision_events = np.inf
        try:
            f_score_events  = 2 * precision_events * recall_events / (precision_events + recall_events)
        except ZeroDivisionError:
            f_score_events = np.inf
        self.log.info('Done computing recall/precision/F1 at event granularity.')

        results = {'signal_treatment'    :    signal_treatment,
                   'num_elephant_events' :    num_elephant_events,
                   'num_detected_events' :    num_detected_events,
                   'recall_events' :          recall_events,
        		   'precision_events' :       precision_events,
        		   'f1score_events' :         f_score_events,
        		   'recall_samples' :         recall_samples,
        		   'precision_samples' :      precision_samples,
        		   'f1score_samples' :        f_score_samples,
        		   'true_pos_samples' :       num_true_positive_samples,
        		   'false_pos_samples' :      num_false_positive_samples,
        		   'true_neg_samples' :       num_true_negative_samples,
        		   'false_neg_samples' :      num_false_negative_samples,
        		   'true_pos_events' :        num_verified_audio_events,
        		   'false_pos_events' :       num_false_pos_detected_events,
        		   'true_neg_events' :        num_true_neg_detected_events,
        		   'false_neg_events' :       num_false_neg_detected_events,
                   
                   'true_pos_any_overlap_events':    num_true_pos_detected_events,
                   'num_true_pos_detected_non_events': num_verified_audio_non_events,
                   'num_false_pos_detected_non_events': num_false_pos_detected_non_events,
                   'num_false_neg_detected_non_events': num_false_neg_detected_non_events,
                   'true_pos_any_overlap_non_event': num_true_detected_non_events,
                   'recall_events_any_overlap' : recall_events_any_overlap,
                   'precision_events_any_overlap' : precision_events_any_overlap,
                   'f_score_events_any_overlap' : f_score_events_any_overlap
                   
                   }
        
        results['min_required_overlap'] = overlap_perc_requirement 
        performance_result = PerformanceResult(results, percent_overlaps)

        return performance_result

    #------------------------------------
    # compute_overlap_percentage
    #-------------------
    
    def compute_overlap_percentage(self, 
                                   audio_burst_indices, 
                                   elephant_burst_indices):
        '''
        Given start/end indices into the discovered audio 
        events, and the same for the labeled events, compute
        the percentage overlaps for each. A complete example,
        verified in unittests:
        
        daud = \                                                   Index in Ele   Absolute Overlap
        np.array([[ 5, 10],      # +     covers exactly                  0             4            
                  [40, 46],      # +     covers beyond borders           2             2
                  [47, 49],      # -                                                    
                  [51, 66],      # +     lower within borders            3            13
                  [70, 75],      # +     upper within borders            5             1    [ 4  2 13 10  0  2]
                  [77, 80],      # +     use ele burst twice             5             3
                  [90, 95]       # -     more in aud than in labels      
                  ])
        dele = \                                                   Index in Aud    Absolute Overlap
        np.array([[ 5, 10],                                              0             4
                 [10, 11],       # abutting intervals                    -
                 [42, 45],                                               1             2
                 [50, 65],                                               3            13
                 [55, 70],       # overlapping intervals                 3            
                 [74, 80]                                                5
                 ])
        
        Intermediate result: the matches between aud and ele bursts:
        
           Audio    Elephants Overlap       Highest Low Bounds            Lowest High Bounds           
         [[ 5 10]   [[ 5 10]    5          [ 5 42 51 55 74 77]          [10 45 65 66 75 80]
         [40 46]     [42 45]    3
         [51 66]     [50 65]    14
         [51 66]     [55 70]    11
         [70 75]     [74 80]     1
         [77 80]]    [74 80]]    3
        
        Percentage overlap: [100. 100. 93.33333333 73.33333333  16.66666667 50.]
        
        So:
        result for percent_overlaps:
        array([100.        , 100.        ,  93.33333333,  73.33333333,
                16.66666667,  50.        ])
        
        
        result for 'matches':
        array([[ 5, 10],
               [40, 46],
               [51, 66],   should be 0.928571429 percent
               [51, 66],   remove dups
               [70, 75],   should be 20%, : 1/5
               [77, 80]])  should be 40%  : 2/5
        
        Results returned: an ordered dict:
        
        results = {'percent_overlaps' : percent_overlaps,
                   'matches_aud' : matches_aud,
                   'true_pos_events' : true_pos_events,
                   'false_neg_events' : false_neg_events,
                   'false_pos_events' : false_pos_events
                   }
                           
        @param audio_burst_indices: list of discovered burst start/stops
        @type audio_burst_indices: [(start,stop)]
        @param elephant_burst_indices: list of burst start/stops from labels
        @type elephant_burst_indices: [(start,stop)]
        @return: dict with percent_overlaps, matches_aud, 
            true_pos_events, false_neg_events, false_pos_events)
        '''

        # Wouldn't have dreamed up the solution below myself
        # in a million years. searchsort(sorted_list, element)
        # finds the index into sorted_list where element would
        # go if one would insert it.
        # Result will look like:
        #    [[0,4],[1,6],[2,7],[2,8]]]
        # Note that in this ex. audio interval pointer
        # 7 and 8 are both associated with labeled burst
        # interval pt 2.
        
        # Get start,stop,start,stop indices into the samples
        # array as flat list:
        aud_indices_flat = audio_burst_indices.flatten()
        labeled_with_aud_interval_ptr_matches = []
        true_pos_events  = 0
        # Collect unique aud indices that are involved
        # in an ele burst:
        good_aud_indices = set()
        
        false_neg_events = 0
        
        self.log.info('Finding which audio events have overlap with labeled events...')
        for (el_burst_indices_pt, (lower, upper)) in enumerate(elephant_burst_indices):
            i = np.searchsorted(aud_indices_flat, lower, side='right')
            j = np.searchsorted(aud_indices_flat, upper, side='left')
            audio_burst_indices_pts = np.arange(i // 2, (j + 1) // 2)
            
            # Number of audio intervals found for the 
            # current labeled burst:
            num_aud_burst_indices = audio_burst_indices_pts.size
            # If no audio burst event, move on to the 
            # next bona fide (i.e. labeled) burst, else:
            if num_aud_burst_indices > 0:
                
                # One more elephant burst matched by at least one
                # audio burst:
                true_pos_events += 1
                good_aud_indices = good_aud_indices.union(audio_burst_indices_pts)
                
                # Handle multiple aud burst associated with one
                # labeled burst: Replicate the pt into the labeled burst
                # intervals once for each such multi-match:
                ele_match_indices = np.array([el_burst_indices_pt] * num_aud_burst_indices)
                # Now have something like: [[3,5],[3,5]]; add 
                # to our accumulating result:
                labeled_with_aud_interval_ptr_matches.extend(np.column_stack((ele_match_indices, 
                                                                              audio_burst_indices_pts
                                                                              )))
            else:
                # No aud event for this ele event:
                false_neg_events += 1
                
        self.log.info('Done finding which audio events have overlap with labeled events.')
        
        # False positives:
        false_pos_events = audio_burst_indices[:,0].size - len(good_aud_indices)
        
        # Now compute the percentage of overlap.
        # Think of the code as being just like the 
        # following equivalent function applied in
        # a loop, except for vectorization.
        
        # def overlaps(labeled_event, audio_event):
        #     '''
        #     Return the amount of overlap for a single
        #     labeled vs. audio event. An event is a two
        #     tuple: [lower_sample_bound, upper_sample_bound).
        #
        #     If >0, the number of samples of overlap
        #     If 0,  audio and labeled bursts abut [Won't happen]
        #     If <0, no overlap.                   [Won't happen]
        #     '''
        #     # Earliest ending - latest beginning:
        #     overlap = min(audio_event[1], labeled_event[1]) - max(audio_event[0], labeled_event[0])
        #     percent_overlap = max(0, overlap / (labeled_event[1] - labeled_event[0]))
        #     return percent_overlap

        # Take a some intermediate steps for sanity:
        
        # Turn the matches we found above into an np array:
        matches     = np.array(labeled_with_aud_interval_ptr_matches)
        
        if len(matches) == 0:
            # No match found at all?
            results = OrderedDict({'percent_overlaps' : [],
                                   'matches_aud'      : [],
                                   'true_pos_events'  : 0,
                                   'false_neg_events' : 0,
                                   'false_pos_events' : 0
                                  })
            return results
        
        # The matches are pts into the elephant_burst_indices,
        # and audio_burst_indices. De-reference to get the 
        # actual sample-index-low-bound/sample-index-high-bound
        # pairs for elephant and audio events:
        matches_ele = elephant_burst_indices[matches[:,0]]
        matches_aud = audio_burst_indices[matches[:,1]]
        
        # A single ele burst may be 'discovered' multiple
        # times. Like this:
        #    matches_ele   matches_aud:
        #    darray: [     ndarray: [  
        #     [50 65]       [56 57]    
        #     [50 65]       [58 59]    
        #     [50 65]       [64 65]    
        #     [55 70]       [56 57]    
        #     [55 70]       [58 59]  
        # We only give credit to the first match, i.e.
        #    matches_ele   matches_aud:
        #    darray: [     ndarray: [  
        #     [50 65]       [56 57]    
        #     [55 70]       [58 59]  
        
        # Generate indices of the first instance of
        # each elephant burst that is matched. Ref above:
        # [0,3]:\
        (matches_ele, first_uniq_indices) = np.unique(matches_ele, return_index=True, axis=0)
        # For both ele and aud matches, keep only these uniques:
        
        matches_aud = matches_aud[first_uniq_indices]
        
        # Get 1d arrays for low and high bounds for both
        # audio and eles:
        low_bounds_ele  = matches_ele[:,0]
        high_bounds_ele = matches_ele[:,1]
        low_bounds_aud  = matches_aud[:,0]
        high_bounds_aud = matches_aud[:,1]
        
        # For each ele/aud interval pair, find
        # the lowest of the two high bounds:
        lowest_high_bounds = np.select([high_bounds_ele <= high_bounds_aud,
                                        high_bounds_ele  > high_bounds_aud],
                                       [high_bounds_ele, high_bounds_aud]
                                       )
        
        # For each ele/aud interval pair, find
        # the highest of the two low bounds:
        highest_low_bounds = np.select([low_bounds_ele >= low_bounds_aud,
                                        low_bounds_ele <  low_bounds_aud],
                                         [low_bounds_ele, low_bounds_aud]
                                         )
        
        overlaps = lowest_high_bounds - highest_low_bounds

        # For each elephant interval: get its width
        # in samples:
        ele_interval_widths = matches_ele[:,1] - matches_ele[:,0]
        
        # Finally!!!! The overlaps in percent.
        percent_overlaps = 100 * overlaps/ele_interval_widths

        
        results = OrderedDict({'percent_overlaps' : percent_overlaps,
                               'matches_aud'      : matches_aud,
                               'true_pos_events'  : true_pos_events,
                               'false_neg_events' : false_neg_events,
                               'false_pos_events' : false_pos_events
                              })
        return results


    #------------------------------------
    # collect_audio_events
    #-------------------    

    def collect_audio_events(self, samples):
        '''
        Given a 1d array of audio samples, 
        returns a 2-column array containing the
        start and stop indices of audio bursts.
        An audio burst is a series of non-zero
        volt samples, followed by one or more 
        zero volt samples.
        
        Example:
           samples =  np.array([20, 40, 0, 0, 0, 30, 0, 60, 50, 0])
           
        Returns:
           [[0,2],
            [5,6],
            [7,9]
            ]
        
        @param samples: array of audio samples
        @type samples: numpy.array
        @return: 2-col array with start/stop indices into samples. 
        @rtype: np.array(2D)
        '''
        # Add a zero before and after the signal array:
        samples_padded = np.insert(samples,(0,samples.size),0)
        
        # To use only vector ops, we replicate the padded samples
        # array with one shifted left:
        samples_padded_after = np.roll(samples_padded,-1)
        
        # Start indices are the ones in which samples_padded is a 0,
        # followed by a non-zero:
        self.log.info("Finding audio burst start indices...")
        start_indices = 1 + np.where(np.logical_and(samples_padded == 0, samples_padded_after != 0))[0]
        self.log.info("Done finding audio burst start indices.")
        
        # Analogously for the end indices of each burst:
        self.log.info("Finding audio burst end indices...")
        end_indices   = 1 + np.where(np.logical_and(samples_padded != 0, samples_padded_after == 0))[0]
        self.log.info("Done finding audio burst end indices.")

        self.log.info(f"Creating {start_indices.size} audio burst specs...")
        burst_index_pairs = np.column_stack((start_indices, end_indices))
        self.log.info(f"Done creating {start_indices.size} audio burst specs.")
        
        self.log.info("Creating audio event mask...")
        audio_mask = ma.masked_not_equal(samples_padded, 0).mask
        self.log.info("Done creating audio event mask.")
        
        # If entire audio event seq was 0, we are done.
        # The audio_mask is still an np array. But 
        # it has 'no shape':
        if audio_mask.shape == ():
            return([], 0)
         
        # We have one extra mask bit at the front, and the
        # end: chop those off:
        audio_mask = audio_mask[1:-1]
        # And correct the burst start/ends accordingly:
        burst_index_pairs -= 1
        return (burst_index_pairs, audio_mask.astype(int))
    
    #------------------------------------
    # label_file_reader
    #-------------------
    
    def label_file_reader(self, label_file_path):
        '''
        Given the path to a Raven export, return two representations
        of the labeled bursts: A set of start/stop indices
        into the samples that underly the labels. And a mask
        the length of the number of samples, where burst 
        regions have 1s, and other regions have 0s.
        
        Note that the Raven label files have start/stop in 
        fractional seconds. We convert to samples.
        
        The start_stop indices are of the form:
           [[burst1_start_idx_into_samples, burst1_end_idx_into_samples], 
            [burst2_start_idx_into_samples, burst2_end_idx_into_samples], 
                ...
           ]
           
        Assumption: self.framerate has framerate of corresponding
        .wav file.
        
        @param label_file_path: path to Raven label file
        @type label_file_path: str
        @returns two-tuples of indices into samples, and burst mask
        @rtype: ([[int,int]], np.array(int))
        '''
        
        # For result:
        self.events = []
        
        self.log.info("Reading label file...")
        with open(label_file_path, 'r') as fd:
            csv_reader = csv.reader(fd, delimiter='\t')
            # Discard col header:
            _header = next(csv_reader)
            
            COL_SELECTION_INDEX = 0
            COL_BEGIN_TIME_INDEX = 3
            COL_END_TIME_INDEX = 4
            start_end_list = []
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
                begin_time_secs = float(line[COL_BEGIN_TIME_INDEX])
                end_time_secs = float(line[COL_END_TIME_INDEX])
                self.events.append(ElephantEvent(selection_index, begin_time_secs, end_time_secs))
                begin_time_samples = int(np.ceil(self.framerate * begin_time_secs))
                # Don't know whether Cornell end time is inclusive or not,
                # so for safety, add one to make the sample end time exclusive:
                end_time_samples   = int(np.ceil(1 + self.framerate * end_time_secs))
                start_end_list.append([begin_time_samples,end_time_samples])
                
        last_end_time_samples = end_time_samples
        # Make np.array of sample-level start-stop:
        elephant_burst_indices = np.array(start_end_list)
        self.log.info("Done reading label file.")
        
        self.log.info("Creating elephant burst mask...")
        # Mask is as long as the end of the last burst in
        # sample space:    
        el_mask = np.zeros(last_end_time_samples + 1, dtype=int)
        for (burst_start, burst_end) in elephant_burst_indices:
            el_mask[burst_start:burst_end] = 1
        self.log.info("Done creating elephant burst mask.")

        return (elephant_burst_indices, el_mask)


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

class PerformanceResult(OrderedDict):
    '''
    Instances hold results from performance
    computations. Quantities ending in '_events'
    refer to performance at the level of 'burst'.
    For labels this means all labeled sets
    of samples. For samples it means clusteres of
    non-zero sample values.
    
    '''

    # Name of all properties stored in instances
    # of this class. Needed when reading from
    # previously saved csv file of a PerformanceResult
    # instance. 
    props = OrderedDict({'signal_treatment'      : SignalTreatmentDescriptor,
                         'recall_events'         : float,
                         'precision_events'      : float,
                         'f1score_events'        : float,
                         'recall_samples'        : float,
                         'precision_samples'     : float,
                         'f1score_samples'       : float,
                         'overlaps_summary'      : dict,
                         'true_pos_samples'      : int,
                         'false_pos_samples'     : int,
                         'true_neg_samples'      : int,
                         'false_neg_samples'     : int,
                         
                         'true_pos_events'       : int,
                         'false_pos_events'      : int,
                         'true_neg_events'       : int,
                         'false_neg_events'      : int,

                         'min_required_overlap'  : float,
                        })

    # In the final instance overlap_percentages will
    # be replaced with a dict of stats about the 
    # (usually very long) array:
    
    def __init__(self, all_results={}, overlap_percentages=[]):
        '''
        Created instance will have all information
        in the all_results dict. If overlap_percentages
        is given, it is a (possibly very long) np.array
        of overlap percentage floats. Summary statistics
        will be computed from that array, and retained
        under the key 'overlaps_summary'
        
        @param all_results: results to be contained in the instance
        @type all_results: dict
        @param overlap_percentages: percentages of overlap for each
            audio burst, (already multiplied by 100)
        @type overlap_percentages: np.array([float])
        '''
        self.log = LoggingService(logfile=PrecRecComputer.logfile)
        if len(overlap_percentages) == 0:
            self.log.warn("Overlap percentages is empty; so no overlaps summary stats computed")
        # Summarize the overlap_percentages, which can be
        # a huge array:
        try:
            overlaps_summary = self.summarize_overlaps(overlap_percentages, 
                                                       all_results['min_required_overlap'])
        except KeyError:
            # min_required_overlap was not passed in:
            self.log.warn("No min_required_overlap provide; PerformanceResult won't contain an overlaps summary.")
            overlaps_summary = {}
        all_results['overlaps_summary'] = overlaps_summary
        # Install the initial results
        super().__init__(all_results)

    #------------------------------------
    # __eq__
    #-------------------

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented

        for (key, val) in self.items():
            # Use ==, rather than != in case
            # other has no __ne__
            try:
                if val == other[key]:
                    continue
                else:
                    return False
            except KeyError:
                # Missing key, so: not equal
                return False
        return True

    def _ne__(self, other):
        return not self.__eq__(other)
    
    #------------------------------------
    # diff
    #-------------------
    
    def diff(self, perf_res_left, perf_res_right):
        '''
        Prints differences between two performance
        result instances. Returns True if no difference
        
        @param perf_res_left:
        @type perf_res_left:
        @param perf_res_right:
        @type perf_res_right:
        '''
        errors = []
        for (key,val) in perf_res_left.items():
            if isinstance(val, np.ndarray):
                # Values are np.arrays; simple Python
                # equality won't work. Also, for floats,
                # array_equal also fails over slight diffs.
                # Rather than using isclose(), just round
                # to 2 decimals for both:
                if not np.array_equal(np.round(val,2), 
                                      np.round(perf_res_right[key], 2)
                                      ):
                    errors.append(key)
                continue
            if val != perf_res_right[key]:
                errors.append(key)
                continue
        if len(errors) == 0:
            return True
        for problem_key in errors:
            print(f"{problem_key}: Left: {perf_res_left[problem_key]}; Right: {perf_res_right[problem_key]}")
        return False

    
    #------------------------------------
    # to_flat_dict
    #-------------------
    
    def to_flat_dict(self):
        '''
        Returns an OrderedDict where the top level
        key/val pairs of self are again top level.
        But the overlaps_summary is flattened. The
        attr names will be overlaps_summary_min,
        overlaps_summary_max, etc.
        '''
        res_dict = OrderedDict()
        for prop_key, prop_val in self.items():
            if isinstance(prop_val, dict):
                if len(prop_val) == 0:
                    # Empty dict:
                    res_dict[prop_key] = {}
                else:
                    for (key, val) in prop_val.items():
                        res_dict[f"{prop_key}_{key}"] = val
            elif isinstance(prop_val, SignalTreatmentDescriptor):
                res_dict[prop_key] = prop_val.to_flat_str()
            else:
                res_dict[prop_key] = prop_val
                
        return res_dict
        


    #------------------------------------
    # from_str
    #-------------------
    
    @classmethod
    def from_str(cls, stringified_perf_result):
        instance = eval(stringified_perf_result,
                        {'__builtins__' : None},
                        {'PerformanceResult' : PerformanceResult,
                         'OrderedDict' : OrderedDict}
                        )
        if not isinstance(instance, PerformanceResult):
            # Passed-in string is a primitive, such as
            # '30.2', rather than a str that makes
            # a PerformanceResult inst:
            raise ValueError(f"String '{stringified_perf_result}' does not evaluate to a PerformanceResult") 
            
        # Find the overlaps_summary dict embedded in
        # the stringified PerformanceResult. I.e. dig
        # out the dict from: 
        #  ...('overlaps_summary', {'min': 0.0, 'max': 74.6, ... 'perc_ge_min': 0.0})
       
        pat = re.compile(r".*(\{[^}]*\})")
        match = pat.match(stringified_perf_result)
        try:
            dict_str = match.groups()[0]
        except IndexError:
            # Did not find it:
            cls.log.warn(f"Could not find overlap_summary dict in {stringified_perf_result}")
        try:
            instance['overlaps_summary'] = eval(dict_str,
                                                {'__builtins__' : None},
                                                {}
                                                )
        except Exception:
            cls.log.warn(f"Could not eval overlap_summary dict in {stringified_perf_result}")
        return instance        

    #------------------------------------
    # _is_number
    #-------------------
            
    def _is_number(self, word):
        p = re.compile('^[0-9.]*$')
        return p.search(word)

    #------------------------------------
    # __str__
    #-------------------

    def __str__(self):
        '''
        Make str evaluatable into a PerformanceResult instance:
        '''
        tmp_dict = OrderedDict(self.items())
        tmp_dict['signal_treatment'] = str(tmp_dict['signal_treatment'])
        return f"PerformanceResult({str(tmp_dict)})"

    #------------------------------------
    # print
    #-------------------

    def print(self, outfile=None):
        
        try:
            if outfile is None:
                out_fd = sys.stdout
            else:
                out_fd = open(outfile, 'w')
                
            # For nice col printing: find longest 
            # property name:
            col_width = max(len(prop_name) for prop_name in self.keys())
            for prop in self.keys():
                print(prop.ljust(col_width), self[prop])

        finally:
            if out_fd != sys.stdout:
                out_fd.close()

    #------------------------------------
    # confusion_matrix_samples 
    #-------------------

    def confusion_matrix_samples(self):
        
        try:
            if self['confusion_matrix_samples'] is not None:
                return self['confusion_matrix_samples']
        except KeyError:
            pass
        self['confusion_matrix_samples'] = np.array([[self['true_pos_samples'], self['false_pos_samples']],
                                                    [self['false_neg_samples'], self['true_neg_samples']]
                                                    ]
                                                   )
        return self['confusion_matrix_samples']


    #------------------------------------
    # confusion_matrix_events
    #-------------------

    def confusion_matrix_events(self):
        
        try:
            if self['confusion_matrix_events'] is not None:
                return self['confusion_matrix_events']
        except KeyError:
            pass
        self['confusion_matrix_events'] = np.array([[self['true_pos_events'], self['false_pos_events']],
                                                    [self['false_neg_events'], self['true_neg_events']]
                                                    ]
                                                   )
        return self['confusion_matrix_events']

    #------------------------------------
    # summarize_overlaps
    #-------------------
    
    def summarize_overlaps(self, overlap_percentages, min_required_overlap):
        '''
        Computes summary statistics of an np array of
        overlap percentages:
           min
           max
           mean
           med
           sd
           num_ge_min
           perc_ge_min 
        
        @param overlap_percentages: array of percentages (already multiplied by 100)
        @type overlap_percentages: np.array([float])
        @param min_required_overlap: minimum percentage overlap required for a 
            detection to be counted as true-positive
        @type min_required_overlap: {int | float}
        @return: summary statistics
        @rtype: {str : float}
        '''
        res_dict = {}
        if len(overlap_percentages) == 0:
            return res_dict
        res_dict['min'] = np.round(np.min(overlap_percentages), 1)
        res_dict['max'] = np.round(np.max(overlap_percentages), 1)
        res_dict['mean'] = np.round(np.mean(overlap_percentages), 1)
        res_dict['med'] = np.round(np.median(overlap_percentages), 1)
        res_dict['sd'] = np.round(np.std(overlap_percentages), 1)
        qualifying_overlaps = overlap_percentages[overlap_percentages >= min_required_overlap]
        res_dict['num_ge_min'] = qualifying_overlaps.size
        res_dict['perc_ge_min'] = np.round(qualifying_overlaps.size / overlap_percentages.size, 0)
        return res_dict 
