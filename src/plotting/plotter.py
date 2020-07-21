'''
Created on Mar 19, 2020

@author: paepcke
'''

from datetime import timedelta

from matplotlib.font_manager import FontProperties
from matplotlib.patches import ConnectionPatch
from matplotlib.text import Text
from scipy.signal.filter_design import freqz, sosfreqz

from DSP.dsp_utils import DSPUtils
import matplotlib.gridspec as grd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Plotter(object):
    '''
    Plotting for spectrograms and other simple plots,
    including overplotting two wave forms.
    You can add plotting tasks by name through
    the PlotterTask class in this file.
    '''
    DEFAULT_TITLE = 'Spectrogram'
    
    #------------------------------------
    # Constructor
    #-------------------

    def __init__(self):
        self.title     = Plotter.DEFAULT_TITLE
        
    #------------------------------------
    # plot
    #------------------- 
    
    def plot(self, x_arr, y_arr, title='My Title', xlabel='X-axis', ylabel='Y-axis'):
        '''
        Just a basic default.
        
        @param x_arr:
        @type x_arr:
        @param y_arr:
        @type y_arr:
        @param title:
        @type title:
        @param xlabel:
        @type xlabel:
        @param ylabel:
        @type ylabel:
        '''
        fig = plt.figure()
        ax  = fig.add_subplot(1,1,1)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.plot(x_arr,y_arr)
        
        fig.show()

    #------------------------------------
    # plot_spectrogram_from_magnitudes
    #-------------------
    
    def plot_spectrogram_from_magnitudes(self,
                         spectro,
                         time_intervals_to_cover=None,
                         title='Spectrogram (secs)',
                         block=False
                         ):
        '''
        Given a long spectrogram (frequency strengths (rows) and time (columns)),
        plot a (series of) spectrogram snippets. 
        
        Three options: 
        
        Option 1: caller can provide time_intervals_to_cover.
        That must be a tuple of pairs. Each pair is the start and 
        end second of one spectrogram to display. These snippets do not
        need to be adjacent.
        
        Option 2: If time_intervals_to_cover is None, the time-length
        of the spectrogram is partitioned into 18 (6x3) time
        periods, equally spaced over the entire time period.
        
        Option 3: If an empty array is passed for time_intervals_to_cover,
        then the entire spectrogram is plotted.
        
        Result is a matrix of plots, with the true labels
        for frequencies and times.
        
        @param spectro: datagrame whose rows are frequency energy,
            and whose columns are time
        @type spectro: pd.Dataframe
        @param time_intervals_to_cover: a list of 2-tuples whose values are  
             time intervals in seconds.
        @type time_intervals_to_cover: [({int | float}, {int | float})]
        @param block: whether or not to wait with return
            till user dismisses the figure window.
        @type block: bool
        '''

        time_labels = spectro.columns
        
        # Define the grid of spectrogram plots:
        
        default_spectrogram_segment_secs = 30 # seconds

        max_spectrogram_time = time_labels[-1]
        
        if type(time_intervals_to_cover) == list \
            and len(time_intervals_to_cover) == 0:
            # Show the whole spectrogram:
            time_intervals_to_cover = [(0, max_spectrogram_time)]
            time_label_indices_to_cover = [(0,len(time_labels)-1)]
            time_indices = np.arange(0,len(time_labels))
            plot_grid_width = 1
            plot_grid_height = 1
            num_plots = 1

        elif time_intervals_to_cover is not None:
        
            # Find index pairs into time_labels that contain 
            # the passed-in time intervals. We'll fill the following
            # with start-stop index pairs:
            time_indices = []
        
            # Case 1: caller specified particular time intervals:
            num_spectrograms = len(time_intervals_to_cover)
            plot_grid_width = 3
            # Number of 3-column spectrogram rows we'll need:
            plot_grid_height = int(np.ceil(num_spectrograms / plot_grid_width))
            num_plots = plot_grid_height * plot_grid_width

            # Find the requested times in the time labels:
            for (min_sec, max_sec) in time_intervals_to_cover:
                # Get a two-el array with indices into
                # time labels for which spectrogram is to 
                # be constructed. First, find all the indices
                # that point to values within the min/max time:
                time_interval_indices = np.nonzero(np.logical_and(time_labels >= min_sec,
                                                                  time_labels < max_sec)
                )
                # Remember the first and last index:
                time_indices.append((time_interval_indices[0], time_interval_indices[-1]))
                
            # To match datatype of time_indices in the else 
            # branch:
            time_indices = np.array(time_indices)
        else:
            # No time intervals given, spread 30-second spectrogram
            # excerpts equally across the full width of the total
            # spectrogram:
            plot_grid_width = 3
            plot_grid_height = 6
            num_plots = plot_grid_height * plot_grid_width
            time_indices = []
            plot_mid_point_every = time_labels[-1] / default_spectrogram_segment_secs
            half_interval        = default_spectrogram_segment_secs / 2.
            curr_mid_point = plot_mid_point_every
            while curr_mid_point < max_spectrogram_time:
                min_sec = max(0, curr_mid_point - half_interval)
                max_sec = min(max_spectrogram_time, curr_mid_point + half_interval)
                # Get a two-el array with indices into
                # time labels for which spectrogram is to 
                # be constructed. np.nonzero() returns a 
                # *tuple* of arrays; therefore the [0]:
                time_interval_indices = np.nonzero(np.logical_and(time_labels >= min_sec,
                                                                  time_labels < max_sec)
                )[0]
                # Record the first and last time index
                # to cover in this window:
                time_indices.append((time_interval_indices[0], 
                                                    time_interval_indices[-1]))
                curr_mid_point += plot_mid_point_every
           
            # Now we have indices into the times for many 
            # adjacent 30-sec  segments. But we only want num_plots
            # spectrograms. Pick segments evenly spaced:
            num_30sec_segments = len(time_indices)
            num_30sec_segments_wanted = int(np.ceil(num_30sec_segments / num_plots))
            # For convenient selection using range(), make
            # time_indices into a numpy array:
            time_indices = np.array(time_indices)
            time_label_indices_to_cover = time_indices[range(0,
                                                       num_30sec_segments,
                                                       num_30sec_segments_wanted
                                                       )
                                                      ]
            
        # We now have an array of start-top *indices* into
        # the time labels array. We create a spectrogram
        # plot for each of them, placing each plot into a
        # matrix of plots:

        self._plot_spectrogram_excerpts(spectro,
                                        time_label_indices_to_cover, 
                                        plot_grid_width,
                                        plot_grid_height,
                                        title=title)
        
        if block:
            self.block_till_figs_dismissed()


    #------------------------------------
    # plot_spectrogram_from_audio 
    #-------------------
        
    def plot_spectrogram_from_audio(self, 
                                    raw_audio, 
                                    samplerate, 
                                    start_sec, 
                                    end_sec, 
                                    plot):
        '''
        Given an audio file, create a spectrogram
        via plt.specgram(). If plot is True, create
        two plots on top of each other: the wav file
        and the spectrogram. Works with an excerpt
        by specifying start_sec and end_sec.
        
        @param raw_audio:
        @type raw_audio:
        @param samplerate:
        @type samplerate:
        @param start_sec:
        @type start_sec:
        @param end_sec:
        @type end_sec:
        @param plot:
        @type plot:
        '''

        (spectrum, freqs, t_bins, im) = plt.specgram(raw_audio, 
                                                     Fs=samplerate,
                                                     #cmap='jet'
                                                     cmap='gray'
                                            		 )
        if plot:
            t = np.arange(start_sec, end_sec, 1/samplerate)
            _fig = plt.Figure()
            grid_spec = grd.GridSpec(nrows=2,
                                     ncols=1
                                     ) 
            ax_audio = plt.subplot(grid_spec[0])
            plt.xlabel('Time')
            plt.ylabel('Audio Units')
            ax_audio.plot(t, raw_audio)
            
            plt.show()
        return (spectrum,freqs,t_bins,im)
    
    #------------------------------------
    # plot_spectrogram_from_dataframe_file 
    #-------------------
    
    def plot_spectrogram_from_dataframe_file(self, 
                                             dff_file,
                                             *args,
                                             **kwargs
                                             ):
        '''
        Read spectrogram dataframe from file, and
        call plot_spectrogram_from_magnitudes().
        Additional args and kwargs are passed through
        to that latter method.
         
        @param dff_file:
        @type dff_file:
        '''
        spectro = DSPUtils.load_spectrogram(dff_file)
        
        self.plot_spectrogram_from_magnitudes(spectro,
                                              *args,
                                              **kwargs
                                              )

    #------------------------------------
    # plot_spectrogram_with_labels_truths 
    #-------------------
    
    def plot_spectrogram_with_labels_truths(self,
                                            spectrogram, 
                                            predictions=None, 
                                            label_mask=None, 
                                            binary_preds=None, 
                                            title=None, 
                                            vert_lines=None,
                                            filters=[]
                                            ):
        '''
        Visualizes the spectogram and associated predictions/label_mask. 
        features is the entire spectrogram that will be visualized
    
        For now this just has placeholder plots for outputs and label_mask
        when they're not passed in. 
    
        Inputs are numpy arrays
    
        @param spectrogram:
        @type spectrogram:
        @param predictions:
        @type predictions:
        @param label_mask:
        @type label_mask:
        @param binary_preds:
        @type binary_preds:
        @param title:
        @type title:
        @param vert_lines:
        @type vert_lines:
        @param filters:
        @type filters:
        '''
        
        spectro_magnitudes_copy = np.copy(spectrogram.to_numpy())
        if filters is not None:
            if type(filters) != list:
                filters = [filters]
            for _filter in filters:
                # Apply given filters to spectrogram.
                # Use a safe method for applying eval():
                spectro_magnitudes_copy = eval(_filter(spectro_magnitudes_copy),
                   {"__builtins__":None},    # No built-ins at all
                   {}                        # No additional func needed
                   )
        
        self._plot_truth(spectro_magnitudes_copy,
                         outputs=predictions,
                         labels=label_mask,
                         binary_preds=binary_preds,
                         title=title,
                         vert_lines=vert_lines
                         )

    #------------------------------------
    # _plot_truth 
    #-------------------

    def _plot_truth(self,
                        spectrogram, 
                        predictions=None, 
                        label_mask=None, 
                        title=None, 
                        vert_lines=True,
                        ):
        '''
        Derived from Jonathan's visualize() method.
        
        Visualizes the spectogram and associated predictions/labels. 
    
        For now this just has placeholder plots for outputs and labels
        when they're not passed in. 
    
        Inputs are numpy arrays
        
        @param spectrogram: entire, or partial spectrogram
        @type spectrogram: pd.DataFrame
        @param predictions: 1/0 for every time column. Reflects
            predictions of models: call/not-call
        @type predictions: np.array
        @param label_mask: mask 1/0 for where labels state an elephant
            call occurred
        @type label_mask: np.array
        @param binary_preds: ???
        @type binary_preds: ???
        @param title: for plot
        @type title: str
        @param vert_lines: If true, vertical lines are drawn
            across all stacked charts to mark the start and
            end of calls:
        @type vert_lines: bool

        '''
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3,
                                            ncols=1,
                                            sharex=True
                                            )
        
        #*************
        spectrogram = spectrogram.iloc[0:60,:]
        #*************
        magnitudes  = spectrogram.to_numpy()
        # The x-axis labels in seconds:
        time_labels = spectrogram.columns
        # The x-axis indices (i.e. bin number):
        x_indices   = range(len(time_labels))
        
        # How many x-axis tick (time) labels to keep:
        # Arbitrarily: 5. That's just the labels,
        # not the data columns, which are more:
        
        keep_every_nth_x = int(len(time_labels) / 5)
        # np-array slicing: [first:last:step]:
        sparse_time_labels = time_labels[1::keep_every_nth_x]
        sparse_x_indices   = x_indices[1::keep_every_nth_x]

        sparse_hr_min_sec_labels = [DSPUtils.hrs_mins_secs_from_secs(time_label) 
                        for time_label in sparse_time_labels]
        
        #sparse_label_mask = label_mask[1::keep_every_nth_x]
        
        # Create two-line x-labels: military time, and
        # time in seconds since start of recording:
        x_labels = [f"{mil_time}\n{round(secs,1)}s" for 
                    (mil_time, secs) in zip(sparse_hr_min_sec_labels, 
                                            sparse_time_labels)]
        plt.xticks(sparse_x_indices,
                   x_labels,
                   fontsize=9)
        
        # How many y-labels (frequency) to keep:
        freq_labels = spectrogram.index
        
        # hz_per_row = max(freq_labels) / len(freq_labels)
        keep_every_nth_y    = int(len(freq_labels) / 10)
        sparse_freq_labels  = freq_labels[1::keep_every_nth_y]
        sparse_freq_indices = range(1,int(max(freq_labels)),keep_every_nth_y)
        
        plt.yticks(sparse_freq_indices,
                   sparse_freq_labels,
                   fontsize=9
                   )
        
        ax1.imshow(magnitudes, 
                   cmap="magma_r", 
                   #*****interpolation='none', 
                   interpolation='hanning', 
                   origin="lower", 
                   aspect="auto"
                   )
        
        if predictions is not None:
            ax2.plot(np.arange(predictions.shape[0]), predictions)
            ax2.set_ylim([0,1])
            ax2.axhline(y=0.5, color='r', linestyle='-')
    
        if label_mask is not None:
            plt.yticks([0,1], [0,1], fontsize=9)
            ax3.plot(np.arange(0, len(label_mask)), label_mask)
            if vert_lines:
                call_intervals = self.get_calls_from_mask(label_mask)

                # Draw pairs of vertical lines for each call:
                top_of_spectro = max(spectrogram.index)
                
                for call_interval in call_intervals:
                    conn_line_left = ConnectionPatch(xyA=(call_interval.left,0), 
                                                     coordsA=ax3.transData, 
                                                     xyB=(call_interval.left,top_of_spectro), 
                                                     coordsB=ax1.transData)
                    conn_line_right = ConnectionPatch(xyA=(call_interval.right,0), 
                                                      coordsA=ax3.transData, 
                                                      xyB=(call_interval.right,top_of_spectro), 
                                                      coordsB=ax1.transData)
                    fig.add_artist(conn_line_left)
                    fig.add_artist(conn_line_right)
                    
                    ax3.axvline(x=call_interval.left, color='green', linestyle=':')
                    ax3.axvline(x=call_interval.right, color='green', linestyle=':')
        
        # Make the plot appear in a specified location on the screen
        if plt.get_backend() == "TkAgg":
            mngr = plt.get_current_fig_manager()
            #geom = mngr.window.geometry()  
            mngr.window.wm_geometry("+400+150")
    
        if title is not None:
            ax1.set_title(title)
    
        plt.show()
        
    
    #------------------------------------
    # visualize_predictions
    #-------------------

    def visualize_predictions(self,
                              snippet_intervals, 
                              spectrogram, 
                              prediction_mask=None, 
                              label_mask=None,
                              #*****chunk_size=20, # seconds  
                              chunk_size=80, # ?seconds  
                              title=None,
                              filters=None):
        '''
        Derived from Jonathan's method in visualization.py
        
        Visualize the predicted labels and gt labels for the calls provided.
        This is used to visualize the results of predictions on for example
        the full spectrogram. 
        
        @param snippet_intervals: intervals in seconds
            for segments of spectro to show. Could be
            all the starts/ends of calls.
        @type snippet_intervals: [pd.Interval]
        @param spectrogram: entire spectrogram; usually 24 hrs. 
        @type spectrogram: pd.DataFrame
        @param prediction_mask: mask call/no-call from Raven
            label file.
        @type prediction_mask: np.array[int]
        @param chunk_size:
        @type chunk_size:
        @param label:
        @type label:
        '''
        '''

        Parameters:
        - calls: Assumed to be list of tuples of (start_time, end_time, len) where start_time
        and end_time are for now in spect frames
        - label: gives what the calls represent (i.e. true_pos, false pos, false_neg) 


        Down the road maybe we should include the % overlap
        '''
                
        if filters is not None:
            spectro_magnitudes_copy = np.copy(spectrogram.to_numpy())
            if type(filters) != list:
                filters = [filters]
            for _filter in filters:
                # Apply given filters to spectrogram.
                # Use a safe method for applying eval():
                spectro_magnitudes_copy = eval(_filter(spectro_magnitudes_copy),
                   {"__builtins__":None},    # No built-ins at all
                   {}                        # No additional func needed
                   )
            spectrogram = pd.DataFrame(spectro_magnitudes_copy,
                                       columns=spectrogram.columns,
                                       index=spectrogram.index
                                       )

        for snippet_interval in snippet_intervals:
            
            start_time = snippet_interval.left
            end_time = snippet_interval.right
            # List of all snippet x-axis time labels
            #all_snippet_times = time_labels[time_labels>=snippet_interval.left]\
            #    .append(time_labels[time_labels<snippet_interval.right])
                
            # np.where returns a tuple of arrays; in this case,
            # something like (array([3]),), where 3 is the time bin
            # where the start time is the x-axis label. The int() pulls
            # the single number out of its array:
            start_bin_idx = int(np.where(spectrogram.columns == start_time)[0])
            end_bin_idx   = int(np.where(spectrogram.columns == end_time)[0])
                
            num_time_slots  = end_bin_idx - start_bin_idx
    
            # Let us position the call in the middle
            # Then visualize
            padding = (chunk_size - num_time_slots) // 2
            window_start = int(max(start_bin_idx - padding, 0))
            window_end = int(min(end_bin_idx + padding, spectrogram.shape[1]))
            
            #prediction_excerpt = prediction_mask[window_start: window_end] if \
            #    prediction_mask is not None else None
                
            spectrogram_snippet = spectrogram.iloc[:,window_start:window_end]
            # the truth mask snippet that corresponds to the
            # spectrogram snippet:
            label_mask_snippet  = label_mask[window_start:window_end]

            #print (spectrogram.shape)
            self._plot_truth(spectrogram_snippet,
                             prediction_mask,
                             label_mask_snippet,
                             title=title, 
                             vert_lines=True
                             )

    #------------------------------------
    # _plot_spectrogram_excerpts
    #-------------------
    
    def _plot_spectrogram_excerpts(self,
             spectro,
             time_label_indices_to_cover,
             plot_grid_width=3,
             plot_grid_height=6,
             title='Spectrogram'
             ):
        '''
        Workhorse for plotting pieces from
        one spectgrogram. Used by plot_spectrogram_from_magnitudes() 
        
        @param time_label_indices_to_cover:
        @type time_label_indices_to_cover:
        @param freq_time:
        @type freq_time:
        @param time_labels:
        @type time_labels:
        @param freq_labels:
        @type freq_labels:
        @param plot_grid_width:
        @type plot_grid_width:
        @param plot_grid_height:
        @type plot_grid_height:
        @param title:
        @type title:
        '''

        freq_time = spectro.to_numpy()
        time_labels = spectro.columns
        freq_labels = spectro.index
        
        fig, axes = plt.subplots(ncols=plot_grid_width,
                                 nrows=plot_grid_height,
                                 constrained_layout=True)
        fig.suptitle(title)

        fig.show()
        
        # For multi-panel plots axes will be a
        # 2d array:
        if type(axes) == np.ndarray:
            flat_axes = axes.flatten()
        else:
            flat_axes = [axes]
        plot_position = 0
        for (min_sec_index, max_sec_index) in time_label_indices_to_cover:
            
            # Take next segment to be displayed from spectrogram:
            # i.e. all frequencies (y-axis), and the time interval:
            matrix_excerpt = freq_time[0:, min_sec_index:max_sec_index]
            ax = flat_axes[plot_position]
            ax.set_autoscaley_on(True)
            
            # Determine with and height from the 
            # min/max of freq and time of the excerpt:
            
            highest_freq = np.ceil(freq_labels[-1])
            lowest_freq = np.floor(freq_labels[0])

            lowest_time  = np.floor(time_labels[min_sec_index])
            highest_time = np.ceil(time_labels[max_sec_index])
            
            #lowest_time_smpte = str(timedelta(seconds=lowest_time))
            #highest_time_smpte = str(timedelta(seconds=highest_time))
            
            # Origin "lower" makes y axis start at 0:
            _axes_image = ax.imshow(matrix_excerpt, 
                                    #cmap='jet', 
                                    cmap='gray', 
                                    origin='lower',
                                    aspect='auto',
                                    extent=(lowest_time-0.5, 
                                            highest_time,
                                            lowest_freq-0.5,
                                            highest_freq
                                            )
                                    )
            
            plot_position += 1

        # Make sure all the spectrograms have been drawn:
        plt.pause(0.1)
        
        # The tick labels will all be in seconds since
        # start of the spectrogram. Add a second line
        # to each label, giving the corresponding hh:mm:ss time.
        # Also: make font smaller:
        for ax in flat_axes:
            font_props = FontProperties(size=8)
            new_xticklabels = []
            for xtick_txt_obj in ax.get_xticklabels():
                label_txt = xtick_txt_obj.get_text()
                # Negative tick labels use en-dash, rather
                # than minus sign. Replace if needed:
                if label_txt[0].encode('utf-8') == b'\xe2\x88\x92':
                    label_txt = f"-{label_txt[1:]}"
                time_in_secs = float(label_txt)
                time_in_smpt = str(timedelta(seconds=time_in_secs))
                new_txt = f"{label_txt}\n{time_in_smpt}"
                xtick_text = Text(text=new_txt)
                xtick_text.set_fontproperties(font_props)
                new_xticklabels.append(xtick_text)
            ax.set_xticklabels(new_xticklabels)

    #------------------------------------
    # plot_frequency_response
    #-------------------
    
    def plot_frequency_response(self, 
                                filter_coeffs,
                                framerate,
                                cutoffs, 
                                title=None):
        '''
        Plot response to a low/high/bandpass filter.
        Filter coefficients from from calling iirfilter().
        If that function was called such that it returned
        nominator/denominator of the transfer function, namely
        a and b, hen filter_coeffs should be a tuple (a,b)
        if the function was called requesting second order segments,
        (sos), then filter_coeffs should be the sos.
        
        @param filter_coeffs: definiting filter characteristics
        @type filter_coeffs: {(np_arr, np_arr) | np_arr}
        @param framerate: sample frequency of the audio file.
        @type int
        @param cutoffs: frequencies (Hz) at which filter is supposed to cut off
        @type cutoffs: {int | [int]}
        @param title: optional title of the figure. If None,
            a title is created from the cutoff freqs.
        @type str
        '''
        if type(cutoffs) != list:
            cutoffs = [cutoffs]
        if type(filter_coeffs) == tuple:
            (a,b) = filter_coeffs
            w, h = freqz(b, a, worN=8000)
        else:
            w, h = sosfreqz(filter_coeffs, worN=8000)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(0.5 * framerate * w/np.pi, np.abs(h), 'b') # Blue
        ax.plot(cutoffs[0], 0.5*np.sqrt(2), 'ko')
        ax.axvline(cutoffs[0], color='k')        
        if len(cutoffs) > 1:
            ax.plot(cutoffs[1], 0.5*np.sqrt(2), 'ko')
            ax.axvline(cutoffs[1], color='k')
        # Since x axis will be log, cannot start
        # x vals at 0. To make best use of the 
        # horizontal space, start plotting at 
        # 5Hz below the low cutoff freq:
        ax.set_xlim(cutoffs[0] - 5, max(cutoffs) + max(cutoffs))  #0.5*framerate)
        ax.set_xscale('log')
        ax.set_yscale('log')
        if title is not None:
            fig.suptitle(title)
        else:
            fig.suptitle(f"Filter Frequency Response cutoff(s): {cutoffs} Hz")
        ax.set_xlabel('Log frequency [Hz]')
        ax.grid()
        fig.show()

        
#         plt.plot(cutoffs[0], 0.5*np.sqrt(2), 'ko')
#         plt.axvline(cutoffs[0], color='k')        
#         if len(cutoffs) > 1:
#             plt.plot(cutoffs[1], 0.5*np.sqrt(2), 'ko')
#             plt.axvline(cutoffs[1], color='k')
#         # Since x axis will be log, cannot start
#         # x vals at 0. To make best use of the 
#         # horizontal space, start plotting at 
#         # 5Hz below the low cutoff freq:
#         plt.xlim(cutoffs[0] - 5, max(cutoffs) + max(cutoffs))  #0.5*framerate)
#         plt.xscale('log')
#         plt.yscale('log')
#         if title is not None:
#             plt.title = title
#         else:
#             plt.title(f"Filter Frequency Response cutoff(s): {cutoffs} Hz")
#         plt.xlabel('Log frequency [Hz]')
#         plt.grid()
#         plt.show()

    #------------------------------------
    # over_plot
    #-------------------    
    
    def over_plot(self, y_arr, legend_entry, title="Filters", xlabel="Time", ylabel='Voltage'):
        '''
        Call multiple times with different 
        y_arr values each time. Will keep plotting
        over what's already there.
        
        @param y_arr:
        @type y_arr:
        @param legend_entry:
        @type legend_entry:
        @param title:
        @type title:
        @param xlabel:
        @type xlabel:
        @param ylabel:
        @type ylabel:
        '''

        try:
            self.ax
        except AttributeError:
            fig = plt.figure()
            self.ax  = fig.add_subplot(1,1,1)
            self.ax.set_title(title)
            self.ax.set_xlabel(xlabel)
            self.ax.set_ylabel(ylabel)
            
        self.ax.plot(np.arange(y_arr.size),
                     y_arr,
                     label=legend_entry
                     )
        self.ax.legend()

    #------------------------------------
    # get_calls_from_mask 
    #-------------------
    
    def get_calls_from_mask(self, label_mask):

        # Will find indices to the starts and ends
        # of calls in the mask:
        call_intervals = []

        # Get indices into label_mask where the mask
        # is 1. np.where returns a one-tuple:
        ones_indices = np.where(label_mask == 1)[0]

        if not len(ones_indices) == 0:
            call_start_idx = ones_indices[0]
            prev_one_idx = ones_indices[0]
            
            for next_one_idx in ones_indices[1:]:
                if next_one_idx - prev_one_idx > 1:
                    # End of one call:
                    call_intervals.append(pd.Interval(left=call_start_idx, 
                                                      right=prev_one_idx))
                    prev_one_idx = next_one_idx
                    call_start_idx = next_one_idx
                    continue
                else:
                    prev_one_idx = next_one_idx
            # Record the last one:
            call_intervals.append(pd.Interval(left=call_start_idx, 
                                              right=next_one_idx))
        return call_intervals

    #------------------------------------
    # block_till_figs_dismissed
    #-------------------

    @classmethod
    def block_till_figs_dismissed(cls):
        '''
        Block execution until all currently drawn figure
        windows are dismissed by the user.
        
        @param cls: Plotter class
        @type cls: Plotter
        '''
        
        
        try:
            # Try to raise the figure window:
            plt.get_current_fig_manager().window.raise_()
        except Exception:
            # Sometimes works, sometimes not:
            pass
        input("Press ENTER to close the figures and exit...")
        

# ------------------------------ class PlotterTasks ----------------

class PlotterTasks(object):
    '''
    All methods are class level. Add plot
    names, such as 
      o 'gated_wave_excerpt',
      o 'samples_plus_envelope',
      o 'spectrogram_excerpts',
      o 'filter_response'
    by calling 'PlotterTasks.add_task(<plotName>, **kwargs) 
    '''
    
    
    plotter_tasks = {}
    
    #------------------------------------
    # add_task
    #-------------------

    @classmethod
    def add_task(cls, plot_name, **kwargs):
        cls.plotter_tasks[plot_name] = kwargs
        
    #------------------------------------
    # has_task
    #-------------------
    
    @classmethod
    def has_task(cls, plot_name):
        try:
            return cls.plotter_tasks[plot_name]
        except KeyError:
            return None
