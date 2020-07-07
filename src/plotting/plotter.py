'''
Created on Mar 19, 2020

@author: paepcke
'''

import matplotlib.pyplot as plt
from matplotlib.text import Text
import matplotlib.gridspec as grd
from matplotlib.font_manager import FontProperties

import numpy as np
from scipy.signal.filter_design import freqz, sosfreqz
from datetime import timedelta

from visualization import visualize
from DSP.dsp_utils import DSPUtils


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
                         freq_labels,
                         time_labels,
                         freq_time,
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
        
        @param freq_labels: labels for the frequency axis 
        @type freq_labels: np.array
        @param time_labels: labels for the time axis
        @type time_labels: np.array
        @param freq_time: matrix whose rows are frequency energy,
            and whose columns are time
        @type freq_time: np.array(real, real)
        @param time_intervals_to_cover: a list of 2-tuples whose values are  
             time intervals in seconds.
        @type time_intervals_to_cover: [({int | float}, {int | float})]
        @param block: whether or not to wait with return
            till user dismisses the figure window.
        @type block: bool
        '''
        
        # Define the grid of spectrogram plots:
        plot_grid_width  = 3
        plot_grid_height = 6
        default_spectrogram_segment_secs = 30 # seconds

        num_plots = plot_grid_height * plot_grid_width
        max_spectrogram_time = time_labels[-1]
        
        if type(time_intervals_to_cover) == list \
            and len(time_intervals_to_cover) == 0:
            # Show the whole spectrogram:
            time_intervals_to_cover = [(0, max_spectrogram_time)]
            time_label_indices_to_cover = [(0,len(time_labels)-1)]
            plot_grid_width = 1
            plot_grid_height = 1
            num_plots = 1
        
        # Find index pairs into time_labels that contain 
        # the passed-in time intervals. We'll fill the following
        # with start-stop index pairs:
        time_indices = []
        
        if time_intervals_to_cover is not None:
            # Case 1: caller specified particular time intervals:
            num_spectrograms = len(time_intervals_to_cover)
            # Number of 3-column spectrogram rows we'll need:
            plot_grid_height = int(np.ceil(num_spectrograms / plot_grid_width))
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

        self.plot_spectrogram_excerpts(time_label_indices_to_cover, 
                                       freq_time, 
                                       time_labels, 
                                       freq_labels,
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
        df = DSPUtils.load_spectrogram(dff_file)
        
        self.plot_spectrogram_from_magnitudes(df.index,
                                              df.columns,
                                              df.values,
                                              *args,
                                              **kwargs
                                              )

    #------------------------------------
    # plot_spectrogram_with_labels_truths 
    #-------------------
    
    def plot_spectrogram_with_labels_truths(self,
                                            features, 
                                            outputs=None, 
                                            labels=None, 
                                            binary_preds=None, 
                                            title=None, 
                                            vert_lines=None,
                                            filters=[]
                                            ):
        new_features = np.copy(features)
        if filters is not None:
            if type(filters) != list:
                filters = [filters]
            for _filter in filters:
                new_features = eval(_filter(new_features),
                   {"__builtins__":None},    # No built-ins at all
                   {}                        # No additional func needed
                   )
        
        visualize(new_features,
                  outputs=outputs,
                  labels=labels,
                  binary_preds=binary_preds,
                  title=title,
                  vert_lines=vert_lines
                  )

    #------------------------------------
    # plot_spectrogram_excerpts
    #-------------------
    
    def plot_spectrogram_excerpts(self,
             time_label_indices_to_cover,
             freq_time,
             time_labels,
             freq_labels,
             plot_grid_width=3,
             plot_grid_height=6,
             title='Spectrogram'
             ):

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
    # compute_timeticks 
    #-------------------
    
    def compute_timeticks(self, framerate, spectrogram_t):
        
        # Number of columns in the spectrogram:
        estimate_every_samples = self.n_fft - self.overlap * self.n_fft
        estimate_every_secs    = estimate_every_samples / framerate
        one_sec_every_estimates= 1/estimate_every_secs
        (_num_freqs, num_estimates) = spectrogram_t.shape
        num_second_ticks = num_estimates / one_sec_every_estimates
        num_minute_ticks = num_second_ticks / 60
        num_hour_ticks   = num_second_ticks / 3600
        num_time_ticks = {'seconds': int(num_second_ticks),
                          'minutes': int(num_minute_ticks),
                          'hours'  : int(num_hour_ticks)
                          }
        
        #time_labels       = np.array(range(num_timeticks))
        return num_time_ticks

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
        plt.ioff()
        plt.show()

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
