'''
Created on Mar 19, 2020

@author: paepcke
'''

import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.font_manager import FontProperties

import numpy as np
from scipy.signal.filter_design import freqz
from datetime import timedelta

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

    def __init__(self, framerate):
        self.framerate = framerate
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
    # plot_spectrogram
    #-------------------
    
    def plot_spectrogram(self,
                         freq_labels,
                         time_labels,
                         freq_time,
                         time_intervals_to_cover=None,
                         title='Spectrogram (secs)'
                         ):
        '''
        Given a long spectrogram (frequency strengths (rows) and time (columns)),
        plot a (series of) spectrogram snippets. 
        
        Two options: caller can provide time_intervals_to_cover.
        That must be a tuple of pairs. Each pair is the start and 
        end second of one spectrogram to display. These snippets do not
        need to be adjacent.
        
        If no time_intervals_to_cover times are provided, the time-length
        of the spectrogram is partitioned into 18 (6x3) time
        periods, equally spaced over the entire time period.
        
        Result is a matrix of plots, with the true labels
        for frequencies and times.
        
        @param freq_time: matrix whose rows are frequency energy,
            and whose columns are time
        @type freq_time: np.array(real, real)
        @param time_intervals_to_cover: a list of 2-tuples whose values are  
             time intervals in seconds.
        @type time_intervals_to_cover: [({int | float}, {int | fload})]
        '''
        
        # Define the grid of spectroram plots:
        plot_grid_width  = 3
        plot_grid_height = 6
        default_spectrogram_segment_secs = 30 # seconds

        num_plots = plot_grid_height * plot_grid_width
        max_spectrogram_time = time_labels[-1]
        
        # Find index pairs into time_labels that contain 
        # the passed-in time intervals. We'll fill the following
        # with start-stop index pairs:
        time_indices = []
        
        if time_intervals_to_cover is not None:
            # Case 1: caller specified particular time intervals:
            num_spectrograms = len(time_intervals_to_cover)
            # Number of 3-column spectrogram rows we'll need:
            plot_grid_height = np.ceil(num_spectrograms / plot_grid_width)
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
                                       title=self.title)
        
        
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

        fig.show()

        flat_axes = axes.flatten()
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
                                    cmap='jet', 
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
    
    def plot_frequency_response(self, b, a, cutoff, order):
        '''
        b,a come from call to get_butter_lowpass_parms()
        
        @param b:
        @type b:
        @param a:
        @type a:
        @param cutoff: frequency (Hz) at which filter is supposed to cut off
        @type cutoff: int
        @param order: order of the filter: 1st, 2nd, etc.; i.e. number of filter elements.
        @type order: int
        '''
    
        w, h = freqz(b, a, worN=8000)
        plt.subplot(1, 1, 1)
        plt.plot(0.5 * self.framerate * w/np.pi, np.abs(h), 'b')
        plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
        plt.axvline(cutoff, color='k')
        plt.xlim(0, 0.5*self.framerate)
        plt.title(f"Lowpass Filter Frequency Response (order: {order}, cutoff: {cutoff} Hz")
        plt.xlabel('Frequency [Hz]')
        plt.grid()

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
      o 'low_pass_filter'
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
