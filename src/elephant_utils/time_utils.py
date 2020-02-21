#!/usr/bin/env python
'''
Created on Oct 28, 2019

@author: paepcke
'''
import argparse
import csv
import math
import os
import re
import sys
from datetime import datetime
from datetime import date

import requests


# -------------------------------- Class Point -------------------    
class Point(object):
    
    def __init__(self, lat, long):
        self._latitude  = lat
        self._longitude = long
        
    @property
    def latitude(self):
        return self._latitude
    
    @property
    def longitude(self):
        return self._longitude
    
# -------------------------------- Class TimeUtils -------------------    
        

class TimeUtils(object):
    '''
    classdocs
    '''

    LOC_LAT_LONG_DICT = {'central_africa' : Point(19.5687, 2.3185),
                         'congo' : Point(21.7587, 4.0383)
                         }
    DATE_PAT = re.compile(r'[0-9]{8}')
    
    #-------------------------
    # Constructor 
    #--------------


    def __init__(self, in_file_path):
        '''
        Constructor
        '''
        # A cache for sunrise/sunset of all audio locations.
        # Will be {site name : (sunrise, sunset)}. Site names
        # are like nna1:
        
        self.daylight_cache = {}
        
        self.location_info = self.get_location_info()
        self.augment_elephant_call_table(in_file_path)

        
    #-------------------------
    # augment_elephant_call_table 
    #--------------

    def augment_elephant_call_table(self, in_file):
        '''
        Given an elephant call label file, add convenience
        columns:
        
            o LabelDate     string yyyy-mm-dd from 'Begin File' field
            o StartDateTime string 'yyyy-mm-dd hh:mm:ss', which is local
                               time at recording site. Derived from 
                               'Begin Time', which is seconds since midnight
            o StopDateTime  like StartTime
            o Sunrise       local sunrise on day LabelDate
            o Sunset        local sunset on day LabelDate
            o latitude
            o longitude

            
        The conversion from fractional seconds to hh:mm:ss works like this:

			    H = floor(T/3600)
			    M = floor((T % 3600)/60)
			    S = floor((T % 3600) % 60)

        @param in_file: Original TSV file of elephant labels 
        @type in_file: str
        '''
        
        with open(in_file, 'r') as in_fd:
            reader = csv.DictReader(in_fd,
                                    delimiter = '\t'
                                    )
            col_names = reader.fieldnames
            # Add the new cols we will add:
            col_names.extend(['LabelDate',
                              'StartTime 24hr Format',
                              'StopTime 24hr Format',
                              'Sunrise',
                              'Sunset',
                              'NightTimeStart',
                              'NightTimeStop',
                              'Site',
                              'Latitude',
                              'Protection',
                              'Habitat',
                              'Longitude'
                              ])
            
            writer = csv.DictWriter(sys.stdout,
                                    col_names,
                                    delimiter='\t')
            
            # Col header:
            sys.stdout.write('\t'.join(writer.fieldnames) + '\n')
            
            
            # For error reporting:
            row_num = 0
            for row in reader:
                row_num += 1
                
                # Get date from myfile_20190320_....txt
                label_date_obj = row['Begin File'].split('_')[1]
                
                if TimeUtils.DATE_PAT.search(label_date_obj) is None:
                    raise ValueError(f"Could not extract date from row {row_num}'s begin-file.")

                # Add site name, lat/long, habitat, and protection:                
                site_loc_dict = self.get_site_loc_dict(row)
                
                # Get the site code: e.g. nn06a, and put its lat/long into 
                # the lookup dict. This info will be used for the sunset/sunrise
                # computations:
                
                TimeUtils.LOC_LAT_LONG_DICT[site_loc_dict['Site']] = Point(site_loc_dict['Latitude'],
                                                                           site_loc_dict['Longitude']
                                                                           )
                
                # Get a date obj from the date:
                label_date_obj = date(int(label_date_obj[0:4]), int(label_date_obj[4:6]), int(label_date_obj[6:8]))
                
                time_dict = self.get_call_time_info(row, label_date_obj)
                
                row['LabelDate'] = label_date_obj
                row['StartTime 24hr Format'] = time_dict['start_time_24']  
                row['StopTime 24hr Format']   = time_dict['stop_time_24']  
                row['Sunrise'] = time_dict['sunrise']
                row['Sunset'] = time_dict['sunset']
                row['NightTimeStart'] = time_dict['night_time_start']
                row['NightTimeStop'] = time_dict['night_time_stop']

                row.update(site_loc_dict)

                writer.writerow(row)


    #-------------------------
    # get_site_loc_dict 
    #--------------
    
    def get_site_loc_dict(self, row):
        '''
        Given a row from the elephant call labels, 
        get is "Begin File' name, which is encoded as
        <site_code>_<call_date>_...
        Extract the site code, and find the dict 
        {site/lat/long/habitat/protection} created earlier
        in self.location_info
        
        Return that dict
        
        @param row: dict with one row from the elephant call labels
        @type row: {str : str}
        @return: dict with call location info: site-code, lat, long, habitat, protection 
        @rtype: {str : str}
        '''
        
        site_name = row['Begin File'].split('_')[0]
        return self.location_info[site_name]

    #-------------------------
    #  get_call_time_info
    #--------------
    
    def get_call_time_info(self, row, label_date_obj):
        '''
        From a row in elephant call data, extract/compute
        the following dict:
        
               {'start_time_24'    : start_time_24,
                'stop_time_24'     : stop_time_24,
                'sunrise'          : sunrise,
                'sunset'           : sunset,
                'night_time_start' : night_time_start,
                'night_time_stop'  : night_time_stop
                }
        
        @param row: dict of one row in elephant call data
        @type row: {str : str}
        @param label_date_obj: date of call 
        @type label_date_obj: datetime.date
        '''
        
        # Get begin time as seconds since midnight: 2543.4433
        
        begin_time_secs = float(row['Begin Time (s)'])
        end_time_secs = float(row['End Time (s)'])
        start_time_24 = f"{label_date_obj} {math.floor(begin_time_secs / 3600)}:" + f"{math.floor((begin_time_secs % 3600) / 60)}:" + f"{math.floor((begin_time_secs % 3600) % 60)}"
        stop_time_24 = f"{label_date_obj} {math.floor(end_time_secs / 3600)}:" + f"{math.floor((end_time_secs % 3600) / 60)}:" + f"{math.floor((end_time_secs % 3600) % 60)}"
        
        # Turn into datetime objs:
        
        start_time_24_obj = datetime.strptime(start_time_24, '%Y-%m-%d %H:%M:%S')
        stop_time_24_obj = datetime.strptime(stop_time_24, '%Y-%m-%d %H:%M:%S')
        
        recording_loc = self.get_site_loc_dict(row)['Site']
        sunrise, sunset = self.get_sunrise_sunset(recording_loc, label_date_obj)
        
        night_time_start = start_time_24_obj < sunrise or start_time_24_obj > sunset
        night_time_stop = stop_time_24_obj < sunrise or stop_time_24_obj > sunset
        
        return {'start_time_24'    : start_time_24,
                'stop_time_24'     : stop_time_24,
                'sunrise'          : sunrise,
                'sunset'           : sunset,
                'night_time_start' : night_time_start,
                'night_time_stop'  : night_time_stop
                }

    #-------------------------
    # get_location_info 
    #--------------
    
    def get_location_info(self):
        '''
        Read location file for all the audio collection sites:
           Site    Latitude        longitude       Habitat Protection
           
        Build a new dict: {site : {site : xxx,
                                   latitude : xxx,
                                   longitude : xxx,
                                   Habitat : xxx,
                                   Protection : xxx
                                   }
                          }
                          
        @return: dict keyed by site, with info about each site
        @rtype: {str : {str : str}}
        '''
        
        location_file = os.path.join(os.path.dirname(__file__), 'FinalARUlocs_simple.tsv')
        with open(location_file, 'r') as in_fd:
            reader = csv.DictReader(in_fd,
                                    delimiter = '\t'
                                    )
            all_locs  = {}
            for row in reader:
                # Get, the Site key/val pair
                # from row to use site as a key to the 
                # row-dict:
                site = row['Site']
                
                # Make capitalization of latitude/longitude
                # be consistently upper case:
                row['Longitude'] = row['longitude']
                del row['longitude']
                
                # New entry: 
                all_locs[site] = row
        return all_locs
        

    #-------------------------
    # get_sunrise_sunset 
    #--------------
    
    def get_sunrise_sunset(self, location_name, date_obj):
        '''
        Given a recording location site, like 'nna1', and
        a date string like '2018-02-28', return a tuple
        with sunrise and sunset at that location, 
        on that date.
        
        @param location_name: location of the recording. Must
            be a key into LOC_LAT_LONG_DICT
        @type location_name: str
        @param date_obj: date when sunrise/sunset is requested
        @type date_obj: datetime.datetime
        @param location: name of location, which must be a key in 
            dict TimeUtils.LOC_LAT_LONG_DICT
        @type location: str
        @return: tuple of sunset/sunrise
        @rtype: (datetime, datetime)
        '''
        
        # Already in the cache?
        try:
            return self.daylight_cache[location_name]
        except KeyError:
            # Not in cache:
            pass
        
        lat_long   = TimeUtils.LOC_LAT_LONG_DICT[location_name]
        lat        = lat_long.latitude
        long       = lat_long.longitude
        req = requests.get(f'https://api.sunrise-sunset.org/json?lat={lat}&lng={long}&date={date_obj}')
        # Get: '6:26:52 AM'
        sunrise_time = req.json()['results']['sunrise']
        sunset_time  = req.json()['results']['sunset']
        
        # Turn '2018-02-23 '6:26:52 AM' into a datetime: 
        sunrise = datetime.strptime(f"{date_obj} {sunrise_time}", '%Y-%m-%d %I:%M:%S %p')
        sunset  = datetime.strptime(f"{date_obj} {sunset_time}", '%Y-%m-%d %I:%M:%S %p')
        
        self.daylight_cache[location_name] = (sunrise, sunset)
        
        return(sunrise, sunset)


# --------------------------- Main ------------------        
if __name__ == '__main__':


    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Add info to elephant label files; write tsv to stdout"
                                     )


    parser.add_argument('label_file',
                        help='Path to elephant call tsv labels path',
                        )
    
    args = parser.parse_args()

    label_file = args.label_file
    if not os.path.exists(label_file):
        print(f"File {label_file} does not exist.")
        sys.exit(1)

    #*********TimeUtils(label_file)
    #*********sys.exit(0)
    # TESTING
    TimeUtils('/Users/paepcke/Project/Wildlife/Data/Elephants/labels_2018_all.tsv')
                        
                