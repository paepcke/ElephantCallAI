#!/usr/bin/env python

import argparse
import calendar
from datetime import date, datetime
from enum import Enum
import os
import re
import sys
from urllib import request


class Increments(Enum):
	MONTHLY = 1
	YEARLY  = 2
	ALL     = 3
	
class WeatherPuller(object):

	#------------------------------------
	# Constructor
	#-------------------

	def __init__(self, 
				start_mmm_yyyy,  # inclusive e.g. 'Feb-2019' 
				end_mmm_yyyy,    # exclusive
				lat,
				long,
				increment=Increments.MONTHLY,
				outfile=None
				):
		
		(start_month, start_year) = start_mmm_yyyy.split('-')
		# Example: 'Jan-01-2020', '%b-%d-%Y')

		dt_start = datetime.strptime(f"{start_month}-01-{start_year}", '%b-01-%Y')
		
		(end_month, end_year) = end_mmm_yyyy.split('-')
		dt_end = datetime.strptime(f"{end_month}-01-{end_year}", '%b-01-%Y')

		self.get_data(dt_start, dt_end, lat, long, increment, outfile)
		

	#------------------------------------
	# get_data 
	#-------------------

	def get_data(self, start_dt, end_dt, lat, long, increment, outfile):
		
		# https://cleanedobservations.weather.com/v2/wsi/metar/[2.32474,16.45236]?startDate=01/01/2015&endDate=12/31/2019&interval=hourly&units=metric&format=csv&time=lwt&userKey=2084dbaba8713db0723fbd7f96d1231f

		if increment == Increments.MONTHLY:
			inc_in_months = 1
		elif increment == Increments.YEARLY:
			inc_in_months = 12
		else:
			inc_in_months = 0
		
		next_start_dt   = start_dt
		next_start_date = datetime.strftime(next_start_dt, '%m/01/%Y')

		if outfile is None:
			out_fd = sys.stdout
		else:
			# Clear out destination file:
			out_fd = open(outfile, 'w')
			out_fd.truncate()
			out_fd.close()
			out_fd = open(outfile, 'a')
			
		done = False
		try:
			while not done:
				if increment == Increments.ALL:
					next_end_date = datetime.strftime(end_dt, '%m/01/%Y')
				else:
					next_end_dt   = self.add_months(next_start_dt, inc_in_months)
					next_end_date = datetime.strftime(next_end_dt, '%m/01/%Y')
					
				
				url = f"https://cleanedobservations.weather.com/v2/wsi/metar/" +\
					  f"[{lat},{long}]?startDate={next_start_date}&endDate={next_end_date}&" +\
					  f"interval=hourly&units=metric&format=csv&time=lwt&userKey=2084dbaba8713db0723fbd7f96d1231f"
	
				print(url)
				
				(tmp_fn, headers) = request.urlretrieve(url)
				print(tmp_fn)
		except Exception as e:
			print(f"Call failed: {repr(e)}")
			print(f"URL used: {url}")
		finally:
			try:
				if outfile is not None:
					out_fd.close()		
			except Exception:
				pass	
			

	#------------------------------------
	# add_months 
	#-------------------

	def add_months(self, start_date, months):

		month = start_date.month - 1 + months
		year = start_date.year + month // 12
		month = month % 12 + 1
		day = min(start_date.day, calendar.monthrange(year,month)[1])
		return date(year, month, day)
	
# --------------------------- Main ----------------------

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Pull weather data from weather.com month by month"
                                     )

# 	parser.add_argument('-l', '--errLogFile',
#                         help='fully qualified log file name to which info and error messages \n' +\
#                              'are directed. Default: stdout.',
#                         dest='errLogFile',
#                         default=None);
	parser.add_argument('-o', '--outfile',
                        help='fully qualified log file name for output. Default: stdout.',
                        default=None);                        
	parser.add_argument('-i' '--increments',
                        help='Granularity of requests (case insensitive): {MONTHLY | YEARLY | ALL}. Default: MONTHLY',
                        choices=['MONTHLY', 'monthly', 'YEARLY', 'yearly', 'ALL', 'all'],
                        default="MONTHLY",
                        dest='increments'
                        )

	parser.add_argument('start_mmm_yyy',
						help='Start month/year (inclusive); example: Feb-2019'
						)
	parser.add_argument('end_mmm_yyy',
						help='End month/year (exclusive); example: Feb-2019'
						)
	parser.add_argument('lat',
						type=float,
						help='Latiture like: 2.32410'
						)
	parser.add_argument('long',
						type=float,
						help='Longitude like: 16.55124'
						)
		
	args = parser.parse_args()
	
	# Check date format:
	p = re.compile(r'{JAN|jan|Jan|FEB|feb|Feb|MAR|mar|Mar|'+\
	               r'APR|apr|Apr|MAY|may|May|JUN|jun|Jun|'+\
	               r'JUL|jul|Jul|AUG|aug|Aug|SEP|sep|Sep}'+\
	               r'OCT|oct|Oct|NOV|nov|Nov|DEC|dec|Dec}'+\
	               r'-[0-9]{4}'
               )
	if p.search(args.start_mmm_yyy) is None:
		print(f"Bad start date format; should be like FEB-2001, not {args.start_mmm_yyy}")
		sys.exit()
	if p.search(args.end_mmm_yyy) is None:
		print(f"Bad end date format; should be like FEB-2001, not {args.end_mmm_yyy}")
		sys.exit()
	
	if args.increments.lower() == 'monthly':
		incs = Increments.MONTHLY
	elif args.increments.lower() == 'yearly':
		incs = Increments.YEARLY
	else:
		incs = Increments.ALL
		
	WeatherPuller(args.start_mmm_yyy,
				  args.end_mmm_yyy,
				  args.lat,
				  args.long,
				  increment=incs,
				  outfile=args.outfile
		)
	
	