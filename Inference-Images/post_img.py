"""
post_img.py
Authors: Sin Yong Tan and Maggie Jacoby
Edited: 2020-09-01

Input: Folder with img inferences on the 1S (86400 per day) in day-wise CSVS
Output: Day-wise CSVs with inferences on the 10 seconds

To run: python3 post_img.py -path /Volumes/TOSHIBA-18/H6-black/ 
	optional parameters: -hub, -start_date, -save_location
"""


import os
import sys
import glob
import argparse

import numpy as np
import pandas as pd

import time
from datetime import datetime

from my_functions import *



def create_timeframe(start_date, end_date=None, freq="10S"):
    
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    if end_date == None:
        end_date = start_date + pd.Timedelta(days=1)
    else:
        end_date = datetime.strptime(end_date, '%Y-%m-%d')

    timeframe = pd.date_range(start_date, end_date, freq=freq).strftime('%Y-%m-%d %H:%M:%S')[:-1]
    timeframe = pd.to_datetime(timeframe)
    
    return timeframe




if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-path','--path', default='AA', type=str, help='path of stored data')
	parser.add_argument('-hub', '--hub', default='', type=str, help='if only one hub... ')
	parser.add_argument('-save_location', '--save', default='', type=str, help='location to store files (if different from path')
	parser.add_argument('-start_date', '--start', default='', type=str, help='type day to start')

	args = parser.parse_args()

	path = args.path
	save_path = args.save if len(args.save) > 0 else path
	start_date = args.start
	home_system = os.path.basename(path.strip('/'))
	H = home_system.split('-')
	H_num, color = H[0], H[1][0].upper()
	hubs = [args.hub] if len(args.hub) > 0 else sorted(mylistdir(path, bit=f'{color}S', end=False))
	print(f'List of Hubs: {hubs}')


	for hub in hubs:
		start = time.time()

		print(f'Reading image inferences from from hub: {hub}')
		read_root_path = os.path.join(path, 'Inference_DB', hub, 'img_1sec', '20*')
		dates = sorted(glob.glob(read_root_path))
		dates = [x for x in dates if os.path.basename(x) >= start_date]


		save_root_path = make_storage_directory(os.path.join(save_path,'Inference_DB', hub, 'img_inf'))
		print("save_root_path: ", save_root_path)

		for date_folder_path in dates:
			date = os.path.basename(date_folder_path).strip('.csv')
			print(f"Loading date folder: {date} ...")

			data = pd.read_csv(date_folder_path, squeeze=True, index_col=0) # Read in as pd.Series, for resample()
			data.index = pd.to_datetime(data.index)
			data = data.resample('10S', label='right', closed='right').max() # include right end value, labeled using right end(timestamp)

			timeframe = create_timeframe(date)
			data = data.reindex(timeframe, fill_value=np.nan)
			data.index.name = "timestamp"
			data.to_csv(os.path.join(save_root_path,f'{date}.csv'))
			