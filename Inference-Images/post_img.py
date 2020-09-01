'''
cd "to_maggie"
'''
# csv10
import os
from datetime import datetime
import pandas as pd

from glob import glob
from natsort import natsorted
import numpy as np


# Function creates the "timeframe" for 1 day, in 1 sec(changable) freq. 
def create_timeframe(file_path):
	# "filepath" example: G:\H1-black\BS5\csv\H1_BS5_2019-02-23.csv
	file_name = os.path.basename(file_path)  # csv file name with extension
	index = file_name.find("-")
	start_date = file_name[index-4:index+6]  # takes the "2019-02-23" part

	end_date = datetime.strptime(start_date, '%Y-%m-%d')
	start_date = datetime.strptime(start_date, '%Y-%m-%d')
	end_date = end_date + pd.Timedelta(days=1)
# 	time_frame = pd.date_range(start_date, end_date, freq = '10S').strftime('%Y-%m-%d %H:%M:%S').tolist()
	time_frame = pd.date_range(start_date, end_date, freq = '10S').strftime('%Y-%m-%d %H:%M:%S')
	time_frame = time_frame[:-1]

	return time_frame, file_name




# ==== Add Arg parsers ====

H_num = 6
station_color = "B"
station_nums = [2]

if station_color == "B":
	sta_col = "black"
elif station_color == "R":
	sta_col = "red"
elif station_color == "G":
	sta_col = "green"


# for station_num in station_nums:
station_num = station_nums[0]

data_path = f"C:/Users/Sin Yong Tan/Desktop/to_maggie/H{H_num}-{sta_col}/Inference_DB/{station_color}S{station_num}/img_inf"
save_path = os.path.join(data_path,"processed")

# Create Folder
if not os.path.exists(save_path):
	os.makedirs(save_path)

days = natsorted(glob(os.path.join(data_path,"*.csv")))

for day in days:
# day = days[0]
# 	print(day)

	data = pd.read_csv(day,squeeze=True,index_col=0) # Read in as pd.Series, for resample()
	data.index = pd.to_datetime(data.index)
	data = data.resample('10S', label='right', closed='right').max() # include right end value, labeled using right end(timestamp)
	
	timeframe, fname = create_timeframe(day)
	timeframe = pd.to_datetime(timeframe)
	
	# https://stackoverflow.com/questions/19324453/add-missing-dates-to-pandas-dataframe
	data = data.reindex(timeframe, fill_value=np.nan)
	data.index.name = "timestamp"
	data.to_csv(os.path.join(save_path,fname))
	

