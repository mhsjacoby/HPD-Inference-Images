"""
confidence.py (formerly detect.py -> now archived)
Author: Sin Yong Tan 2020-08-04
Based on code from: https://github.com/ultralytics/yolov5
Updates by Maggie Jacoby 
	2020-09-16: nuse gen_argparser, change file location

This is the first step of inferencing code for images in the HPDmobile
Inputs: path the folder (home, system, hub specific), which contain 112x112 png images
Outputs: csv with 0/1  occupancy by day (thresholded acording to detection_options.py occ_threshold) and prob (greater than detection_options.py prob_threshold)

A median filter is first applied to the images (default filter size = 3)

Run this:
python3 confidence.py -path /Users/maggie/Desktop/
	optional arguments: 
						-hub (default is to run all in the folder)
						-save_location (if different from read path)
						-img_file_name (default is whatever starts with "img" eg, "img-downsized" or "img-unpickled")
"""

import argparse
import torch.backends.cudnn as cudnn
from yolov5.utils.datasets import *
from yolov5.utils.utils import *

import cv2
import datetime
import time
import sys

import warnings
warnings.filterwarnings("ignore")

from gen_argparse import *
from my_functions import *
import detection_options as opt

def detect():

	minute_fname, minute_occupancy, minute_conf = [], [], []
	source, save_txt, imgsz = opt.source, opt.save_txt, opt.img_size

	dataset = LoadImages(source, img_size=imgsz)

	# Run inference
	for path, img, _, _ in dataset:
		fname = os.path.basename(path).split("_")
		fname = fname[0] + " " + fname[1]
		minute_fname.append(fname)

		# ==== Added Median Filtering Here ====
		img = cv2.medianBlur(img, opt.filter_size)

		img = torch.from_numpy(img).to(device)		
		img = img.float()  # uint8 to fp32
		img /= 255.0  # 0 - 255 to 0.0 - 1.0
		if img.ndimension() == 3:
			img = img.unsqueeze(0)

		# Inference
		pred = model(img, augment=opt.augment)[0]
		pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=[0], agnostic=opt.agnostic_nms)

		# Process detections
		for i, det in enumerate(pred):  # detections per image
			M = 0
			if det is not None:
				M = max([float(x[4]) for x in det])
			minute_conf.append(M)
			minute_occupancy.append(0 if det is None or M < opt.occ_threshold else 1)
	return minute_fname, minute_occupancy, minute_conf



if __name__ == '__main__':
	# uses arguments specifed by gen_argparse.py
	print(f'List of Hubs: {hubs}')

	for hub in hubs:
		print(hub)
		img_fnames = [img_name] if len(img_name) > 0 else mylistdir(os.path.join(path, hub), bit='img', end=False)
		if len(img_fnames) > 1:
			print(f'Multiple images files found in hub {hub}: {img_fnames}. \n System Exiting.')
			sys.exit()
		
		print(f'Reading images from file: {img_fnames[0]}')

		read_root_path = os.path.join(path,hub,img_fnames[0],"*")
		dates = sorted(glob.glob(read_root_path))
		dates = [x for x in dates if os.path.basename(x) >= start_date]
		print('Dates: ', [os.path.basename(d) for d in dates])

		save_path = make_storage_directory(os.path.join(save_root,'Inference_DB', hub, 'img_1sec'))
		print("save_root_path: ", save_path)

		# ================ Move Model loading etc here ================
		# Initialize
		device = torch_utils.select_device(opt.device) # Prints "Using CPU"

		# Load model
		model = torch.load(opt.weights, map_location=device)['model'].float()  # load to FP32
		model.to(device).eval()
		start = time.time()

		for date_folder_path in dates:
			date = os.path.basename(date_folder_path)
			if not date.startswith('20'):
				print(f'passing folder: {date}')
				continue

			print(f"Loading date folder: {date} ...")

			''' Check if Directory is empty '''
			times = os.listdir(date_folder_path)
			if len(times) == 0: 
				print(f"Date folder {os.path.basename(date_folder_path)} is empty")
			else:

				day_start = datetime.datetime.now()
				day_fname, day_occupancy, day_conf = [], [], []			# Create placeholders
				date_folder_path = os.path.join(date_folder_path, "*")
				
				for time_folder_path in sorted(glob.glob(date_folder_path)):
					time_f = os.path.basename(time_folder_path)
					if int(time_f)%100 == 0:
						print(f"Checking time folder: {time_f} ...")

					imgs = os.listdir(time_folder_path)

					if len(imgs) == 0:
						print(f"Time folder {os.path.basename(time_folder_path)} is empty")

					else:
						# Update source folder
						opt.source = time_folder_path # 1 min folder ~ 60 img

						with torch.no_grad():
							min_fname, min_occ, min_conf = detect() # detect this time folder
							day_fname.extend(min_fname)
							day_occupancy.extend(min_occ)
							day_conf.extend(min_conf)


				day_fname = [date_[:11]+date_[11:13]+":"+date_[13:15]+":"+date_[15:17] for date_ in day_fname] # date formatting
				day_fname = [datetime.datetime.strptime(date_, '%Y-%m-%d %H:%M:%S') for date_ in day_fname] # date formatting

				save_data = np.vstack((day_fname,day_occupancy, day_conf))
				save_data = np.transpose(save_data)
				np.savetxt(os.path.join(save_path,date+".csv"), save_data, delimiter=',',fmt='%s',header="timestamp,occupied,probability",comments='')
				day_end = datetime.datetime.now()
				print(f"Time to process day {date} on hub {hub}: {str(day_end-day_start).split('.')[0]}")
				print(f'Current time is: {datetime.datetime.now().strftime("%m-%d %H:%M")}')

		end = time.time()
		total_time = (end-start)/3600
		print(f'Total time taken to process hub {hub} in home {H_num}: {total_time:.02} hours')
		
