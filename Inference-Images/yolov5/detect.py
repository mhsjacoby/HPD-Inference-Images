"""
detect.py
Author: Sin Yong Tan 2020-08-04
Based on code from: https://github.com/ultralytics/yolov5
Updates by Maggie Jacoby 
	2020-08-05: replaced if/else with dictionary, changed read paths, changed format of save date name

This is the first step of inferencing code for images in the HPDmobile
Inputs: path the folder (home, system, hub specific), which contain 112x112 png images
Outputs: csv with 0/1  occupancy by day

A median filter is first applied to the images (default filter size = 3)

Run this:
python detect.py -path /Users/maggie/Desktop/ -H 1 -sta_num 1 -sta_col G


==== SY Notes ====
Keep save_img and save_txt, update them to save_occ and save_csv(or save_json) later, depending on need

How to save the labeled data with >1 bounding box?

Check chronological order of the saving in csv (should be sorted ady) 

runs around 14~16 FPS
"""

import argparse
import torch.backends.cudnn as cudnn
from utils.datasets import *
from utils.utils import *
import cv2
import datetime
import time

import warnings
warnings.filterwarnings("ignore")

def detect():

	minute_fname, minute_occupancy, minute_conf = [], [], []
	source, save_txt, imgsz = opt.source, opt.save_txt, opt.img_size

	dataset = LoadImages(source, img_size=imgsz)

	# Run inference
	for path, img, _, _ in dataset:
		print(path)
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
		
		# Apply NMS
		pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=[0], agnostic=opt.agnostic_nms)
		# if len(pred) > 0:
		# 	print(f'pred: {len(pred[0])}')
		# Process detections
		for i, det in enumerate(pred):  # detections per image
			M = 0
			if det is not None:
				M = max([float(x[4]) for x in det])
			minute_conf.append(M)
			minute_occupancy.append(0 if det is None else 1)			
	return minute_fname, minute_occupancy, minute_conf






if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	# Model arg: No changes needed
	parser.add_argument('--weights', type=str, default='weights/yolov5s.pt', help='model.pt path')
	# parser.add_argument('--weights', type=str, default='weights/yolov5x.pt', help='model.pt path')
	parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder to read the img (constantly loops in the code)
	parser.add_argument('--img-size', type=int, default=128, help='inference size (pixels)')
	parser.add_argument('--conf-thres', type=float, default=0.2, help='object confidence threshold')
	parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
	parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
	parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
	parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
	parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
	parser.add_argument('--augment', action='store_true', help='augmented inference')

	parser.add_argument('-f_sz', '--filter_size', type=int, default=3, help='Apply median filter to input img') # Added median filtering option


	# Loading arg
	parser.add_argument('-path','--path', default="AA", type=str, help='path of stored data')
	parser.add_argument('-H','--H_num', default='1', type=int, help='House number: 1,2,3,4,5,6')
	parser.add_argument('-sta_num','--station_num', default=1, type=int, help='Station Number')
	parser.add_argument('-sta_col','--station_color', default="B", type=str, help='Station Color')

	parser.add_argument('-start','--start_date_index', default=0, type=int, help='Processing START Date index')
	# parser.add_argument('-end','--end_date_index', default=-1, type=int, help='Processing END Date index')

	opt = parser.parse_args()
	opt.img_size = check_img_size(opt.img_size)

	path = opt.path
	H_num = opt.H_num
	station_num = opt.station_num
	station_color = opt.station_color
	start_date_index = opt.start_date_index
	# end_date_index = opt.end_date_index # Not implemented yet

	color_index = {"B": "black", "R": "red", "G": "green"}
	sta_col = color_index[station_color]



	read_root_path = os.path.join(path,"H%s-%s"%(H_num,sta_col),"%sS%s"%(station_color,station_num),"img-downsized","*")
	print("read_root_path: ", read_root_path)
	
	save_root_path = os.path.join(path,"Inference_DB","H%s-%s"%(H_num,sta_col),"%sS%s"%(station_color,station_num),"img")
	print("save_root_path: ", save_root_path)

	if not os.path.exists(save_root_path):
		os.makedirs(save_root_path)

	dates = sorted(glob.glob(read_root_path))
	dates = dates[start_date_index:]
	print("Starting from:", os.path.basename(dates[0]))
	# dates = dates[start_date_index:end_date_index] # Not implemented


	# ================ Move Model loading etc here ================
	# Initialize
	device = torch_utils.select_device(opt.device) # Prints "Using CPU"

	# Load model
	model = torch.load(opt.weights, map_location=device)['model'].float()  # load to FP32
	model.to(device).eval()

	start = time.time()

	for date_folder_path in dates:
		date = os.path.basename(date_folder_path)
		print("Loading date folder: " + date + "...")

		''' Check if Directory is empty '''
		times = os.listdir(date_folder_path)

		if len(times) == 0: 
			print("Date folder "+ os.path.basename(date_folder_path) + " is empty")
		else:
			# Created day-content placeholder
			day_fname, day_occupancy, day_conf = [], [], []			
			date_folder_path = os.path.join(date_folder_path,"*")
			
			for time_folder_path in sorted(glob.glob(date_folder_path)):
				time_f = os.path.basename(time_folder_path)
				print("Checking time folder: "+ time_f +"...")

				imgs = os.listdir(time_folder_path)

				if len(imgs) == 0:
					print("Time folder "+ os.path.basename(time_folder_path) + " is empty")

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

			save_data = np.vstack((day_fname, day_occupancy, day_conf))
			save_data = np.transpose(save_data)
			np.savetxt(os.path.join(save_root_path,date+".csv"), save_data, delimiter=',',fmt='%s',header="timestamp,occupancy, confidence",comments='')


	end = time.time()
	print("Time taken:",end-start)


