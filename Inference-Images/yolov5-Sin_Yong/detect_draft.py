import argparse

import torch.backends.cudnn as cudnn

# from utils import google_utils # To download weights. Can wremove google util file altogether
from utils.datasets import *
from utils.utils import *

import cv2
# from time import time
import datetime

# import pandas as pd


def detect(save_img=False):

	minute_fname = []
	minute_occupancy = []

	source, weights, view_img, save_txt, imgsz = \
		opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size

	webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

	# Initialize
	device = torch_utils.select_device(opt.device) # Prints "Using CPU"

	# half = device.type != 'cpu'  # half precision only supported on CUDA
	half = False

	# Load model
	# google_utils.attempt_download(weights) # download if model not found 
	model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
	# torch.save(torch.load(weights, map_location=device), weights)  # update model if SourceChangeWarning
	# model.fuse()
	model.to(device).eval()
	# if half:
		# model.half()  # to FP16

	# Second-stage classifier
	# classify = False
	# if classify:
		# modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
		# modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
		# modelc.to(device).eval()

	# Set Dataloader
	vid_path, vid_writer = None, None
	# if webcam:
		# view_img = True
		# cudnn.benchmark = True  # set True to speed up constant image size inference
		# dataset = LoadStreams(source, img_size=imgsz)
	# else:
	save_img = True
	dataset = LoadImages(source, img_size=imgsz)

	# Get names and colors
	names = model.module.names if hasattr(model, 'module') else model.names
	colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

	# Run inference
	t0 = time.time()
	
	img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
	# _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
	_ = model(img.half() if half else img) if device.type != 'cpu' else None  # How to change?

	for path, img, im0s, vid_cap in dataset:


		fname = os.path.basename(path).split("_")[0]

		minute_fname.append(fname)


		# ==== Added Median Filtering Here ====
		# prev_time = time.time()
		# img = cv2.medianBlur(img, opt.filter_size)
		# inference_time = datetime.timedelta(seconds=time.time() - prev_time)
		# print('Median filter time: %s' % (inference_time))
		# print('Median filter time: %.9f' % (time.time()-prev_time))


		img = torch.from_numpy(img).to(device)
		
		# img = img.half() if half else img.float()  # uint8 to fp16/32
		img = img.float()  # uint8 to fp32
		
		img /= 255.0  # 0 - 255 to 0.0 - 1.0
		if img.ndimension() == 3:
			img = img.unsqueeze(0)

		# Inference
		# t1 = torch_utils.time_synchronized() # timing the prediction
		pred = model(img, augment=opt.augment)[0]

		# Apply NMS
		pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
		# t2 = torch_utils.time_synchronized()

		# Apply Classifier
		# if classify:
			# pred = apply_classifier(pred, modelc, img, im0s)

		# Process detections
		for i, det in enumerate(pred):  # detections per image
			# if webcam:  # batch_size >= 1
				# p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
			# else:
			p, s, im0 = path, '', im0s

			minute_occupancy.append(0 if det is None else 1)
			
			# if det is None: # if no "person" was detected
				# minute_occupancy.append(0)
			# else: # "person" was detected
				# minute_occupancy.append(1)



	return minute_fname, minute_occupancy










	# 		save_path = str(Path(out) / Path(p).name)
	# 		txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
	# 		s += '%gx%g ' % img.shape[2:]  # print string
	# 		gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
	# 		if det is not None and len(det):
	# 			# Rescale boxes from img_size to im0 size
	# 			det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

	# 			# Print results
	# 			for c in det[:, -1].unique():
	# 				n = (det[:, -1] == c).sum()  # detections per class
	# 				s += '%g %ss, ' % (n, names[int(c)])  # add to string

	# 			# Write results
	# 			for *xyxy, conf, cls in det:
	# 				if save_txt:  # Write to file
	# 					xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
	# 					with open(txt_path + '.txt', 'a') as f:
	# 						f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

	# 				if save_img or view_img:  # Add bbox to image
	# 					label = '%s %.2f' % (names[int(cls)], conf)
	# 					plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

	# 		# Print time (inference + NMS)
	# 		print('%sDone. (%.3fs)' % (s, t2 - t1))

	# 		# Stream results
	# 		if view_img:
	# 			cv2.imshow(p, im0)
	# 			if cv2.waitKey(1) == ord('q'):  # q to quit
	# 				raise StopIteration

	# 		# Save results (image with detections)
	# 		if save_img:
	# 			if dataset.mode == 'images':
	# 				cv2.imwrite(save_path, im0)
	# 			else:
	# 				if vid_path != save_path:  # new video
	# 					vid_path = save_path
	# 					if isinstance(vid_writer, cv2.VideoWriter):
	# 						vid_writer.release()  # release previous video writer

	# 					fps = vid_cap.get(cv2.CAP_PROP_FPS)
	# 					w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	# 					h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	# 					vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
	# 				vid_writer.write(im0)

	# if save_txt or save_img:
	# 	print('Results saved to %s' % os.getcwd() + os.sep + out)
	# 	if platform == 'darwin':  # MacOS
	# 		os.system('open ' + save_path)

	# print('Done. (%.3fs)' % (time.time() - t0))






if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	# Model arg
	parser.add_argument('--weights', type=str, default='weights/yolov5s.pt', help='model.pt path')
	parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder to read the img (constantly loops in the code)
	# parser.add_argument('--output', type=str, default='inference/output/x3', help='output folder')  # output folder - Not used
	parser.add_argument('--img-size', type=int, default=128, help='inference size (pixels)')
	parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
	parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
	parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
	parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
	parser.add_argument('--view-img', action='store_true', help='display results')
	parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
	parser.add_argument('--classes', default=[0], type=int, help='filter by class') # By default detects only "person"
	parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
	parser.add_argument('--augment', action='store_true', help='augmented inference')

	parser.add_argument('-f_sz', '--filter_size', type=int, default=1, help='Apply median filter to input img') # Added filtering option


	# Loading arg
	parser.add_argument('-drive','--drive_letter', default="AA", type=str, help='Hard Drive Letter')
	parser.add_argument('-H','--H_num', default='1', type=int, help='House number: 1,2,3,4,5,6')
	parser.add_argument('-sta_num','--station_num', default=1, type=int, help='Station Number')
	parser.add_argument('-sta_col','--station_color', default="B", type=str, help='Station Color')

	parser.add_argument('-start','--start_date_index', default=0, type=int, help='Processing START Date index')
	parser.add_argument('-end','--end_date', default=None, type=str, help='Processing END Date')


	opt = parser.parse_args()
	opt.img_size = check_img_size(opt.img_size)


	# ==========================================================
	# day_fname = []
	# day_occupancy = []


	# with torch.no_grad():
	# 	min_fname, min_occ = detect() # detect this time folder
	# 	day_fname.extend(min_fname)
	# 	day_occupancy.extend(min_occ)

	# print("min_fname:",min_fname)
	# print("min_occ:",min_occ)

	# save_data = np.vstack((day_fname,day_occupancy))

	# np.savetxt(path, save_data, delimiter=',',fmt='%s',header="timestamp,occupancy",comments='')

	# ==========================================================



	drive_letter = opt.drive_letter
	H_num = opt.H_num
	station_num = opt.station_num
	station_color = opt.station_color
	# filter_sz = opt.filter_size
	# img_sz = opt.img_sz
	start_date_index = opt.start_date_index
	# end_date = opt.end_date # Not implemented yet

	if station_color == "B":
		sta_col = "black"
	elif station_color == "R":
		sta_col = "red"
	elif station_color == "G":
		sta_col = "green"

	read_root_path = os.path.join(drive_letter+":/","H%s-%s"%(H_num,sta_col),"%sS%s"%(station_color,station_num),"img","*")
	print("read_root_path: ", read_root_path)
	
	save_root_path = os.path.join(drive_letter+":/","Inference_DB","H%s-%s"%(H_num,sta_col),"%sS%s"%(station_color,station_num),"img")
	print("save_root_path: ", save_root_path)

	if not os.path.exists(save_root_path):
		os.makedirs(save_root_path)

	dates = sorted(glob.glob(read_root_path))
	dates = dates[start_date_index:]


	for date_folder_path in dates:
		date = os.path.basename(date_folder_path)
		print("Loading date folder: " + date + "...")

		'''
		Check if Directory is empty
		'''
		times = os.listdir(date_folder_path)

		if len(times) == 0: 
			print("Date folder "+ os.path.basename(date_folder_path) + " is empty")
		else:

			# Created day-content placeholder
			day_fname = []
			day_occupancy = []


			date_folder_path = os.path.join(date_folder_path,"*")
			
			for time_folder_path in sorted(glob.glob(date_folder_path)):
				time_f = os.path.basename(time_folder_path)
				print("Checking time folder: "+ time_f +"...")

				wavs = os.listdir(time_folder_path)

				if len(wavs) == 0:
					print("Time folder "+ os.path.basename(time_folder_path) + " is empty")

				else:
					# Update source folder
					opt.source = time_folder_path

					with torch.no_grad():
						min_fname, min_occ = detect() # detect this time folder
						day_fname.extend(min_fname)
						day_occupancy.extend(min_occ)


			day_fname = [date_[:11]+date_[11:13]+":"+date_[13:15]+":"+date_[15:17] for date_ in day_fname]
			day_fname = [datetime.datetime.strptime(date_, '%Y-%m-%d %H:%M:%S') for date_ in day_fname]

			save_data = np.vstack((day_fname,day_occupancy))
			save_data = np.transpose(save_data)
			np.savetxt(os.path.join(save_root_path,date+".csv"), save_data, delimiter=',',fmt='%s',header="timestamp,occupancy",comments='')



'''
python detect.py -drive G -H 1 -sta_num 1 -sta_col G -f_sz 1

python detect.py --img-size 256 --filter_size 3 --conf-thres 0.5 --weights weights/yolov5s.pt --source 0 --output inference/output2/s-cam

python detect.py --img-size 640 --filter_size 3 --conf-thres 0.5 --weights weights/yolov5s.pt --source inference/images --output inference/output2/s-640



Note: 
Keep save_img and save_txt, update them to save_occ and save_csv(or save_json) later, depending on need

How to save the labeled data with >1 bounding box?


Check chronological order of the saving - should be sorted ady



'''