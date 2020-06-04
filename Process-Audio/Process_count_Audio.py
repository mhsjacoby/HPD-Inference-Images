"""
Process_count_Audio.py
Base code written by Sin Yong Tan, based on processing technique developed by Ali Saffair, 
Editied and audio counting added by Maggie Jacoby
Most recent edit: 6/4/2020
"""


import numpy as np
from glob import glob
import os
import scipy.io.wavfile
import argparse
from scipy.fftpack import dct
from sklearn.preprocessing import scale
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from datetime import datetime


def mylistdir(directory, bit='', end=True):
    filelist = os.listdir(directory)
    if end:
        return [x for x in filelist if x.endswith(f'{bit}')]
    else:
         return [x for x in filelist if x.startswith(f'{bit}')]

def make_storage_directory(target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    return target_dir



# ==================================================================
from scipy.signal import butter, lfilter, freqz

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# --------------------------------------------------------------------

def butter_lowpass(highcut, fs, order=5):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, high, btype='low')
    return b, a

def butter_lowpass_filter(data, highcut, fs, order=5):
    b, a = butter_lowpass(highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# ==================================================================

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    else:
        print(folder_name, "folder exist\n")


parser = argparse.ArgumentParser(description='Convert Wav to 12 DCTs')

parser.add_argument('-root','--root_dir', default='/Users/maggie/Desktop/HPD_mobile_data/HPD-env/', type=str,
                    help='Root Directory') 		# For running on Maggie's system

# parser.add_argument('-drive','--drive_letter', default="AA", type=str,
# 					help='Hard Drive Letter')

parser.add_argument('-H','--H_num', default='1', type=int,
                    help='House number: 1,2,3,4,5,6')
parser.add_argument('-sta_num','--station_num', default=1, type=int,
                    help='Station Number')
parser.add_argument('-sta_col','--station_color', default="B", type=str,
                    help='Station Color')

parser.add_argument('-start','--start_date_index', default=0, type=int,
                    help='Processing START Date index')
parser.add_argument('-end','--end_date', default=None, type=str,
                    help='Processing END Date')

args = parser.parse_args()


if __name__ == '__main__':

    root_dir = args.root_dir		# For running on Maggie's system
    # drive_letter = args.drive_letter
    H_num = args.H_num
    station_num = args.station_num
    station_color = args.station_color
    start_date_index = args.start_date_index
    end_date = args.end_date

    if station_color == "B":
        sta_col = "black"
    elif station_color == "R":
        sta_col = "red"

    filter_banks = []
    
    # filter_banks.append([0, 100])
    filter_banks.append([100, 200])
    filter_banks.append([200, 300])
    filter_banks.append([300, 400])
    filter_banks.append([395, 505])
    filter_banks.append([510, 630])
    filter_banks.append([630, 770])
    filter_banks.append([765, 915])
    filter_banks.append([920, 1080])
    filter_banks.append([1075, 1265])
    filter_banks.append([1265, 1475])
    filter_banks.append([1480, 1720])
    filter_banks.append([1710, 1990])
    filter_banks.append([1990, 2310])
    filter_banks.append([2310, 2690])
    filter_banks.append([2675, 3125])


    # Sample rate and desired cutoff frequencies (in Hz).
    fs = 8000
    # fh = 1600 # not used, but keepping as a record

    read_root_path = os.path.join(root_dir, "H%s-%s"%(H_num,sta_col),"%sS%s"%(station_color,station_num),"audio","*") 	# For running on Maggie's system
    # read_root_path = os.path.join(drive_letter+":/","H%s-%s"%(H_num,sta_col),"%sS%s"%(station_color,station_num),"audio","*")
    print("read_root_path: ", read_root_path)

   
    save_dir = '/Users/maggie/Desktop/Audio_test_save'
    save_root = os.path.join(save_dir, f'H{H_num}-{sta_col}',f'{station_color}{station_num}')  #"H%s-%s"%(H_num,sta_col),"%sS%s"%(station_color,station_num)) # For running on Maggie's system
    print('save root: ', save_root)

    # processed_save = os.path.join('/Users/maggie/Desktop/Audio_test_save',"H%s-%s"%(H_num,sta_col),"%sS%s"%(station_color,station_num),"audio_processed") # For running on Maggie's system
    # downsampled_save = os.path.join('/Users/maggie/Desktop/Audio_test_save',"H%s-%s"%(H_num,sta_col),"%sS%s"%(station_color,station_num),"audio_downsampled") # For running on Maggie's system
    # save_root_path = os.path.join(drive_letter+":/","H%s-%s"%(H_num,sta_col),"%sS%s"%(station_color,station_num),"audio_processed")
    # print("save_root_path: ", save_root_path)

    dates = sorted(glob(read_root_path))
    dates = dates[start_date_index:]

    # ==== Downsampling params ====
    '''
    NOTE: 
    These params changes if 
    - initial fs is not 8000 
    - desired fh is not 1600 
    - num. filters is not 16
    - wav file length is not 10 seconds

    These params was hard-coded to save computation time.
    '''
    number_of_filters = len(filter_banks)+1 # +1 for loww-pass (0~100hz that was not in filterbanks)
    filter_start_index = [i for i in range(0,number_of_filters*5,5)]
    increment = 80
    num_final_datapoint = 1000
    temp = np.asarray([i for i in range(num_final_datapoint)])*increment # [0, 80, 160,...]
    filter_i_sampling_index = np.zeros((number_of_filters,num_final_datapoint))

    for j in range(number_of_filters):
        filter_i_sampling_index[j] = temp + filter_start_index[j]

    filter_i_sampling_index = np.transpose(filter_i_sampling_index)
    filter_i_sampling_index = filter_i_sampling_index.astype(int)

    # ==== Start Looping Folders ====
    for date_folder_path in dates:
        date = os.path.basename(date_folder_path)
        print("Loading date folder: " + date + "...")

        '''
        Check if Directory is empty
        '''
        all_mins = sorted(mylistdir(date_folder_path))

        if len(all_mins) == 0:
            print("Date folder "+ date + " is empty")

        else:
            hours = [str(x).zfill(2) + '00' for x in range(0,24)]
            hourly_content = {hour:np.nan for hour in hours}
            # Read date folders
            print(f'full path: {date_folder_path}')

            # Make storage directories
            downsampled_folder = make_storage_directory(os.path.join(save_root, 'audio_downsampled', date))
            processed_folder = make_storage_directory(os.path.join(save_root, 'audio_processed', date))
            
            for hour in hours:
                print('hour', hour)
                content_ds = {f'{str(hour[0:2]).zfill(2)}:{str(M).zfill(2)}:{str(S).zfill(2)}':np.zeros((0,0)) for M in range(0,60) for S in range(0,60,10)}
                content_ps = {f'{str(hour[0:2]).zfill(2)}:{str(M).zfill(2)}:{str(S).zfill(2)}':np.zeros((0,0)) for M in range(0,60) for S in range(0,60,10)}

                this_hour = [x for x in all_mins if x[0:2]==hour[0:2]]
                for minute in sorted(this_hour):
                    minute_path = os.path.join(date_folder_path, minute)
                    # print(f'Checking time folder: {minute_path} ...')
                    wavs = sorted(mylistdir(minute_path, bit='.wav'))

                    if len(wavs) == 0:
                        print("Time folder "+ os.path.basename(minute_path) + " is empty")

                    else:

                        for wav_name in wavs:
                            wav_path = os.path.join(date_folder_path, minute, wav_name)
                            t = wav_name.split(' ')[-1].strip('._audio.wav')
                            time_file = f'{t[0:2]}:{t[2:4]}:{t[4:6]}'
                            # print('time:', time_file, t) # Loading this wav file                       

                            try:  
                                _, wav = scipy.io.wavfile.read(wav_path)
                                audio_len_seconds = len(wav)/fs # length of audio clip in seconds

                                assert (audio_len_seconds == 10.0)
                                
                                ## Process Audio
                                processed_audio = np.zeros((int(len(wav)),number_of_filters)) # Placeholder
                                
                                temp = butter_lowpass_filter(wav, 100, fs, order=6) # low pass filter (first filter)
                                temp -= np.mean(temp) # Mean Shift
                                processed_audio[:,0] = abs(temp) # Full wave rectify

                                for idx, Filter in enumerate(filter_banks):
                                    # temp = butter_bandpass_filter(wav, lowcut, highcut, fs, order=6)
                                    temp = butter_bandpass_filter(wav, Filter[0], Filter[1], fs, order=6) # Band pass filter
                                    processed_audio[:, idx+1] = abs(temp) # Full wave rectify


                                ## Downsample:
                                downsampled = np.zeros((num_final_datapoint,number_of_filters))

                                for i in range(number_of_filters):
                                    downsampled[:,i] = processed_audio[filter_i_sampling_index[:,i],i]



                                ################ Uncomment if don't want to perform dct ################
                                # fname, _ = os.path.splitext(os.path.basename(wav_path))
                                # downsample_save = os.path.join(downsampled_folder, fname+"_ds.npy")
                                # np.save(downsample_save, downsampled)

                                # modify below line for donwsampled
                                content_ds[time_file] = downsampled # store content, similar timestamp format as env. csv(s) timestamp

                                ################################################################



                                ################ Comment the following lines if don't want to perform dct ################
                                processed_audio = dct(downsampled) # Perform DCT across different filter on each timepoint
                                processed_audio = processed_audio[:,:12] # Keep only first 12 coefficients
                                processed_audio = scale(processed_audio,axis=1) # Normalizing/Scaling to zero mean & unit std
                                
                                # fname, _ = os.path.splitext(os.path.basename(wav_path))
                                # save_path = os.path.join(save_folder, fname+".npy")
                                # np.save(save_path, processed_audio)
                                ################################################################

                                # print()
                                ## Maggie write timestamp
                                ################ collect desired data(before and/or after dct) into dictionary(placeholder) for npz_zompressed saving ################
                                content_ps[time_file] = processed_audio # store content, similar timestamp format as env. csv(s) timestamp
                                ################################################################
                                # 2019-10-19 00:40:30 

                            except Exception as e:
                                print(f'Audio error: {e}')

                # create save folders

                    # ################ npz_compressed saving at the end of each hour (or desired saving interval) ################
                fname_ds = f'{hour}_ds.npz' # ==> intended to produce "0000", "0100", .....
                fname_ps = f'{hour}_ps.npz' # ==> intended to produce "0000", "0100", .....

                downsampled_save_path = os.path.join(downsampled_folder, fname_ds)
                processed_save_path = os.path.join(processed_folder, fname_ps)

                np.savez_compressed(downsampled_save_path, **content_ds)
                np.savez_compressed(processed_save_path, **content_ps)
                    # ################################################################
            
        print("======================================")

    print("Done")


'''
Example Expected Input Audio Folder structure:
F:/H1-red/RS1/audio/2019-02-10/2229/2019-02-10 222900_audio.wav

Example Output Audio Folder structure:
F:/H1-red/RS1/audio_processed/2019-02-10/2229/2019-02-10 222900_audio.npy


Run the line:
python cleaned_main_with_pseudo.py -drive G -H 1 -sta_num 1 -sta_col R -start 0 -end test

'''
