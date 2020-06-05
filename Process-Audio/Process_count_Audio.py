"""
Process_count_Audio.py
Base code written by Sin Yong Tan, based on processing technique developed by Ali Saffair, 
Helper functions stored in AudioFunctions.py
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

from AudioFunctions import *




# ==================================================================

def check_pi(home, hub, day, hour, pi_path):
    found_on_pi = []
    pi_dir = os.path.join(pi_path, day)
    all_mins = sorted(mylistdir(pi_dir))
        all_seconds = []

        if len(all_mins) == 0:
            print("Date folder "+ date + " is empty")

    this_hour = [x for x in all_mins if x[0:2]==hour[0:2]]

                for minute in sorted(this_hour):
                    minute_path = os.path.join(date_folder_path, minute)
                    # print(f'Checking time folder: {minute_path} ...')
                    wavs = sorted(mylistdir(minute_path, bit='.wav'))



# ==================================================================
# Filter parameters

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
# ==================================================================

# Parse user input

parser = argparse.ArgumentParser(description='Convert Wav to 12 DCTs')

parser.add_argument('-root','--root_dir', default='/Users/maggie/Desktop/HPD_mobile_data/HPD-env/', type=str,
                    help='Root Directory') 
parser.add_argument('-hub','--hub', default='RS1', type=str,
                    help='Hub: RS1, BS2, etc')
parser.add_argument('-pi_audio', '--pi_audio', default=False, type=bool,
                    help='does pi audio exist? True if so, else leave blank')

# If adding start/end
parser.add_argument('-start','--start_date_index', default=0, type=int,
                    help='Processing START Date index')
parser.add_argument('-end','--end_date', default=None, type=str,
                    help='Processing END Date')

args = parser.parse_args()


if __name__ == '__main__':

    root_dir = args.root_dir
    hub = args.hub
    pi_audio = args.pi_audio
    start_date_index = args.start_date_index
    end_date = args.end_date

    home_name = root_dir.split('/')[-1]
    print(f'Home: {home_name}, hub: {hub}, pi_audio: {pi_audio}')


    read_root_path = os.path.join(root_dir, hub, 'audio')

    if pi_audio:
        pi_path = os.path.join(root_dir, hub, 'audio_from_pi')
        print(f'Pi audio path: {pi_path}')
    else:
        print('No audio from pi')
    print("read_root_path: ", read_root_path)

    save_dir = '/Users/maggie/Desktop/Audio_test_save'
    save_root = os.path.join(save_dir, hub)
    print('save root: ', save_root)

    dates = sorted(mylistdir(read_root_path, bit='2019', end=False))
    print(dates)
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

    all_days_data = {}

    # ==== Start Looping Folders ====
    for date in dates:
        date_folder_path = os.path.join(read_root_path, date)
        print("Loading date folder: " + date + "...")
        all_mins = sorted(mylistdir(date_folder_path))
        all_seconds = []

        if len(all_mins) == 0:
            print("Date folder "+ date + " is empty")

        else:
            hours = [str(x).zfill(2) + '00' for x in range(0,24)]
            hourly_content = {hour:np.nan for hour in hours}
            print(f'full path: {date_folder_path}')

            # Make storage directories
            downsampled_folder = make_storage_directory(os.path.join(save_root, 'audio_downsampled', date))
            processed_folder = make_storage_directory(os.path.join(save_root, 'audio_processed', date))
            
            for hour in hours:
                print('hour', hour)
                this_hr_count = 0
                content_ds = {f'{str(hour[0:2]).zfill(2)}:{str(M).zfill(2)}:{str(S).zfill(2)}' : np.zeros((0,0)) for M in range(0,60) for S in range(0,60,10)}
                content_ps = {f'{str(hour[0:2]).zfill(2)}:{str(M).zfill(2)}:{str(S).zfill(2)}' : np.zeros((0,0)) for M in range(0,60) for S in range(0,60,10)}

                this_hour = [x for x in all_mins if x[0:2]==hour[0:2]]
                for minute in sorted(this_hour):
                    minute_path = os.path.join(date_folder_path, minute)
                    wavs = sorted(mylistdir(minute_path, bit='.wav'))

                    if len(wavs) == 0:
                        print("Time folder "+ os.path.basename(minute_path) + " is empty")

                    else:
                        for wav_name in wavs:
                            wav_path = os.path.join(date_folder_path, minute, wav_name)
                            t = wav_name.split(' ')[-1].strip('._audio.wav')
                            time_file = f'{t[0:2]}:{t[2:4]}:{t[4:6]}'

                            try:  
                                _, wav = scipy.io.wavfile.read(wav_path)
                                audio_len_seconds = len(wav)/fs # length of audio clip in seconds
                                all_seconds.append(time_file)
                                assert (audio_len_seconds == 10.0)
                                
                                ## Process Audio
                                processed_audio = np.zeros((int(len(wav)),number_of_filters)) # Placeholder
                                
                                temp = butter_lowpass_filter(wav, 100, fs, order=6) # low pass filter (first filter)
                                temp -= np.mean(temp) # Mean Shift
                                processed_audio[:,0] = abs(temp) # Full wave rectify

                                for idx, Filter in enumerate(filter_banks):
                                    temp = butter_bandpass_filter(wav, Filter[0], Filter[1], fs, order=6) # Band pass filter
                                    processed_audio[:, idx+1] = abs(temp) # Full wave rectify

                                ## Downsample:
                                downsampled = np.zeros((num_final_datapoint,number_of_filters))

                                for i in range(number_of_filters):
                                    downsampled[:,i] = processed_audio[filter_i_sampling_index[:,i],i]

                                ################################################################
                                content_ds[time_file] = downsampled # store downsampled (not dct)

                                ################ Comment the following lines if don't want to perform dct ################
                                processed_audio = dct(downsampled) # Perform DCT across different filter on each timepoint
                                processed_audio = processed_audio[:,:12] # Keep only first 12 coefficients
                                processed_audio = scale(processed_audio,axis=1) # Normalizing/Scaling to zero mean & unit std                   
                                ################################################################

                                ################ collect desired data(before and/or after dct) into dictionary(placeholder) for npz_zompressed saving ################
                                content_ps[time_file] = processed_audio # store content, similar timestamp format as env. csv(s) timestamp
                                ################################################################

                            except Exception as e:
                                print(f'Audio error: {e}')

                # ################ npz_compressed saving at the end of each hour (or desired saving interval) ################
                fname_ds = f'{hour}_ds.npz' # ==> intended to produce "0000", "0100", .....
                fname_ps = f'{hour}_ps.npz' # ==> intended to produce "0000", "0100", .....

                downsampled_save_path = os.path.join(downsampled_folder, fname_ds)
                processed_save_path = os.path.join(processed_folder, fname_ps)

                np.savez_compressed(downsampled_save_path, **content_ds)
                np.savez_compressed(processed_save_path, **content_ps)
                # ################################################################

        all_seconds_set = sorted(list(set(all_seconds)))
        total = len(all_seconds_set)
        summary = {'total': len(all_seconds_set), 'start_end': (all_seconds_set[0], all_seconds_set[-1])}
        print(f'start: {all_seconds_set[0]}, end: {all_seconds_set[-1]}, len: {total}' )
        all_days_data[date] = summary

        print("======================================")

    write_summary(home_name, hub, all_days_data)

    print("Done")


'''
Example Expected Input Audio Folder structure:
F:/H1-red/RS1/audio/2019-02-10/2229/2019-02-10 222900_audio.wav

Example Output Audio Folder structure:
F:/H1-red/RS1/audio_processed/2019-02-10/2229/2019-02-10 222900_audio.npy


Run the line:
python cleaned_main_with_pseudo.py -drive G -H 1 -sta_num 1 -sta_col R -start 0 -end test

'''
