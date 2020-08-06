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

import time

# ==================================================================

# Filter parameters

# Sample rate and desired cutoff frequencies (in Hz).
fs = 8000   # defined in process_wav function inputs
# fh = 1600 # not used, but keepping as a record

filter_banks = create_filter_banks()

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

# ==================================================================

# Parse user input

parser = argparse.ArgumentParser(description='Convert Wav to 12 DCTs')

parser.add_argument('-root','--root_dir', default='/Users/maggie/Desktop/HPD_mobile_data/HPD-env/', type=str,
                    help='Root Directory') 
# parser.add_argument('-hub','--hub', default='RS1', type=str,
#                     help='Hub: RS1, BS2, etc')
parser.add_argument('-pi_audio', '--pi_audio', default=False, type=bool,
                    help='does pi audio exist? True if so, else leave blank')

# If adding start/end
parser.add_argument('-start','--start_date_index', default=0, type=int,
                    help='Processing START Date index')
parser.add_argument('-end','--end_date', default=None, type=str,
                    help='Processing END Date')

args = parser.parse_args()




def process_wav(wav_name, date_folder_path, minute, fs=8000):
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

        ################ Comment the following lines if don't want to perform dct ################
        processed_audio = dct(downsampled) # Perform DCT across different filter on each timepoint
        processed_audio = processed_audio[:,:12] # Keep only first 12 coefficients
        processed_audio = scale(processed_audio,axis=1) # Normalizing/Scaling to zero mean & unit std                   
        ################################################################
  
        return processed_audio, downsampled, time_file
    
    except Exception as e:
        print(f'Error processing file {wav_path}: {e}')
        return [], [], time_file

# ==================================================================
## Check pi for all audio files

def check_pi(pi_path):
    found_on_pi = {}
    for day in mylistdir(pi_path, '2019-', end=False):
        pi_dir = os.path.join(pi_path, day)
        all_mins = sorted(mylistdir(pi_dir))
        if len(all_mins) == 0:
            print(f'No pi files for {day}')
        else:
            for minute in sorted(all_mins):
                # found_on_pi = found_on_pi + mylistdir(os.path.join(pi_dir, minute), bit='.wav')
                this_minute = mylistdir(os.path.join(pi_dir, minute), bit='.wav')
                # print(f'found: {this_minute}')
                for audio_f in this_minute:
                    name = audio_f.split('_')[0]
                    found_on_pi[name] = (day, minute)
    return found_on_pi
# ==================================================================






if __name__ == '__main__':

    # hubs = ['RS1', 'RS2', 'RS3', 'RS4', 'RS5']
    # hubs = ['BS2', 'BS3', 'BS4', 'RS5']
    hubs = ['BS4', 'BS5']


    for hub in hubs:
        root_dir = args.root_dir
        # hub = args.hub
        pi_audio = args.pi_audio
        start_date_index = args.start_date_index
        end_date = args.end_date

        home_name = root_dir.strip('/').split('/')[-1]

        h_num = home_name.split('-')[0]
        print(f'Home: {home_name}, num: {h_num}, hub: {hub}, pi_audio: {pi_audio}')

        paths = make_read_write_paths(root_dir, hub, pi_audio)
        read_root_path, save_root = paths['read'], paths['write']
        print(f'reading from: {read_root_path}')
        print(f'saving to: {save_root}')


        dates = sorted(mylistdir(read_root_path, bit='2019', end=False))
        print(dates)
        # dates = dates[start_date_index:]

        all_days_data = {}

        if pi_audio == True:
            print('Checking pi ...')
            found_on_pi = check_pi(paths['pi'])
            print(f'Number of files found on pi: {len(found_on_pi)}')
        else:
            print('No pi audio files received')
            found_on_pi = []

        t_start = time.perf_counter()
        # ==== Start Looping Folders ====
        for date in dates:
            t1 = time.perf_counter()
            date_folder_path = os.path.join(read_root_path, date)
            print("Loading date folder: " + date + "...")
            all_mins = sorted(mylistdir(date_folder_path))

            all_seconds = []

            if len(all_mins) == 0:
                print(f'Date folder {date} is empty')

            else:
                # Make storage directories
                downsampled_folder = make_storage_directory(os.path.join(save_root, 'audio_downsampled', date))
                processed_folder = make_storage_directory(os.path.join(save_root, 'audio_dct', date))
                            
                hours = [str(x).zfill(2) + '00' for x in range(0,24)]

                for hour in hours:
                    print('hour', hour)

                    # create dictionaries to store downsampled (*_ds) and processed (*_ps) audio
                    # content_ds = make_empty_dict(hour)
                    # content_ps = make_empty_dict(hour)
                    content_ds = {}
                    content_ps = {}
                    full_list = make_all_seconds(hour)
                    this_hour = [x for x in all_mins if x[0:2]==hour[0:2]]

                    for minute in sorted(this_hour):
                        minute_path = os.path.join(date_folder_path, minute)
                        wavs = sorted(mylistdir(minute_path, bit='.wav'))

                        if len(wavs) == 0:
                            print("Time folder "+ os.path.basename(minute_path) + " is empty")

                        else:
                            for wav_name in wavs:
                                processed_audio, downsampled_audio, time_file = process_wav(wav_name, date_folder_path, minute)

                                if len(processed_audio) > 0:
                                    all_seconds.append(time_file)
                                    content_ds[time_file] = downsampled_audio # store downsampled (not dct)
                                    content_ps[time_file] = processed_audio # store content, similar timestamp format as env. csv(s) timestamp                      
                                else:
                                    print(f'no audio for {time_file}')
                                    ################################################################     


                    ### Check for missing files on pi and read in
                    list_hour_actual = [x for x in content_ps.keys()]
                    print(f'length of processed this hour: {len(content_ps)}')

                    missing = list(set(full_list)-set(list_hour_actual))
                    missing = [f'{date} {x.replace(":", "")}' for x in missing]
                    print(f'len of missing: {len(missing)}')

                    
                    
                    if len(missing) > 0:
                        if len(found_on_pi) > 0:
                            this_hour_on_pi = [x for x in found_on_pi if (x.split(' ')[0] == date and x.split(' ')[1][0:2] == hour[0:2])]
                            for m in missing:
                                if m in this_hour_on_pi:
                                    print(f'found! {m}, {found_on_pi[m]}') 
                                    day, minute = found_on_pi[m]

                                    # print(day, minute, f'{m}_audio.wav', os.path.join(paths['pi'], day))
                                    processed_pi, downsampled_pi, time_file = process_wav(f'{m}_audio.wav', os.path.join(paths['pi'], day), minute)

                                    if len(processed_pi) > 0:
                                        content_ds[time_file] = downsampled_pi # store downsampled (not dct)
                                        content_ps[time_file] = processed_pi # store content, similar timestamp format as env. csv(s) timestamp  

                    list_hour_after = [x for x in content_ps.keys()]
                    missing_after = list(set(full_list)-set(list_hour_after))
                    missing_after = [f'{date} {x.replace(":", "")}' for x in missing_after]
                    print(f'found on pi: {len(missing)-len(missing_after)}')

                    full_content_ds = make_fill_full(content_ds, hour)
                    # print(f'len of full ds: {len(full_content_ds)}, full with value: {sum(1 for i in full_content_ds.values() if len(i) > 0)}, len of content_ds: {len(content_ds)}')
                    full_content_ps = make_fill_full(content_ps, hour)
                    # print(f'len of full ps: {len(full_content_ps)}, full with value: {sum(1 for i in full_content_ps.values() if len(i) > 0)}, len of content_ps: {len(content_ps)}')


                    # ################ npz_compressed saving at the end of each hour (or desired saving interval) ################
                    fname_ds = f'{date}_{hour}_{hub}_{home_name.split("-")[0]}_ds.npz' # ==> intended to produce "0000", "0100", .....
                    fname_ps = f'{date}_{hour}_{hub}_{home_name.split("-")[0]}_ps.npz' # ==> intended to produce "0000", "0100", .....
                    print(fname_ds)
                    downsampled_save_path = os.path.join(downsampled_folder, fname_ds)
                    processed_save_path = os.path.join(processed_folder, fname_ps)

                    np.savez_compressed(downsampled_save_path, **full_content_ds)
                    np.savez_compressed(processed_save_path, **full_content_ps)
                    # np.savez_compressed(downsampled_save_path, **content_ds)
                    # np.savez_compressed(processed_save_path, **content_ps)
                    # ################################################################
            
            all_seconds_set = sorted(list(set(all_seconds)))
            total = len(all_seconds_set)
            if total == 0:
                summary = {'total': 0, 'start_end': (0,0)}
            else:
                summary = {'total': len(all_seconds_set), 'start_end': (all_seconds_set[0], all_seconds_set[-1])}
            print(date, summary)
            all_days_data[date] = summary

            t_now = time.perf_counter()

            print(f'======= end of day {date} --- time to processing one day: {t_now-t1} total so far: {t_now-t_start} =======')

        write_summary(home_name, hub, all_days_data, root_dir)

        print("Done")


'''
Example Expected Input Audio Folder structure:
F:/H1-red/RS1/audio/2019-02-10/2229/2019-02-10 222900_audio.wav

Example Output Audio Folder structure:
F:/H1-red/RS1/audio_processed/2019-02-10/2229/2019-02-10 222900_audio.npy


Run the line:
python cleaned_main_with_pseudo.py -drive G -H 1 -sta_num 1 -sta_col R -start 0 -end test

'''
