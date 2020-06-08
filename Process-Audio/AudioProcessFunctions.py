"""
AudioProcessFunctions.py
Functions for facilitiating audio processing files written by Sin Yong Tan, Ali Saffari, Maggie Jacoby
Maggie Jacoby, June 2020
"""

import os
import numpy as np
import scipy.io.wavfile
from scipy.signal import butter, lfilter, freqz

# ==================================================================

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
# Prepare the processing parameters

def create_filter_banks():

    filter_banks = []

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

    return filter_banks

# --------------------------------------------------------------------
# ==== Downsampling params ====
'''
NOTE: 
These params changes if 
- initial fs is not 8000 
- desired fh is not 1600 
- num. filters is not 16
- wav file length is not 10 seconds

These params were hard-coded to save computation time.
'''

def create_filter_index(bank_len):
    # Sample rate and desired cutoff frequencies (in Hz).
    fs = 8000   # defined in process_wav function inputs
    # fh = 1600 # not used, but keepping as a record

    number_of_filters = bank_len+1 # +1 for loww-pass (0~100hz that was not in filterbanks)
    filter_start_index = [i for i in range(0,number_of_filters*5,5)]
    increment = 80
    num_final_datapoint = 1000
    temp = np.asarray([i for i in range(num_final_datapoint)])*increment # [0, 80, 160,...]
    filter_i_sampling_index = np.zeros((number_of_filters,num_final_datapoint))

    for j in range(number_of_filters):
        filter_i_sampling_index[j] = temp + filter_start_index[j]

    filter_i_sampling_index = np.transpose(filter_i_sampling_index)
    filter_i_sampling_index = filter_i_sampling_index.astype(int)

    return filter_i_sampling_index

# ==================================================================

# Process the wav files and return downsampled files and DCT files

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