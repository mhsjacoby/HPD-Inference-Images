"""
AudioFunctions.py
Functions for facilitiating audio processing files written by Sin Yong Tan, Ali Saffari, Maggie Jacoby
Maggie Jacoby, June 2020
"""


# ==================================================================
# Maggie's file handling Functions
import os
import numpy as np

def mylistdir(directory, bit='', end=True):
    filelist = os.listdir(directory)
    if end:
        return [x for x in filelist if x.endswith(f'{bit}') and not x.endswith('.DS_Store')]
    else:
         return [x for x in filelist if x.startswith(f'{bit}') and not x.endswith('.DS_Store')]


def make_storage_directory(target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    return target_dir


def make_empty_dict(hour, mins=60, secs=60, freq=10):
    d = {f'{str(hour[0:2]).zfill(2)}:{str(M).zfill(2)}:{str(S).zfill(2)}':
        np.zeros((0,0)) for M in range(0,mins) for S in range(0,secs,freq)}
    return d

def make_all_seconds(hour, mins=60, secs=60, freq=10):
    L = [f'{str(hour[0:2]).zfill(2)}:{str(M).zfill(2)}:{str(S).zfill(2)}' for M in range(0,mins) for S in range(0,secs,freq)]
    return L

# --------------------------------------------------------------------

def make_fill_full(content, hour):
    full_content = make_empty_dict(hour)
    for x in content.keys():
        full_content[x] = content[x]
    return full_content


def make_read_write_paths(root_dir, hub, pi_audio=False):
    read_root_path = os.path.join(root_dir, hub, 'audio')
    # save_dir = '/Users/maggie/Desktop/Audio_test_save'
    save_root_path = os.path.join(root_dir, hub, 'processed_audio')
    paths = {'read': read_root_path, 'write': save_root_path}

    if pi_audio:
        pi_path = os.path.join(root_dir, hub, 'audio_from_pi')
        print(f'Pi audio path: {pi_path}')
        paths['pi'] = pi_path
    else:
        print('No audio from pi')
    
    return paths

# --------------------------------------------------------------------

def write_summary(home, hub, days, write_dir):
    total_per_day = 8640
    H = home.split('-')[0]
    # store_dir = make_storage_directory(f'/Users/maggie/Desktop/summary_test/HPD_mobile-{H}/{home}/Summaries/')
    store_dir = make_storage_directory(os.path.join(write_dir,'Summaries'))
    fname = os.path.join(store_dir, f'{H}-{hub}-audio-summary.txt')

    with open(fname, 'w+') as writer:
        for day in days:
            try:
                total_audio = days[day]['total']/total_per_day
                perc = f'{total_audio:.2}'
            except Exception as e:
                print(f'except: {e}')
                perc = 0.00

            F = days[day]["start_end"]
            details = f'{hub} {day} {F[0], F[1]} {perc}'
            print(details)
            writer.write(details + '\n')
    writer.close()
    print(f'{fname}: Write Successful!')


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

# --------------------------------------------------------------------
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

# ==================================================================
