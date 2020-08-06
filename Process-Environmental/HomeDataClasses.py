"""
HomeDataClasses.py
Maggie Jacoby, June 2020
import into env data cleaning jupyter notebooks
"""


import os
import sys
import csv
import ast
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

class HomeData():
    def __init__(self, path):
        self.root_dir = path
        self.summary_dir = self.make_storage_directory(os.path.join(self.root_dir, 'DataSummaries'))
        self.home = path.split('/')[-1].split('-')[-2]
        self.system = path.split('/')[-1].split('-')[-1]

    
    def mylistdir(self, directory, bit='', end=True):
        filelist = os.listdir(directory)
        if end:
            return [x for x in filelist if x.endswith(f'{bit}') and not x.endswith('.DS_Store') and not x.startswith('Icon')]
        else:
             return [x for x in filelist if x.startswith(f'{bit}') and not x.endswith('.DS_Store') and not x.startswith('Icon')]

    def make_storage_directory(self, target_dir):
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        return target_dir


class ReadEnv(HomeData):
    
    def __init__(self, path, sensor_hubs, from_pi=True):
        HomeData.__init__(self, path)
        self.hubs = sensor_hubs
        self.pi = from_pi
        self.day_length = 8640
        self.unwritten_days = {}

        
    def read_in_data(self, path, measurements):
        with open(path, 'r') as f:
            try:
                data_dicts = json.loads(f.read())
                for time_point in data_dicts:
                    for measure in time_point:
                        measurements[measure].append(time_point[measure])
            except Exception as e:
                pass
            return measurements

    def read_all(self, path, days):
        print(f'> reading in data from: {path}')
        measurements = {
            'time':[], 'tvoc_ppb':[], 'temp_c':[], 'rh_percent':[], 
            'light_lux':[],'co2eq_ppm':[], 'dist_mm':[], 'co2eq_base':[], 'tvoc_base':[]}
        for day in days:
            all_mins = sorted(self.mylistdir(os.path.join(path, day)))
            for m in all_mins:
                files = sorted(self.mylistdir(os.path.join(path, day, m), '.json'))
                for f in files:
                    measurements = self.read_in_data(os.path.join(path, day, m, f), measurements)
        df = pd.DataFrame.from_dict(measurements)
        return df
            
        
    def get_all_data(self, hub):
        env_dir = os.path.join(self.root_dir, hub, 'env_params')
        main_days = sorted(self.mylistdir(env_dir))
        df = self.read_all(env_dir, main_days)
        df = self.clean_dates(df)
        if self.pi:
            print(f'> gathering data from pi...')
            from_pi = os.path.join(self.root_dir, hub, 'env_params_from_pi') 
            pi_days = sorted(self.mylistdir(from_pi))
            pi_df = self.read_all(from_pi, pi_days)
            pi_df = self.clean_dates(pi_df)
            print(f'> merging dfs...')
            df = df.append(pi_df)
        df['timestamp'] = df.index
        df['home'] = self.home
        df['hub'] = hub
        df = df.drop_duplicates(subset=['timestamp']).set_index('timestamp').sort_index()
        print(f'> final df of length: {len(df)}')
        return df
    
    
    def clean_dates(self, df): 
        print(f'> cleaning dates on df of length: {len(df)}')
        df['time'] = df['time'].str.strip('Z').str.replace('T',' ')
        df['timestamp'] = pd.to_datetime(df['time'])
        df = df.drop(columns = ['time'])
        df = df.set_index('timestamp')
        # df.index = df.index.floor('10s')
        df.fillna(np.nan)
        return df
     
        
    def make_day_dfs(self, df, hub):
        print(f'> Making and writing csvs by day.')
        dates = sorted(list(set([d.strftime('%Y-%m-%d') for d in df.index])))
        day_lens = {}
        counts = {}
        unwritten = []
        for day in dates:
            day_df = df.loc[day:day]
            length = len(day_df)
            day_lens[day] = length
            counts[day] = day_df.notnull().sum().to_dict()
            if length > 0.3*self.day_length:   
                self.write_data(hub, df, day)
            else:
                unwritten.append((day, length))
#                 print(f'Not enough data to write. Day {day} only has {length} entries')
        print(f'> Completed writing csvs.')
        self.unwritten_days[hub] = unwritten 
        self.write_summary(hub, day_lens, counts)
        

    
    def write_data(self, hub, df_to_write, day):
        storage_path = self.make_storage_directory(os.path.join(self.root_dir, hub, 'CSV'))
        target_fname = os.path.join(storage_path, f'{self.home}_{hub}_{day}.csv')
        if not os.path.isfile(target_fname):
            df_to_write.to_csv(target_fname, index_label = 'timestamp', index = True)
#         else:
#             print(f'{target_fname} already exists')
                     
                     
    def write_summary(self, hub, dates, counts):
        fname = os.path.join(self.summary_dir, f'{self.home}-{hub}-data-summary.txt')
        with open(fname, 'w+') as writer:
            writer.write('Hx Hub Date       %    [tvoc, temp, rh, lux, co2, dist, co2_b, tvoc_b]' + '\n')
            for day in dates:
                percent = self.get_day_details(dates[day])
                if not percent:
                    c = percent
                else:
                    c = [float(f'{counts[day][x]/8640:.2f}') for x in counts[day]]
                details = f'{self.home} {hub} {day} {percent} {c}'
                writer.write(details + '\n')
        writer.close()
        print(f'{fname} : Write Sucessful!')
  
                     
    def get_day_details(self, len_day):
        try:
            total = len_day/self.day_length
            perc = f'{total:.2f}'
        except:
            perc = 0.00
        return perc
    
    def write_unwritten(self):
        fname = os.path.join(self.summary_dir, f'{self.home}-{self.system}-unwritten-files.txt')
        with open(fname, 'w+') as writer:
            for hub in self.unwritten_days:
                all_missed = self.unwritten_days[hub]
                for pair in all_missed:
                    writer.write(f'{self.home} {hub} {pair[0]} {pair[1]} \n')
                writer.write('\n')
        writer.close()
        print(f'{fname} : Write Sucessful!')
                    

    def main(self):
        self.all_dfs = {}
        for hub in self.hubs:
            print(f'\n> working on: {self.home}-{self.system} {hub}')
            hub_df = self.get_all_data(hub)
            self.make_day_dfs(hub_df, hub)
            self.all_dfs[hub] = hub_df
        self.write_unwritten()
        all_hubs = [self.all_dfs[hub] for hub in self.hubs]
        self.full_df = pd.concat(all_hubs)
            
  