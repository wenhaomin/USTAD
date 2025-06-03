# -*- coding: utf-8 -*-


#-------------------------------------------------------------------------------------------------------------------------#
#for linux import package
import sys
import os
import platform

file = 'your_project_directory'
sys.path.append(file)
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk(file) for name in dirs])
#-------------------------------------------------------------------------------------------------------------------------#

import numpy as np
import pandas as pd
import pickle as pk
from utils.util import ws
import datetime


def pre_process(fin, fout, mode, first_day): 
    print('Raw input file:', fin)

    # read parquet files
    if mode == 'test':
        datapath = fin + 'stay_points_test_anomalous.parquet'
    elif mode == 'train':
        datapath = fin + 'stay_points_train.parquet'

    stay = pd.read_parquet(datapath, engine='pyarrow')


    if 'anomaly' and 'anomaly_type' in stay.columns:
        stay['label'] = stay['anomaly'].astype(int)
    else:
        stay['label'] = 0
        stay['anomaly_type'] = 0
    stay['start_datetime'] = pd.to_datetime(stay['start_datetime']).dt.tz_localize(None)
    stay['end_datetime'] = pd.to_datetime(stay['end_datetime']).dt.tz_localize(None)
    print('Number of Records:', len(stay))

    # merge lat, lng, act_types (from poi file) to stay
    poi_df = pd.read_parquet(fin + 'poi.parquet')
    stay = stay.merge(poi_df[['poi_id', 'latitude', 'longitude', 'act_types']], on='poi_id', how='left')
    nan_rows = stay[stay[['latitude', 'longitude', 'act_types']].isna().any(axis=1)] # check if there are any NaN values (unmatched poi_id)
    print(f"Number of rows with NaN values after merging: {nan_rows.shape[0]}")
    if nan_rows.shape[0] > 0:
        stay['latitude'] = stay['latitude'].fillna(0)
        stay['longitude'] = stay['longitude'].fillna(0)
        # For 'act_types', fill NaN values with a list. Use .apply() method for lists.
        stay['act_types'] = stay['act_types'].apply(lambda x: x if pd.notna(x) else [-1])

    print('Number of Records after merging:', len(stay))
    print('Number of unique agents:', len(stay['agent_id'].unique()))

    stay['stay_duration'] = (stay['end_datetime'] - stay['start_datetime']).dt.total_seconds() / 60
    
    print('Expand basic information...')

    first_day_date = datetime.datetime.strptime(first_day, "%Y-%m-%d")
    first_day_date = first_day_date.replace(tzinfo=None)
    stay['day'] = (stay['start_datetime'] - first_day_date).dt.days
    date, start_hour, start_time_minute = [], [], []
    for x in stay['start_datetime'].tolist():
        x = pd.to_datetime(x)
        date.append(x.date().__str__())
        start_hour.append(x.hour)
        start_time_minute.append(x.hour * 60 + x.minute)

    stay['date'] = date
    stay['start_hour'] = start_hour
    stay['start_time_minute'] = start_time_minute
    stay = stay.rename(columns={'act_types': 'poi', 'latitude': 'lat', 
                                'longitude': 'lng', 'agent_id': 'uid', 
                                'end_datetime': 'end_time', 'start_datetime': 'start_time'})
    # sort data by user, date, user, and time
    stay = stay[['uid', 'lat', 'lng', 'poi_id', 'start_time', 'stay_duration','poi', 'day', 'date', 'start_hour', 'start_time_minute','label','anomaly_type']] # removing end_time column
    stay = stay.sort_values(by=['uid', 'date', 'start_time'])

    save_path = fout + '/stay_{}.csv'.format(mode)
    stay.to_csv(save_path, index = False)
    # save the statistics information
    stay_stat_path = fin + '/stay_describle_{}.csv'.format(mode)
    stay.describe().to_csv(stay_stat_path)

    return stay



if __name__ == "__main__":
    if 1:
        fin = ws + '/data/raw/NUMOSIM/'
        fout = ws + '/data/raw/NUMOSIM/'
        for mode in ['train', 'test']:
            print('Processing {} data...'.format(mode))
            df = pre_process(fin=fin, fout=fout, mode = mode, first_day="2024-01-01")




