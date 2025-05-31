# -*- coding: utf-8 -*-

#-------------------------------------------------------------------------------------------------------------------------#
#for linux import package
import sys
import os
import platform
file = 'D:\Study\Lab\project\CMU project\haystac\temp_sync\D-Transformer' if platform.system() == 'Windows'  else '/data/shurui/D-Transformer/'
sys.path.append(file)
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk(file) for name in dirs])
#-------------------------------------------------------------------------------------------------------------------------#

import os.path

import json
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from collections import defaultdict

import datetime
from datetime import timedelta



import torch
from torch import nn
from torch.utils.data import Dataset

from utils.util import ws, dir_check, gps_to_normed_xy, min_max_norm


# Overall information about the features
FEATURE_LIST = [
    # continous
    {'idx': 0, 'name': 'x', 'type': 'continuous', },
    {'idx': 1, 'name': 'y', 'type': 'continuous',},
    {'idx': 2, 'name': 'start_time_minute', 'type': 'continuous',},
    {'idx': 3, 'name': 'stay_duration', 'type': 'continuous',},
    # discrete
    {'idx': 4, 'name': 'poi', 'type': 'discrete',},
    {'idx': 5, 'name': 'dow', 'type': 'discrete',},
]

FEATURE_COLS = [f['name'] for f in FEATURE_LIST]          # feature name of all features

POSITION_LIST = [
    {'idx': 0, 'name': 'seq_pos'},
    {'idx': 1, 'name': 'within_day_pos'},
    {'idx': 2, 'name': 'day_pos'},
]
position2idx = {f['name']: f['idx'] for f in POSITION_LIST}


# Feature2Idx = {f['name']:f['idx']  for f in FEATURE_LIST} # map the feature name to its index in the numpy array

# CONTINUOUS_FEATURE_NAME = [f['name'] for f in FEATURE_LIST if f['type'] == 'continuous'] # feature name of all continuous features
# CONTINUOUS_FEATURE_IDX = [f['idx'] for f in FEATURE_LIST if f['type'] == 'continuous']

# DISCRETE_FEATURE_NAME = [f['name'] for f in FEATURE_LIST if f['type'] == 'discrete']
# DISCRETE_FEATURE_IDX = [f['idx'] for f in FEATURE_LIST if f['type'] == 'discrete']

# only need to normalize continous features that need to be predicted
# NeedNormFeature = CONTINUOUS_FEATURE_NAME[:]# only continuous features need to be normalized


def get_day_of_week(date_str):
    date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    return date.strftime("%A")

def get_after_dates(start_date, after_days):
    """
    Generates a list of dates starting from a given date and adding a specified number of days.

    Parameters:
    start_date_str (str): The start date in the format "YYYY-MM-DD".
    after_days (list): A list of integers representing the number of days after the start date.

    Returns:
    list: A list of dates in the format "YYYY-MM-DD".
    """
    start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    date_list = [(start_date + datetime.timedelta(days=day)).strftime("%Y-%m-%d") for day in after_days]
    return date_list

class Normalizer(nn.Module):
    def __init__(self, stat, feat_names, feat_cols, norm_type='zscore'):
        """
        Args:
            stat: pd.Dataframe, that records the statistics of all features
            feat_names: list, feature that need to be normalized, e.g.,  ['x', 'y']
            feat_cols: list,  feature's index in the input arr, e.g., [0, 1]
            norm_type: str, 'minmax' or 'zscore'
        """
        super().__init__()

        self.stat = stat
        self.feat_names = feat_names
        self.feat_cols = feat_cols
        self.norm_type = norm_type

    def _norm_col(self, x_col, col_name):
        if self.norm_type == 'zscore':
            x_col = (x_col - self.stat.loc['mean', col_name]) / self.stat.loc['std', col_name]
        elif self.norm_type == 'minmax':
            x_col = (x_col - self.stat.loc['min', col_name]) / \
                    (self.stat.loc['max', col_name] - self.stat.loc['min', col_name])
            # x_col = x_col * 2 - 1
        else:
            raise NotImplementedError(self.norm_type)
        return x_col

    def forward(self, arr):
        """ Normalize the input array. """
        if isinstance(arr, torch.Tensor):
            x = torch.clone(arr)
        else:
            x = np.copy(arr)
        for col, name in zip(self.feat_cols, self.feat_names):
            x[..., col] = self._norm_col(x[..., col], name)
        return x


class Denormalizer(nn.Module):
    def __init__(self, stat, feat_names, feat_cols, norm_type='zscore'):
        """
        Args:
           stat: pd.Dataframe, that records the statistics of all features
           feat_names: list, feature that need to be denormalized, e.g.,  ['x', 'y']
           feat_cols: list,  feature's index in the input arr, e.g., [0, 1]
           norm_type: str, 'minmax' or 'zscore'
        """
        super().__init__()

        self.stat = stat
        self.feat_names = feat_names
        self.feat_cols =  feat_cols
        self.norm_type = norm_type

    def _denorm_col(self, x_col, col_name):
        if self.norm_type == 'zscore':
            x_col = x_col * self.stat.loc['std', col_name] + self.stat.loc['mean', col_name]
        elif self.norm_type == 'minmax':
            # x_col = (x_col + 1) / 2
            x_col = x_col * (self.stat.loc['max', col_name] - self.stat.loc['min', col_name]) + self.stat.loc['min', col_name]
        else:
            raise NotImplementedError(self.norm_type)
        return x_col

    # def forward(self, select_cols, arr):
    #     """ Denormalize the input batch. """
    #     if isinstance(arr, torch.Tensor):
    #         x = torch.clone(arr)
    #     else:
    #         x = np.copy(arr)
    #     for col, name in zip(self.feat_cols, self.feat_names):
    #         if col in select_cols:
    #             x[..., col] = self._denorm_col(x[..., col], name)
    #     return x

    def forward(self, arr):
        """ Denormalize the input batch. """
        if isinstance(arr, torch.Tensor):
            x = torch.clone(arr)
        else:
            x = np.copy(arr)
        for col, name in zip(self.feat_cols, self.feat_names):
            x[..., col] = self._denorm_col(x[..., col], name)
        return x



def get_dateset_stat(data, mode, fout):
    """
       get the statistics of the dataset
       data is the return of the EventDataset.get_dataset, e.g., a dict: {'X': array, 'X_len': array}
       mode: train, val, or test
    """
    X = data['X']

    B = len(X) # B: number of samples
    d_x = len(X[0][0]) # # d_x: number of features

    X_len = data['X_len']
    L_avg = round(sum(X_len)/len(X_len),2) # average number of sequence length
    L_max = max(X_len) # max number of sequence length
    L_min = min(X_len) # min number of sequence length

    results = [mode, B, L_max, d_x, L_avg, L_min]
    df = pd.DataFrame(data = [results], columns=['dataset','#sample', 'L_max', '#feature', 'L_avg', 'L_min'])
    if not os.path.exists(fout):
        df.to_csv(fout, index=False)
    else:
        df.to_csv(fout, mode='a', index=False, header=False)


def get_discrete_feature_cat_num(data_name, feature):
    """return the category number of a discrete feature"""
    path = ws + f'/data/dataset/{data_name}/discrete_feature.json'
    with open(path, 'r') as f:
        data_dict = json.load(f)
    return len(data_dict[feature])

class EventDataset(object):
    """
    Read data from the preprocessed file and then generate three .npy files that store dataset for train, val, and test
    """

    def __init__(self, params):
        self.params = params

        self.fin = self.params['fin'] # the input dir of the raw data

        self.fout = self.params.get('fout', '') # the path to save the dataset

        self.mode = params['mode'] # train, val, test

        self.is_test = params.get('is_test', False) # is test or not

        # the statistics information of the whole raw data
        self.stay_stat_path =  self.fin + "/stay_describle.csv"

        # # the statistics information of the train dataset
        self.stay_train_stat_path = os.path.dirname(self.fout) + '/stay_train_describle.csv'
        self.stay_val_stat_path = os.path.dirname(self.fout) + '/stay_val_describle.csv'
        self.stay_test_stat_path = os.path.dirname(self.fout) + '/stay_test_describle.csv'

        # the statistics information of the dataset
        self.dataset_stat_path = ws + f'/data/dataset/{params["data_name"]}/dataset_describle.csv'

        # the category to idx mapping dict of the discrete features
        self.discrete_map_path =  ws + f'/data/dataset/{params["data_name"]}/discrete_feature.json'

        self.data = defaultdict(list)


    def get_previous_dates(self, date: str, previous_days: list):
        """
        get the previous dates of given date
        Args:
            date (string): 2023-03-05
            previous_days (int list): [1, 2], means previous 1 and 2 days
        Returns:
            [2023-03-04, 2023-03-03]
        """
        return [(pd.to_datetime(date) - timedelta(days=day)).strftime("%Y-%m-%d") for day in previous_days]


    def get_features(self, df: pd.DataFrame):
        """
        The most important function in this class,
        which construct all information for one input sequence, i.e., the features of each event, input length, user id, position encoding, etc.
        Args:
            df: the selected dataframe of one courier's event.
        Returns:
            the information dict of a sample
        """
        X_len = len(df) # number of events

        uid = df['uid'].tolist()[0] # user id

        features = FEATURE_COLS # event features
        X = df[features].values


        seq_pos = np.array([i for i in range(X_len)])
        day_pos = np.array((df['day'] - df['day'].min()).tolist())

        day_lst = df['day'].tolist()
        day_positions = {}
        within_day_pos = []
        for day in day_lst:
            if day not in day_positions:
                day_positions[day] = 0
            else:
                day_positions[day] += 1
            within_day_pos.append(day_positions[day])
        within_day_pos = np.array(within_day_pos)

        position = np.stack((seq_pos, within_day_pos, day_pos), axis=0)

        # position = np.zeros(X_len) # position encoding
        # for i in range(X_len):
        #     position[i] = i+1

        return {'X': X, 'X_len': X_len, 'uid': uid, 'position': position}
        # note: if new information is added, remember to add the key into feature_lst in results_merge function below

    def results_merge(self, results):
        """
        Designed along with the multi-thread mechanism, used to aggregate results from different threads
        """
        all_result = []
        for r in results:
            all_result += r

        feature_lst = ['X',  'X_len', 'uid', 'position']  # label information,

        for r in all_result:
            for f in feature_lst:
                self.data[f].append(r[f])

        # record the dataset statistics
        get_dateset_stat(self.data, self.mode, self.dataset_stat_path)

        return self.data

    def multi_thread_work(self, parameter_queue, function_name, thread_number=5):
        from multiprocessing import Pool
        """
        For parallelization
        """
        pool = Pool(thread_number)
        result = pool.map(function_name, parameter_queue)
        pool.close()
        pool.join()
        return result


    def get_feature_kernel(self, args={}):

        u_lst = args['u_lst']
        df = args['df']
        previous_days = args['previous_days'] #e.g., [0, 1, 2]

        pbar = tqdm(total=len(u_lst))
        result_lst = []

        ## For single thread, used to test the code, or locate bug
        # for u in u_lst:
        #     pbar.update(1)
        #     result = self.get_features()

        for u in u_lst:
            pbar.update(1)
            df_u = df[df['uid'] == u]  # all data of a user

            dates_of_u = df_u['date'].unique()
            dates_of_u.sort()

            for ds in dates_of_u:
                # get the previous dates of a given date
                input_dates = self.get_previous_dates(ds, previous_days)
                df_temp = df_u[df_u['date'].isin(input_dates)]
                result = self.get_features(df_temp)
                result_lst.append(result)

        return result_lst


    def get_dataset(self):
        '''
        the main function for data construction.
        construct train / val / test dataset based on stay.csv (the output of preprocess.py)
        '''

        if self.mode in ['train', 'val']: # our train and val comes from train dataset
            df = pd.read_csv(self.params['fin'] + "/stay_train.csv", sep=',', encoding='utf-8')
        elif self.mode == 'test': 
            df = pd.read_csv(self.params['fin'] + "/stay_test.csv", sep=',', encoding='utf-8')

        if self.is_test:
            df = df[:10000]
        else:
            df = df

        df = df.rename(columns={'stay_time': 'stay_duration'})
        print('Finished loading data from:', self.params['fin'])

        # filter users to be our selected 20k agents
        u_lst = np.load(self.params['fin'] + 'selected_agents_20k.npy', allow_pickle=True)
        df = df[df['uid'].isin(u_lst)]

        # convert the gps to x,y point (unit: km) centerd at the middle point
        if 'x' not in df.columns.tolist():
            df['x'], df['y'] = gps_to_normed_xy(df['lng'].tolist(), df['lat'].tolist(), region='LA', scale=1000) 

        # calcuate stay_duration hour and start time hour
        if 'stay_duration_hour' not in df.columns.tolist():
            df['stay_duration_hour'] = df['stay_duration'].apply(lambda x: round(x/60, 2))
        if 'start_time_hour' not in df.columns.tolist():
            df['start_time_hour'] = df['start_time_minute'].apply(lambda x: round(x/60, 2))
        # calcuate day of week
        if 'dow' not in df.columns.tolist():
            df['dow'] = pd.to_datetime(df['start_time']).dt.dayofweek

        # # save the statistics information
        # if not os.path.exists(self.stay_stat_path):
        #     df.describe().to_csv(self.stay_stat_path)

        # convert the discrete features to id
        discrete_map = {}
        # for f in DISCRETE_FEATURE_NAME:
        for fea in FEATURE_LIST:
            f, f_type = fea['name'], fea['type']
            if f_type!= 'discrete': continue
            df[f], unique_categories = pd.factorize(df[f])
            # Create a mapping dictionary
            mapping = {idx: category for idx, category in enumerate(unique_categories)}
            discrete_map[f] = mapping
        # save the discrete mapping to file
        dir_check(self.discrete_map_path)
        with open(self.discrete_map_path, 'w') as fp:
            json.dump(discrete_map, fp)

        # get the dates for train, val and test, according to the parameter type
        if isinstance(self.params['train_ratio'], float):
            print('Using date ratio to split data')
            all_dates = df['date'].unique()
            all_dates.sort()
            m = len(all_dates)
            idx1, idx2 =  int(m * self.params['train_ratio']),  int(m * (self.params['train_ratio'] + self.params['val_ratio']))
            train_dates = all_dates[:idx1].tolist()
            val_dates = all_dates[idx1 + 1: idx2].tolist()
            test_dates = all_dates[idx2 + 1:].tolist()
        elif isinstance(self.params['train_ratio'], list): # directly give the dates
            print('Use given date list to split data')
            train_dates = self.params['train_ratio']
            val_dates = self.params['val_ratio']
            test_dates = self.params['test_ratio']
        else:
            raise  NotImplementedError


        if self.is_test == False:
            if self.mode == 'train':
                df = df[df['date'].isin(train_dates)]
                df.describe().to_csv(self.stay_train_stat_path) # save the statistics information in train
            elif self.mode == 'val':
                df = df[df['date'].isin(val_dates)]
                df.describe().to_csv(self.stay_val_stat_path) # save the statistics information in train
            elif self.mode == 'test':
                df = df[df['date'].isin(test_dates)]
                df.describe().to_csv(self.stay_test_stat_path) # save the statistics information in train
            else:
                raise RuntimeError('please specify a mode')

        # contruct dataset parallel
        u_lst = df['uid'].unique()
        # u_lst = u_lst[: int(len(u_lst) / 10)]
        num_thread = self.params['num_thread']
        n = len(u_lst)
        task_num = n // num_thread
        previous_days = list(range(self.params['day_window'] + 1))
        print('previous days:', previous_days)
        args_lst = []
        for i in range(0, n, task_num):
            u_lst_a_task = u_lst[i: min(i + task_num, n)]
            df_a_task = df[df['uid'].isin(u_lst_a_task)]
            args_lst.append({'u_lst': u_lst_a_task, 'df': df_a_task, "previous_days": previous_days})

        ## one single thread, used to locate bug
        # for args in args_lst:
        #     self.get_feature_kernel(args)

        result_lst = self.multi_thread_work(args_lst, self.get_feature_kernel, num_thread)

        data = self.results_merge(result_lst)

        # save data into fout
        if self.fout != '':
            dir_check(self.fout)
            np.save(self.fout, data)

        return data


def get_params():
    parser = argparse.ArgumentParser()
    # dataset parameters
    parser.add_argument('--fin',  type=str, default=ws + '/data/raw/HighPrev/')
    parser.add_argument('--day_window', type=int, default=2)
    parser.add_argument('--data_name', type=str, default='9_9_HighPrev')
    is_test = True if platform.system() == 'Windows' else False
    # is_test = True
    parser.add_argument('--is_test', type=bool, default=is_test)
    num_thread = 2 if platform.system() == 'Windows' else 40
    parser.add_argument('--num_thread', type=int, default= num_thread)

    # data split, according to date
    parser.add_argument('--train_ratio', type=float, default=0.4)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.5)
    args, _ = parser.parse_known_args()
    return args

def main():
    params = vars(get_params())
    if params['is_test']: params['data_name'] += '_test'
    data_name = params['data_name']
    first_day = "2024-01-01"
    first_day_date =datetime.datetime.strptime(first_day, "%Y-%m-%d")


    for mode in [ 'train', 'val', 'test']: #'test', 'train', 'val',
        if mode == 'train':
            fout = ws + f'/data/dataset/{data_name}/train.npy'
            params['mode'] = mode
            params['fout'] =  fout
            week = 3
            date_to_use = first_day
            params['train_ratio'] = get_after_dates(date_to_use, list(range(7 * week)))
            EventDataset(params).get_dataset()
            print('train file saved at: ', fout)
        elif mode == 'val':
            fout = ws + f'/data/dataset/{data_name}/val.npy'
            params['mode'] = mode
            params['fout'] = fout
            week = 1
            date_to_use = first_day_date + timedelta(weeks=3)  # week = 3 --> Start from the 4th week
            params['val_ratio'] = get_after_dates(date_to_use.strftime("%Y-%m-%d"), list(range(7 * week)))
            EventDataset(params).get_dataset()
            print('val file saved at: ', fout)
        else:
            fout = ws + f'/data/dataset/{data_name}/test.npy'
            params['mode'] = mode
            params['fout'] = fout
            week = 2
            date_to_use = first_day_date + timedelta(weeks=4)  # week = 4 --> Start from the 5th week from first_day which should be the test period
            params['test_ratio'] = get_after_dates(date_to_use.strftime("%Y-%m-%d"), list(range(7 * week)))
            EventDataset(params).get_dataset()
            print('test file saved at: ', fout)

    # save the feature meta information to file
    fea_path = ws + f'/data/dataset/{data_name}/feature.json'
    dir_check(fea_path)
    with open(fea_path, 'w') as fp:
        json.dump(FEATURE_LIST, fp)

    print('Dataset constructed...')


if __name__ == '__main__':
    # the main function to get the dateset
    if 1: main()
