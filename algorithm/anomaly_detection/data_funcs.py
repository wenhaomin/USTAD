# inputs: test data, trained model, gts list of events
# outputs: anomaly score for each event, evaluation metrics

#-------------------------------------------------------------------------------------------------------------------------#
#for linux import package
import sys
import os
import platform

import numpy as np
import pandas as pd

file = 'D:\Study\Lab\project\CMU project\haystac\temp_sync\D-Transformer' if platform.system() == 'Windows'  else '/data/shurui/D-Transformer/'
sys.path.append(file)
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk(file) for name in dirs])
#-------------------------------------------------------------------------------------------------------------------------#


# laod the pre-trained model, then conduct the prediction or evaluation

import numpy as np
import pandas as pd
import json
import random
from tqdm import tqdm
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
import pickle
import matplotlib.pyplot as plt
import datetime
from collections import defaultdict
import pyperclip
from algorithm.dataset_loader import UNKNOWN_TOKEN,FEATURE_PAD, MASK_TOKEN, PAD_TOKEN,KNOWN_TOKEN,pad_batch_3d,pad_batch_2d,pad_batch_1d
import  torch
from utils.util import ws, get_common_params, dict_merge, get_dataset_path,dir_check,gps_to_normed_xy
from torch.utils.data import DataLoader,Dataset
from joblib import Parallel, delayed
from data.dataset import Normalizer, POSITION_LIST, FEATURE_COLS
from torch.nn import functional as F
from multiprocessing import Pool


# Function to aggregate results for a single key
def aggregate_key(key, results):
    aggregated = []
    for result in results:
        aggregated.extend(result[key])
    return key, aggregated

def get_params():
    parser = get_common_params()
    args, _ = parser.parse_known_args()
    return args
def multi_thread_work(parameter_queue, function_name, thread_number=5):
    from multiprocessing import Pool
    """
    For parallelization
    """
    pool = Pool(thread_number)
    result = pool.map(function_name, parameter_queue)
    pool.close()
    pool.join()
    return result
def process_user_sequences(uid, user_df, gts_df, window_size_before, window_size_after, feature_cols):
    """
    Creates test sequences with the specified window size and masks the target event for each user.
    Also records the label for the masked test event.

    Args:
        uid (int): User ID.
        user_df (pd.DataFrame): The test dataframe of specific user containing events.
        gts_df (pd.DataFrame): The dataframe containing ground truth labels.
        window_size_before (int): The number of days before the test event.
        window_size_after (int): The number of days after the test event.
        feature_cols (list): List of feature columns.

    Returns:
        dict: A dictionary containing arrays for 'X', 'X_len', 'position', 'mask_idx', 'uid', and 'label'.
    """
    sequences = {
        'X': [],
        'X_len': [],
        'position': [],
        'mask_idx': [],
        'uid': [],
        'label': []
    }
    
    # position2idx = {f['name']: f['idx'] for f in POSITION_LIST}

    # Ensure the dates are sorted
    user_df =user_df.sort_values(['date', 'start_time'])

    # Get all unique dates for the user
    unique_dates = user_df['date'].unique()
    for test_date in unique_dates:
        start_date = max(user_df['date'].min(), test_date - datetime.timedelta(days=window_size_before))
        end_date = min(user_df['date'].max(), test_date + datetime.timedelta(days=window_size_after))
        sequence_df = user_df[(user_df['date'] >= start_date) & (user_df['date'] <= end_date)]

        X = sequence_df[feature_cols].values
        seq_pos = np.arange(len(sequence_df))
        day_pos = (sequence_df['day'] - sequence_df['day'].min()).values

        day_lst = sequence_df['day'].tolist()
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

        # Iterate over each event in the test date
        test_event_indices = sequence_df.index[sequence_df['date'] == test_date].tolist()
        for global_mask_idx in test_event_indices:
            # Adjust mask_idx to be relative to the current sequence
            mask_idx = sequence_df.index.get_loc(global_mask_idx)
            # Determine the label for the test event
            mask_start_time = sequence_df.loc[global_mask_idx, 'start_time']
            mask_uid = sequence_df.loc[global_mask_idx, 'uid']
            if "label" not in sequence_df.columns:
                if "start_time" in gts_df.columns:
                    label = int(((gts_df['uid'] == uid) & (gts_df['start_time'] <= mask_start_time)& (gts_df['end_time'] >= mask_start_time)).any())
                else:
                    label = int(mask_uid in gts_df['agent'].unique())
            elif "label" in sequence_df.columns:
                label = int(sequence_df.loc[global_mask_idx, 'label'])

            sequences['X'].append(X)
            sequences['X_len'].append(len(X))
            sequences['position'].append(position)
            sequences['mask_idx'].append(mask_idx)
            sequences['uid'].append(uid)
            sequences['label'].append(label)

    return sequences

def process_user_sequences_parallel(args):
    uids, user_df, gts_df, window_size_before, window_size_after, feature_cols = args['uids_a_task'], args['df_a_task'], args['gts_df'], args['window_size_before'], args['window_size_after'], args['feature_cols']
    sequences = {
        'X': [],
        'X_len': [],
        'position': [],
        'mask_idx': [],
        'uid': [],
        'label': []
    }

    for uid in tqdm(uids):
        user_sequences = process_user_sequences(uid, user_df[user_df['uid'] == uid], gts_df, window_size_before, window_size_after, feature_cols)
        for key in sequences.keys():
            sequences[key].extend(user_sequences[key])
    
    return sequences


def create_test_sequences_parallel(window_size_before, window_size_after, feature_cols, num_threads, fin,fout, test_mode,feature_list,subsetnum, dataset_name, mode):
    if 'trial3' in fin:
        test_data_path = fin + '/stay_test.parquet'
        test_df = pd.read_parquet(test_data_path, engine='pyarrow')
        gts_path = fin + '/inserted_agents.csv'
        gts_df = pd.read_csv(gts_path)
        test_df = test_df.rename(columns={'stay_time': 'stay_duration'})
        test_df['date'] = pd.to_datetime(test_df['date'])
        test_df['start_time'] = pd.to_datetime(test_df['start_time'])
        print('Finished loading data from:', fin)
        regionname = 'Jordan'
        if test_mode == 'subset': # use subset agents (the same as the training data)
            # filter users
            u_lst = test_df['uid'].unique()
            u_lst = set(u_lst[:20000]) # control the user number
            gts_agent = set(gts_df['agent'])
            combined_uid = u_lst.union(gts_agent)
            test_df = test_df[test_df['uid'].isin(combined_uid)]
            print(f'there are {len(combined_uid)} agents in the test data')

    elif 'NUMOSIM_LA' in fin:
        # discrete_map_path = fin + '/discrete_feature.json'
        if mode == 'test':
            test_data_path = fin + '/stay_test.csv'
        elif mode == 'train':
            test_data_path = fin + '/stay_train.csv'
        test_df = pd.read_csv(test_data_path)
        test_df = test_df.rename(columns={'stay_time': 'stay_duration'})
        print('Finished loading data from:', fin)
        if test_mode == 'test': # use all test agents
            test_df = test_df
        elif test_mode == 'subset': # use subset agents (the same as the training data)
            # filter users to be our selected 20k agents
            agentids = np.load(fin + '/selected_agents_20k.npy', allow_pickle=True)
            print(f'there are {len(agentids)} agents in the test data')
            test_df = test_df[test_df['uid'].isin(agentids)] # filter user ids
        if mode == 'test':
            test_df = test_df[test_df['date']>= "2024-01-29"] # slice out test period only
        elif mode == 'train':
            test_df = test_df[test_df['date'] <= "2024-01-21"] # slice out train period only
        test_df['start_time'] = pd.to_datetime(test_df['start_time']).dt.tz_localize(None)
        test_df['date'] = pd.to_datetime(test_df['date'])
        test_df = test_df.rename(columns={'gt': 'label'})
        regionname = 'LA'
        print(f'Length of test data: {len(test_df)}')
    elif 'trial2' in fin:
        test_data_path = fin + '/stay.csv'
        # test_df = pd.read_csv(test_data_path, usecols=['uid', 'lat', 'lng', 'start_time', 'stay_time','day', 'poi','date','start_hour','start_time_minute'])
        test_df = pd.read_csv(test_data_path)
        
        gts_path = fin + '/gts_test_grouped.csv'
        gts_df = pd.read_csv(gts_path)
        gts_df['start_time'] = pd.to_datetime(gts_df['start_time']).dt.tz_localize(None)
        gts_df['end_time'] = pd.to_datetime(gts_df['end_time']).dt.tz_localize(None)
        if test_mode == 'test':
            agentids = test_df['uid'].unique()
            agentids = agentids[:subsetnum]
            print(f'there are {len(agentids)} agents in the test data')
            test_df = test_df[test_df['uid'].isin(agentids)] # filter user ids
        elif test_mode == 'restrictive': # only use agents that have anomaly events during test + some normal agents
            agentids = np.load(fin + '/selected_agents_254.npy', allow_pickle=True)
            agentids = set(agentids)
            print(f'there are {len(agentids)} agents in the test data')
            test_df = test_df[test_df['uid'].isin(agentids)] # filter user ids
            subsetnum = len(agentids)
        elif test_mode == 'subset':
            agentids = pd.read_pickle(fin + '/combined_agentids.pkl')
            agentids = set(agentids)
            print(f'there are {len(agentids)} agents in the test data')
            test_df = test_df[test_df['uid'].isin(agentids)] # filter user ids
            subsetnum = len(agentids)
        elif test_mode == 'inject':
            test_data_path = fin + '/injection/train_stay_injected.csv'
            test_df = pd.read_csv(test_data_path)
            print(f'reading injection data from {test_data_path}')
            print(test_df.columns)
            agentids = test_df['uid'].unique()
            subsetnum = len(agentids)
            print(f'there are {len(agentids)} agents in the test data')
        test_df['date'] = pd.to_datetime(test_df['date'])
        if mode == 'test':
            test_df = test_df[test_df['date']>= "2024-01-29"] # slice out test period only
        elif mode == 'train':
            test_df = test_df[test_df['date'] <= "2024-01-21"] # slice out train period only
        test_df['start_time'] = pd.to_datetime(test_df['start_time']).dt.tz_localize(None)
        test_df = test_df.rename(columns={'stay_time': 'stay_duration'})
        regionname = 'LA'

    elif 'trial4_2' in fin:
        test_data_path = fin + '/test_stay.parquet'
        test_df = pd.read_parquet(test_data_path, engine='pyarrow')
        print('Finished loading data from:', fin)
        regionname = 'Tokyo'
        test_df['date'] = pd.to_datetime(test_df['date'])
        test_df['start_time'] = pd.to_datetime(test_df['start_time'])
        test_df = test_df.rename(columns={'stay_time': 'stay_duration'})
        test_df = test_df.rename(columns={'gt': 'label'})
        print(f'Columns in test_df: {test_df.columns}')

    elif 'trial4' in fin:
        test_data_path = fin + '/stay_test.parquet'
        test_df = pd.read_parquet(test_data_path, engine='pyarrow')
        print('Finished loading data from:', fin)
        regionname = 'Tokyo'
        if test_mode == 'subset':
            agentids = pd.read_pickle(fin + '/combined_agentids.pkl')
            agentids = set(agentids)
            print(f'there are {len(agentids)} agents in the test data')
            test_df = test_df[test_df['uid'].isin(agentids)] # filter user ids
            subsetnum = len(agentids)
        test_df['date'] = pd.to_datetime(test_df['date'])
        test_df['start_time'] = pd.to_datetime(test_df['start_time'])
        test_df = test_df.rename(columns={'stay_time': 'stay_duration'})
        test_df = test_df.rename(columns={'gt': 'label'})

    # convert the gps to x,y point (unit: km) centerd at the middle point
    if 'x' not in test_df.columns.tolist():
        test_df['x'], test_df['y'] = gps_to_normed_xy(test_df['lng'].tolist(), test_df['lat'].tolist(), region=regionname, scale=1000) # region: trial 3 in Jordan, scale 1000, means x, y unit is 1 km.
    print('Finished converting gps to x,y')

    # calcuate day of week
    if 'dow' not in test_df.columns.tolist():
        test_df['dow'] = test_df['start_time'].dt.dayofweek
    if 'start_time_minute' not in test_df.columns.tolist():
        test_df['start_time_minute'] = test_df['start_time'].dt.hour * 60 + test_df['start_time'].dt.minute
    if 'time_x' not in test_df.columns.tolist():
        test_df['angle'] = (test_df['start_time_minute'] / 1440) * 2 * np.pi
        # Calculate time_x and time_y
        r = 10 
        test_df['time_x'] = r * np.cos(test_df['angle'])
        test_df['time_y'] = r * np.sin(test_df['angle'])
        test_df = test_df.drop(columns=['angle'])
    print('Finished calculating day of week')
    # read the mapping file and convert the discrete features to id
    # discrete_map = {}
    # # for f in DISCRETE_FEATURE_NAME:
    # for fea in feature_list:
    #     f, f_type = fea['name'], fea['type']
    #     if f_type!= 'discrete': continue
    #     test_df[f], unique_categories = pd.factorize(test_df[f])
    #     # Create a mapping dictionary
    #     mapping = {idx: category for idx, category in enumerate(unique_categories)}
    #     discrete_map[f] = mapping
    # # save the discrete mapping to file
    # discrete_map_path = fout + '/discrete_map.json'
    # dir_check(discrete_map_path)
    # with open(discrete_map_path, 'w') as fp:
    #     json.dump(discrete_map, fp)


    discrete_map_path = ws + f'/data/dataset/{dataset_name}/discrete_feature.json'
    with open(discrete_map_path, 'r') as fp:
        discrete_map = json.load(fp)
                
    if 'trial2' in fin:
        for f, mapping in discrete_map.items():
            # Create a reverse mapping from category to index
            reverse_mapping = {v: int(k) for k, v in mapping.items()}
            # Apply the mapping to the DataFrame
            test_df[f] = test_df[f].map(reverse_mapping).astype(int)
    elif 'trial3' in fin:
        # Convert keys back to integers for 'dow' mapping
        if 'dow' in discrete_map:
            discrete_map['dow'] = {int(k): v for k, v in discrete_map['dow'].items()}
        for feature in discrete_map:
            test_df[feature] = test_df[feature].map(discrete_map[feature])
    elif 'NUMOSIM' in fin:
        if 'dow' in discrete_map:
            discrete_map['dow'] = {int(k): v for k, v in discrete_map['dow'].items()}
        if 'poi' in discrete_map:
            discrete_map['poi'] = {int(k): v for k, v in discrete_map['poi'].items()}
        for feature in discrete_map:
            test_df[feature] = test_df[feature].map(discrete_map[feature])
    elif 'trial4' in fin:
        if 'dow' in discrete_map:
            discrete_map['dow'] = {int(k): v for k, v in discrete_map['dow'].items()}
        for feature in discrete_map:
            test_df[feature] = test_df[feature].map(discrete_map[feature])

    test_df = test_df.sort_values(['uid', 'start_time']).reset_index(drop=True)
    print('Finished converting discrete features to IDs')
    uids = test_df['uid'].unique()
    task_num = len(uids) // num_threads


    # args_lst = []
    # for i in tqdm(range(0, len(uids), task_num), desc="Preparing arguments"):
    #     uids_a_task = uids[i:min(i + task_num, len(uids))]
    #     df_a_task = test_df[test_df['uid'].isin(uids_a_task)]
    #     args_lst.append((uids_a_task, df_a_task, gts_df, window_size_before, window_size_after, feature_cols))
    # print('Finished preparing arguments for parallel processing')

    args_lst = []
    for i in tqdm(range(0, len(uids), task_num), desc="Preparing arguments"):
        uids_a_task = uids[i:min(i + task_num, len(uids))]
        df_a_task = test_df[test_df['uid'].isin(uids_a_task)]
        if 'trial2' in fin or 'trial3' in fin:
            args_tmp = {"uids_a_task":uids_a_task, "df_a_task":df_a_task, "gts_df":gts_df, "window_size_before":window_size_before, "window_size_after":window_size_after, "feature_cols":feature_cols}
        else:
            args_tmp = {"uids_a_task":uids_a_task, "df_a_task":df_a_task, "gts_df":[], "window_size_before":window_size_before, "window_size_after":window_size_after, "feature_cols":feature_cols}
        args_lst.append(args_tmp)
    # with Pool(num_threads) as pool:
    #     result_lst = list(tqdm(pool.starmap(process_user_sequences_parallel, args_lst), total=len(args_lst), desc="Processing user chunks"))
    result_lst = multi_thread_work(args_lst, process_user_sequences_parallel, num_threads)

    # Aggregate the results from all parallel tasks
    final_sequences = {
        'X': [],
        'X_len': [],
        'position': [],
        'mask_idx': [],
        'uid': [],
        'label': []
    }

    for result in tqdm(result_lst, desc="Aggregating results"):
        for key in final_sequences.keys():
            final_sequences[key].extend(result[key])


    testsavepath = fout + '/{}_sequences_{}_{}_{}_{}.npy'.format(mode,window_size_before, window_size_after, test_mode,subsetnum)
    dir_check(testsavepath)
    np.save(testsavepath, final_sequences)

    return final_sequences


class TransformerDatasetTest(Dataset):
    def __init__(self, sequences, params: dict):
        super().__init__()
        self.data = sequences
        self.feature_list = params['feature_list']

        # feature2idx = {f['name']: f['idx'] for f in self.feature_list}
        # use_feature_idx = [feature2idx[f['name']] for f in params['feature_list']]

        # read feature list
        # data_dir =  os.path.dirname(params['dataset'])
        # data_dir = '/data/shurui/D-Transformer/data/dataset/7_22_trial2_ver2'
        # feature_dict = json.load(open(data_dir + '/feature.json'))
        # # Note: feature_dict is the feature in the dataset, which does not mean the features used for the training
        # feature2idx = {f['name']: f['idx'] for f in feature_dict}
        # print('Feature2idx:', feature2idx)
        # use_feature_idx = [feature2idx[f['name']] for f in params['feature_list']]
        # print('Use_feature_idx:', use_feature_idx)
        # # Feature2idx: {'x': 0, 'y': 1, 'start_time_minute': 2, 'stay_duration': 3, 'poi': 4, 'dow': 5}
        # # Use_feature_idx: [0, 1, 2, 3, 4, 5]

        # # only use
        # self.data['X'] = [x[:, use_feature_idx] for x in self.data['X']]

        # evaluate the test for multiple times, so that we are evaluate a very average behavior

        # multiple_times = 5
        # keys = self.data.keys()
        # for k in keys:
        #     self.data[k]  = self.data[k] * multiple_times
        
        # Feature dimension
        self.d_x = len(self.data['X'][0][0])
        print('Number of features:', self.d_x, "|| Feature example:", self.data['X'][0][0], "|| Feature names:", [f['name'] for f in self.feature_list])

        # Load user embedding dict: (not used in the current version, remove this part or just use an all zeros vector as placeholders)
        if 'trial2' in params['dataset']:
            from utils.util import ws
            u_path = ws + '/data/raw/trial2/user_feature/user_emb.json'
        else: # since we are not using u_emb anyways, I'll just load this all zeros version as placeholder
            from utils.util import ws
            u_path = ws + '/data/raw/trial2/user_feature/user_emb.json'
            # u_path = os.path.dirname(params['test_path']) + '/user_emb.json'
        self.u_emb_dict = json.load(open(u_path))

    def __len__(self):
        return len(self.data['X_len'])

    def __getitem__(self, index):
        X = self.data['X'][index]
        X_len = self.data['X_len'][index]
        uid = int(self.data['uid'][index])
        u_emb = np.array(self.u_emb_dict.get(uid, np.zeros(64)))
        position = self.data['position'][index]
        maskids = self.data['mask_idx'][index]
        label = self.data['label'][index]
        return X, X_len, u_emb, position, maskids, uid, label


class PretrainPadderTest:
    """
    Collate function for the pre-training
    """
    def __init__(self, norm='minmax', norm_path='', need_norm_feature=[], need_norm_feature_idx=[]):
        self.need_norm_feature = need_norm_feature
        self.need_norm_feature_idx = need_norm_feature_idx
        self.norm = norm

        if norm == 'minmax' and not norm_path:
            raise ValueError("norm path should not be empty for norm type: minmax")

        if self.norm == 'minmax':
            assert self.need_norm_feature, "need_norm_feature should not be empty"
            assert self.need_norm_feature_idx, "need_norm_feature_idx should not be empty"
            assert len(self.need_norm_feature) == len(self.need_norm_feature_idx), "len(need_norm_feature) should equal len(need_norm_feature_idx)"
            stat = pd.read_csv(norm_path, index_col=0)
            self.normalizer = Normalizer(stat, feat_names=self.need_norm_feature, feat_cols=self.need_norm_feature_idx, norm_type='minmax')

    def __call__(self, raw_batch):
        position2idx = {f['name']: f['idx'] for f in POSITION_LIST}

        input_batch, output_batch, pos_batch, user_emb_batch, uid_batch = [], [], [], [], []
        labels = []
        for X, X_len, u_emb, position, mask_idx, uid, label in raw_batch:
            if self.norm in ['minmax']:
                X = self.normalizer(X)
            
            prompt_arr = []
            trg_arr = []
            for i, x in enumerate(X):
                if i == mask_idx:  # Apply the mask to this specific event
                    mask_values = np.ones_like(x) * FEATURE_PAD
                    x_token = np.full_like(x, MASK_TOKEN)
                    arr = np.stack((mask_values, x_token), axis=-1)
                    prompt_arr.append(arr)

                    x_token = np.full_like(x, UNKNOWN_TOKEN)
                    arr = np.stack((x, x_token), axis=-1)
                    trg_arr.append(arr)
                else:
                    x_token = np.full_like(x, KNOWN_TOKEN)
                    arr = np.stack((x, x_token), axis=-1)
                    prompt_arr.append(arr)

                    x_token = np.full_like(x, KNOWN_TOKEN)
                    arr = np.stack((x, x_token), axis=-1)
                    trg_arr.append(arr)


            prompt_arr = np.stack(prompt_arr, axis=0)
            trg_arr = np.stack(trg_arr, axis=0)

            input_batch.append(prompt_arr)
            output_batch.append(trg_arr)
            pos_batch.append(position.T)
            user_emb_batch.append(u_emb)
            uid_batch.append(int(uid))
            labels.append(label)

        input_batch = torch.tensor(pad_batch_3d(input_batch)).float()
        output_batch = torch.tensor(pad_batch_3d(output_batch)).float()
        pos_batch = torch.tensor(pad_batch_2d(pos_batch)).long()
        user_emb_batch = torch.tensor(pad_batch_1d(user_emb_batch)).float()
        uid_batch = torch.from_numpy(np.array(uid_batch))
        # user_emb_batch = torch.tensor(pad_batch_1d(user_emb_batch)).long()
        labels_batch = torch.from_numpy(np.array(labels))  

        return input_batch, output_batch, pos_batch, user_emb_batch, uid_batch, labels_batch




def posthoc_uncertainty(model, test_loader, device, params, UC_mode, version, time_version, show_progress=True):
    # version: which version of classification uc
    # time_version: which version of time encoding? V1 is start time minute; V2 is time_x and time_y
    from utils.eval import Metric, AverageMeter
    from data.dataset import Denormalizer
    from timeit import default_timer as timer
    from utils.util import to_device, dict_merge, run, ws
    from tqdm import tqdm

    model.eval()

    uc_types = []
    if UC_mode in ['all', 'au']:
        uc_types.append('au')
    if UC_mode in ['all', 'eu']:
        uc_types.append('eu')       
    # if UC_model is 'None', then uc_types is []

    data_dict = {}
    if time_version == 'V2':
        # data_dict['time_angle'] = {'y': [], 'y_hat_prob': [], 'y_hat': [], 'eu_score': [], 'au_score': []}
        if not any(f['name'] == 'time_angle' for f in model.feature_list):
            new_feature = {'idx': 10, 'name': 'time_angle', 'type': 'continuous', 'is_predict': 'yes'}
            model.feature_list.append(new_feature)
        
    for f in model.feature_list:
        f_idx, f_name = f['idx'], f['name']
        data_dict[f_name] = {'y': [], 'y_hat_prob': [], 'y_hat': [], 'eu_score': [], 'au_score': [], 'loss':[]}
    with torch.no_grad():
        all_labels = []
        all_uids = []
        for batch in tqdm(test_loader, disable=not show_progress):
            # input_seq, target_seq, position, u_emb, uid, labels = to_device(batch, device)
            input_seq, target_seq, position, u_emb, uid,labels = to_device(batch, device)
            all_labels.extend([label.cpu().numpy() for label in labels])
            all_uids.extend([idx.cpu().numpy() for idx in uid])

            target_token = target_seq[..., 1].long()
            feature_mask = target_token != UNKNOWN_TOKEN

            target_con = target_seq[:, :, :model.d_c, 0]
            if not uc_types:
                pred_con, pred_dis_mu = model.forward(input_seq, position, u_emb, return_raw = True)
                pred_dis_sigma = [torch.zeros_like(x) for x in pred_dis_mu]
            elif 'au' in uc_types:
                pred_con, log_var_con, pred_dis_mu, pred_dis_sigma = model.sample(input_seq, position, u_emb, return_raw=True, n_sample=params['T'])
            else:
                pred_con, pred_dis_mu = model.sample(input_seq, position, u_emb, return_raw=True, n_sample=params['T'])
                log_var_con = torch.zeros_like(pred_con)
                pred_dis_sigma = [torch.zeros_like(x) for x in pred_dis_mu]
            d_c = pred_con.shape[-1]
            pred_con_denorm = pred_con
            # print(f'pred con: {pred_con_denorm.shape}')
            target_seq_denorm = target_seq[:, :, :d_c, 0]

            # Process continuous features
            for f in model.con_feature_list:
                f_idx, f_name = f['idx'], f['name']
                
                f_msk = feature_mask[..., f_idx]
                valid_indices = ~f_msk
                y = target_seq_denorm[..., f_idx]  # (B, L)
                y_hat = pred_con_denorm[..., f_idx]  # (B, L)
                                    
                data_dict[f_name]['y'] += y[valid_indices].cpu().numpy().tolist()
                data_dict[f_name]['y_hat_prob'] += y[valid_indices].cpu().numpy().tolist() # for continuous features, y_hat_prob is the same as y since there is no proabbilities
                
                if not uc_types:
                    data_dict[f_name]['y_hat'] += y_hat[valid_indices].cpu().numpy().tolist()
                else:
                    y_hat_mean = torch.mean(pred_con_denorm, dim=1)[..., f_idx]
                    data_dict[f_name]['y_hat'] += y_hat_mean[valid_indices].cpu().numpy().tolist()

                if 'eu' in uc_types:
                    f_eu = torch.mean(torch.square(y_hat), dim=1) - torch.square(torch.mean(y_hat, dim=1))
                    data_dict[f_name]['eu_score'] += f_eu[~f_msk].cpu().numpy().tolist()
                else:
                    data_dict[f_name]['eu_score'] = [0] * len(data_dict[f_name]['y'])

                if 'au' in uc_types:
                    log_var = log_var_con[..., f_idx]
                    var = torch.exp(log_var)
                    var_avg = torch.mean(var, dim=1)
                    data_dict[f_name]['au_score'] += var_avg[~f_msk].cpu().numpy().tolist()
                else:
                    data_dict[f_name]['au_score'] = [0] * len(data_dict[f_name]['y'])

                
                if 'au' in uc_types:
                    loss_con = F.mse_loss(y, y_hat_mean, reduction='none')
                    loss_con_reweighted = loss_con/var_avg
                    loss_con_reweighted = 0.5 * loss_con_reweighted
                    data_dict[f_name]['loss'] += loss_con_reweighted[valid_indices].cpu().numpy().tolist()
                else:
                    loss_con = F.mse_loss(y, y_hat, reduction='none')
                    data_dict[f_name]['loss'] += loss_con[valid_indices].cpu().numpy().tolist()

            if time_version == 'V2':
                # data_dict['time_angle'] = {'y': [], 'y_hat_prob': [], 'y_hat': [], 'eu_score': [], 'au_score': []}
                time_x_idx = None
                time_y_idx = None
                for f in model.con_feature_list:
                    if f['name'] == 'time_x':
                        time_x_idx = f['idx']
                    elif f['name'] == 'time_y':
                        time_y_idx = f['idx']
                time_msk = feature_mask[..., time_x_idx]
                # print(f'shape of time_msk: {time_msk.shape}')
                # time_y_msk = feature_mask[..., time_y_idx]
                # time_msk = time_x_msk | time_y_msk
                time_x_real = target_seq_denorm[..., time_x_idx]
                time_y_real = target_seq_denorm[..., time_y_idx]
                time_x_pred = pred_con_denorm[..., time_x_idx] # (B, T, L)
                time_y_pred = pred_con_denorm[..., time_y_idx] # (B, T, L)
                # **Denormalize time_x and time_y**
                time_x_real = time_x_real * 20 - 10
                time_y_real = time_y_real * 20 - 10
                time_x_pred = time_x_pred * 20 - 10
                time_y_pred = time_y_pred * 20 - 10
                if not uc_types:
                    time_x_pred_mean = time_x_pred
                    time_y_pred_mean = time_y_pred
                else:
                    # Average over T forward passes for time_x and time_y
                    time_x_pred_mean = torch.mean(time_x_pred, dim=1)  # Average over T, (B, L)
                    time_y_pred_mean = torch.mean(time_y_pred, dim=1)  # Average over T

                # Compute the final predicted angle from averaged time_x and time_y
                time_angle_pred = torch.atan2(time_y_pred_mean, time_x_pred_mean) #(B, L)
                time_angle_pred = torch.where(time_angle_pred < 0, time_angle_pred + 2 * np.pi, time_angle_pred)
                time_angle_real = torch.atan2(time_y_real, time_x_real)#(B, L)
                time_angle_real = torch.where(time_angle_real < 0, time_angle_real + 2 * np.pi, time_angle_real)
                # print(f'shape of time_angle_pred: {time_angle_pred.shape}, shape of time_angle_real: {time_angle_real.shape}')
                assert time_msk.shape == time_angle_real.shape, "Mask shape must match time_angle_real and time_angle_pred shapes!"

                data_dict['time_angle']['y'] += time_angle_real[~time_msk].cpu().numpy().tolist()
                data_dict['time_angle']['y_hat_prob'] += time_angle_real[~time_msk].cpu().numpy().tolist()  # Same as real for continuous features
                data_dict['time_angle']['y_hat'] += time_angle_pred[~time_msk].cpu().numpy().tolist()


                if 'au' in uc_types:
                    log_var_x = log_var_con[..., time_x_idx] # (B,T,L)
                    log_var_y = log_var_con[..., time_y_idx]
                    var_x = torch.exp(log_var_x)
                    var_y = torch.exp(log_var_y)
                    var_avg = torch.mean((var_x + var_y) / 2, dim=1)
                    data_dict['time_angle']['au_score'] += var_avg[~time_msk].cpu().numpy().tolist()
                else:
                    data_dict['time_angle']['au_score'] = [0] * len(data_dict['time_angle']['y'])


                # Compute epistemic uncertainty (EU) as the variance of the angles
                if 'eu' in uc_types:
                    time_angle_mean = torch.atan2(time_y_pred_mean, time_x_pred_mean)  # (B,L)
                    # time_angle_mean = torch.where(time_angle_mean < 0, time_angle_mean + 2 * np.pi, time_angle_mean)
                    # print(f'shape of time_angle_mean: {time_angle_mean.shape}')
                    time_angle_pred_all = torch.atan2(time_y_pred, time_x_pred)  # (B,T,L)
                    time_angle_pred_all = torch.where(time_angle_pred_all < 0, time_angle_pred_all + 2 * np.pi, time_angle_pred_all)
                    # print(f'shape of time_angle_pred_all: {time_angle_pred_all.shape}')
                    angular_differences = torch.abs(time_angle_pred_all - time_angle_mean.unsqueeze(1)) # (B,T,L)
                    # print(f'shape of angular_differences: {angular_differences.shape}')
                    angular_differences = torch.where(
                        angular_differences > np.pi, 2 * np.pi - angular_differences, angular_differences) # Normalize angular differences to the shorter arc
                    eu_score = torch.mean(angular_differences**2, dim=1)
                    # print(f'shape of eu_score: {eu_score.shape}')
                    data_dict['time_angle']['eu_score'] += eu_score[~time_msk].cpu().numpy().tolist()
                else:
                    data_dict['time_angle']['eu_score'] = [0] * len(data_dict['time_angle']['y'])

                
                if 'au' in uc_types:
                    loss_con = F.mse_loss(time_angle_real, time_angle_pred, reduction='none')
                    loss_con_reweighted = loss_con/var_avg
                    loss_con_reweighted = 0.5 * loss_con_reweighted
                    data_dict['time_angle']['loss'] += loss_con_reweighted[~time_msk].cpu().numpy().tolist()
                else:
                    loss_con = F.mse_loss(time_angle_real, time_angle_pred, reduction='none')
                    data_dict['time_angle']['loss'] += loss_con[~time_msk].cpu().numpy().tolist()

                    
            # Process discrete features
            for f, mu, sigma in zip(model.dis_feature_list, pred_dis_mu, pred_dis_sigma):
                f_idx, f_name = f['idx'], f['name']
                f_msk = feature_mask[..., f_idx]  # (B, L)
                logit = mu  # (B, L, class_num_d_i) or (B, T, L, class_num_d_i) if T > 1
                valid_indices = ~f_msk
                y = target_seq[:, :, f_idx, 0].long()  # (B, L)
                if not uc_types:
                    prob_avg = F.softmax(logit, dim=-1)  # (B, L, class_num_d_i)
                else: 
                    prob = F.softmax(logit, dim=-1) # (B, T, L, class_num_d_i) if T > 1
                    prob_avg = torch.mean(prob, dim=1) # (B, L, class_num_d_i)

                y_hat_prob = prob_avg.gather(dim=2, index=y.unsqueeze(-1)).squeeze(-1)  # (B, L)
                data_dict[f_name]['y'] += y[valid_indices].cpu().numpy().tolist()
                data_dict[f_name]['y_hat_prob'] += y_hat_prob[valid_indices].cpu().numpy().tolist()
                data_dict[f_name]['y_hat'] += torch.argmax(prob_avg, dim=2)[valid_indices].cpu().numpy().tolist()
   
                if 'au' in uc_types:
                    if version == 'V1': # au = avg(sigma_c) over T
                        # Find the sigma of the true (observed) class
                        sigma_c = sigma.gather(dim=-1, index=y.unsqueeze(1).unsqueeze(-1).expand(-1, sigma.shape[1], -1, 1)).squeeze(-1)  # Shape: (B, T, L)
                        avg_sigma_c = sigma_c.mean(dim=1)  # (B, L)
                        data_dict[f_name]['au_score'] += avg_sigma_c[~f_msk].cpu().numpy().tolist()
                    elif version == 'V2': # au = avg_entropy
                        # avg(entropy(softmaxt(mu+sigma))) over T
                        epsilon = torch.randn_like(sigma)
                        pred_f = mu + torch.mul(sigma, epsilon) # Shape: (B, T, L, class_num_d_i)
                        softmax_pred_f = F.softmax(pred_f, dim=-1)

                        H_t = torch.sum(-softmax_pred_f * torch.log(softmax_pred_f + 1e-10), dim=-1)  # Entropy per class per T, shape (B, T, L)
                        H_t_avg = torch.mean(H_t, dim=1)  # Average entropy per T, shape shape (B, L)
                        data_dict[f_name]['au_score'] += H_t_avg[~f_msk].cpu().numpy().tolist()
                    elif version == 'V3': # au = avg(sigma_c) over T
                        sigma_c = sigma.gather(dim=-1, index=y.unsqueeze(1).unsqueeze(-1).expand(-1, sigma.shape[1], -1, 1)).squeeze(-1)  # Shape: (B, T, L)
                        avg_sigma_c = sigma_c.mean(dim=1)  # (B, L)
                        data_dict[f_name]['au_score'] += avg_sigma_c[~f_msk].cpu().numpy().tolist()
                else:
                    data_dict[f_name]['au_score'] = [0] * len(data_dict[f_name]['y'])

                if 'eu' in uc_types:
                    if version == 'V1': # eu = entropy(avg_softmax), kendall paper
                        H_p = torch.sum(-prob_avg * torch.log(prob_avg), dim=-1)
                        data_dict[f_name]['eu_score'] += H_p[~f_msk].cpu().numpy().tolist()
                    elif version == 'V2': # eu = tu - au
                        # total uc = entropy(avg(softmax(f+sigma)))
                        softmax_pred_f_avg = softmax_pred_f.mean(dim=1)  # (B, L, class_num_d_i)
                        total_uc = torch.sum(-softmax_pred_f_avg * torch.log(softmax_pred_f_avg + 1e-10), dim=-1)  # Entropy per class, shape (B, L)
                        eu = total_uc - H_t_avg
                        data_dict[f_name]['eu_score'] += eu[~f_msk].cpu().numpy().tolist()
                    elif version == 'V3':
                        logit_c = logit.gather(dim=-1, index=y.unsqueeze(1).unsqueeze(-1).expand(-1, logit.shape[1], -1, 1)).squeeze(-1)  # Shape: (B, T, L)
                        eu_c = torch.mean(torch.square(logit_c), dim=1) - torch.square(torch.mean(logit_c, dim=1))
                        data_dict[f_name]['eu_score'] += eu_c[~f_msk].cpu().numpy().tolist()
                else:
                    data_dict[f_name]['eu_score'] = [0] * len(data_dict[f_name]['y'])

                if not uc_types:
                    epsilon = torch.randn_like(sigma)
                    pred_f_1 = mu + torch.mul(sigma, epsilon)
                    prob_avg_1 = F.softmax(pred_f_1, dim=-1) # (B, L, class_num_d_i)
                else:
                    epsilon = torch.randn_like(sigma)
                    pred_f_1 = mu + torch.mul(sigma, epsilon)
                    softmax_pred_f_1 = F.softmax(pred_f_1, dim=-1) # (B,T,L,class_num_d_i)
                    prob_avg_1 = torch.mean(softmax_pred_f_1, dim=1) # (B, L, class_num_d_i)
                    # loss_f = F.nll_loss(torch.log(prob_avg), y, reduction='none')

                # Reshape prob_avg and y for loss computation
                prob_avg_1 = prob_avg_1.view(-1, prob_avg_1.size(-1))  # Shape: [B*L, class_num_d_i]
                y = y.view(-1)  # Shape: [B*L]

                # Compute loss
                loss_f = F.nll_loss(torch.log(prob_avg_1), y, reduction='none')  # Shape: [B*L]

                # Filter loss using mask
                f_msk = f_msk.view(-1)  # Shape: [B*L]
                filtered_loss = loss_f[~f_msk]  # Filtered loss values
                data_dict[f_name]['loss'] += filtered_loss.cpu().numpy().tolist()
                # data_dict[f_name]['loss'] += loss_f[~f_msk].cpu().numpy().tolist()

    # Ensure all_labels is a list of tensors
    # all_labels = [label.cpu().numpy() for label in all_labels]
    # all_labels = [label.cpu().numpy() if isinstance(label, torch.Tensor) else label for label in all_labels]
    # all_uids = [uid.cpu().numpy() if isinstance(uid, torch.Tensor) else uid for uid in all_uids]
    
    result = {}
    for f in model.feature_list:
        f_idx, f_name = f['idx'], f['name']
        for key, value in data_dict[f_name].items():
            print(f"{f_name} Key: {key}, Length: {len(value)}")

        df = pd.DataFrame(data_dict[f_name])
        df['label'] = all_labels
        df['uid'] = all_uids
        result[f_name] = df
    return result
