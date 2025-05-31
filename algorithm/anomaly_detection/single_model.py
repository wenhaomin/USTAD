#-------------------------------------------------------------------------------------------------------------------------#
import sys
import os
import platform
import re
import numpy as np
import pandas as pd
import torch
import pickle
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc
from collections import defaultdict,Counter

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import itertools
# Set up paths
file = 'D:\\Study\\Lab\\project\\CMU project\\haystac\\temp_sync\\D-Transformer' if platform.system() == 'Windows' else '/data/shurui/D-Transformer/'
sys.path.append(file)
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk(file) for name in dirs])

# Import necessary modules

from utils.util import ws
from output.model_inference_uc import get_path
from algorithm.anomaly_detection.data_funcs import *
from scipy.stats import rankdata
from torch.utils.data import DataLoader
torch.set_num_threads(12)

torch.cuda.empty_cache()


FEATURE_LIST = [
    # continous
    {'idx': 0, 'name': 'x', 'type': 'continuous', 'is_predict': 'yes'},
    {'idx': 1, 'name': 'y', 'type': 'continuous', 'is_predict': 'yes'},
    {'idx': 2, 'name': 'time_x', 'type': 'continuous', 'is_predict': 'yes'},
    {'idx': 3, 'name': 'time_y', 'type': 'continuous', 'is_predict': 'yes'},
    # {'idx': 10, 'name': 'time_angle', 'type': 'continuous', 'is_predict': 'yes'},
    {'idx': 4, 'name': 'stay_duration', 'type': 'continuous', 'is_predict': 'yes'},
    # discrete
    {'idx': 5, 'name': 'poi', 'type': 'discrete', 'class_num': 15, 'is_predict': 'yes'}, # 40 for trial2 and numosim, 7 for trial3
    {'idx': 6, 'name': 'dow', 'type': 'discrete', 'class_num': 7 ,'is_predict': 'yes'}
    ]


FEATURE_COLS = [f['name'] for f in FEATURE_LIST if f['type']!='trip_emb'] 
settings_map = {
    "trial4_full": {
        'window_size_before': 1,
        'window_size_after': 1,
        'test_mode': 'full',
        'subsetnum': 691945,
        'datasetname': 'trial4',
        'fin': ws + '/data/raw/trial4_2',
        'fout': ws + '/data/dataset/5_4_trial4_inference', # 12_18 contains time_x and time_y version without trip
        'mode': 'test', # which data to run inference
        'poi_version':'V1', #V1 au is the same as V3 but eu is in kendall paper
        'time_version': 'V2'
    }}


# Global Parameters
output_dir = ws + '/output/pred_uc/'
seed = 2021
is_test = False
T = 50

def slice_dict(d, start, end):
    return {k: v[start:end] for k, v in d.items()}

def get_merged_output_files(model_id, mode, poi_version, time_version, datasetname, test_mode, window_size_after, window_size_before, subsetnum):
    pattern = f'uc_result_{mode}_{poi_version}_{time_version}_{model_id}_{datasetname}_{test_mode}_{window_size_after}_{window_size_before}_{subsetnum}'
    files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.startswith(pattern) and f.endswith('.pkl')]
    files.sort()
    return files

def merge_and_clean_up_files(model_id, mode,poi_version, time_version,datasetname, test_mode, window_size_after, window_size_before, subsetnum):
    merged_files = get_merged_output_files(model_id, mode, poi_version, time_version, datasetname, test_mode, window_size_after, window_size_before, subsetnum)
    merged_results = {}

    for file in tqdm(merged_files, desc=f"Merging results for model {model_id}"):
        with open(file, 'rb') as f:
            uc_result = pickle.load(f)
        if not merged_results:
            merged_results = {k: v.copy() for k, v in uc_result.items()}
        else:
            for k, v in uc_result.items():
                merged_results[k] = pd.concat([merged_results[k], v], ignore_index=True)

    final_output_filename = os.path.join(output_dir, f'uc_result_{mode}_{poi_version}_{time_version}_{model_id}_{datasetname}_{test_mode}_{window_size_after}_{window_size_before}_{subsetnum}.pkl')
    with open(final_output_filename, 'wb') as f:
        pickle.dump(merged_results, f)
    print(f'Saved merged results for model {model_id} to {final_output_filename}')

    for file in merged_files:
        os.remove(file)

    return final_output_filename

   
def extract_errors_and_uc(result, feature_list):
    # LOSS for numerical features = MSE 
    data_dict = {}
    for feature in feature_list:
        f_name = feature['name']
        if f_name in ['time_x', 'time_y']:
            continue
        data_dict[f_name] = {'y':[], 'y_hat_prob':[], 'y_hat':[],'error': [], 'loss': [], 'label': [], 'au_score': [], 'eu_score': [], 'uid': []}
        # if f_name == 'dow':
        #     continue
        df = result[f_name].sort_values(by='uid', kind='stable').reset_index(drop=True)
        data_dict[f_name]['label'] = df['label'].values
        if feature['type'] == 'continuous':
            if f_name == 'time_angle':
                delta = np.abs(df['y'] - df['y_hat'])
                error = np.minimum(delta, 2 * np.pi - delta)
                squared_error = error ** 2
            else:
                error = np.abs(df['y'] - df['y_hat'])
                squared_error = (df['y'] - df['y_hat']) ** 2
            loss = 0.5 * squared_error / df['au_score']  # au_score = sigma^2
        elif feature['type'] == 'discrete':
            error = 1 - df['y_hat_prob']
            # loss = -np.log(df['y_hat_prob'])  # Cross-entropy loss
            epsilon = 1e-8
            adjusted_prob = np.clip(df['y_hat_prob'] - 0.5 * (df['au_score']), epsilon, 1.0)
            # Compute approximate NLL loss
            loss = -np.log(adjusted_prob)
        elif feature['type'] == 'trip_emb':
            y_values = np.stack(df['y'].values)
            y_hat_values = np.stack(df['y_hat'].values)
            
            # Compute MSE across the embedding dimension (axis=1) in a vectorized way
            error = np.mean((y_values - y_hat_values) ** 2, axis=1)
            loss = 0.5 * error / df['au_score']
            # if f_name == 'shape':
            #     error = error * 0.01
            # else:
            #     error = error
        data_dict[f_name]['error'] = error
        data_dict[f_name]['loss'] = loss
        data_dict[f_name]['au_score'] = df['au_score'].values
        data_dict[f_name]['eu_score'] = df['eu_score'].values
        data_dict[f_name]['uid'] = df['uid'].values
        data_dict[f_name]['y'] = df['y'].values
        data_dict[f_name]['y_hat_prob'] = df['y_hat_prob'].values
        data_dict[f_name]['y_hat'] = df['y_hat'].values
    return data_dict

if __name__ == '__main__':

    import random
    from utils.util import Platforms
    import torch
    if platform.platform() == Platforms.la3:
        cuda_id = random.choice([3])
        device = f"cuda:{cuda_id}"
        torch.cuda.set_device(device)
    else:
        device = "cuda:0"  # for windows

    import  torch
    from utils.util import ws, get_common_params, dict_merge, get_dataset_path

    def get_params():
            parser = get_common_params()
            args, _ = parser.parse_known_args()
            return args

    seed = 2021
    is_test = False
    args_lst = []
    T = 50 # number of samples

    # common args
    params = vars(get_params())


    if 1: # dual trasnformer uc 0512
        from algorithm.dual_transformer_uc.model import USTAD, save2file
        from algorithm.dual_transformer_uc.train import test_model_with_loss as test_model

        model_id = 'CLLA'
        UC_mode = 'all'
        pretrained_model_path = get_path('./output/model/', model_id)

        model_params = {'mask_ratio': 0.3, 'd_h': 128, 'num_layer': 3, 'dataset': '5_4_trial4', 'seed': 2021, 'dropout': 0.05}
        params = dict_merge([params, model_params])

    # model related setting:

    MODEL = USTAD
    print('model path:', pretrained_model_path)
    POSITION_LIST = [
        {'idx': 0, 'name': 'seq_pos'},
        {'idx': 1, 'name': 'within_day_pos'},
        {'idx': 2, 'name': 'day_pos'},
    ]

    position2idx = {f['name']: f['idx'] for f in POSITION_LIST}
    common_params = { 'user_mode': 'no_use', 'pos_mode': 'vanilla_pe', 'num_head': 4,
                        'is_test': is_test, 'early_stop_start_epoch': 20, 'early_stop': 5, 'norm': 'minmax',
                        'batch_size': 32, 'workers': 8, 'position2idx': position2idx, 'T': T, 'model_id': model_id}

    params = dict_merge([params, common_params])
    Feature2Idx = {f['name']: f['idx'] for f in FEATURE_LIST}
    NeedNormFeature = [f['name'] for f in FEATURE_LIST if f['type'] == 'continuous']
    params['feature_list'] = FEATURE_LIST
    params['feature2idx'] = Feature2Idx
    params['need_norm_feature'] = NeedNormFeature
    params['need_norm_feature_idx'] = [Feature2Idx[f] for f in NeedNormFeature]
    norm_path = ws + f'/data/dataset/{params["dataset"]}/stay_train_describle.csv'
    params['norm_path'] = norm_path

    model = MODEL(params).float()
    model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
    model.to(device)

    need_norm_feature = params['need_norm_feature']
    need_norm_feature_idx = params['need_norm_feature_idx']

    # dataset setting 
    selected_setting = "trial4_full" #trial2_restrictive
    test_setting_params = settings_map[selected_setting]
    test_setting_params['testsavepath'] = (
    f"{test_setting_params['fout']}/{test_setting_params['mode']}_sequences_"
    f"{test_setting_params['window_size_before']}_{test_setting_params['window_size_after']}_"
    f"{test_setting_params['test_mode']}_{test_setting_params['subsetnum']}.npy")

    if os.path.exists(test_setting_params['testsavepath']):
        test_sequences = np.load(test_setting_params['testsavepath'], allow_pickle=True).item()
        print(f'Loaded test sequences from {test_setting_params["testsavepath"]}')
    else:
        test_sequences = create_test_sequences_parallel(
            test_setting_params['window_size_before'], 
            test_setting_params['window_size_after'], 
            FEATURE_COLS, 20, 
            test_setting_params['fin'], 
            test_setting_params['fout'], 
            test_setting_params['test_mode'], 
            FEATURE_LIST, 
            test_setting_params['subsetnum'],
            params['dataset'],
            test_setting_params['mode']
        )
        print(f'Saved test sequences to {test_setting_params["testsavepath"]}')

    num_segments = 10
    for segment_index in range(num_segments):
        print(f'Running inference for model {model_id} segment {segment_index}')
        segment_start = segment_index * len(test_sequences['uid']) // num_segments
        segment_end = (segment_index + 1) * len(test_sequences['uid']) // num_segments if segment_index < num_segments - 1 else len(test_sequences['uid'])
        test_sequences_segment = slice_dict(test_sequences, segment_start, segment_end)
        #uc_result = run_inference(test_sequences_segment, model, params, device, uc_mode, test_setting_params['poi_version'], test_setting_params['time_version'])
        test_collate_fn = PretrainPadderTest(
            norm=params['norm'], 
            norm_path=params['norm_path'], 
            need_norm_feature=params['need_norm_feature'], 
            need_norm_feature_idx=params['need_norm_feature_idx'])
        test_dataset = TransformerDatasetTest(sequences=test_sequences_segment, params=params)
        print(f"Test dataset size: {len(test_dataset)}")
        if len(test_dataset) == 0:
            raise ValueError("Test dataset is empty!")

        print(f'Prepared data for model {model_id} segment {segment_index} ----- ')

        test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False, collate_fn=test_collate_fn, num_workers=10)

        # for batch in test_loader:
        #     print("Collate function worked, batch received!")
        #     break

        # try:
        #     print("Trying to fetch first batch...")
        #     first_batch = next(iter(test_loader))
        #     print("First batch received!")
        # except StopIteration:
        #     print("Test loader is empty!")
        # except Exception as e:
        #     print(f"Error while fetching batch: {e}")

        uc_result = posthoc_uncertainty(model, test_loader, device, params, UC_mode,  test_setting_params['poi_version'], test_setting_params['time_version'], show_progress=True)
        final_output_filename_segment = os.path.join(output_dir, f'uc_result_{test_setting_params["mode"]}_{test_setting_params["poi_version"]}_{test_setting_params["time_version"]}_{model_id}_{test_setting_params["datasetname"]}_{test_setting_params["test_mode"]}_'
                                                 f'{test_setting_params["window_size_after"]}_{test_setting_params["window_size_before"]}_'
                                                 f'{test_setting_params["subsetnum"]}_{segment_index}.pkl')
        print(f'Inference for model {model_id} segment {segment_index} done')
        with open(final_output_filename_segment, 'wb') as f:
            pickle.dump(uc_result, f)
    
    merge_and_clean_up_files(
            model_id, test_setting_params["mode"], test_setting_params["poi_version"],test_setting_params["time_version"], test_setting_params["datasetname"], test_setting_params["test_mode"], test_setting_params["window_size_after"], 
            test_setting_params["window_size_before"], test_setting_params["subsetnum"])
    print(f'Inference for model {model_id} done')

    # --------- Anomaly Detection Evaluation ---------
    print(f'Loading results for model {model_id}')
    final_output_filename = os.path.join(output_dir, f'uc_result_{test_setting_params["mode"]}_{test_setting_params["poi_version"]}_{test_setting_params["time_version"]}_{model_id}_{test_setting_params["datasetname"]}_{test_setting_params["test_mode"]}_'
                                                 f'{test_setting_params["window_size_after"]}_{test_setting_params["window_size_before"]}_'
                                                 f'{test_setting_params["subsetnum"]}.pkl')

    uc_result = pd.read_pickle(final_output_filename)

    F_LIST = [f for f in FEATURE_LIST if f['name'] != 'time_x' and f['name'] != 'time_y']
    error_dict = extract_errors_and_uc(uc_result, F_LIST)
    error_savepath = (
        f'{ws}/output/ADscore/error_dict_{model_id}_'
        f'{test_setting_params["datasetname"]}_{test_setting_params["test_mode"]}_'
        f'{test_setting_params["window_size_after"]}_{test_setting_params["window_size_before"]}_'
        f'{test_setting_params["subsetnum"]}.pkl')
    with open(error_savepath, 'wb') as f:
        pickle.dump(error_dict, f)

    # Anomaly detection evaluation
    
    # compute aggregated scores
    log_dict = error_dict
    first_key = next(iter(log_dict))
    # Assume log_dict[first_key] is a DataFrame
    df = log_dict[first_key]

    labels = np.array([x.item() if isinstance(x, np.ndarray) else x for x in df['label']]).astype(int)
    uids = np.array([x.item() if isinstance(x, np.ndarray) else x for x in df['uid']]).astype(int)

    log_features = [f['name'] for f in params['feature_list']  if f['is_predict'] == 'yes']
    num_samples = len(labels)  # Number of events
    df_data = {'label': labels, 'uid': uids}  # Initialize with label & uid
    log_features_score = [ f for f in log_features if f != 'dow']
    for f_name in log_features_score:
        df_data[f'{f_name}_pe'] = log_dict[f_name]['error']  # Store prediction error (pe)
    log_df = pd.DataFrame(df_data)
    for f_name in log_features_score:
        log_df[f'{f_name}_percentile'] = rankdata(log_df[f'{f_name}_pe'], method='average') / num_samples
    log_df['anomaly_score'] = log_df[[f'{f}_percentile' for f in log_features_score]].max(axis=1)

    event_labels = log_df['label'].values
    anomaly_scores = log_df['anomaly_score'].values 
    event_roc_auc = roc_auc_score(event_labels, anomaly_scores)
    event_aupr = average_precision_score(event_labels, anomaly_scores)
    log_df['uid'] = log_df['uid'].astype(int)
    log_df['label'] = log_df['label'].astype(int)
    agent_df = log_df.groupby('uid').agg({'anomaly_score': 'max', 'label': 'max'}).reset_index()
    agent_labels = agent_df['label'].values
    agent_roc_auc = roc_auc_score(agent_labels, agent_df['anomaly_score'].values)
    agent_aupr = average_precision_score(agent_labels, agent_df['anomaly_score'].values)
    
    print(f"Results for model {model_id} for data setting {selected_setting}:")
    print("Event-level ROCAUC:", event_roc_auc)
    print("Event-level PR AUC:", event_aupr)
    print("Agent-level ROCAUC:", agent_roc_auc)
    print("Agent-level PR AUC:", agent_aupr)


