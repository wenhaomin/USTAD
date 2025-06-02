# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

#-------------------------------------------------------------------------------------------------------------------------#
#for linux import package
import sys, os, platform
file_lst = ['your_project_directory']
for file in file_lst:
    sys.path.append(file)
    sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk(file) for name in dirs])
#-------------------------------------------------------------------------------------------------------------------------#

import torch
from tqdm import tqdm
from timeit import default_timer as timer

from utils.util import to_device, dict_merge,run, ws
from algorithm.dataset_loader import EventDataset, PretrainPadder
from algorithm.ustad.model import USTAD, save2file
from utils.eval import Metric, AverageMeter

from data.dataset import Denormalizer


def process_batch(batch, model, device, params):
    # input_seq, target_seq, position, u_emb, uid = batch
    return model.loss(*to_device(batch, device))

def test_model_with_loss(model, test_loader, device, params, save2file, mode):

    model.eval()
    evaluators = [Metric(feature_lst = params['feature_list'])]
    time_start = timer()

    batch_input_seq, batch_pred_con, batch_target_seq, batch_uid = [], [],  [], []

    val_epoch_log = {}
    log_features = params.get('log_features', [])
    log_features += ['total']
    feature_loss = {f: AverageMeter() for f in log_features}
    with torch.no_grad():
        for batch in tqdm(test_loader):

            input_seq, target_seq, position, u_emb, uid = to_device(batch, device)

            T = params.get('T', 1)
            if T == 1:
                pred, _ = model(input_seq, position, u_emb, return_raw=False)
            else:
                # print('sample number:', T)
                pred = model.forward_avg(input_seq, position, u_emb, n_sample=T)

            # add data into record
            batch_input_seq.append(input_seq.cpu().numpy())
            batch_pred_con.append(pred.cpu().numpy())
            batch_target_seq.append(target_seq.cpu().numpy())
            batch_uid.append(uid.cpu().numpy())

            # Denormalize data
            if params['norm']=='minmax':
                # number of continous features
                d_c = len(params['need_norm_feature'])
                stay_stat_path = params['norm_path']
                stat = pd.read_csv(stay_stat_path, index_col=0)
                denormalizer = Denormalizer(stat, params['need_norm_feature'], params['need_norm_feature_idx'], norm_type=params['norm']) # importance, need to give a parameter
                pred[:, :, : d_c] = denormalizer(pred[:, :, :d_c])
                target_seq[:, :, : d_c, 0]  = denormalizer(target_seq[:, :, :d_c, 0] )

            for e in evaluators:
                e.update_metrics(target_seq.cpu().numpy(), pred.cpu().numpy())

            # get the loss value
            loss, loss_value_dict = model.loss(*to_device(batch, device))
            # loss, loss_value_dict = model.loss(input_seq, target_seq, position, u_emb)
            for f in log_features:
                if f not in loss_value_dict.keys():
                    print( f'Feature {f} not in return loss dict, current keys of loss dict: {loss_value_dict.keys()}')
                else:
                    feature_loss[f].update(loss_value_dict[f])


    # save the input, predcition, and target
    if mode == 'test':
        import pickle
        with open(params['forecast_path'], 'wb') as f:
            pickle.dump([batch_input_seq, batch_pred_con, batch_target_seq, batch_uid, params['need_norm_feature'], params['norm']], f)
        print('Save forecast results to:', params['forecast_path'])

    if mode == 'val':
        for k, v in feature_loss.items():
            val_epoch_log[k] = v.avg
        return evaluators[-1], val_epoch_log

    params['test_time'] = timer() - time_start
    for e in evaluators:
        params_save = dict_merge([e.to_dict(), params])
        save2file(params_save)
    return evaluators[-1]

def main(params):
    params['model'] = 'vanilla Transformer'
    params['model_save2file'] = (USTAD, save2file)
    norm_path = ws + f'/data/dataset/{params["dataset"]}/stay_train_describle.csv'
    params['norm_path'] = norm_path
    params['log_features'] = [f['name'] for f in params['feature_list']]
    need_norm_feature = params['need_norm_feature']
    need_norm_feature_idx = params['need_norm_feature_idx']

    train_collate_fn = PretrainPadder(mode='train', event_mask_ratio=params['mask_ratio'], norm=params['norm'], norm_path = norm_path, need_norm_feature = need_norm_feature, need_norm_feature_idx = need_norm_feature_idx)
    val_collate_fn = PretrainPadder(mode='val',  norm=params['norm'], norm_path = norm_path, need_norm_feature = need_norm_feature, need_norm_feature_idx = need_norm_feature_idx)
    test_collate_fn = PretrainPadder(mode='test',  norm=params['norm'], norm_path = norm_path, need_norm_feature = need_norm_feature, need_norm_feature_idx = need_norm_feature_idx)

    return run(params, EventDataset, process_batch, test_model_with_loss, collate_fn=(train_collate_fn, val_collate_fn, test_collate_fn))


def get_params():
    from utils.util import get_common_params
    parser = get_common_params()
    # Model parameters
    parser.add_argument('--model', type=str, default='USTAD')
    parser.add_argument('--user_mode', type=str, default= 'no_use') # how to utilize user information, choice: no_use, before_transformer, after_transformer
    parser.add_argument('--pos_mode', type=str, default='vanilla_pe') # to set different types of pos,  choice: no_use, vanilla_pe,
    parser.add_argument('--num_head', type=int, default=4) # number of heads in transformer
    parser.add_argument('--num_layer', type=int, default=2) # number of layers in transformer
    parser.add_argument('--dropout', type=int, default=0.05) # number of layers in transformer

    parser.add_argument('--norm', type = str, default='minmax') # conduct feature norm or not, minmax, no_norm

    # training parameters
    parser.add_argument('--mask_ratio', type=float, default=0.2)

    args, _ = parser.parse_known_args()
    return args

if __name__ == '__main__':

    FEATURE_LIST = [
        # continous
        {'idx': 0, 'name': 'x', 'type': 'continuous', 'is_predict': 'yes'},
        {'idx': 1, 'name': 'y', 'type': 'continuous', 'is_predict': 'yes'},
        {'idx': 2, 'name': 'start_time_minute', 'type': 'continuous', 'is_predict': 'yes'},
        {'idx': 3, 'name': 'stay_duration', 'type': 'continuous', 'is_predict': 'yes'},
        # discrete
        {'idx': 4, 'name': 'poi', 'type': 'discrete', 'class_num': 40, 'is_predict': 'yes'},
        {'idx': 5, 'name': 'dow', 'type': 'discrete', 'class_num': 7 ,'is_predict': 'yes'},
    ]
    Feature2Idx = {f['name']: f['idx'] for f in FEATURE_LIST}
    NeedNormFeature = [f['name'] for f in FEATURE_LIST if f['type'] == 'continuous']

    POSITION_LIST = [
        {'idx': 0, 'name': 'seq_pos'},
        {'idx': 1, 'name': 'within_day_pos'},
        {'idx': 2, 'name': 'day_pos'},
    ]
    position2idx = {f['name']: f['idx'] for f in POSITION_LIST}


    params = vars(get_params())

    # add feature configure
    params['feature_list'] =  FEATURE_LIST
    params['feature2idx'] = Feature2Idx
    params['position2idx'] = position2idx

    params['need_norm_feature'] = NeedNormFeature
    params['need_norm_feature_idx'] = [Feature2Idx[f] for f in NeedNormFeature]

    # for test the code
    params['dataset'] = 'test_dataset'
    params['batch_size'] = 32 if platform.system() == 'Windows' else 128
    params['hidden_size'] = 32
    params['early_stop'] = 1 if platform.system() == 'Windows' else 20
    params['is_test'] = True
    workers = 1 if platform.system() == 'Windows' else 16
    params['workers'] = workers
    main(params)

