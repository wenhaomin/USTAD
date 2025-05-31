# -*- coding: utf-8 -*-


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
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
import pickle
import matplotlib.pyplot as plt

from collections import defaultdict
import pyperclip
from algorithm.dataset_loader import UNKNOWN_TOKEN
import torch 
from torch.nn import functional as F



def get_path(dir, key):
    file_paths = []
    # Walk through directory
    for root, directories, files in os.walk(dir):
        for filename in files:
            # Create the full file path
            file_path = os.path.join(root, filename)
            file_paths.append(file_path)
    for f in file_paths:
        if key in f:
            return f
    return None

def posthoc_evaluation(model, test_loader, device, params, save2file, mode, show_progress=True):
    from utils.eval import Metric, AverageMeter
    from data.dataset import Denormalizer
    from timeit import default_timer as timer
    from utils.util import to_device, dict_merge, run, ws
    from tqdm import tqdm

    model.eval()
    time_start = timer()

    # Initialize the data collection dictionary
    data_dict = {}
    for f in model.feature_list:
        f_idx, f_name = f['idx'], f['name']
        data_dict[f_name] = {'y': [], 'y_hat_prob':[],'y_hat': []}

    with torch.no_grad():
        all_labels = []
        for batch in tqdm(test_loader, disable=not show_progress):

            input_seq, target_seq, position, u_emb, uid, labels = to_device(batch, device)
            all_labels.extend([label.cpu().numpy() for label in labels])

            target_token = target_seq[..., 1].long()  # (B, L, F)
            feature_mask = target_token != UNKNOWN_TOKEN  # (B, L, F)

            target_con = target_seq[:, :, :model.d_c, 0]  # (B, L, d_c)

            # Use the forward function to get predictions
            pred_con, pred_dis_mu = model.forward(input_seq, position, u_emb, return_raw = True)
            # pred_con, pred_dis_mu = model.forward(input_seq, position, uid, return_raw = True)

            d_c = pred_con.shape[-1]
            pred_con_denorm = pred_con
            target_seq_denorm = target_seq[:, :, :d_c, 0]

            # Process continuous features
            for f in model.con_feature_list:
                f_idx, f_name = f['idx'], f['name']
                f_msk = feature_mask[..., f_idx]  # (B, L)

                y = target_seq_denorm[..., f_idx]  # (B, L)
                y_hat = pred_con_denorm[..., f_idx]  # (B, L)

                valid_indices = ~f_msk
                data_dict[f_name]['y'] += y[valid_indices].cpu().numpy().tolist()
                data_dict[f_name]['y_hat_prob'] += y[valid_indices].cpu().numpy().tolist()
                data_dict[f_name]['y_hat'] += y_hat[valid_indices].cpu().numpy().tolist()


            # Process discrete features
            for f, mu in zip(model.dis_feature_list, pred_dis_mu):
                f_idx, f_name = f['idx'], f['name']
                f_msk = feature_mask[..., f_idx]  # (B, L)
                logit = mu  # (B, L, class_num_d_i)
                prob_avg = F.softmax(logit, dim=-1)  # (B, L, class_num_d_i)

                y = target_seq[:, :, f_idx, 0].long()  # (B, L)
                # y_hat = torch.argmax(prob_avg, dim=2)  # (B, L)
                y_hat_prob = prob_avg.gather(dim=2, index=y.unsqueeze(-1)).squeeze(-1)  # (B, L)

                valid_indices = ~f_msk
                data_dict[f_name]['y'] += y[valid_indices].cpu().numpy().tolist()
                data_dict[f_name]['y_hat_prob'] += y_hat_prob[valid_indices].cpu().numpy().tolist()
                data_dict[f_name]['y_hat'] += torch.argmax(prob_avg, dim=2)[valid_indices].cpu().numpy().tolist()

    # Compile results into DataFrames
    result = {}
    for f_name in data_dict.keys():
        df = pd.DataFrame(data_dict[f_name])
        df['label'] = all_labels
        result[f_name] = df

    # # Optionally save the results
    # from utils.util import dir_check
    # result_path = f'/data/shurui/D-Transformer/output/predictions/pred_{params["model_id"]}.pkl'
    # dir_check(result_path)
    # with open(result_path, 'wb') as f:
    #     pickle.dump(result, f)

    return result


def posthoc_uncertainty(model, test_loader, device, params, save2file, mode,UC_mode, show_progress=True):
    from utils.eval import Metric, AverageMeter
    from data.dataset import Denormalizer
    from timeit import default_timer as timer
    from utils.util import to_device, dict_merge, run, ws
    from tqdm import tqdm
    

    model.eval()
    time_start = timer()
    log_features = params.get('log_features', [])
    log_features += ['total']

    uc_types = []
    if UC_mode in ['all', 'au']:
        uc_types.append('au')
    if UC_mode in ['all', 'eu']:
        uc_types.append('eu')

    uc_dict = {}
    for f in model.feature_list:
        f_idx, f_name = f['idx'], f['name']
        for uc in uc_types:
            key = f'{f_name}-{uc}'
            uc_dict[key] = {'y':[], 'y_hat':[], 'y_hat_prob':[],'uc_score':[]}

    y_hat_eu_dict = {}
    for f in model.feature_list:
        f_idx, f_name = f['idx'], f['name']
        key = f'{f_name}-eu'
        y_hat_eu_dict[key] = {'y_hat':[]}

    with torch.no_grad():
        all_labels = []
        for batch in tqdm(test_loader, disable=not show_progress):

            input_seq, target_seq, position, u_emb, uid,labels = to_device(batch, device)
            # Collect labels
            all_labels.extend(labels)

            # pred_samples = model.sample(input_seq, position, u_emb, return_raw=False, n_sample=5)
            # get the target token, for calculating the loss
            target_token = target_seq[..., 1].long()  # (B, L, F)
            # print(f'target token: {target_token.shape}')
            feature_mask = target_token != UNKNOWN_TOKEN  # (B, L, F)
            # print(f'feature mask: {feature_mask}')

            target_con = target_seq[:, :, :model.d_c, 0]  # (B, L, d_c)
            if 'au' in uc_types:
                pred_con, log_var_con, pred_dis_mu, pred_dis_sigma = model.sample(input_seq, position, u_emb, return_raw=True, n_sample=params['T'])
            else:
                pred_con, pred_dis_mu = model.sample(input_seq, position, u_emb, return_raw=True, n_sample=params['T'])
                log_var_con = torch.zeros_like(pred_con)
                pred_dis_sigma = [torch.zeros_like(x) for x in pred_dis_mu]

            # T: means the number of samples
            # pred_con:    (B, T, L, d_c), tensor, e.g., (128, 5, 8, 4)
            # log_var_con: (B, T, L, d_c), tensor, e.g., (128, 5, 8, 4)
            # pred_dis_mu: list of d_d, each item is (B, T, L, class_num_d_i),    e.g., [(128,5,8,40), (128,5,8,7)]
            # pred_dis_sigma: list of d_d, each item is (B, T, L, class_num_d_i), e.g., [(128,5,8,40), (128,5,8,7)]

            # # Denormalize data
            # if params['norm']=='minmax':
            #     # number of continous features
            #     d_c = pred_con.shape[-1]
            #     stay_stat_path = params['norm_path']
            #     stat = pd.read_csv(stay_stat_path, index_col=0)
            #     denormalizer = Denormalizer(stat, params['need_norm_feature'], params['need_norm_feature_idx'], norm_type=params['norm']) # importance, need to give a parameter
            #     pred_con_denorm = denormalizer(pred_con)
            #     target_seq_denorm  = denormalizer(target_seq[:, :, :d_c, 0] )
            
            # I do not want to denormalize, hand code this part for now; will udpate this part with a parameter later
            d_c = pred_con.shape[-1]
            pred_con_denorm = pred_con
            # print(f'pred con: {pred_con_denorm.shape}')
            target_seq_denorm = target_seq[:, :, :d_c, 0]
            # print(f'target_seq: {target_seq_denorm.shape}')
            for f in model.con_feature_list:
                f_idx, f_name = f['idx'], f['name']
                y_hat = pred_con[..., f_idx]
                
                f_msk = feature_mask[..., f_idx]

                if 'eu' in uc_types:
                    key = f'{f_name}-eu'
                    f_eu = torch.mean(torch.square(y_hat), dim=1) - torch.square(torch.mean(y_hat, dim=1))
                    uc_dict[key]['uc_score'] += f_eu[~f_msk].cpu().numpy().tolist()
                    y = target_seq_denorm[..., f_idx]
                    y_hat_mean = torch.mean(pred_con_denorm, dim=1)[..., f_idx]
                    uc_dict[key]['y'] += y[~f_msk].cpu().numpy().tolist()
                    uc_dict[key]['y_hat'] += y_hat_mean[~f_msk].cpu().numpy().tolist()
                    uc_dict[key]['y_hat_prob'] += y[~f_msk].cpu().numpy().tolist() # for con_features, yhatprob=y, just as a placeholder
                    y_hat_eu_dict[key]['y_hat'] += y_hat.cpu().numpy().tolist()

                if 'au' in uc_types:
                    key = f'{f_name}-au'
                    log_var = log_var_con[..., f_idx]
                    var = torch.exp(log_var)
                    var_avg = torch.mean(var, dim=1)
                    uc_dict[key]['uc_score'] += var_avg[~f_msk].cpu().numpy().tolist()
                    y = target_seq_denorm[..., f_idx]
                    y_hat_mean = torch.mean(pred_con_denorm, dim=1)[..., f_idx]
                    uc_dict[key]['y'] += y[~f_msk].cpu().numpy().tolist()
                    uc_dict[key]['y_hat'] += y_hat_mean[~f_msk].cpu().numpy().tolist()
                    uc_dict[key]['y_hat_prob'] += y[~f_msk].cpu().numpy().tolist()



            for f, mu, sigma in zip(model.dis_feature_list, pred_dis_mu, pred_dis_sigma):
                f_idx, f_name = f['idx'], f['name']
                f_msk = feature_mask[..., f_idx]
                logit = mu
                prob = F.softmax(logit, dim=-1)
                prob_avg = torch.mean(prob, dim=1)

                if 'eu' in uc_types:
                    key = f'{f_name}-eu'
                    H_p = torch.sum(-prob_avg * torch.log(prob_avg), dim=-1)
                    uc_dict[key]['uc_score'] += H_p[~f_msk].cpu().numpy().tolist()
                    y = target_seq[:, :, f_idx, 0].long()
                    y_hat_prob = prob_avg.gather(dim=2, index=y.unsqueeze(-1)).squeeze(-1)
                    uc_dict[key]['y'] += y[~f_msk].cpu().numpy().tolist()
                    # uc_dict[key]['y_hat'] += y_hat_prob[~f_msk].cpu().numpy().tolist()
                    uc_dict[key]['y_hat_prob'] += y_hat_prob[~f_msk].cpu().numpy().tolist()
                    uc_dict[key]['y_hat'] += torch.argmax(prob_avg, dim=2)[~f_msk].cpu().numpy().tolist()

                if 'au' in uc_types:
                    key = f'{f_name}-au'
                    var = sigma * sigma
                    var_avg = torch.mean(var, dim=3)
                    var_avg = torch.mean(var_avg, dim=1)
                    uc_dict[key]['uc_score'] += var_avg[~f_msk].cpu().numpy().tolist()
                    y = target_seq[:, :, f_idx, 0].long()
                    y_hat_prob = prob_avg.gather(dim=2, index=y.unsqueeze(-1)).squeeze(-1)
                    uc_dict[key]['y'] += y[~f_msk].cpu().numpy().tolist()
                    # uc_dict[key]['y_hat'] += y_hat_prob[~f_msk].cpu().numpy().tolist()
                    uc_dict[key]['y_hat_prob'] += y_hat_prob[~f_msk].cpu().numpy().tolist()
                    uc_dict[key]['y_hat'] += torch.argmax(prob_avg, dim=2)[~f_msk].cpu().numpy().tolist()



    # check if uc dict is empty

    # Ensure all_labels is a list of tensors
    all_labels = [label.cpu().numpy() for label in all_labels]
    result = {}
    for f in model.feature_list:
        f_idx, f_name = f['idx'], f['name']
        for uc in uc_types:
            key = f'{f_name}-{uc}'
            df = pd.DataFrame(uc_dict[key])
            df['label'] = all_labels
            result[key] = df

    # # save result, so no need to recompute
    # from utils.util import dir_check
    # result_path = ws + f'output/uc_score/uc_{params["model_id"]}.pkl'
    # dir_check(result_path)
    # with open(result_path, 'wb') as f:
    #     pickle.dump(reuslt, f)
    # Save y_hat_dict separately
    y_hat_file_path = ws + f'/output/pred_uc/y_hat_EFYJ_trial2_.pkl'
    with open(y_hat_file_path, 'wb') as f:
        pickle.dump(y_hat_eu_dict, f)

    return result







