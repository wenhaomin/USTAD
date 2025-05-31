# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from timeit import default_timer as timer
from collections import defaultdict

from algorithm.dataset_loader import UNKNOWN_TOKEN


def masked_mean(values, mask):
    values[mask] = 0
    values = values.sum()
    # values = values.masked_fill(mask, 0).sum() # no masked fill function for np.array
    count = (~mask).sum()
    return values / count

def masked_mae_np(y_true, y_pred, mask):
    mae = np.abs(y_true - y_pred)
    masked_mae = masked_mean(mae, mask)
    return masked_mae

def masked_mse_np(y_true, y_pred, mask):
    mse = (y_true - y_pred) ** 2
    masked_mse = masked_mean(mse, mask)
    return masked_mse


def masked_mape_np(y_true, y_pred, mask):
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.abs((y_pred - y_true) / y_true)
        mape = np.nan_to_num(mape)
        # mape = np.nan_to_num(mask * mape)
        masked_mape = masked_mean(mape, mask)
        return masked_mape * 100


def masked_acc_np(y_true, y_pred, mask):
    correct = (y_true == y_pred).astype(int)
    masked_accuracy = masked_mean(correct, mask)
    return masked_accuracy


def time_to_str(t, mode='min'):
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d min'%(hr,min)

    elif mode=='sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%2d min %02d sec'%(min,sec)
    else:
        raise NotImplementedError


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Metric(object):
    """
    computes the metric for each feature
    """
    def __init__(self, feature_lst: list):
        """
        Args:
            feature_lst: list
                e.g.,  [
                            {'idx': 0, 'name': 'lat', 'type': 'continuous'},
                            {'idx': 1, 'name': 'lng', 'type': 'continuous'},
                            {'idx': 2, 'name': 'start_time_minute', 'type': 'continuous'},
                            {'idx': 3, 'name': 'stay_time', 'type': 'continuous'},
                            {'idx': 4, 'name': 'poi', 'type': 'discrete'}
                        ]
        """
        self.time_start = timer()
        # self.best_metrics = {'mae': np.inf, 'rmse': np.inf, 'mape': np.inf, 'epoch': np.inf}

        self.feature_lst = feature_lst
        self.continuous_fea = [f for f in feature_lst if f['type'] == 'continuous']
        self.discrete_fea = [f for f in feature_lst if f['type'] == 'discrete']

        self.continuous_metrics = ['mae', 'mse', 'rmse', 'mape']
        self.discrete_metrics = ['acc']

        self.metrics = {}

        # init continuous metrics
        for f in self.continuous_fea:
            f_name = f['name']
            for m_name in self.continuous_metrics:
                self.metrics[f'{f_name}-{m_name}'] = AverageMeter()

        # init discrete metrics
        for f in self.discrete_fea:
            f_name = f['name']
            for m_name in self.discrete_metrics:
                self.metrics[f'{f_name}-{m_name}'] = AverageMeter()


        # addtional information for prediction analysis
        # for error bar
        self.error_bar = defaultdict(list)

        # for discrete feature prediction confusion matrix
        self.descrete_pred_target = defaultdict(list)




    def update_metrics(self, y_true, y_pred):
        # both y_true and y_pred should be numpy, which, can contain both continous and discrete features
        # :param y_true: (B, L, F, 2) where y_true[:, :, :, 1] is the target token
        # :param y_pred: (B, L, F)


        target_token = y_true[..., 1] # (B, L, d_x)
        target_value = y_true[..., 0] # (B, L, d_x)


        # calculate continuous feature metrics
        for f in self.continuous_fea:
            idx, f_name = f['idx'], f['name']

            y_true_i, y_pred_i = target_value[:, :, idx], y_pred[:, :, idx]
            token_i = target_token[:, :, idx]

            token_mask =  token_i != UNKNOWN_TOKEN # mask = True, means no need to evaluate the location,
            zero_mask = y_true_i == 0 #do not evaluate when the true value is zero, because it will cost the mape to inf
            mask = np.logical_or(token_mask, zero_mask)

            cnt = (~mask).sum()


            mae = masked_mae_np(y_true_i, y_pred_i, mask)
            mse = masked_mse_np(y_true_i, y_pred_i, mask)
            rmse = mse ** 0.5
            mape = masked_mape_np(y_true_i, y_pred_i, mask)
            self.metrics[f'{f_name}-mae'].update(mae, cnt)
            self.metrics[f'{f_name}-mse'].update(mse, cnt)
            self.metrics[f'{f_name}-rmse'].update(rmse, cnt)
            self.metrics[f'{f_name}-mape'].update(mape, cnt)

            # for the error bar plot:
            mae = np.abs(y_true_i - y_pred_i)
            mae_lst = mae[~mask].tolist()
            self.error_bar[f_name] += mae_lst


        # calculate discrete feature metrics
        for f in self.discrete_fea:
            idx, f_name =  f['idx'], f['name']

            y_true_i, y_pred_i = target_value[:, :, idx], y_pred[:, :, idx] # y_true_i: (B, L)
            token_i = target_token[:, :, idx] # (B, L)

            mask = token_i != UNKNOWN_TOKEN  # mask = True, means no need to evaluate the location
            cnt = (~mask).sum()

            acc = masked_acc_np(y_true_i, y_pred_i, mask)
            self.metrics[f'{f_name}-acc'].update(acc, cnt)

            # record the prediction and target
            self.descrete_pred_target[f'{f_name}-pred'] += y_pred_i[~mask].astype(int).tolist()
            self.descrete_pred_target[f'{f_name}-target'] += y_true_i[~mask].astype(int).tolist()



    def to_str(self):
        """For print"""
        dict = {k: round(v.avg,2) for k, v in self.metrics.items()}
        return str(dict)

    def to_dict(self):
        dict = {k:  round(v.avg,4) for k,v in self.metrics.items()}
        return dict




