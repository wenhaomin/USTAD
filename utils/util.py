import json

# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os, string
from tqdm import  tqdm
from datetime import datetime
from easydict import EasyDict as edict

def get_workspace():
    """
    get the workspace path
    :return:
    """
    cur_path = os.path.abspath(__file__)
    file = os.path.dirname(cur_path)
    file = os.path.dirname(file)
    return file
ws =  get_workspace()


"""record the platform for resoures we have"""
#  different platforms may have different output directory strategy
Platforms = edict()
Platforms.la3 = 'Linux-5.4.0-139-generic-x86_64-with-glibc2.31' # for la3
Platforms.psc = 'Linux-4.18.0-477.27.1.el8_8.x86_64-x86_64-with-glibc2.28' # for psc


def get_datetime_key():
    """ Get a string key based on current datetime. """
    return 'D' + datetime.now().strftime("%Y_%m_%dT%H_%M_%S_") + get_random_string(4)

def get_random_string(length):
    letters = string.ascii_uppercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

def dir_check(path):
    """
    check weather the dir of the given path exists, if not, then create it
    """
    import os
    dir = path if os.path.isdir(path) else os.path.split(path)[0]
    if not os.path.exists(dir): os.makedirs(dir)
    return path

def to_device(batch, device):
    batch = [x.to(device) for x in batch]
    return batch

def setup_seed(seed):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True


def dict_merge(dict_list = []):
    dict_ =  {}
    for dic in dict_list:
        assert isinstance(dic, dict), "object is not a dict!"
        dict_ = {**dict_, **dic}
    return dict_

# trial2 data comes from Los Angeles (UTM region: 10)
# crs code of a region: (for LA search WGS_1984_UTM_Zone_10N)
#Projected Coordinate Systems: https://developers.arcgis.com/javascript/3/jshelp/pcs.htm
# Geographic Coordinate Systems: https://developers.arcgis.com/javascript/3/jshelp/gcs.htm
def gps2xy(lng, lat, region = 'LA'):  # Universal Transverse Mercator
    # convert GPS point to x, y
    from pyproj import Transformer
    utm_region = {
        'LA': 'EPSG:32610',
        'Jordan': 'EPSG:32636'
    }
    tf = Transformer.from_crs("EPSG:4326", utm_region[region])
    x, y = tf.transform(lat, lng)
    return x, y

def gps_to_normed_xy(lngs, lats, region, scale=1000):
    """
    convert given gps points to normalized x,y centered at the middle point
    Args:
        lngs: list of lng
        lats: list of lat
        region: str
        scale: float, scale the returned x,y. When scale = 1000, means 1 km
    Returns:
    """
    xs, ys = gps2xy(lngs, lats, region)

    # calculate middle point
    x_middle = (min(xs) + max(xs)) / 2
    y_middle = (min(ys) + max(ys)) / 2

    # convert others points into (x, y) coordinate centered at the middle point
    x_norm = [ round((x-x_middle)/scale, 3) for x in xs]
    y_norm = [ round((y-y_middle)/scale, 3) for y in ys]

    return x_norm, y_norm

def min_max_norm(df, column):
    """
    Min-Max normalizes a given column in a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the column to normalize.
    column (str): The name of the column to normalize.

    Returns:
    pd.Series: the normalized column.
    """
    min_val = df[column].min()
    max_val = df[column].max()
    # df[column] = (df[column] - min_val) / (max_val - min_val)
    normed = (df[column] - min_val) / (max_val - min_val)
    return normed

def whether_stop(metric_lst = [], n=2, mode='maximize'):
    '''
    For fast parameter search, judge wether to stop the training process according to metric score
    n: Stop training for n consecutive times without rising
    mode: maximize / minimize
    '''
    if len(metric_lst) < 1:return False # at least have 2 results.
    if mode == 'minimize': metric_lst = [-x for x in metric_lst]
    max_v = max(metric_lst)
    max_idx = 0
    for idx,v in enumerate(metric_lst):
        if v == max_v:max_idx = idx
    return max_idx < len(metric_lst) - n

class EarlyStop():
    """
    For training process, early stop strategy
    """
    def __init__(self, mode='maximize', patience = 1):
        self.mode = mode
        self.patience =  patience
        self.metric_lst = []
        self.stop_flag = False
        self.best_epoch = -1 # the best epoch
        self.is_best_change = False # whether the best change compare to the last epoch

    def append(self, x):
        self.metric_lst.append(x)
        #update the stop flag
        self.stop_flag = whether_stop(self.metric_lst, self.patience, self.mode)
        #update the best epoch
        best_epoch = self.metric_lst.index(max(self.metric_lst)) if self.mode == 'maximize'  else self.metric_lst.index(min(self.metric_lst))
        if best_epoch != self.best_epoch:
            self.is_best_change = True
            self.best_epoch = best_epoch#update the wether best change flag
        else:
            self.is_best_change = False
        return self.is_best_change

    def best_metric(self):
        if len(self.metric_lst) == 0:return -1
        else:
            return self.metric_lst[self.best_epoch]


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

#----- Training Utils----------
import argparse
import random, torch
from torch.optim import Adam
from pprint import pprint
from torch.utils.data import DataLoader


def get_common_params():
    # Training settings
    parser = argparse.ArgumentParser(description='Entry Point of the code')
    parser.add_argument('--is_test', type=bool, default=False, help='test the code')

    # dataset
    # parser.add_argument('--min_task_num', type=int, default=0, help = 'minimal number of task')
    # parser.add_argument('--max_task_num',  type=int, default=25, help = 'maxmal number of task')
    parser.add_argument('--dataset', default='trial1', type=str, help='training dataset')
    # parser.add_argument('--pad_value', type=int, default=24, help='logistics: max_num - 1, pd: max_num + 1')

    ## for model setting
    parser.add_argument('--d_h', default=32, type=int, help='hidden dimension of the data')


    ## common settings for deep models
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 256)')
    parser.add_argument('--num_epoch', type=int, default=300, help='number of epochs to train (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 1e-4)')
    parser.add_argument('--seed', type=int, default=2021, metavar='S', help='random seed (default: 6)')
    parser.add_argument('--wd', type=float, default=1e-5, help='weight decay (default: 1e-5)')
    parser.add_argument('--early_stop', type=int, default=5, help='early stop at')
    parser.add_argument('--early_stop_metric', type=str, default='loss', help='metric for early stop') #stay_duration-rmse
    parser.add_argument('--early_stop_start_epoch', type=int, default=0, help='the epoch that starts the early stop')

    parser.add_argument('--workers', type=int, default=16, help='number of data loading workers (default: 4)')
    parser.add_argument('--is_eval', type=str, default=False, help='True means load existing model')
    parser.add_argument('--model_path', type=str, default=None, help='best model path in logistics')
    return parser

# import nni, time
import time
def train_val_test(train_loader, val_loader, test_loader, model, device, process_batch, test_model, params, save2file):
    # torch.manual_seed(params['seed'])
    model_path = params.get('model_path', None)
    resume_train = params.get('resume_train', False)
    if model_path != None and not resume_train:
        try:
            print('Loaded model from:', model_path)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            print('Model loaded !!!')
        except:
            print('Load model failed')
        test_result = test_model(model, test_loader, device, params, save2file, 'test')
        print('\n-------------------------------------------------------------')
        print(f'{params["model"]} Evaluation in test:', test_result.to_str())
        # nni.report_final_result(test_result.to_dict()['krc'])
        return params

    if resume_train:
        try:
            print('Loaded model from:', model_path)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            print('Model loaded, resume training start')
        except:
            print('Load model failed')

    model.to(device)
    optimizer = Adam(model.parameters(), lr=params['lr'], weight_decay=params['wd'])

    early_stop = EarlyStop(mode='minimize', patience=params['early_stop']) # [note] mode of  early stop should depend on the metric

    # get experiment datetime key, used as experiment id
    date_key = get_datetime_key()

    # model path
    model_name = model.model_file_name()
    # model_path = ws + f'/data/dataset/{params["dataset"]}/model/{params["model"]}/{model_name}-{date_key}.pkl'
    model_path = ws + f'/output/model/{model_name}-{date_key}.pkl'
    params['model_path'] = model_path

    # log path
    trial_name = model_name + "_" + '+'.join([str(params[k]) for k in ['batch_size', 'lr', 'wd']]) +  "_" + date_key
    log_path = ws +  f"/output/log/{trial_name}.json"
    params['log_path'] = log_path

    # forecast path
    forecast_path = ws + '/output/forecast/' + trial_name + '.pkl'
    params['forecast_path'] = forecast_path


    dir_check(log_path)
    dir_check(model_path)
    dir_check(forecast_path)
    train_log = []
    log_features = params.get('log_features', [])
    best_val_result = {} # used to record the results in val with the best epoch
    for epoch in range(params['num_epoch']):
        if early_stop.stop_flag: break
        train_epoch_log = {'epoch': epoch, 'mode': 'train', 'total': 0} # 'total': means total loss
        feature_loss = {f: AverageMeter() for f in log_features}

        postfix = {"epoch": epoch, "loss": 0.0, "current_loss": 0.0}
        with tqdm(train_loader, total=len(train_loader), postfix=postfix) as t:
            ave_loss = None
            model.train()
            for i, batch in enumerate(t):
                loss, loss_value_dict = process_batch(batch, model, device, params)

                # update feature loss and total loss
                for f in log_features:
                    if f not in loss_value_dict.keys():
                        print(f'Feature {f} not in return loss dict, current keys of loss dict: {loss_value_dict.keys()}')
                    else:
                        feature_loss[f].update(loss_value_dict[f])
                        postfix[f'{f}_loss'] = feature_loss[f].avg

                if ave_loss is None:
                    ave_loss = loss.item()
                else:
                    ave_loss = ave_loss * i / (i + 1) + loss.item() / (i + 1)

                postfix["loss"] = ave_loss
                postfix["current_loss"] = loss.item()
                t.set_postfix(**postfix)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # write log
        # add value to epoch log
        train_epoch_log['total'] = ave_loss
        for k, v in feature_loss.items():
            train_epoch_log[k] = v.avg
        # add the parameters into in log
        for k, v in params.items():
            if type(v) not in [int, str, bool]: continue
            train_epoch_log[k] = v
        # add epoch log to train log
        train_log.append(train_epoch_log)

        if params['is_test']: break


        # Let the model first train early_stop_start_epoch epochs, then start the early stop process
        # doing so allow the model to be fully trained as well as saving the training time
        if epoch < params['early_stop_start_epoch']: continue

        val_result, val_epoch_log = test_model(model, val_loader, device, params, save2file, 'val')# no need to save results on val；
        # add the parameters into log
        for k, v in params.items():
            if type(v) not in [int, str, bool]: continue
            val_epoch_log[k] = v

        val_epoch_log['epoch'] = epoch
        val_epoch_log['mode'] = 'val'
        train_log.append(val_epoch_log)

        early_stop_metric = params['early_stop_metric']
        if early_stop_metric != 'loss':
            is_best_change = early_stop.append(val_result.to_dict()[early_stop_metric])
            if is_best_change:
                print('value:', val_result.to_dict()[early_stop_metric], early_stop.best_metric())
                torch.save(model.state_dict(), model_path)
                print('best model saved to:', model_path)
                best_val_result = val_result
        elif early_stop_metric == 'loss':
            is_best_change = early_stop.append(val_epoch_log['total']) # here val_epoch_log['total'] means the loss in val, need to refine later
            if is_best_change:
                print('value', val_epoch_log['total'], early_stop.best_metric())
                torch.save(model.state_dict(), model_path)
                print('best model saved to:', model_path)
                best_val_result = val_result

    with open(log_path, 'w') as f:
        json.dump(train_log, f, indent=4)

    try:
        print('loaded model path:', model_path)
        model.load_state_dict(torch.load(model_path))
        print('best model loaded !')
    except:
        print('load best model failed')

    params['best_epoch'] = early_stop.best_epoch
    params['best_metric'] = early_stop.best_metric()

    test_result = test_model(model, test_loader, device, params, save2file, 'test')
    print('\n-------------------------------------------------------------')
    print('Best epoch: ', early_stop.best_epoch, f'|| Best {params["early_stop_metric"]}: {early_stop.best_metric()}')
    print(f'{params["model"]} Evaluation in test:', test_result.to_str())
    print('model_path: ', model_path)
    print('log_path: ', log_path)
    print('forecast_path: ', forecast_path)

    # update the best val results to params
    # temp = {f'val_{k}':v for k,v in best_val_result.to_dict().items()}
    temp = {f'val_{k}':v for k,v in best_val_result.items()}

    params = dict_merge([params, temp])

    params = dict_merge([params, test_result.to_dict()])

    return params


def get_dataset_path(params = {}):
    dataset = params['dataset']
    file = ws + f'/data/dataset/{dataset}'
    train_path = file + f'/train.npy'
    val_path = file + f'/val.npy'
    test_path = file + f'/test.npy'
    return train_path, val_path, test_path

# gpu_id = GPU().get_usefuel_gpu(max_memory=6000, condidate_gpu_id=[0,1,2,3,4,6,7,8])
# config.gpu_id = gpu_id
# if gpu_id != None:
#     cuda_id = GpuId2CudaId(gpu_id)
#     torch.cuda.set_device(f"cuda:{cuda_id}")
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def run(params, DATASET, PROCESS_BATCH, TEST_MODEL, collate_fn = None):


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    params['device'] = device

    setup_seed(params['seed'])
    torch.set_num_threads(params['workers'])

    params['train_path'], params['val_path'],  params['test_path'] = get_dataset_path(params)
    pprint(params)  # print the parameters

    if collate_fn is not None:
        train_collate_fn, val_collate_fn, test_collate_fn =  collate_fn
    else:
        train_collate_fn, val_collate_fn, test_collate_fn = None, None, None

    train_dataset = DATASET(mode='train', params=params)
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=False, collate_fn=train_collate_fn, num_workers=params['workers'] )  # num_workers=2,

    val_dataset = DATASET(mode='val', params=params)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False, collate_fn=val_collate_fn, num_workers=params['workers'])  # cfg.batch_size

    test_dataset = DATASET(mode='test', params=params)
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False, collate_fn=test_collate_fn, num_workers=params['workers'])#, collate_fn=collate_fn

    print(f'Number of Train:{len(train_dataset)} | Number of val:{len(val_dataset)} | Number of test:{len(test_dataset)}')

    model_save2file = params.get('model_save2file', None) # one can directly pass the model and save2file function to the parameter, without register in the utils
    if  model_save2file is not None:
        model, save2file = model_save2file
    # else:
    #     model, save2file = get_model_function(params['model'])
    model = model(params).float()
    result_dict = train_val_test(train_loader, val_loader, test_loader, model, device, PROCESS_BATCH, TEST_MODEL, params, save2file)
    params = dict_merge([params, result_dict])
    return params



def save2file_meta(params, file_name, head):
    def timestamp2str(stamp):
        utc_t = int(stamp)
        utc_h = utc_t // 3600
        utc_m = (utc_t // 60) - utc_h * 60
        utc_s = utc_t % 60
        hour = (utc_h + 8) % 24
        t = f'{hour}:{utc_m}:{utc_s}'
        return t
    import csv, time, os
    dir_check(file_name)
    if not os.path.exists(file_name):
        f = open(file_name, "w", newline='\n')
        csv_file = csv.writer(f)
        csv_file.writerow(head)
        f.close()
    # write_to_hdfs(file_name, head)
    with open(file_name, "a", newline='\n') as file:  #   linux:\n    windows:\r\n    mac:\r
        csv_file = csv.writer(file)
        # params['log_time'] = str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) #
        params['log_time'] = timestamp2str(time.time())
        data = [params[k] for k in head]
        csv_file.writerow(data)
        # write_to_hdfs(file_name, data)


if __name__ == '__main__':

    if 1: # test of gps to x,y
        lng, lat = -118.051398, 33.779514
        x, y = gps2xy(lng, lat)
        print(x, y)

        lngs, lats = [-118.051398, -118.051398], [33.779514, 33.779514]
        x, y = gps2xy(lngs, lats)
        print(x, y)
