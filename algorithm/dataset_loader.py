# -*- coding: utf-8 -*-
import json, math
import os.path
import numpy as np
import pandas as pd

import torch
from data.dataset import Normalizer, POSITION_LIST
from torch.utils.data import Dataset


# load dataset, and padding
class EventDataset(Dataset):
    def __init__(self,
                 mode: str,
                 params: dict,
                 ):
        super().__init__()
        if mode not in ["train", "val", "test"]:
            raise ValueError
        path_key = {'train':'train_path', 'val':'val_path', 'test':'test_path'}[mode]
        path = params[path_key]
        self.data = np.load(path, allow_pickle=True).item()

        # read feature list
        data_dir =  os.path.dirname(path)
        feature_dict = json.load(open(data_dir + '/feature.json'))
        # Note: feature_dict is the feature in the dataset, which does not mean the features used for the training
        feature2idx = {f['name']: f['idx'] for f in feature_dict}
        use_feature_idx = [feature2idx[f['name']] for f in params['feature_list']]

        # only use
        self.data['X'] = [x[:, use_feature_idx] for x in self.data['X']]

        # evaluate the val and test for multiple times, so that we are evaluate a very average behavior
        if mode in ['val', 'test']:
            multiple_times = 5
            # multiple_times = 1
            print(mode, f'extend {multiple_times} times.')
            keys = self.data.keys()
            for k in keys:
                self.data[k]  = self.data[k] * multiple_times

        # feature dimension
        self.d_x = len(self.data['X'][0][0])
        print('Number of features:', self.d_x, "|| Featuer example:", self.data['X'][0][0], "|| Feature names:", [f['name'] for f in params['feature_list']])

        # load user embedding dict:
        # if 'trial2' in params['dataset']:
        #     from utils.util import ws
        #     u_path = ws + '/data/raw/trial2/user_feature/user_emb.json'
        # else:
        u_path = os.path.dirname(path) + '/user_emb.json'
        self.u_emb_dict = json.load(open(u_path))


    def __len__(self):
        return  len(self.data['X_len'])

    def __getitem__(self, index):
        X = self.data['X'][index]
        X_len = self.data['X_len'][index]
        uid = str(self.data['uid'][index])
        u_emb = np.array(self.u_emb_dict[uid])
        position = self.data['position'][index]
        return X, X_len, u_emb, position, uid


def pad_batch_3d(batch):
    """
    Pad the batch to the maximum length of the batch.

    Args:
        batch (list): the batch of arrays to pad, [(L1, F, 2), (L2, F, 2), ...].

    Returns:
        np.array: the padded array.
    """
    max_len = max([arr.shape[0] for arr in batch])
    padded_batch = np.stack((
        np.full((len(batch), max_len, batch[0].shape[1]), FEATURE_PAD, dtype=float),
        np.full((len(batch), max_len, batch[0].shape[1]), PAD_TOKEN, dtype=float)
    ), axis=-1)
    for i, arr in enumerate(batch):
        padded_batch[i, :arr.shape[0]] = arr

    return padded_batch

def pad_batch_2d(batch):
    """
    Pad the batch to the maximum length of the batch.

    Args:
        batch (list): the batch of arrays to pad, [(L1, 2), (L2, 2), ...].

    Returns:
        np.array: the padded array.
    """
    max_len = max([arr.shape[0] for arr in batch])
    padded_batch =   np.full((len(batch), max_len, batch[0].shape[1]), FEATURE_PAD, dtype=float)
    # padded_batch = np.stack((
    #     np.full((len(batch), max_len), FEATURE_PAD, dtype=float),
    #     # np.full((len(batch), max_len), FEATURE_PAD, dtype=float)
    # ), axis=-1)
    for i, arr in enumerate(batch):
        padded_batch[i, :arr.shape[0]] = arr

    return padded_batch

def pad_batch_1d(batch):
    """
    Pad the batch to the maximum length of the batch.

    Args:
        batch (list): the batch of arrays to pad, [(L1), (L2), ...].

    Returns:
        np.array: the padded array.
    """
    max_len = max([arr.shape[0] for arr in batch])
    padded_batch = np.full((len(batch), max_len), FEATURE_PAD, dtype=float)

    for i, arr in enumerate(batch):
        padded_batch[i, :arr.shape[0]] = arr

    return padded_batch

FEATURE_PAD = 0

KNOWN_TOKEN = 0
MASK_TOKEN = 1
UNKNOWN_TOKEN = 2
PAD_TOKEN = 3



class PretrainPadder:
    """
    collate function for the pre-training
    """
    def __init__(self, mode, event_mask_ratio = 0.2,  norm = 'minmax', norm_path = '', need_norm_feature = [], need_norm_feature_idx = []): #device,
        """
        Args:
            event_mask_ratio:
            norm: choose the normalization
        """

        self.mode = mode # mode: train, val, test. They can have different mask mechanism

        self.event_mask_ratio = event_mask_ratio

        self.need_norm_feature = need_norm_feature
        self.need_norm_feature_idx = need_norm_feature_idx

        # Load the statictics information
        self.norm = norm
        if norm == 'minmax' and norm_path=='':
            raise f"norm path should not be empty for norm type:{norm}"

        if self.norm == 'minmax':
            assert len(self.need_norm_feature) != 0, "the need_norm_feature should not be empty"
            assert len(self.need_norm_feature_idx) != 0, "the need_norm_feature idx should not be empty"
            assert len(self.need_norm_feature) == len(self.need_norm_feature_idx), "len(need_norm_feature) should equal need_norm_feature_idx"
            stat = pd.read_csv(norm_path, index_col=0)

            #init normalizer
            self.normalizer = Normalizer(stat, feat_names=self.need_norm_feature, feat_cols=self.need_norm_feature_idx, norm_type = 'minmax')





    def __call__(self, raw_batch):
        # A function for padding the provided raw batch into a standard array.
        """

        Args:
            raw_batch (list): each item is a pd.DataFrame representing one trajectory.

        Returns:
            np.array: the padded input array, with shape (B, L, F, 2)
            np.array: the padded output array, with shape (B, L, F, 2)
            np.array: the padded dual-layer positions, with shape (B, L, 2)
        """
        position2idx = {f['name']:f['idx']  for f in POSITION_LIST}

        input_batch, output_batch, pos_batch, user_emb_batch, uid_batch = [], [], [], [], []
        for X, X_len, u_emb, position, uid in raw_batch:
            valid_len = X.shape[0]
            prompt_arr = []
            trg_arr = []

            if self.mode == 'train':
                mask_idxs = sorted(set(np.random.choice(valid_len, math.ceil(valid_len * self.event_mask_ratio), replace=False)))

            elif self.mode in ['val', 'test']:
                np.random.seed(2019) # note: import to make sure the val data are the same at each epoch

                # There are three mask mechanism in validation data
                val_mask_mode = 'random_mask' # 'random_rask_last_day', 'mask_middle_event'
                if val_mask_mode == 'random_rask_last_day':
                    # for val and test, we only mask the event at the last day, to avoid the data leakage.
                    # (Since we use the sliding window to construct the val and test dataset, that means the input seqnece
                    # in val and test may contain events in the train dataset, especially when the time window get larger)
                    day_pos = position[position2idx['day_pos']]
                    mask_idx_candidates = [i for i in range(valid_len) if day_pos[i] == day_pos[-1]]
                    mask_idxs = sorted(set(np.random.choice(mask_idx_candidates, 1, replace=False))) # only mask one for val and test
                elif val_mask_mode == 'random_mask':
                    mask_idxs = sorted(set(np.random.choice(valid_len, 1, replace=False))) # only mask one for val and test
                elif val_mask_mode == 'mask_middle_event':
                    pass # todo
            else:
                raise  NotImplementedError

            # conduct the normalization, note that only continous features need to be normalized
            if self.norm in ['minmax']:
                X = self.normalizer(X)


            for i, x in enumerate(X):
                if i in mask_idxs:
                    mask_values = np.ones_like(x) * FEATURE_PAD
                    # add mask tokens to input
                    x_token = np.full_like(x, MASK_TOKEN)
                    arr = np.stack((mask_values, x_token), axis=-1)
                    prompt_arr.append(arr)

                    # add data to target
                    x_token = np.full_like(x, UNKNOWN_TOKEN)
                    arr = np.stack((x, x_token), axis=-1)
                    trg_arr.append(arr)
                else:
                    x_token = np.full_like(x, KNOWN_TOKEN)
                    arr = np.stack((x, x_token), axis=-1)
                    prompt_arr.append(arr)

                    # add data to target
                    x_token = np.full_like(x, KNOWN_TOKEN)
                    arr = np.stack((x, x_token), axis=-1)
                    trg_arr.append(arr)

            prompt_arr = np.stack(prompt_arr, axis=0)
            trg_arr = np.stack(trg_arr, axis=0)

            input_batch.append(prompt_arr)
            output_batch.append(trg_arr)
            # pos_batch.append(position)
            pos_batch.append(position.T)
            user_emb_batch.append(u_emb)
            uid_batch.append(int(uid))


        input_batch = torch.tensor(pad_batch_3d(input_batch)).float()#.to(self.device)
        output_batch = torch.tensor(pad_batch_3d(output_batch)).float()#.to(self.device)
        # pos_batch = torch.tensor(pad_batch_1d(pos_batch)).long()#.to(self.device)
        pos_batch = torch.tensor(pad_batch_2d(pos_batch)).long()  # .to(self.device)
        user_emb_batch = torch.tensor(pad_batch_1d(user_emb_batch)).float()#.to(self.device)
        uid_batch = torch.from_numpy(np.array(uid_batch))#.to(self.device)
        return input_batch, output_batch, pos_batch, user_emb_batch, uid_batch


