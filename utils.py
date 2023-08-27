#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import torch
import numpy as np
import random

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

def random_splits(data, num_classes, percls_trn, val_lb, Flag=0):
    indices = []
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)

    if Flag == 0:
        rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(rest_index[:val_lb], size=data.num_nodes)
        data.test_mask = index_to_mask(
            rest_index[val_lb:], size=data.num_nodes)
    else:
        val_index = torch.cat([i[percls_trn:percls_trn+val_lb]
                               for i in indices], dim=0)
        rest_index = torch.cat([i[percls_trn+val_lb:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(val_index, size=data.num_nodes)
        data.test_mask = index_to_mask(rest_index, size=data.num_nodes)
    return data

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def geom_mask(dataset_name, data, split_id=0):
    name = dataset_name
    split_path = './splits'
    file_path = f'{split_path}/{name}_split_0.6_0.2_{split_id}.npz'
    splits_lst = np.load(file_path, allow_pickle=True)
    mask = {'train_mask':[], 'val_mask':[], 'test_mask':[]}
    for key in splits_lst:
        if not torch.is_tensor(splits_lst[key]):
            mask[key] = torch.as_tensor(splits_lst[key])
    train_mask = mask['train_mask']
    val_mask = mask['val_mask']
    test_mask = mask['test_mask']
    data.train_mask = train_mask.bool()
    data.val_mask = val_mask.bool()
    data.test_mask = test_mask.bool()
    return data