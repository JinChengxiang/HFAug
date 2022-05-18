#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author: jcx
@contact: 2112103081@zjut.edu.cn
@file: load_data.py
@time: 2022/4/7 13:49
@desc:
'''
import csv
import json
import sys
import os.path as osp
from itertools import repeat

import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, StratifiedKFold
from torch_sparse import coalesce as coalesce_fn, SparseTensor
from torch_geometric.data import Data
from torch_geometric.io import read_txt_array
from torch_geometric.utils import remove_self_loops
import random

from tqdm import tqdm


def my_graph(folder, dataset):
    names = ['node_attributes', 'node_label']
    items = [read_file(folder, name) for name in names]
    node_attributes, node_label = items
    edge_index = read_file(folder, 'A').t()
    indices = []
    for i in range(2):
        index = (node_label == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    m = np.array(list(range(0, 382)))
    n = np.array([1] * 191 + [0] * 191)
    skf = StratifiedKFold(n_splits=5, random_state=2022, shuffle=True)
    train_masks = []
    val_masks = []
    test_masks = []
    y_trains = []
    y_vals = []
    y_tests = []
    kfold_train_test = []
    for train_index, test_index in skf.split(m, n):
        kfold_train_test.append([train_index, test_index])
    for i, item in enumerate(kfold_train_test):
        train_index, test_index = item[0], item[1]
        val_index = np.array([])

        for val_index1 in random.sample(list(train_index)[:153], 38):
            val_index = np.append(val_index, int(val_index1))
        for val_index2 in random.sample(list(train_index)[153:], 38):
            val_index = np.append(val_index, int(val_index2))

        for value in val_index:
            b = list(train_index)
            train_index = np.delete(train_index, b.index(value))

        y_trains.append(train_index)
        y_vals.append(val_index)
        y_tests.append(test_index)
        train_mask = index_to_mask(train_index, size=node_label.size(0))
        val_mask = index_to_mask(val_index, size=node_label.size(0))
        test_mask = index_to_mask(test_index, size=node_label.size(0))
        train_masks.append(train_mask)
        val_masks.append(val_mask)
        test_masks.append(test_mask)

    data = Data(x=node_attributes, edge_index=edge_index, y=node_label)
    data.x_HFAug = node_attributes
    data.train_index = y_trains
    data.val_index = y_vals
    data.test_index = y_tests
    data.train_mask = train_masks
    data.val_mask = val_masks
    data.test_mask = test_masks

    if dataset == 'raw':
        return data
    else:
        final_data = aggr_mp(data, dataset)
        return final_data


def read_file(folder, name):
    if name == 'node_attributes':
        path = osp.join(folder, f'{name}.txt')
        return read_txt_array(path, sep=',', dtype=torch.float)
    else:
        path = osp.join(folder, f'{name}.txt')
        return read_txt_array(path, sep=',', dtype=torch.long)


def index_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = 1
    return mask


def row_csv2dict(name):
    dict_club = {}
    with open(name) as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            dict_club[row[0]] = row[1]
    return dict_club


def aggr_mp(data, dataset):
    node_dic = row_csv2dict(f'../../data/pre/node_b.csv')
    if dataset == 'all':
        with open(f'../../data/data_het/metapath/eoa_ca_eoa_GNN.txt', 'r') as f:
            with open(f'../../data/data_het/metapath/ca_ca_eoa_GNN.txt', 'r') as g:
                for rows in tqdm(f.readlines()):
                    row = rows.strip().split(',')
                    if len(row) == 4:
                        data.x[int(node_dic.get(row[0]))] = (data.x_HFAug[int(node_dic.get(row[0]))] +
                                                             data.x_HFAug[int(node_dic.get(row[1]))] +
                                                             data.x_HFAug[int(node_dic.get(row[2]))] +
                                                             data.x_HFAug[int(node_dic.get(row[3]))])
                    elif len(row) == 3:
                        data.x[int(node_dic.get(row[0]))] = (data.x_HFAug[int(node_dic.get(row[0]))] +
                                                             data.x_HFAug[int(node_dic.get(row[1]))] +
                                                             data.x_HFAug[int(node_dic.get(row[2]))])
                    elif len(row) == 2:
                        data.x[int(node_dic.get(row[0]))] = (data.x_HFAug[int(node_dic.get(row[0]))] +
                                                             data.x_HFAug[int(node_dic.get(row[1]))])
                for rows in tqdm(g.readlines()):
                    row = rows.strip().split(',')
                    if len(row) == 4:
                        data.x[int(node_dic.get(row[0]))] = (data.x_HFAug[int(node_dic.get(row[0]))] +
                                                             data.x_HFAug[int(node_dic.get(row[1]))] +
                                                             data.x_HFAug[int(node_dic.get(row[2]))] +
                                                             data.x_HFAug[int(node_dic.get(row[3]))])
                    elif len(row) == 3:
                        data.x[int(node_dic.get(row[0]))] = (data.x_HFAug[int(node_dic.get(row[0]))] +
                                                             data.x_HFAug[int(node_dic.get(row[1]))] +
                                                             data.x_HFAug[int(node_dic.get(row[2]))])
                    elif len(row) == 2:
                        data.x[int(node_dic.get(row[0]))] = (data.x_HFAug[int(node_dic.get(row[0]))] +
                                                             data.x_HFAug[int(node_dic.get(row[1]))])
                return data
    else:
        with open(f'../../data/data_het/metapath/{dataset}.txt', 'r') as f:
            for rows in tqdm(f.readlines()):
                row = rows.strip().split(',')
                if len(row) == 4:
                    data.x[int(node_dic.get(row[0]))] = (data.x_HFAug[int(node_dic.get(row[0]))] +
                                                         data.x_HFAug[int(node_dic.get(row[1]))] +
                                                         data.x_HFAug[int(node_dic.get(row[2]))] +
                                                         data.x_HFAug[int(node_dic.get(row[3]))])
                elif len(row) == 3:
                    data.x[int(node_dic.get(row[0]))] = (data.x_HFAug[int(node_dic.get(row[0]))] +
                                                         data.x_HFAug[int(node_dic.get(row[1]))] +
                                                         data.x_HFAug[int(node_dic.get(row[2]))])
                elif len(row) == 2:
                    data.x[int(node_dic.get(row[0]))] = (data.x_HFAug[int(node_dic.get(row[0]))] +
                                                         data.x_HFAug[int(node_dic.get(row[1]))])
            return data










