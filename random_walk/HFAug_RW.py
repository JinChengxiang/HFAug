import csv
import os
import numpy as np
import pandas as pd
from tqdm import tqdm


def mkdir(path):
    """
    :param path:
    :return:
    """
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
    return path


def row_csv2dict():
    dict_club = {}
    with open(r'../data/pre/node_b.csv') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            dict_club[row[0]] = row[1]
    return dict_club


def add_metapath_CA_DW(path, method, dimension):
    fea = pd.read_csv(path, index_col=False)
    columns_list = [str(x) for x in range(dimension)]
    columns_list.append('node')
    taget_node = pd.DataFrame(columns=columns_list)
    with open(f'../data/data_het/metapath/ca_ca_eoa_.txt', 'r') as metapath:
        for row in tqdm(metapath.readlines()):
            nodes = row.strip().split(',')   # to_list
            dict_club = row_csv2dict()
            node = []
            for n in nodes:
                node.append(dict_club.get(n))
            f = fea[(fea['node']==int(node[0]))].values.tolist()
            dic = {}
            for i in range(dimension):
                dic[f'{i}'] = f[0][i]
                dic['node'] = f[0][dimension]
            for n in range(1, len(node)):
                g = fea[(fea['node'] == int(node[n]))].values.tolist()
                for i in range(dimension):
                    dic[f'{i}'] += g[0][i]
            taget_node = taget_node.append(dic, ignore_index=True)
        # print(taget_node)
    taget_node['label'] = [1] * 191 + [0] * 191
    midpath = mkdir(f'./{method}/dimension_{dimension}/')
    taget_node.to_csv(midpath + f'G_emb_ca_ca_eoa.csv', index=False)


def add_metapath_CA_N2V(path, method, p, q, dimension):
    fea = pd.read_csv(path, index_col=False)
    columns_list = [str(x) for x in range(dimension)]
    columns_list.append('node')
    taget_node = pd.DataFrame(columns=columns_list)
    with open(f'../data/data_het/metapath/ca_ca_eoa_.txt', 'r') as metapath:
        for row in tqdm(metapath.readlines()):
            nodes = row.strip().split(',')   # to_list
            dict_club = row_csv2dict()
            node = []
            for n in nodes:
                node.append(dict_club.get(n))
            f = fea[(fea['node']==int(node[0]))].values.tolist()
            dic = {}
            for i in range(dimension):
                dic[f'{i}'] = f[0][i]
                dic['node'] = f[0][dimension]
            for n in range(1, len(node)):
                g = fea[(fea['node'] == int(node[n]))].values.tolist()
                for i in range(dimension):
                    dic[f'{i}'] += g[0][i]
            taget_node = taget_node.append(dic, ignore_index=True)
        # print(taget_node)
    taget_node['label'] = [1]*191 + [0]*191
    midpath = mkdir(f'./{method}/dimension_{dimension}/{p}_{q}/')
    taget_node.to_csv(midpath + f'G_emb_ca_ca_eoa.csv', index=False)


def add_metapath_EOA_DW(path, method, dimension):
    fea = pd.read_csv(path, index_col=False)
    columns_list = [str(x) for x in range(dimension)]
    columns_list.append('node')
    taget_node = pd.DataFrame(columns=columns_list)
    with open(f'../data/data_het/metapath/eoa_ca_eoa_.txt', 'r') as metapath:
        for row in tqdm(metapath.readlines()):
            nodes = row.strip().split(',')   # to_list
            dict_club = row_csv2dict()
            node = []
            for n in nodes:
                node.append(dict_club.get(n))
            if len(node) > 1:
                f = fea[(fea['node']==int(node[1]))].values.tolist()
                dic = {}
                for i in range(dimension):
                    dic[f'{i}'] = f[0][i]
                    dic['node'] = f[0][dimension]
            for n in range(len(node)):
                if n != 1:
                    g = fea[(fea['node'] == int(node[n]))].values.tolist()
                    for i in range(dimension):
                        dic[f'{i}'] += g[0][i]
            taget_node = taget_node.append(dic, ignore_index=True)
        taget_node = taget_node[(taget_node['node'] <= 381)]
        taget_node = taget_node.drop_duplicates(subset='node')
    taget_node['label'] = [1] * 191 + [0] * 191
    midpath = mkdir(f'./{method}/dimension_{dimension}/')
    taget_node.to_csv(midpath + f'G_emb_eoa_ca_eoa.csv', index=False)


def add_metapath_EOA_N2V(path, method, p, q,dimension):
    fea = pd.read_csv(path, index_col=False)
    columns_list = [str(x) for x in range(dimension)]
    columns_list.append('node')
    taget_node = pd.DataFrame(columns=columns_list)
    with open(f'../data/data_het/metapath/eoa_ca_eoa_.txt', 'r') as metapath:
        for row in tqdm(metapath.readlines()):
            nodes = row.strip().split(',')   # to_list
            dict_club = row_csv2dict()
            node = []
            for n in nodes:
                node.append(dict_club.get(n))
            if len(node) > 1:
                f = fea[(fea['node']==int(node[1]))].values.tolist()
                dic = {}
                for i in range(dimension):
                    dic[f'{i}'] = f[0][i]
                    dic['node'] = f[0][dimension]
            # for n in range(2, len(node)):
            for n in range(len(node)):
                if n != 1:
                    g = fea[(fea['node'] == int(node[n]))].values.tolist()
                    for i in range(dimension):
                        dic[f'{i}'] += g[0][i]
            taget_node = taget_node.append(dic, ignore_index=True)
        taget_node = taget_node[(taget_node['node'] <= 381)]
        taget_node = taget_node.drop_duplicates(subset='node')
    taget_node['label'] = [1]*191 + [0]*191
    midpath = mkdir(f'./{method}/dimension_{dimension}/{p}_{q}/')
    taget_node.to_csv(midpath + f'G_emb_ca_ca_eoa.csv', index=False)


if __name__ == '__main__':
    # methods = ['deepwalk', 'node2vec']
    methods = ['deepwalk']
    p_=[0.5, 1, 2]
    q_=[0.5, 1, 2]
    dimensions = [128]

    for method in methods:
        for dimension in dimensions:
            if method == 'node2vec':
                for p in p_:
                    for q in q_:
                        path = rf'./{method}/dimension_{dimension}/{p}_{q}/G_emb_raw.csv'
                        add_metapath_CA_N2V(path, method, p, q, dimension)
                        add_metapath_EOA_N2V(path, method, p, q,dimension)

            else:
                path = rf'./{method}/dimension_{dimension}/G_emb_raw.csv'
                add_metapath_CA_DW(path, method, dimension)
                add_metapath_EOA_DW(path, method,dimension)
