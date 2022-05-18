import pandas as pd
import numpy as np
import torch
import dgl
import tqdm

def construct_graph():
    f_CA = pd.read_csv('CA_feature.csv')
    f_EOA = pd.read_csv('EOA_feature.csv')
    CA_type = f_CA['Address'].to_list()
    EOA_type = f_EOA['Address'].to_list()
    CA_index_id_map = {int(i): x for i, x in enumerate(CA_type)}
    EOA_index_id_map = {int(i): x for i, x in enumerate(EOA_type)}

    call_eoa_ca1 = []
    call_eoa_ca2 = []
    with open('call_eoa_ca.txt') as f:
        for row in f.readlines():
            edge = row.strip().split(',')
            call_eoa_ca1.append(int(edge[0]))
            call_eoa_ca2.append(int(edge[1]))

    call_ca_ca1 = []
    call_ca_ca2 = []
    with open('call_ca_ca.txt') as f:
        for row in f.readlines():
            edge = row.strip().split(',')
            call_ca_ca1.append(int(edge[0]))
            call_ca_ca2.append(int(edge[1]))

    trans_ca_ca1 = []
    trnas_ca_ca2 = []
    with open('trans_ca_ca.txt') as f:
        for row in f.readlines():
            edge = row.strip().split(',')
            trans_ca_ca1.append(int(edge[0]))
            trnas_ca_ca2.append(int(edge[1]))

    trans_ca_eoa1 = []
    trnas_ca_eoa2 = []
    with open('trans_ca_eoa.txt') as f:
        for row in f.readlines():
            edge = row.strip().split(',')
            trans_ca_eoa1.append(int(edge[0]))
            trnas_ca_eoa2.append(int(edge[1]))

    trans_eoa_ca1 = []
    trnas_eoa_ca2 = []
    with open('trans_eoa_ca.txt') as f:
        for row in f.readlines():
            edge = row.strip().split(',')
            trans_eoa_ca1.append(int(edge[0]))
            trnas_eoa_ca2.append(int(edge[1]))

    graph_dict = {
        ('EOA', 'call1', 'CA'): (torch.tensor(call_eoa_ca1), torch.tensor(call_eoa_ca2)),
        ('CA', 'call2', 'CA'): (torch.tensor(call_ca_ca1), torch.tensor(call_ca_ca2)),
        ('EOA', 'trans1', 'CA'): (torch.tensor(trans_eoa_ca1), torch.tensor(trnas_eoa_ca2)),
        ('CA', 'trans2', 'EOA'): (torch.tensor(trans_ca_eoa1), torch.tensor(trnas_ca_eoa2)),
        ('CA', 'trans3', 'CA'): (torch.tensor(trans_ca_ca1), torch.tensor(trnas_ca_ca2))
             }
    hg = dgl.heterograph(graph_dict)
    # print(hg)
    return hg, CA_index_id_map, EOA_index_id_map


def parse_trace1(trace, CA_index_id_map, EOA_index_id_map):
    s = []
    for index in range(trace.size):
        if index == 0 or index == 1 or index == 3:
            s.append(CA_index_id_map.get(trace[index]))
        elif index == 2:
            s.append(EOA_index_id_map.get(trace[index]))
    return ','.join(s)

def parse_trace2(trace, CA_index_id_map, EOA_index_id_map):
    s = []
    for index in range(trace.size):
        if index == 1 or index == 3:
            s.append(CA_index_id_map.get(trace[index]))
        elif index == 0 or index == 2:
            s.append(EOA_index_id_map.get(trace[index]))
    return ','.join(s)


def ca_ca_eoa_ca():
    hg, CA_index_id_map, EOA_index_id_map = construct_graph()
    meta_path = [
        ['call2', 'trans2', 'call1'],  # 0_ ca ca eoa ca
    ]
    num_walks_per_node = 1
    for metapath in meta_path:
        f = open(f"./metapath/ca_ca_eoa_.txt", "w")
        for contract1_idx in tqdm.trange(382):  # 以带标签CA开头的metapath
            traces = dgl.contrib.sampling.metapath_random_walk(
                hg=hg, etypes=metapath, seeds=[contract1_idx, ], num_traces=num_walks_per_node)  # 根据元路径随机游走 形成没有起点的一条路径
            tr = traces[0][0].numpy() # 转为numpy形式
            tr = np.insert(tr, 0, contract1_idx) # 将初始点加入到路径中
            res = parse_trace1(tr, CA_index_id_map, EOA_index_id_map)
            f.write(res + '\n')



def eoa_ca_eoa_ca():
    hg, CA_index_id_map, EOA_index_id_map = construct_graph()
    meta_path = [
        ['call1', 'trans2', 'trans1'],  # 0_ eoa ca eoa ca
    ]
    num_walks_per_node = 1
    for metapath in meta_path:
        f = open(f"./metapath/eoa_ca_eoa_.txt", "w")
        for contract1_idx in tqdm.trange(hg.number_of_nodes('EOA')):  # 以EOA开头的metapath
            traces = dgl.contrib.sampling.metapath_random_walk(
                hg=hg, etypes=metapath, seeds=[contract1_idx, ], num_traces=num_walks_per_node)  # 根据元路径随机游走 形成没有起点的一条路径
            tr = traces[0][0].numpy() # 转为numpy形式
            tr = np.insert(tr, 0, contract1_idx) # 将初始点加入到路径中
            res = parse_trace2(tr, CA_index_id_map, EOA_index_id_map)
            f.write(res + '\n')

def ca_ca_eoa_ca_GNN():
    hg, CA_index_id_map, EOA_index_id_map = construct_graph()
    meta_path = [
        ['call2', 'trans2', 'call1'],  # 0_ ca ca eoa ca
    ]
    num_walks_per_node = 1
    for metapath in meta_path:
        f = open(f"./metapath/ca_ca_eoa_GNN.txt", "w")
        for contract1_idx in tqdm.trange(hg.number_of_nodes('CA')):  # 以CA开头的metapath
            traces = dgl.contrib.sampling.metapath_random_walk(
                hg=hg, etypes=metapath, seeds=[contract1_idx, ], num_traces=num_walks_per_node)  # 根据元路径随机游走 形成没有起点的一条路径
            tr = traces[0][0].numpy() # 转为numpy形式
            tr = np.insert(tr, 0, contract1_idx) # 将初始点加入到路径中
            res = parse_trace1(tr, CA_index_id_map, EOA_index_id_map)
            f.write(res + '\n')


def eoa_ca_eoa_ca_GNN():
    hg, CA_index_id_map, EOA_index_id_map = construct_graph()
    meta_path = [
        ['call1', 'trans2', 'trans1'],  # 0_ eoa ca eoa ca
    ]
    num_walks_per_node = 1
    for metapath in meta_path:
        f = open(f"./metapath/eoa_ca_eoa_GNN.txt", "w")
        for contract1_idx in tqdm.trange(hg.number_of_nodes('EOA')):  # 以EOA开头的metapath
            traces = dgl.contrib.sampling.metapath_random_walk(
                hg=hg, etypes=metapath, seeds=[contract1_idx, ], num_traces=num_walks_per_node)  # 根据元路径随机游走 形成没有起点的一条路径
            tr = traces[0][0].numpy() # 转为numpy形式
            tr = np.insert(tr, 0, contract1_idx) # 将初始点加入到路径中
            res = parse_trace2(tr, CA_index_id_map, EOA_index_id_map)
            f.write(res + '\n')


if __name__ == '__main__':
    ca_ca_eoa_ca()
    eoa_ca_eoa_ca()
    ca_ca_eoa_ca_GNN()
    eoa_ca_eoa_ca_GNN()


