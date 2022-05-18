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


def add_metapath_ca_ML():
    fea = pd.read_csv('./data/all_fea_1k.csv', index_col=False)
    columns_list = ['Address','bal','V_inv','V_return','V_meaninv','V_meanreturn','V_stdinv','V_stdreturn',
                    'V_maxinv','V_maxreturn','N_inv','N_return','V_Gini_inv','V_Gini_return','lifetime','paid_rate','label'
    ]
    taget_node = pd.DataFrame(columns=columns_list)

    with open(f'./data/data_het/metapath/ca_ca_eoa_.txt', 'r') as metapath:
        for row in metapath.readlines():
            node = row.strip().split(',')   # to_list
            f = fea[(fea['Address']==str(node[0]))].values.tolist()
            dic = {
                'Address':f[0][0], 'bal':f[0][1], 'V_inv':f[0][2],'V_return':f[0][3],'V_meaninv':f[0][4],'V_meanreturn':f[0][5],
                'V_stdinv':f[0][6],'V_stdreturn':f[0][7],'V_maxinv':f[0][8],'V_maxreturn':f[0][9],'N_inv':f[0][10],
                'N_return':f[0][11],'V_Gini_inv':f[0][12],'V_Gini_return':f[0][13],'lifetime':f[0][14],'paid_rate':f[0][15],'label':f[0][16]
            }
            for n in range(1, len(node)):
                g = fea[(fea['Address'] == str(node[n]))].values.tolist()
                dic['bal'] += g[0][1]
                dic['V_inv'] += g[0][2]
                dic['V_return'] += g[0][3]
                dic['V_meaninv'] += g[0][4]
                dic['V_meanreturn'] += g[0][5]
                dic['V_stdinv'] += g[0][6]
                dic['V_stdreturn'] += g[0][7]
                dic['V_maxinv'] += g[0][8]
                dic['V_maxreturn'] += g[0][9]
                dic['N_inv'] += g[0][10]
                dic['N_return'] += g[0][11]
                dic['V_Gini_inv'] += g[0][12]
                dic['V_Gini_return'] += g[0][13]
                dic['lifetime'] += g[0][14]
                dic['paid_rate'] += g[0][15]

            taget_node = taget_node.append(dic, ignore_index=True)
    taget_node.to_csv(f'./new_fea_ML/ca_ca_eoa_.csv', index=False)
    ca_guiyihua()


def add_metapath_eoa_ML():
    fea = pd.read_csv('./data/all_fea_1k.csv')
    fea1 = fea.iloc[:382]
    columns_list = ['Address','bal','V_inv','V_return','V_meaninv','V_meanreturn','V_stdinv','V_stdreturn',
                    'V_maxinv','V_maxreturn','N_inv','N_return','V_Gini_inv','V_Gini_return','lifetime','paid_rate','label'
    ]
    taget_node = pd.DataFrame(columns=columns_list)
    metapath_name = 'eoa_ca_eoa_'
    with open(f'./data/data_het/metapath/' + metapath_name + '.txt', 'r') as metapath:
    # with open(f'./data/data_het/metapath/eoa_ca_ca_eoa.txt', 'r') as metapath:
        for row in tqdm(metapath.readlines()):
            node = row.strip().split(',')   # to_list
            if len(node) > 1:
                f = fea1[(fea1['Address']==str(node[1]))].values.tolist()
                dic = {
                    'Address':f[0][0], 'bal':f[0][1], 'V_inv':f[0][2],'V_return':f[0][3],'V_meaninv':f[0][4],'V_meanreturn':f[0][5],
                    'V_stdinv':f[0][6],'V_stdreturn':f[0][7],'V_maxinv':f[0][8],'V_maxreturn':f[0][9],'N_inv':f[0][10],
                    'N_return':f[0][11],'V_Gini_inv':f[0][12],'V_Gini_return':f[0][13],'lifetime':f[0][14],'paid_rate':f[0][15],'label':f[0][16]
                }
                for n in range(len(node)):
                    if n != 1:
                        g = fea[(fea['Address'] == str(node[n]))].values.tolist()
                        dic['bal'] += g[0][1]
                        dic['V_inv'] += g[0][2]
                        dic['V_return'] += g[0][3]
                        dic['V_meaninv'] += g[0][4]
                        dic['V_meanreturn'] += g[0][5]
                        dic['V_stdinv'] += g[0][6]
                        dic['V_stdreturn'] += g[0][7]
                        dic['V_maxinv'] += g[0][8]
                        dic['V_maxreturn'] += g[0][9]
                        dic['N_inv'] += g[0][10]
                        dic['N_return'] += g[0][11]
                        dic['V_Gini_inv'] += g[0][12]
                        dic['V_Gini_return'] += g[0][13]
                        dic['lifetime'] += g[0][14]
                        dic['paid_rate'] += g[0][15]
                #     # print(node[n])
                taget_node = taget_node.append(dic, ignore_index=True)

    taget_node.drop_duplicates(subset='Address', inplace=True, keep='first')
    path = f'./new_fea_ML/' + metapath_name +'.csv'
    taget_node.to_csv(path, index=False)
    eoa_guiyihua(path)


def ca_guiyihua():
    edge_num_all_df = pd.read_csv(f'new_fea_ML/ca_ca_eoa_.csv')
    df_edge_num = edge_num_all_df.iloc[:, 1:-1].apply(lambda x: np.arctan(x) * (2 / np.pi))  # 非线性归一化

    edge_num_all_df['bal'] = df_edge_num['bal']
    edge_num_all_df['V_inv'] = df_edge_num['V_inv']
    edge_num_all_df['V_return'] = df_edge_num['V_return']
    edge_num_all_df['V_meaninv'] = df_edge_num['V_meaninv']
    edge_num_all_df['V_meanreturn'] = df_edge_num['V_meanreturn']
    edge_num_all_df['V_stdinv'] = df_edge_num['V_stdinv']
    edge_num_all_df['V_stdreturn'] = df_edge_num['V_stdreturn']
    edge_num_all_df['V_maxinv'] = df_edge_num['V_maxinv']
    edge_num_all_df['V_maxreturn'] = df_edge_num['V_maxreturn']
    edge_num_all_df['N_inv'] = df_edge_num['N_inv']
    edge_num_all_df['N_return'] = df_edge_num['N_return']
    edge_num_all_df['V_Gini_inv'] = df_edge_num['V_Gini_inv']
    edge_num_all_df['V_Gini_return'] = df_edge_num['V_Gini_return']
    edge_num_all_df['lifetime'] = df_edge_num['lifetime']
    edge_num_all_df['paid_rate'] = df_edge_num['paid_rate']
    edge_num_all_df.to_csv(f'./new_fea_ML/ca_ca_eoa_.csv', index=False)


def eoa_guiyihua(path):
    edge_num_all_df = pd.read_csv(path)
    fea = pd.read_csv('./data/all_fea_1k.csv')
    fea = fea.iloc[:382]
    addr_list = edge_num_all_df['Address'].to_list()
    fea = fea[~fea['Address'].isin(addr_list)]
    edge_num_all_df = pd.concat([fea, edge_num_all_df])
    df_edge_num = edge_num_all_df.iloc[:, 1:-1].apply(lambda x: np.arctan(x) * (2 / np.pi))  # 非线性归一化
    edge_num_all_df['bal'] = df_edge_num['bal']
    edge_num_all_df['V_inv'] = df_edge_num['V_inv']
    edge_num_all_df['V_return'] = df_edge_num['V_return']
    edge_num_all_df['V_meaninv'] = df_edge_num['V_meaninv']
    edge_num_all_df['V_meanreturn'] = df_edge_num['V_meanreturn']
    edge_num_all_df['V_stdinv'] = df_edge_num['V_stdinv']
    edge_num_all_df['V_stdreturn'] = df_edge_num['V_stdreturn']
    edge_num_all_df['V_maxinv'] = df_edge_num['V_maxinv']
    edge_num_all_df['V_maxreturn'] = df_edge_num['V_maxreturn']
    edge_num_all_df['N_inv'] = df_edge_num['N_inv']
    edge_num_all_df['N_return'] = df_edge_num['N_return']
    edge_num_all_df['V_Gini_inv'] = df_edge_num['V_Gini_inv']
    edge_num_all_df['V_Gini_return'] = df_edge_num['V_Gini_return']
    edge_num_all_df['lifetime'] = df_edge_num['lifetime']
    edge_num_all_df['paid_rate'] = df_edge_num['paid_rate']
    edge_num_all_df.to_csv(path, index=False)



if __name__ == '__main__':
    add_metapath_ca_ML()
    add_metapath_eoa_ML()




