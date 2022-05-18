import os
import pandas as pd
import numpy as np
# from classify import logistic_classify as lr
from classify import logistic_classify1 as lr1
from classify import randomForest_classifier as rf
from classify import svm_classifier as svc
from classify import logistic_classify_hander,randomForest_classifier_hander, SVM_hander
import warnings
warnings.filterwarnings("ignore")

def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
    return path

def base_fea():
    seed = 123
    columns_list = ['classify', 'F1']
    pd_result = pd.DataFrame(columns=columns_list)
    input_normal = pd.read_csv(f'../data/' + 'all_fea_1k.csv', nrows=382)

    # ================================================================

    score_dic = randomForest_classifier_hander(data=input_normal,kfold=5, seed=seed)
    dic_result = {'classify': 'rf', "F1": score_dic['F1']}
    print(dic_result)
    pd_result = pd_result.append(dic_result, ignore_index=True)

    score_dic = logistic_classify_hander(data=input_normal, kfold=5, seed=seed)
    dic_result = {'classify': 'lr', "F1": score_dic['F1']}
    print(dic_result)
    pd_result = pd_result.append(dic_result, ignore_index=True)

    score_dic = SVM_hander(data=input_normal, kfold=5, seed=seed)
    dic_result = {'classify': 'svc', "F1": score_dic['F1']}
    print(dic_result)
    pd_result = pd_result.append(dic_result, ignore_index=True)

    # ==============================================================

    pd_result.to_csv(mkdir(f'./result/ML/') +'baseline.csv', index=False, mode='a')


def manuel_fea():
    seed = 123
    columns_list = ['classify', 'F1']
    names = ['ca_ca_eoa_', 'eoa_ca_eoa_']
    for name in names:
        pd_result = pd.DataFrame(columns=columns_list)
        input_normal = pd.read_csv(f'../new_fea_ML/' + name + '.csv')

        # ================================================================

        score_dic = randomForest_classifier_hander(data=input_normal,kfold=5, seed=seed)
        dic_result = {'classify': 'rf', "F1": score_dic['F1']}
        print(dic_result)
        pd_result = pd_result.append(dic_result, ignore_index=True)

        score_dic = logistic_classify_hander(data=input_normal, kfold=5, seed=seed)
        dic_result = {'classify': 'lr', "F1": score_dic['F1']}
        print(dic_result)
        pd_result = pd_result.append(dic_result, ignore_index=True)

        score_dic = SVM_hander(data=input_normal, kfold=5, seed=seed)
        dic_result = {'classify': 'svc', "F1": score_dic['F1']}
        print(dic_result)
        pd_result = pd_result.append(dic_result, ignore_index=True)

        # ==============================================================

        pd_result.to_csv(mkdir(f'./result/ML/') + name + '.csv', index=False, mode='a')


def embedding():
    dimenson_select = [128]
    method_select = ['deepwalk', 'node2vec']
    # method_select = ['deepwalk']
    seed = 123
    kfold_list = [5]
    p_q_select_for_node2vec = [0.5, 1, 2]
    columns_list = ['classify', 'dimenson', 'method', 'p', 'q', 'F1']
    pd_result = pd.DataFrame(columns=columns_list)
    names = ['raw', 'ca_ca_eoa', 'eoa_ca_eoa']
    for name in names:
        for dimenson in dimenson_select:
            for method in method_select:
                if method != 'node2vec':
                        inputdata = pd.read_csv(fr'../random_walk/{method}/dimension_{dimenson}/G_emb_{name}.csv', nrows=382)
                        for kfold in kfold_list:
                            score_dic = rf(data=inputdata,method=method,kfold=kfold,seed=seed)
                            dic_result = {'classify': 'rf', 'dimenson': dimenson, 'method': method,
                                          "F1":score_dic['F1']}
                            print(dic_result)
                            pd_result = pd_result.append(dic_result, ignore_index=True)

                            score_dic = lr1(data=inputdata,method=method,kfold=kfold,seed=seed)
                            dic_result = {'classify': 'lr', 'dimenson': dimenson, 'method': method,
                                          "F1":score_dic['F1']}
                            print(dic_result)
                            pd_result = pd_result.append(dic_result, ignore_index=True)

                            score_dic = svc(data=inputdata,method=method,kfold=kfold,seed=seed)
                            dic_result = {'classify': 'svc', 'dimenson': dimenson, 'method': method,
                                          "F1":score_dic['F1']}
                            print(dic_result)
                            pd_result = pd_result.append(dic_result, ignore_index=True)

                elif method == 'node2vec':
                    for kfold in kfold_list:
                        for p in p_q_select_for_node2vec:
                            for q in p_q_select_for_node2vec:
                                inputdata = pd.read_csv(fr'../random_walk/{method}/dimension_{dimenson}/{p}_{q}/G_emb_{name}.csv')
                                score_dic = rf(data=inputdata,method=method,p=p,q=q,kfold=kfold,seed=seed)
                                dic_result = {'classify': 'rf', 'dimenson': dimenson, 'method': method,
                                              'p':score_dic['p'],"q":score_dic['q'],
                                              "F1": score_dic['F1']}
                                print(dic_result)
                                pd_result = pd_result.append(dic_result, ignore_index=True)

                                score_dic = lr1(data=inputdata, method=method, p=p,q=q, kfold=kfold, seed=seed)
                                dic_result = {'classify': 'lr', 'dimenson': dimenson, 'method': method,
                                              'p': score_dic['p'], "q": score_dic['q'],
                                              "F1": score_dic['F1']}
                                print(dic_result)
                                pd_result = pd_result.append(dic_result, ignore_index=True)

                                score_dic = svc(data=inputdata, method=method, p=p,q=q,kfold=kfold, seed=seed)
                                dic_result = {'classify': 'svc', 'dimenson': dimenson, 'method': method,
                                              'p': score_dic['p'], "q": score_dic['q'],
                                              "F1": score_dic['F1']}
                                print(dic_result)
                                pd_result = pd_result.append(dic_result, ignore_index=True)

        pd_result.to_csv(mkdir(f'./result/RW/') + name + '.csv', index=False, mode='a')



if __name__ == '__main__':
    # base_fea()
    # manuel_fea()
    embedding()
