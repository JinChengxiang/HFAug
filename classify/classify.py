from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB      #先验为高斯分布的朴素贝叶斯
from sklearn.naive_bayes import MultinomialNB   #先验为多项式分布的朴素贝叶斯
from sklearn.tree import DecisionTreeClassifier  #决策树

from sklearn.model_selection import RandomizedSearchCV,GridSearchCV  # 网格搜索法
from sklearn import svm  # 支持向量机库
from sklearn import preprocessing
import matplotlib.pyplot as plt

# from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold #交叉验证
from sklearn.model_selection import KFold

from sklearn.neighbors import KNeighborsClassifier

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None) # 展示所有列


def logistic_classify1(data, method,kfold,seed,p=None,q=None):
    data = np.array(data)
    X = data[:, :-2]
    y = data[:, -1]
    # X = data[:, 1:-2]
    # y = data[:, -1]
    Normalizer = preprocessing.Normalizer()
    Normalizer.fit(X)
    X = Normalizer.transform(X)
    model = LogisticRegression(solver='liblinear', multi_class='ovr')

    # y = pd.DataFrame(y)
    # types = y.unique()
    # n_class = types.size
    y = np.array(y)
    types = np.unique(y)
    n_class = types.size
    y = pd.Categorical(y).codes
    if method != 'node2vec':
        ave_f1 = []
        auc_ave = []
        acc_ave = []
        pre_ave = []
        recall_ave = []
        kf = KFold(n_splits=kfold, shuffle=True, random_state=seed)

        X_train_ls = []
        X_test_ls = []
        y_train_ls = []
        y_test_ls = []
        # train,test是索引  划分出节点的测试集和训练集
        for train_id, test_id in kf.split(np.arange(len(X))):
            X_train_ls.append(list(np.array(X)[train_id]))
            X_test_ls.append(list(np.array(X)[test_id]))
            y_train_ls.append(list(np.array(y)[train_id]))
            y_test_ls.append(list(np.array(y)[test_id]))

        for idx in range(len(X_train_ls)):
            X_train = X_train_ls[idx]
            X_test = X_test_ls[idx]
            y_train = y_train_ls[idx]
            y_test = y_test_ls[idx]

            model.fit(X_train, y_train)
            y_test_score = model.predict_proba(X_test)
            y_pred = model.predict(X_test)
            # y_one_hot = label_binarize(y_test, np.arange(n_class))
            # auc = metrics.roc_auc_score(y_one_hot, y_test_score, average='micro', multi_class='ovr')


            f1 = metrics.f1_score(y_test, y_pred, average='micro')
            # auc = metrics.roc_auc_score(y_test, y_test_score[:, 1])
            # acc = metrics.accuracy_score(y_test, y_pred)
            # pre = metrics.precision_score(y_test,y_pred)
            # recall = metrics.recall_score(y_test,y_pred, average='micro')


            ave_f1.append(f1)
            # auc_ave.append(auc)
            # acc_ave.append(acc)
            # pre_ave.append(pre)
            # recall_ave.append(recall)
        score_dic = {
                 # "AUC":np.mean(auc_ave),
                 # "acc":np.mean(acc_ave),
                 # "Pre":np.mean(pre_ave),
                 # "Recall":np.mean(recall_ave),
                 "F1":np.mean(ave_f1)
                 }

    elif method=='node2vec':
        ave_f1 = []
        auc_ave = []
        acc_ave = []
        pre_ave = []
        recall_ave = []
        kf = KFold(n_splits=kfold, shuffle=True, random_state=seed)

        X_train_ls = []
        X_test_ls = []
        y_train_ls = []
        y_test_ls = []
        # train,test是索引  划分出节点的测试集和训练集
        for train_id, test_id in kf.split(np.arange(len(X))):
            X_train_ls.append(list(np.array(X)[train_id]))
            X_test_ls.append(list(np.array(X)[test_id]))
            y_train_ls.append(list(np.array(y)[train_id]))
            y_test_ls.append(list(np.array(y)[test_id]))

        for idx in range(len(X_train_ls)):
            X_train = X_train_ls[idx]
            X_test = X_test_ls[idx]
            y_train = y_train_ls[idx]
            y_test = y_test_ls[idx]

            model.fit(X_train, y_train)
            y_test_score = model.predict_proba(X_test)
            y_pred = model.predict(X_test)
            # y_one_hot = label_binarize(y_test, np.arange(n_class))
            # auc = metrics.roc_auc_score(y_one_hot, y_test_score, average='micro', multi_class='ovr')

            f1 = metrics.f1_score(y_test, y_pred, average='micro')
            # auc = metrics.roc_auc_score(y_test, y_test_score[:, 1])
            # acc = metrics.accuracy_score(y_test, y_pred)
            # pre = metrics.precision_score(y_test, y_pred)
            # recall = metrics.recall_score(y_test, y_pred, average='micro')

            ave_f1.append(f1)
            # auc_ave.append(auc)
            # acc_ave.append(acc)
            # pre_ave.append(pre)
            # recall_ave.append(recall)

        score_dic = {
            'p':p,'q':q,
            # "AUC": np.mean(auc_ave),
            # "acc": np.mean(acc_ave),
            # "Pre": np.mean(pre_ave),
            # "Recall": np.mean(recall_ave),
            "F1": np.mean(ave_f1)
        }
    return score_dic


def randomForest_classifier(data,method,kfold,seed,p=None,q=None):
    data = np.array(data)
    X = data[:, :-2]
    y = data[:, -1]
    # X = data[:, 1:-2]
    # y = data[:, -1]
    Normalizer = preprocessing.Normalizer()
    Normalizer.fit(X)
    X = Normalizer.transform(X)
    # rf_parameters_default = {'n_estimators':300,'min_samples_leaf':2,'max_depth':100,
    #                          'bootstrap':True,'class_weight':'balanced'}
    model = RandomForestClassifier(random_state = seed)
    # y = pd.DataFrame(y)
    # types = y.unique()
    # n_class = types.size
    y = np.array(y)
    types = np.unique(y)
    n_class = types.size
    y = pd.Categorical(y).codes
    if method != 'node2vec':
        ave_f1 = []
        auc_ave = []
        acc_ave = []
        pre_ave = []
        recall_ave = []
        kf = KFold(n_splits=kfold, shuffle=True, random_state=seed)

        X_train_ls = []
        X_test_ls = []
        y_train_ls = []
        y_test_ls = []
        # train,test是索引  划分出节点的测试集和训练集
        for train_id, test_id in kf.split(np.arange(len(X))):
            X_train_ls.append(list(np.array(X)[train_id]))
            X_test_ls.append(list(np.array(X)[test_id]))
            y_train_ls.append(list(np.array(y)[train_id]))
            y_test_ls.append(list(np.array(y)[test_id]))

        for idx in range(len(X_train_ls)):
            X_train = X_train_ls[idx]
            X_test = X_test_ls[idx]
            y_train = y_train_ls[idx]
            y_test = y_test_ls[idx]

            model.fit(X_train, y_train)
            y_test_score = model.predict_proba(X_test)
            y_pred = model.predict(X_test)
            # y_one_hot = label_binarize(y_test, np.arange(n_class))
            # auc = metrics.roc_auc_score(y_one_hot, y_test_score, average='micro', multi_class='ovr')

            f1 = metrics.f1_score(y_test, y_pred, average='micro')
            # auc = metrics.roc_auc_score(y_test, y_test_score[:, 1])
            # acc = metrics.accuracy_score(y_test, y_pred)
            # pre = metrics.precision_score(y_test, y_pred)
            # recall = metrics.recall_score(y_test, y_pred, average='micro')

            ave_f1.append(f1)
            # auc_ave.append(auc)
            # acc_ave.append(acc)
            # pre_ave.append(pre)
            # recall_ave.append(recall)
        score_dic = {
            # "AUC": np.mean(auc_ave),
            # "acc": np.mean(acc_ave),
            # "Pre": np.mean(pre_ave),
            # "Recall": np.mean(recall_ave),
            "F1": np.mean(ave_f1)
        }

    elif method == 'node2vec':
        ave_f1 = []
        auc_ave = []
        acc_ave = []
        pre_ave = []
        recall_ave = []
        kf = KFold(n_splits=kfold, shuffle=True, random_state=seed)

        X_train_ls = []
        X_test_ls = []
        y_train_ls = []
        y_test_ls = []
        # train,test是索引  划分出节点的测试集和训练集
        for train_id, test_id in kf.split(np.arange(len(X))):
            X_train_ls.append(list(np.array(X)[train_id]))
            X_test_ls.append(list(np.array(X)[test_id]))
            y_train_ls.append(list(np.array(y)[train_id]))
            y_test_ls.append(list(np.array(y)[test_id]))

        for idx in range(len(X_train_ls)):
            X_train = X_train_ls[idx]
            X_test = X_test_ls[idx]
            y_train = y_train_ls[idx]
            y_test = y_test_ls[idx]

            model.fit(X_train, y_train)
            y_test_score = model.predict_proba(X_test)
            y_pred = model.predict(X_test)
            # y_one_hot = label_binarize(y_test, np.arange(n_class))
            # auc = metrics.roc_auc_score(y_one_hot, y_test_score, average='micro', multi_class='ovr')

            f1 = metrics.f1_score(y_test, y_pred, average='micro')
            # auc = metrics.roc_auc_score(y_test, y_test_score[:, 1])
            # acc = metrics.accuracy_score(y_test, y_pred)
            # pre = metrics.precision_score(y_test, y_pred)
            # recall = metrics.recall_score(y_test, y_pred, average='micro')

            ave_f1.append(f1)
            # auc_ave.append(auc)
            # acc_ave.append(acc)
            # pre_ave.append(pre)
            # recall_ave.append(recall)

        score_dic = {
            'p': p, 'q': q,
            # "AUC": np.mean(auc_ave),
            # "acc": np.mean(acc_ave),
            # "Pre": np.mean(pre_ave),
            # "Recall": np.mean(recall_ave),
            "F1": np.mean(ave_f1)
        }
    return score_dic


def svm_classifier(data, method, kfold, seed, p=None, q=None):
    data = np.array(data)
    X = data[:, :-2]
    y = data[:, -1]
    # X = data[:, 1:-2]
    # y = data[:, -1]
    # 归一化feature
    Normalizer = preprocessing.Normalizer()
    Normalizer.fit(X)
    X = Normalizer.transform(X)

    # rf_parameters_default = {'n_estimators':100,'min_samples_leaf':2,'max_depth':100,
    #                          'bootstrap':True,'class_weight':'balanced'}
    # model = RandomForestClassifier(**rf_parameters_default,random_state = 1)
    model = svm.SVC(kernel='rbf',probability=True)

    y = np.array(y)
    types = np.unique(y)
    n_class = types.size
    y = pd.Categorical(y).codes
    if method != 'node2vec':
        ave_f1 = []
        auc_ave = []
        acc_ave = []
        pre_ave = []
        recall_ave = []
        kf = KFold(n_splits=kfold, shuffle=True, random_state=seed)

        X_train_ls = []
        X_test_ls = []
        y_train_ls = []
        y_test_ls = []
        # train,test是索引  划分出节点的测试集和训练集
        for train_id, test_id in kf.split(np.arange(len(X))):
            X_train_ls.append(list(np.array(X)[train_id]))
            X_test_ls.append(list(np.array(X)[test_id]))
            y_train_ls.append(list(np.array(y)[train_id]))
            y_test_ls.append(list(np.array(y)[test_id]))

        for idx in range(len(X_train_ls)):
            X_train = X_train_ls[idx]
            X_test = X_test_ls[idx]
            y_train = y_train_ls[idx]
            y_test = y_test_ls[idx]

            model.fit(X_train, y_train)
            y_test_score = model.predict_proba(X_test)
            y_pred = model.predict(X_test)
            # y_one_hot = label_binarize(y_test, np.arange(n_class))
            # auc = metrics.roc_auc_score(y_one_hot, y_test_score, average='micro', multi_class='ovr')

            f1 = metrics.f1_score(y_test, y_pred, average='micro')
            # auc = metrics.roc_auc_score(y_test, y_test_score[:, 1])
            # acc = metrics.accuracy_score(y_test, y_pred)
            # pre = metrics.precision_score(y_test, y_pred)
            # recall = metrics.recall_score(y_test, y_pred, average='micro')

            ave_f1.append(f1)
            # auc_ave.append(auc)
            # acc_ave.append(acc)
            # pre_ave.append(pre)
            # recall_ave.append(recall)
        score_dic = {
            # "AUC": np.mean(auc_ave),
            # "acc": np.mean(acc_ave),
            # "Pre": np.mean(pre_ave),
            # "Recall": np.mean(recall_ave),
            "F1": np.mean(ave_f1)
        }

    elif method == 'node2vec':
        ave_f1 = []
        auc_ave = []
        acc_ave = []
        pre_ave = []
        recall_ave = []
        kf = KFold(n_splits=kfold, shuffle=True, random_state=seed)

        X_train_ls = []
        X_test_ls = []
        y_train_ls = []
        y_test_ls = []
        # train,test是索引  划分出节点的测试集和训练集
        for train_id, test_id in kf.split(np.arange(len(X))):
            X_train_ls.append(list(np.array(X)[train_id]))
            X_test_ls.append(list(np.array(X)[test_id]))
            y_train_ls.append(list(np.array(y)[train_id]))
            y_test_ls.append(list(np.array(y)[test_id]))

        for idx in range(len(X_train_ls)):
            X_train = X_train_ls[idx]
            X_test = X_test_ls[idx]
            y_train = y_train_ls[idx]
            y_test = y_test_ls[idx]

            model.fit(X_train, y_train)
            y_test_score = model.predict_proba(X_test)
            y_pred = model.predict(X_test)
            # y_one_hot = label_binarize(y_test, np.arange(n_class))
            # auc = metrics.roc_auc_score(y_one_hot, y_test_score, average='micro', multi_class='ovr')

            f1 = metrics.f1_score(y_test, y_pred, average='micro')
            # auc = metrics.roc_auc_score(y_test, y_test_score[:, 1])
            # acc = metrics.accuracy_score(y_test, y_pred)
            # pre = metrics.precision_score(y_test, y_pred)
            # recall = metrics.recall_score(y_test, y_pred, average='micro')

            ave_f1.append(f1)
            # auc_ave.append(auc)
            # acc_ave.append(acc)
            # pre_ave.append(pre)
            # recall_ave.append(recall)

        score_dic = {
            'p': p, 'q': q,
            # "AUC": np.mean(auc_ave),
            # "acc": np.mean(acc_ave),
            # "Pre": np.mean(pre_ave),
            # "Recall": np.mean(recall_ave),
            "F1": np.mean(ave_f1)
        }
    return score_dic


def logistic_classify_hander(data,kfold,seed):
    data = np.array(data)
    X = (data[:, 1:-1]).astype(float)
    y = (data[:, -1]).astype(float)

    Normalizer = preprocessing.Normalizer()
    Normalizer.fit(X)
    X = Normalizer.transform(X)

    model = LogisticRegression(solver='liblinear', multi_class='ovr', C=1, penalty='l2')

    ave_f1 = []
    auc_ave = []
    acc_ave = []
    pre_ave = []
    recall_ave = []
    y = np.array(y)
    types = np.unique(y)
    n_class = types.size
    y = pd.Categorical(y).codes

    kf = KFold(n_splits=kfold, shuffle=True, random_state=seed)

    X_train_ls = []
    X_test_ls = []
    y_train_ls = []
    y_test_ls = []
    # train,test是索引  划分出节点的测试集和训练集
    for train_id, test_id in kf.split(np.arange(len(X))):
        X_train_ls.append(list(np.array(X)[train_id]))
        X_test_ls.append(list(np.array(X)[test_id]))
        y_train_ls.append(list(np.array(y)[train_id]))
        y_test_ls.append(list(np.array(y)[test_id]))

    for idx in range(len(X_train_ls)):
        X_train = X_train_ls[idx]
        X_test = X_test_ls[idx]
        y_train = y_train_ls[idx]
        y_test = y_test_ls[idx]


        model.fit(X_train, y_train)
        y_test_score = model.predict_proba(X_test)
        y_pred = model.predict(X_test)

        f1 = metrics.f1_score(y_test, y_pred,average='micro')
        print(f1)
        ave_f1.append(f1)

    score_dic = {
        "F1": np.mean(ave_f1)
    }
    return score_dic


def randomForest_classifier_hander(data,kfold,seed):
    data = np.array(data)
    X = (data[:, 1:-1]).astype(float)
    y = (data[:, -1]).astype(float)

    Normalizer = preprocessing.Normalizer()
    Normalizer.fit(X)
    X = Normalizer.transform(X)

    model = RandomForestClassifier(random_state=seed,
    n_estimators=100, criterion='gini', max_depth=7, min_samples_leaf=2, min_impurity_decrease=0)

    ave_f1 = []
    auc_ave = []
    acc_ave = []
    pre_ave = []
    recall_ave = []

    y = np.array(y)
    types = np.unique(y)
    n_class = types.size
    y = pd.Categorical(y).codes

    kf = KFold(n_splits=kfold, shuffle=True, random_state=seed)

    X_train_ls = []
    X_test_ls = []
    y_train_ls = []
    y_test_ls = []
    # train,test是索引  划分出节点的测试集和训练集
    for train_id, test_id in kf.split(np.arange(len(X))):
        X_train_ls.append(list(np.array(X)[train_id]))
        X_test_ls.append(list(np.array(X)[test_id]))
        y_train_ls.append(list(np.array(y)[train_id]))
        y_test_ls.append(list(np.array(y)[test_id]))

    for idx in range(len(X_train_ls)):
        X_train = X_train_ls[idx]
        X_test = X_test_ls[idx]
        y_train = y_train_ls[idx]
        y_test = y_test_ls[idx]

        model.fit(X_train, y_train)
        y_test_score = model.predict_proba(X_test)
        y_pred = model.predict(X_test)

        f1 = metrics.f1_score(y_test, y_pred,average='micro')
        print(f1)

        ave_f1.append(f1)

    score_dic = {
        "F1": np.mean(ave_f1)
    }
    return score_dic


def SVM_hander(data,kfold,seed):
    data = np.array(data)
    X = (data[:, 1:-1]).astype(float)
    y = (data[:, -1]).astype(float)

    Normalizer = preprocessing.Normalizer()
    Normalizer.fit(X)
    X = Normalizer.transform(X)

    model = svm.SVC(probability=True, kernel='rbf', C=10)

    ave_f1 = []
    y = np.array(y)
    types = np.unique(y)
    n_class = types.size
    y = pd.Categorical(y).codes

    kf = KFold(n_splits=kfold, shuffle=True, random_state=seed)

    X_train_ls = []
    X_test_ls = []
    y_train_ls = []
    y_test_ls = []
    # train,test是索引  划分出节点的测试集和训练集
    for train_id, test_id in kf.split(np.arange(len(X))):
        X_train_ls.append(list(np.array(X)[train_id]))
        X_test_ls.append(list(np.array(X)[test_id]))
        y_train_ls.append(list(np.array(y)[train_id]))
        y_test_ls.append(list(np.array(y)[test_id]))

    for idx in range(len(X_train_ls)):
        X_train = X_train_ls[idx]
        X_test = X_test_ls[idx]
        y_train = y_train_ls[idx]
        y_test = y_test_ls[idx]

        model.fit(X_train, y_train)
        y_test_score = model.predict_proba(X_test)
        y_pred = model.predict(X_test)

        f1 = metrics.f1_score(y_test, y_pred,average='micro')
        print(f1)


        ave_f1.append(f1)

        score_dic = {
            "F1": np.mean(ave_f1)
        }
    return score_dic

