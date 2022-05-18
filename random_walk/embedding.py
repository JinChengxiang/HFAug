from __future__ import print_function
import numpy as np
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from sklearn.linear_model import LogisticRegression
from graph import *
import node2vec
# from gcn import gcnAPI

import time
import ast
import pandas as pd
import scipy.sparse as sp
import os


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--input', default='../node_classification_dataset/flickr/Adj.npz',
                        help='Input graph file')
    parser.add_argument('--output',
                        help='Output representation file')
    parser.add_argument('--number-walks', default=5, type=int,
                        help='Number of random walks to start at each node')
    parser.add_argument('--directed', type=bool, default=False,
                        help='Treat graph as directed.')
    parser.add_argument('--walk-length', default=50, type=int,
                        help='Length of the random walk started at each node')
    parser.add_argument('--workers', default=8, type=int,
                        help='Number of parallel processes.')
    parser.add_argument('--representation-size', default=64, type=int,
                        help='Number of latent dimensions to learn for each node.')
    parser.add_argument('--window-size', default=10, type=int,
                        help='Window size of skipgram model.')
    parser.add_argument('--epochs', default=200, type=int,
                        help='The training epochs of LINE and GCN')
    parser.add_argument('--p', default=1.0, type=float)
    parser.add_argument('--q', default=1.0, type=float)
    parser.add_argument('--method', default='grarep', choices=[
        'node2vec',
        'deepwalk',
    ], help='The learning method')

    parser.add_argument('--label', default='',
                        help='The file of node label')
    parser.add_argument('--feature-file', default='',
                        help='The file of node features')
    parser.add_argument('--graph-format', default='adjlist', choices=['adjlist', 'edgelist'],
                        help='Input graph format')
    parser.add_argument('--weighted', type=bool, default=False,
                        help='Treat graph as weighted')
    parser.add_argument('--clf-ratio', default=0.5, type=float,
                        help='The ratio of training data in the classification')
    parser.add_argument('--order', default=2, type=int,
                        help='Choose the order of LINE, 1 means first order, 2 means second order, 3 means first order + second order')
    parser.add_argument('--no-auto-save', action='store_true',
                        help='no save the best embeddings when training LINE')
    parser.add_argument('--dropout', default=0.5, type=float,
                        help='Dropout rate (1 - keep probability)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight for L2 loss on embedding matrix')
    parser.add_argument('--hidden', default=16, type=int,
                        help='Number of units in hidden layer 1')
    parser.add_argument('--kstep', default=4, type=int,
                        help='Use k-step transition probability matrix')
    parser.add_argument('--lamb', default=0.2, type=float,
                        help='lambda is a hyperparameter in TADW')
    parser.add_argument('--lr', default=0.01, type=float,
                        help='learning rate')
    parser.add_argument('--alpha', default=1e-6, type=float,
                        help='alhpa is a hyperparameter in SDNE')
    parser.add_argument('--beta', default=5., type=float,
                        help='beta is a hyperparameter in SDNE')
    parser.add_argument('--nu1', default=1e-5, type=float,
                        help='nu1 is a hyperparameter in SDNE')
    parser.add_argument('--nu2', default=1e-4, type=float,
                        help='nu2 is a hyperparameter in SDNE')
    parser.add_argument('--bs', default=200, type=int,
                        help='batch size of SDNE')
    parser.add_argument('--encoder-list', default='[1000, 128]', type=str,
                        help='a list of numbers of the neuron at each encoder layer, the last number is the '
                             'dimension of the output node representation')
    args = parser.parse_args()
    return args


def main(args):
    t1 = time.time()
    g = Graph()
    print("Reading...")


    g.read_edgelist1(filename=args.input)

    if args.method == 'node2vec':
        model = node2vec.Node2vec(graph=g, path_length=args.walk_length,
                                  num_paths=args.number_walks, dim=args.representation_size,
                                  workers=args.workers, p=args.p, q=args.q, window=args.window_size)
    elif args.method == 'deepwalk':
        model = node2vec.Node2vec(graph=g, path_length=args.walk_length,
                                  num_paths=args.number_walks, dim=args.representation_size,
                                  workers=args.workers, window=args.window_size, dw=True)
    t2 = time.time()
    print(t2 - t1)

    return g, model


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
    return path


if __name__ == "__main__":
    dimenson_select = [128]
    method_select = ['deepwalk', 'node2vec']
    p_q_select_for_node2vec = [0.5, 1, 2]
    for dimension in dimenson_select:
        for method in method_select:
            arg = parse_args()
            arg.input = f"../data/pre/A.txt"
            arg.representation_size = dimension
            arg.method = str(method)
            walk_long = 50
            if method == 'node2vec':
                for p in p_q_select_for_node2vec:
                    for q in p_q_select_for_node2vec:
                        arg.output = mkdir(
                            f'./{method}/' + 'dimension_' + str(dimension) + '/' + str(p) + '_' + str(q) + '/')
                        arg.p = p
                        arg.q = q
                        g, model = main(arg)

                        embedding_vector = model.vectors
                        for key, value in embedding_vector.items():
                            embedding_vector[key] = np.append(embedding_vector[key], key)
                        embedding_vector = pd.DataFrame(embedding_vector).T

                        if embedding_vector.shape[1] == 17:
                            embedding_vector = embedding_vector.rename(columns={16: 'node'})
                        elif embedding_vector.shape[1] == 33:
                            embedding_vector = embedding_vector.rename(columns={32: 'node'})
                        elif embedding_vector.shape[1] == 65:
                            embedding_vector = embedding_vector.rename(columns={64: 'node'})
                        elif embedding_vector.shape[1] == 129:
                            embedding_vector = embedding_vector.rename(columns={128: 'node'})
                        elif embedding_vector.shape[1] == 257:
                            embedding_vector = embedding_vector.rename(columns={256: 'node'})


                        embedding_vector.to_csv(arg.output + f'G_emb_raw_all.csv', index=False)

                        # add label
                        G_embed = pd.read_csv(arg.output + f'G_emb_raw_all.csv')
                        G_embed = G_embed.sort_values(by='node', ascending=True)
                        G_embed = G_embed.iloc[:382]
                        label = []
                        for index, row in G_embed.iterrows():
                            if row['node'] <= 190 and row['node'] >= 0:
                                label.append('1')
                            elif row['node'] <= 381 and row['node'] >= 191:
                                label.append('0')
                        G_embed['label'] = label
                        G_embed.to_csv(arg.output + f'G_emb_raw.csv', index=False)

            else:
                arg.output = mkdir(f'./{method}/' + 'dimension_' + str(dimension) + '/')

                g, model = main(arg)
                embedding_vector = model.vectors
                for key, value in embedding_vector.items():
                    embedding_vector[key] = np.append(embedding_vector[key], key)
                embedding_vector = pd.DataFrame(embedding_vector).T

                if embedding_vector.shape[1] == 17:
                    embedding_vector = embedding_vector.rename(columns={16: 'node'})
                elif embedding_vector.shape[1] == 33:
                    embedding_vector = embedding_vector.rename(columns={32: 'node'})
                elif embedding_vector.shape[1] == 65:
                    embedding_vector = embedding_vector.rename(columns={64: 'node'})
                elif embedding_vector.shape[1] == 129:
                    embedding_vector = embedding_vector.rename(columns={128: 'node'})
                elif embedding_vector.shape[1] == 257:
                    embedding_vector = embedding_vector.rename(columns={256: 'node'})

                embedding_vector.to_csv(arg.output + f'G_emb_raw_all.csv', index=False)

                G_embed = pd.read_csv(arg.output + f'G_emb_raw_all.csv')
                G_embed = G_embed.sort_values(by='node', ascending=True)
                G_embed = G_embed.iloc[:382]
                label = []
                for index, row in G_embed.iterrows():
                    if row['node'] <= 190 and row['node'] >= 0:
                        label.append('1')
                    elif row['node'] <= 381 and row['node'] >= 191:
                        label.append('0')
                G_embed['label'] = label
                G_embed.to_csv(arg.output + f'G_emb_raw.csv', index=False)





