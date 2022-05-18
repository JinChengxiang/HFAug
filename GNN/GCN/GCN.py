import csv
import numpy as np
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score
import os.path as osp
import argparse
import torch
import torch.nn.functional as F
from tools import EarlyStopping, ColumnNormalizeFeatures, mkdir
from dataset import Planetoid

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv, GCN2Conv  # noqa
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
parser.add_argument('--gpu', type=str, help='gpu id', default='0')
args = parser.parse_args()
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

class Net(torch.nn.Module):
    def __init__(self,input_fea,hidden,output_fea):
        super().__init__()
        self.conv1 = GCNConv(input_fea, hidden,
                             normalize=args.use_gdc)
        self.conv2 = GCNConv(hidden, output_fea,
                             normalize=args.use_gdc)

        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

    def forward(self):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x,p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# ----------------------------


def train(epoch, train_mask):
    model.train()
    optimizer.zero_grad()
    output = model()
    preds_train = output[train_mask]
    reals_train = y_train
    loss_train = F.nll_loss(preds_train, reals_train)
    preds_train_cpu = predict_fn(output[train_mask])
    reals_train_cpu = reals_train.detach().cpu()

    preds_train_cpu = np.hstack(preds_train_cpu)
    reals_train_cpu = np.hstack(reals_train_cpu)

    f1_train = f1_score(reals_train_cpu, preds_train_cpu, average='micro')
    loss_train.backward()
    optimizer.step()

    print('Epoch: {:04d}'.format(epoch),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'f1_train: {:.4f}'.format(f1_train))

    return loss_train, f1_train




def test(test_mask):
    model.eval()
    output = model()

    preds = output[test_mask]
    reals = y_test

    preds_test_cpu = predict_fn(output[test_mask])
    reals_test_cpu = reals.detach().cpu().view_as(preds_test_cpu)

    preds_test_cpu = np.hstack(preds_test_cpu)
    reals_test_cpu = np.hstack(reals_test_cpu)
    loss_test = F.nll_loss(preds, reals)
    f1 = f1_score(reals_test_cpu, preds_test_cpu, average='micro')
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "f1= {:.4f}".format(f1),
          )
    with open(f'./result/gcn_{dataset0}.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        a = []
        a.append(hidden)
        a.append(lr)
        a.append(str(loss_test.item()))
        a.append(str(f1))
        writer.writerow(a)


if __name__ == '__main__':

    datasets = ['raw', 'ca_ca_eoa_GNN', 'eoa_ca_eoa_GNN', 'all']
    # datasets = ['eoa_ca_eoa_GNN', 'all']
    for dataset0 in datasets:
        path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', dataset0)
        dataset = Planetoid(path, dataset0, transform=ColumnNormalizeFeatures())
        data = dataset[0]
        if args.use_gdc:
            gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
                        normalization_out='col',
                        diffusion_kwargs=dict(method='ppr', alpha=0.5),
                        sparsification_kwargs=dict(method='topk', k=128,
                                                   dim=0), exact=True)
            data = gdc(data)

        lr_ = [0.005]
        hidden_ = [128]
        for i in range(10):
            for hidden in hidden_:
                for lr in lr_:
                    with open(mkdir(f'./result/') + f'gcn_{dataset0}.csv', 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([f'hidden',f'lr','loss', 'f1'])
                    for _iter in range(len(data.train_index)):
                        print(f"第{_iter + 1}折")
                        y_train = data.train_index[_iter]
                        y_test = data.test_index[_iter]
                        train_mask = data.train_mask[_iter]
                        test_mask = data.test_mask[_iter]


                        model, data = Net(input_fea=dataset.num_features, hidden=hidden, output_fea=2).to(device), data.to(device)
                        optimizer = torch.optim.Adam([
                            dict(params=model.conv1.parameters(), weight_decay=5e-4),
                            dict(params=model.conv2.parameters(), weight_decay=1e-4)
                        ], lr=lr)  # Only perform weight-decay on first convolution.

                        y_train = torch.tensor(data.y[y_train])
                        y_test = torch.tensor(data.y[y_test])
                        train_mask = torch.tensor(train_mask)
                        test_mask = torch.tensor(test_mask)
                        y_train = y_train.cuda()
                        y_test = y_test.cuda()
                        train_mask = train_mask.cuda()
                        test_mask = test_mask.cuda()
                        predict_fn = lambda output: output.max(1, keepdim=True)[1].detach().cpu()
                        print("Optimization Finished!")

                        early_stopping = EarlyStopping(patience=200, verbose=True)
                        for epoch in range(1, 2001):
                            lost, f1 = train(epoch, train_mask)
                            if f1 > 0.75:
                                early_stopping(lost, model)
                                if early_stopping.early_stop:
                                    print("Early stopping")
                                    break
                            # val(val_mask)
                        model.load_state_dict(torch.load('checkpoint.pt'))
                        test(test_mask)
