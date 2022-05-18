import argparse
import csv
import os.path as osp
import numpy as np
import torch_geometric
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score
import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import GINConv, global_add_pool

from tools1 import EarlyStopping, ColumnNormalizeFeatures, mkdir
from dataset import Planetoid
import torch_geometric.transforms as T
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, help='gpu id', default='3')

args = parser.parse_args()
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

class Net(torch.nn.Module):
    def __init__(self,input_fea, hidden, output_fea):
        super().__init__()
        # self.aggr = 'means'
        self.conv1 = GINConv(
            Sequential(Linear(input_fea, hidden), ReLU(),
                       Linear(hidden, hidden), ReLU()
                       ))
        self.conv2 = GINConv(
            Sequential(Linear(hidden, hidden), ReLU(),
                       ))
        self.conv3 = GINConv(
            Sequential(Linear(hidden, hidden), ReLU(),
                      ))
        self.conv4 = GINConv(
            Sequential(Linear(hidden, hidden), ReLU(),
                      ))
        self.conv5 = GINConv(
            Sequential(Linear(hidden, output_fea)
                      ))
        # self.lin = Linear(hidden, output_fea)

    def forward(self,):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        # x = self.conv2(x, edge_index)
        # x = self.conv3(x, edge_index)
        # x = self.conv4(x, edge_index)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv5(x, edge_index)
        # x = self.lin(x)
        return F.log_softmax(x, dim=-1)

# ----------------------------
def train(train_mask):
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

    return loss_train, f1_train


def val(val_mask):
    model.eval()
    output = model()
    preds_val = output[val_mask]
    reals_val = y_val
    preds_val_cpu = predict_fn(output[val_mask])
    reals_val_cpu = reals_val.detach().cpu().view_as(preds_val_cpu)
    preds_val_cpu = np.hstack(preds_val_cpu)
    reals_val_cpu = np.hstack(reals_val_cpu)
    loss_val = F.nll_loss(preds_val,reals_val)
    f1_val = f1_score(reals_val_cpu, preds_val_cpu, average="micro")
    return loss_val, f1_val


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
    with open(mkdir(f'./result/') + f'gin_{dataset0}.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        a = []
        a.append(hidden)
        a.append(lr)
        a.append(str(loss_test.item()))
        a.append(str(f1))
        writer.writerow(a)


if __name__ == '__main__':
    datasets = ['raw', 'ca_ca_eoa_GNN', 'eoa_ca_eoa_GNN', 'all']
    for dataset0 in datasets:
        path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', dataset0)
        dataset = Planetoid(path, dataset0, transform=T.NormalizeFeatures())
        data = dataset[0]

        lr_ = [0.01]
        hidden_ = [128]
        for hidden in hidden_:
            for lr in lr_:
                for i in range(10):
                    with open(mkdir(f'./result/') + f'gin_{dataset0}.csv', 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([f'hidden',f'lr','loss', 'f1'])
                    for _iter in range(len(data.train_index)):
                        print(f"第{_iter + 1}折")
                        y_train = data.train_index[_iter]
                        # [1,0]  [0,1]
                        y_val = data.val_index[_iter]
                        # y_train = y_train+y_val
                        y_test = data.test_index[_iter]
                        train_mask = data.train_mask[_iter]
                        val_mask = data.val_mask[_iter]
                        # train_mask = train_mask+val_mask
                        test_mask = data.test_mask[_iter]

                        model= Net(input_fea=dataset.num_features, hidden=hidden, output_fea=2).to(device)
                        data = data.to(device)
                        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                        y_train = data.y[y_train].clone().detach()
                        y_val = data.y[y_val].clone().detach()

                        y_test = data.y[y_test].clone().detach()
                        train_mask = train_mask.clone().detach()
                        val_mask = val_mask.clone().detach()

                        test_mask = test_mask.clone().detach()
                        y_train = y_train.to(device)
                        y_val = y_val.to(device)

                        y_test = y_test.to(device)
                        train_mask = train_mask.to(device)
                        val_mask = val_mask.to(device)

                        test_mask = test_mask.to(device)
                        predict_fn = lambda output: output.max(1, keepdim=True)[1].detach().cpu()
                        print("Optimization Finished!")


                        early_stopping = EarlyStopping(patience=300, verbose=True)
                        for epoch in range(1, 1001):
                            train_lost, train_f1 = train(train_mask)
                            lost, f1 = val(val_mask)

                            print('Epoch: {:04d}'.format(epoch),
                                  'loss_train: {:.4f}'.format(train_lost),
                                  'f1_train: {:.4f}'.format(train_f1),
                                  "loss_val= {:.4f}".format(lost),
                                  "f1_val= {:.4f}".format(f1)
                                  )
                            if f1 > 0.7:
                                early_stopping(lost, model)
                                if early_stopping.early_stop:
                                    print("Early stopping")
                                    break
                        model.load_state_dict(torch.load('checkpoint1.pt'))
                        test(test_mask)

