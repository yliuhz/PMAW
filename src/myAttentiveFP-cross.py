
import torch
import torch.nn as nn

from dgllife.model.gnn import AttentiveFPGNN
from dgllife.model.readout import AttentiveFPReadout

import deepchem as dc # 用于数据处理
import os
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

import dgl

from pylab import *
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
import numpy as np
import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import StepLR

class mulAttentiveFP(torch.nn.Module):
    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 num_layers=2,
                 num_timesteps=2,
                 graph_feat_size=200,
                 n_tasks=1,
                 dropout=0.):
        super(mulAttentiveFP, self).__init__()

        self.gnn_1 = AttentiveFPGNN(node_feat_size=node_feat_size,
                                  edge_feat_size=edge_feat_size,
                                  num_layers=num_layers,
                                  graph_feat_size=graph_feat_size,
                                  dropout=dropout)
        self.readout_1 = AttentiveFPReadout(feat_size=graph_feat_size,
                                          num_timesteps=num_timesteps,
                                          dropout=dropout)

        self.gnn_2 = AttentiveFPGNN(node_feat_size=node_feat_size,
                                  edge_feat_size=edge_feat_size,
                                  num_layers=num_layers,
                                  graph_feat_size=graph_feat_size,
                                  dropout=dropout)
        self.readout_2 = AttentiveFPReadout(feat_size=graph_feat_size,
                                          num_timesteps=num_timesteps,
                                          dropout=dropout)
        
        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(graph_feat_size, n_tasks)
        )

    def forward(self, g_1, node_feats_1, edge_feats_1, g_2, node_feats_2, edge_feats_2, 
        get_node_weight=False):
        node_feats_1 = self.gnn_1(g_1, node_feats_1, edge_feats_1)
        # g_feats_1 = self.readout_1(g_1, node_feats_1, get_node_weight)
        g_feats_1 = self.readout_1(g_1, node_feats_1)

        node_feats_2 = self.gnn_2(g_2, node_feats_2, edge_feats_2)
        # g_feats_2 = self.readout_2(g_2, node_feats_2, get_node_weight)
        g_feats_2 = self.readout_2(g_2, node_feats_2)

        g_feats = g_feats_1 + g_feats_2

        return self.predict(g_feats).view(-1,)

def get_batch(inputs, labels, st, batch_size):
    n = len(inputs)
    ed = min(n, st+batch_size)
    inputs = inputs[st:ed]

    g_1s = [inp[0] for inp in inputs]
    g_2s = [inp[1] for inp in inputs]

    g_1s = dgl.batch(g_1s)
    g_2s = dgl.batch(g_2s)

    node_1s = g_1s.ndata['x']
    node_2s = g_2s.ndata['x']
    
    edge_1s = g_1s.edata['edge_attr']
    edge_2s = g_2s.edata['edge_attr']

    # node_1s = [g_1.ndata['x'] for g_1 in g_1s]
    # node_2s = [g_2.ndata['x'] for g_2 in g_2s]

    # edge_1s = [g_1.edata['edge_attr'] for g_1 in g_1s]
    # edge_2s = [g_2.edata['edge_attr'] for g_2 in g_2s]

    labels = torch.as_tensor(labels[st:ed])

    return g_1s, node_1s, edge_1s, g_2s, node_2s, edge_2s, labels
    
def mean_relative_error(y_pred, y_test):
    assert len(y_pred) == len(y_test)
    mre = 0.0
    for i in range(len(y_pred)):
        mre = mre + abs((y_pred[i] - y_test[i]) / y_test[i])
    mre = mre * 100/ len(y_pred)
    return mre

class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))

if __name__=='__main__':

    config = {
      'train_epochs': 2000,
      'eval_epochs': 1, 
      'batch_size': 512,
      'alpha': 1.0,
      'train_shuffle': True,
      'eval_shuffle': False,
      'device': 'cuda:0',
      'number_atom_features': 30,
      'number_bond_features': 11,
      'num_layers': 2,
      'num_timesteps': 2,
      'graph_feat_size': 200,
      'dropout': 0,
      'n_classes': 2,
      'nfeat_name': 'x',
      'efeat_name': 'edge_attr',
      'learning_rate': 1e-3
    }

    datadir = './data/database/'
    mol_file = "22-01-29-mol-smiles-train.csv"
    sol_file = "22-01-29-sol-smiles-train.csv"

    mol_df = pd.read_csv(os.path.join(datadir,mol_file), encoding='gb18030')
    sol_df = pd.read_csv(os.path.join(datadir,sol_file), encoding='gb18030')

    mol_smiles = mol_df['molecule'].tolist()
    mol_labels = mol_df['label'].tolist()
    sol_smiles = sol_df['solvent'].tolist()
    sol_labels = sol_df['label'].tolist()

    assert mol_labels == sol_labels

    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    print('featurizing molecule')
    mol_X = featurizer.featurize(mol_smiles)
    print('featurizing solvent')
    sol_X = featurizer.featurize(sol_smiles)

    dgl_graphs_1 = [
        graph.to_dgl_graph(self_loop=True).to(config['device']) for graph in mol_X
    ]

    dgl_graphs_2 = [
        graph.to_dgl_graph(self_loop=True).to(config['device']) for graph in sol_X
    ]

    inputs = list(zip(dgl_graphs_1, dgl_graphs_2))
    assert len(inputs) == len(mol_labels)
    assert mol_labels == sol_labels

    mol_labels = torch.as_tensor(mol_labels).view(-1,1)

    min_max_scaler_y = MinMaxScaler()
    min_max_scaler_y.fit(mol_labels)
    mol_labels = torch.as_tensor(min_max_scaler_y.transform(mol_labels), dtype=torch.float)
    mol_labels = mol_labels.view(-1,)
    mol_labels = mol_labels.to(config['device'])

    all_eval_logits, all_eval_labels, all_eval_mres = [], [], []
    all_train_logits, all_train_labels = [], []

    for j in range(10):
        kf = KFold(n_splits=10, shuffle=True, random_state=j)

        for train_index, test_index in kf.split(inputs):
            X_train = np.array(inputs)[train_index]
            y_train = mol_labels[train_index]
            X_test = np.array(inputs)[test_index]
            y_test = mol_labels[test_index]

            model = mulAttentiveFP(
                node_feat_size=config['number_atom_features'],
                edge_feat_size=config['number_bond_features'],
                num_layers=config['num_layers'],
                num_timesteps=config['num_timesteps'],
                graph_feat_size=config['graph_feat_size'],
                n_tasks=1,
                dropout=config['dropout']) 

            # print(inputs)
            # print(mol_labels)
            
            model.to(config['device'])
            model.train()
            optimizer = torch.optim.Adam(params=model.parameters(), lr=config['learning_rate'])
            # lossF = nn.MSELoss()
            lossF = LogCoshLoss()
            losses = []
            scheduler = StepLR(optimizer, step_size=500, gamma=0.1)

            for epoch in range(config['train_epochs']):
                # for idx, inp in enumerate(inputs):
                for i in range(0, len(X_train), config['batch_size']):
                    optimizer.zero_grad()
                    
                    g_1s, node_1s, edge_1s, g_2s, node_2s, edge_2s, labels = get_batch(X_train, y_train, i, config['batch_size'])

                    logits = model(g_1s, node_1s, edge_1s, g_2s, node_2s, edge_2s, labels)

                    # logits = torch.as_tensor(logits)
                    # labels = torch.as_tensor(labels)
                    # logits.requires_grad = True

                    loss = lossF(logits, labels)
                    loss.backward()
                    optimizer.step()

                    losses.append(loss.detach().cpu().item())

                    print('epoch:{:d}/{:d} iter:{:d}/{:d} loss:{:.4f} lr={:.6f}'.format(epoch+1, config['train_epochs'], (i+1)//config['batch_size'], len(inputs)//config['batch_size'], sum(losses)/len(losses), scheduler.get_last_lr()[0]))

                scheduler.step()
                print('epoch:{:d} loss:{:.4f}'.format(epoch+1, sum(losses)/len(losses)))

            model.eval()
            logits, labels = [], []
            with torch.no_grad():
                for epoch in range(config['eval_epochs']):
                    for idx, inp in enumerate(X_test):

                        g_1, g_2 = X_test[idx]
                        label = y_test[idx]

                        node_feats_1 = g_1.ndata[config['nfeat_name']]
                        edge_feats_1 = g_1.edata[config['efeat_name']]
                        node_feats_2 = g_2.ndata[config['nfeat_name']]
                        edge_feats_2 = g_2.edata[config['efeat_name']]

                        logit = model(g_1, node_feats_1, edge_feats_1, g_2, node_feats_2, edge_feats_2)

                        logits.append(logit.cpu().item())
                        labels.append(label.cpu().item())

                logits_np = np.reshape(logits, (-1,1))
                labels_np = np.reshape(labels, (-1,1))
                logits_np = min_max_scaler_y.inverse_transform(logits_np)
                labels_np = min_max_scaler_y.inverse_transform(labels_np)

                mre = mean_relative_error(logits_np, labels_np)
                all_eval_logits += list(logits_np)
                all_eval_labels += list(labels_np)
                all_eval_mres.append(mre)

                # 添加训练数据预测真实表
                logits_, labels_ = [], []
                for epoch in range(config['eval_epochs']):
                    for idx, inp in enumerate(X_train):

                        g_1, g_2 = X_train[idx]
                        label = y_train[idx]

                        node_feats_1 = g_1.ndata[config['nfeat_name']]
                        edge_feats_1 = g_1.edata[config['efeat_name']]
                        node_feats_2 = g_2.ndata[config['nfeat_name']]
                        edge_feats_2 = g_2.edata[config['efeat_name']]

                        logit = model(g_1, node_feats_1, edge_feats_1, g_2, node_feats_2, edge_feats_2)

                        logits_.append(logit.cpu().item())
                        labels_.append(label.cpu().item())

                logits_np = np.reshape(logits_, (-1,1))
                labels_np = np.reshape(labels_, (-1,1))
                logits_np = min_max_scaler_y.inverse_transform(logits_np).reshape(-1,)
                labels_np = min_max_scaler_y.inverse_transform(labels_np).reshape(-1,)

                # mre = mean_relative_error(logits_np, labels_np)
                all_train_logits += list(logits_np)
                all_train_labels += list(labels_np)
    
    mre = mean_relative_error(all_eval_logits, all_eval_labels)

    ## 白+绿纯色颜色映射
    clist = ['white', 'purple', 'black']
    newcmp = LinearSegmentedColormap.from_list('chaos',clist)

    in_y_pred = np.reshape(all_eval_logits, (-1,))
    in_y_test = np.reshape(all_eval_labels, (-1,))

    xmin = in_y_test.min()
    # xmin = min(xmin, in_y_pred.min())
    xmax = in_y_test.max()
    # xmax = max(xmax, in_y_pred.max())

    fig = plt.figure(figsize=(14, 10))
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    # plt.grid(linestyle="--")
    plt.xlabel('Real values for lambda(mm)', fontsize=20)
    plt.ylabel('Predicted values for lambda(mm)', fontsize=20)
    plt.yticks(size=16)
    plt.xticks(size=16)
    plt.plot([xmin, xmax], [xmin, xmax], ':', linewidth=1.5, color='gray')
    print('MRE', all_eval_mres)
    print('avg MRE', sum(all_eval_mres) / len(all_eval_mres))
    print('max MRE', max(all_eval_mres))
    print('min MRE', min(all_eval_mres))

    errstr = 'MRE=%.2f%%' % (sum(all_eval_mres) / len(all_eval_mres))
    plt.text(xmin + 50, xmax - 130, errstr, fontsize=20, weight='bold')

    cross_result = {'Real lambda': in_y_test, 'Predicted lambda': in_y_pred}
    cross_result = pd.DataFrame(cross_result)
    cross_result.to_csv('Out/cross_result_attentiveFP.csv', index=False, encoding='gb18030')

    # for i in range(len(in_y_pred)):
        # plt.scatter(in_y_test[i], in_y_pred[i], edgecolors='b')
    hexf = plt.hexbin(in_y_test, in_y_pred, gridsize=20, extent=[xmin, xmax, xmin, xmax],
            cmap=newcmp)
    # xmin = np.array(in_y_test).min()
    # xmax = np.array(in_y_test).max()
    # ymin = np.array(in_y_pred).min()
    # ymax = np.array(in_y_pred).max()
    plt.axis([xmin, xmax, xmin, xmax])
    ax = plt.gca()
    ax.tick_params(top=True, right=True)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=16)
    plt.savefig('pics/fig-attentiveFP.png')
    plt.show()

    cross_result = {'Real lambda': all_train_labels, 'Predicted lambda': all_train_logits}
    cross_result = pd.DataFrame(cross_result)
    cross_result.to_csv('Out/cross_result_attentiveFP_train.csv', index=False, encoding='gb18030')