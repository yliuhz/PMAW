# TIPS: only used to find the best epoch of GCN


from math import gamma
from sklearn.metrics import mean_absolute_percentage_error
import torch
import torch.nn as nn

from dgllife.model.gnn import AttentiveFPGNN
from dgllife.model.readout import AttentiveFPReadout, MLPNodeReadout
from dgllife.model.gnn.gcn import GCNLayer

import deepchem as dc # 用于数据处理
import os
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

import dgl
import numpy as np

import torch.nn.functional as F

import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import StepLR
from torchsummary import summary

class mulGCN(torch.nn.Module):
    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 num_layers=2,
                 num_timesteps=2,
                 graph_feat_size=200,
                 n_tasks=1,
                 dropout=0.):
        super(mulGCN, self).__init__()

        self.gnn_1_1 = GCNLayer(in_feats=node_feat_size,
                                  out_feats=node_feat_size,
                                  gnn_norm='none',
                                  activation=F.relu,
                                  residual=True,
                                  batchnorm=False,
                                  dropout=dropout)
        self.gnn_1_2 = GCNLayer(in_feats=node_feat_size,
                                  out_feats=node_feat_size,
                                  gnn_norm='none',
                                  activation=F.relu,
                                  residual=True,
                                  batchnorm=False,
                                  dropout=dropout)
        self.readout_1 = MLPNodeReadout(node_feats=node_feat_size,
                                        hidden_feats=node_feat_size,
                                        graph_feats=graph_feat_size,
                                        activation=F.relu,
                                        mode='sum')

        self.gnn_2_1 = GCNLayer(in_feats=node_feat_size,
                                  out_feats=node_feat_size,
                                  gnn_norm='none',
                                  activation=F.relu,
                                  residual=True,
                                  batchnorm=False,
                                  dropout=dropout)
        self.gnn_2_2 = GCNLayer(in_feats=node_feat_size,
                                  out_feats=node_feat_size,
                                  gnn_norm='none',
                                  activation=F.relu,
                                  residual=True,
                                  batchnorm=False,
                                  dropout=dropout)
        self.readout_2 = MLPNodeReadout(node_feats=node_feat_size,
                                        hidden_feats=node_feat_size,
                                        graph_feats=graph_feat_size,
                                        activation=F.relu,
                                        mode='sum')
        
        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(graph_feat_size, n_tasks)
        )

    def forward(self, g_1, node_feats_1, edge_feats_1, g_2, node_feats_2, edge_feats_2, 
        get_node_weight=False):
        node_feats_1 = self.gnn_1_1(g_1, node_feats_1)
        # node_feats_1 = self.gnn_1_2(g_1, node_feats_1)
        g_feats_1 = self.readout_1(g_1, node_feats_1)

        node_feats_2 = self.gnn_2_1(g_2, node_feats_2)
        # node_feats_2 = self.gnn_2_2(g_2, node_feats_2)
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
    
def load_from_file(mol_file, sol_file, datadir):
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

    mol_labels = torch.as_tensor(mol_labels, dtype=torch.float)

    return inputs, mol_labels


if __name__=='__main__':

    config = {
      'train_epochs': 2000,
      'eval_epochs': 1, 
      'batch_size': 128,
      'alpha': 1.0,
      'train_shuffle': True,
      'eval_shuffle': False,
      'device': 'cuda:0',
      'number_atom_features': 30,
      'number_bond_features': 11,
      'num_layers': 2,
      'num_timesteps': 2,
      'graph_feat_size': 64,
      'dropout': 0,
      'n_classes': 2,
      'nfeat_name': 'x',
      'efeat_name': 'edge_attr',
      'learning_rate': 1e-2
    }

    datadir = './data/database'
    mol_file = "22-01-29-mol-smiles-train.csv"
    sol_file = "22-01-29-sol-smiles-train.csv"

    # inputs, mol_labels = load_from_file(mol_file, sol_file, datadir)

    model = mulGCN(
        node_feat_size=config['number_atom_features'],
        edge_feat_size=config['number_bond_features'],
        num_layers=config['num_layers'],
        num_timesteps=config['num_timesteps'],
        graph_feat_size=config['graph_feat_size'],
        n_tasks=1,
        dropout=config['dropout'])

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
    exit(0)

    mol_labels = torch.as_tensor(mol_labels).view(-1,1)

    min_max_scaler_y = MinMaxScaler()
    min_max_scaler_y.fit(mol_labels)
    mol_labels = torch.as_tensor(min_max_scaler_y.transform(mol_labels), dtype=torch.float)
    mol_labels = mol_labels.view(-1,)
    mol_labels = mol_labels.to(config['device'])

    # print(inputs)
    # print(mol_labels)
    
    logits, labels = [], []
    mol_test_file = '22-01-29-mol-smiles-test-level-1.csv'
    sol_test_file = '22-01-29-sol-smiles-test-level-1.csv'

    test_inputs, test_labels = load_from_file(mol_test_file, sol_test_file, datadir)

    model.to(config['device'])
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config['learning_rate'])
    lossF = nn.MSELoss()
    losses = []
    eval_mres = []
    scheduler = StepLR(optimizer, step_size=500, gamma=0.1)

    for epoch in range(config['train_epochs']):
        model.train()
        # for idx, inp in enumerate(inputs):
        cur_losses = []
        for i in range(0, len(inputs), config['batch_size']):
            optimizer.zero_grad()

            # logits = []
            # labels = []
            # for idx in range(i, min(i+config['batch_size'],len(inputs))):
            #     g_1, g_2 = inputs[idx]
            #     labels.append(mol_labels[idx])

            #     node_feats_1 = g_1.ndata[config['nfeat_name']]
            #     edge_feats_1 = g_1.edata[config['efeat_name']]
            #     node_feats_2 = g_2.ndata[config['nfeat_name']]
            #     edge_feats_2 = g_2.edata[config['efeat_name']]

            #     logit = model(g_1, node_feats_1, edge_feats_1, g_2, node_feats_2, edge_feats_2)
            #     logits.append(logit)

            #     print('???',logit)
            #     print('???',mol_labels[idx])
            
            g_1s, node_1s, edge_1s, g_2s, node_2s, edge_2s, labels = get_batch(inputs, mol_labels, i, config['batch_size'])

            logits = model(g_1s, node_1s, edge_1s, g_2s, node_2s, edge_2s, labels)

            # logits = torch.as_tensor(logits)
            # labels = torch.as_tensor(labels)
            # logits.requires_grad = True

            loss = lossF(logits, labels)
            loss.backward()
            optimizer.step()

            cur_losses.append(loss.detach().cpu().item())
        losses.append(sum(cur_losses)/len(cur_losses))
            # print('epoch:{:d}/{:d} iter:{:d}/{:d} loss:{:.4f} lr={:.6f}'.format(epoch+1, config['train_epochs'], (i+1)//config['batch_size'], len(inputs)//config['batch_size'], sum(losses)/len(losses), scheduler.get_last_lr()[0]))

        scheduler.step()
        # print('epoch:{:d} loss:{:.4f}'.format(epoch+1, sum(losses)/len(losses)))

        model.eval()
        labels, logits = [], []
        with torch.no_grad():
            for val_epoch in range(config['eval_epochs']):
                for idx, inp in enumerate(test_inputs):

                    g_1, g_2 = test_inputs[idx]
                    label = test_labels[idx]

                    node_feats_1 = g_1.ndata[config['nfeat_name']]
                    edge_feats_1 = g_1.edata[config['efeat_name']]
                    node_feats_2 = g_2.ndata[config['nfeat_name']]
                    edge_feats_2 = g_2.edata[config['efeat_name']]

                    logit = model(g_1, node_feats_1, edge_feats_1, g_2, node_feats_2, edge_feats_2)

                    logits_np = np.reshape(logit.cpu(), (-1,1))
                    logits_np = min_max_scaler_y.inverse_transform(logits_np).reshape(-1,)

                    logits += list(logits_np)
                    labels.append(label.cpu().tolist())

        mre = mean_absolute_percentage_error(labels, logits)
        eval_mres.append(mre)
        
        print('epoch:{:d} loss:{:.4f} val_mre:{:.4f}'.format(epoch+1, sum(losses)/len(losses), mre))

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot([x for x in range(len(losses))], losses, 'b', label='loss')
    ax1.set_ylabel('loss', color='b')
    ax2.plot([x for x in range(len(eval_mres))], eval_mres, 'r', label='eval_mre')
    ax2.set_ylabel('eval_mre', color='r')
    ax1.set_xlabel('epochs')
    plt.title('Training of GCN')
    plt.savefig('pics/Training_of_GCN.png')

    import os
    outdir = 'Out/losses_and_mres'
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, 'gcn.txt'), 'w') as f:
        f.write('loss\n')
        f.write(' '.join([str(x) for x in losses]))
        f.write('\n')
        f.write('mres\n')
        f.write(' '.join([str(x) for x in eval_mres]))