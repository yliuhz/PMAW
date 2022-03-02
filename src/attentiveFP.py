
import os
from re import S
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import deepchem as dc
from deepchem.molnet.load_function.molnet_loader import TransformerGenerator, _MolnetLoader
from deepchem.data import Dataset
from typing import List, Optional, Tuple, Union
import pandas as pd

import pdb
import logging
logging.basicConfig(level = logging.INFO)

from deepchem.metrics import to_one_hot
from deepchem.feat.mol_graphs import ConvMol
import numpy as np
from deepchem.models.layers import GraphConv, GraphPool, GraphGather
import tensorflow as tf
import keras.layers as layers
import keras

# from deepchem.models.torch_models.attentivefp import AttentiveFP, AttentiveFPModel
import torch
from torch import nn
from dgllife.model.gnn import AttentiveFPGNN
from dgllife.model.readout import AttentiveFPReadout

import torch.nn.functional as F

from deepchem.models.losses import Loss, L2Loss, SparseSoftmaxCrossEntropy
from deepchem.models.torch_models.torch_model import TorchModel

TOXCAST_URL = ""
TOXCAST_TASKS = [
    'λabs (nm)',
]

class _DescriptorLoader(_MolnetLoader):
  def __init__(self, featurizer: Union[dc.feat.Featurizer, str], splitter: Union[dc.splits.Splitter, str, None], transformer_generators: List[Union[TransformerGenerator, str]], tasks: List[str], data_dir: Optional[str], save_dir: Optional[str], filename: str, **kwargs):
      super().__init__(featurizer, splitter, transformer_generators, tasks, data_dir, save_dir, **kwargs)
      
      self.filename = filename


  def create_dataset(self) -> Dataset:
    dataset_file = os.path.join(self.data_dir, self.filename)
    if not os.path.exists(dataset_file):
      dc.utils.data_utils.download_url(url=TOXCAST_URL, dest_dir=self.data_dir)
    loader = dc.data.CSVLoader(
        tasks=self.tasks, feature_field="smiles", featurizer=self.featurizer)

    # pdb.set_trace()
    ret = loader.create_dataset(dataset_file)
    # pdb.set_trace()

    return ret

def load_descriptor(
    featurizer: Union[dc.feat.Featurizer, str] = 'ECFP',
    splitter: Union[dc.splits.Splitter, str, None] = 'index', 
    # 'scaffold',
    transformers: List[Union[TransformerGenerator, str]] = ['minmax'],
    # = ['balancing'],
    reload: bool = True,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    **kwargs
) -> Tuple[List[str], Tuple[Dataset, ...], List[dc.trans.Transformer]]:
  """Load Toxcast dataset

  ToxCast is an extended data collection from the same
  initiative as Tox21, providing toxicology data for a large
  library of compounds based on in vitro high-throughput
  screening. The processed collection includes qualitative
  results of over 600 experiments on 8k compounds.

  Random splitting is recommended for this dataset.

  The raw data csv file contains columns below:

  - "smiles": SMILES representation of the molecular structure
  - "ACEA_T47D_80hr_Negative" ~ "Tanguay_ZF_120hpf_YSE_up": Bioassays results.
    Please refer to the section "high-throughput assay information" at
    https://www.epa.gov/chemical-research/toxicity-forecaster-toxcasttm-data
    for details.

  Parameters
  ----------
  featurizer: Featurizer or str
    the featurizer to use for processing the data.  Alternatively you can pass
    one of the names from dc.molnet.featurizers as a shortcut.
  splitter: Splitter or str
    the splitter to use for splitting the data into training, validation, and
    test sets.  Alternatively you can pass one of the names from
    dc.molnet.splitters as a shortcut.  If this is None, all the data
    will be included in a single dataset.
  transformers: list of TransformerGenerators or strings
    the Transformers to apply to the data.  Each one is specified by a
    TransformerGenerator or, as a shortcut, one of the names from
    dc.molnet.transformers.
  reload: bool
    if True, the first call for a particular featurizer and splitter will cache
    the datasets to disk, and subsequent calls will reload the cached datasets.
  data_dir: str
    a directory to save the raw data in
  save_dir: str
    a directory to save the dataset in

  References
  ----------
  .. [1] Richard, Ann M., et al. "ToxCast chemical landscape: paving the road
     to 21st century toxicology." Chemical research in toxicology 29.8 (2016):
     1225-1251.
  """
  loader = _DescriptorLoader(featurizer, splitter, transformers, TOXCAST_TASKS,
                          data_dir, save_dir, **kwargs)

  print(123)

  # pdb.set_trace()
  ret = loader.load_dataset('toxcast', reload=False, frac_train=0.8, frac_val=0.1, frac_test=0.1)
  # pdb.set_trace()

  return ret

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
        g_feats_1 = self.readout_1(g_1, node_feats_1, get_node_weight)

        node_feats_2 = self.gnn_2(g_2, node_feats_2, edge_feats_2)
        g_feats_2 = self.readout_2(g_2, node_feats_2, get_node_weight)

        g_feats = g_feats_1 + g_feats_2

        return self.predict(g_feats)
        # if get_node_weight:
        #     g_feats, node_weights = self.readout(g, node_feats, get_node_weight)
        #     return self.predict(g_feats), node_weights
        # else:
        #     g_feats = self.readout(g, node_feats, get_node_weight)
        #     return self.predict(g_feats)

class mulAttentiveFP1Input(nn.Module):
    def __init__(self,
               n_tasks: int,
               num_layers: int = 2,
               num_timesteps: int = 2,
               graph_feat_size: int = 200,
               dropout: float = 0.,
               mode: str = 'regression',
               number_atom_features: int = 30,
               number_bond_features: int = 11,
               n_classes: int = 2,
               nfeat_name: str = 'x',
               efeat_name: str = 'edge_attr'):
        try:
            import dgl
        except:
            raise ImportError('This class requires dgl.')
        try:
            import dgllife
        except:
            raise ImportError('This class requires dgllife.')

        if mode not in ['classification', 'regression']:
            raise ValueError("mode must be either 'classification' or 'regression'")

        super(mulAttentiveFP1Input, self).__init__()

        self.n_tasks = n_tasks
        self.mode = mode
        self.n_classes = n_classes
        self.nfeat_name = nfeat_name
        self.efeat_name = efeat_name
        if mode == 'classification':
            out_size = n_tasks * n_classes
        else:
            out_size = n_tasks

        # from dgllife.model import AttentiveFPPredictor as DGLAttentiveFPPredictor

        self.model = mulAttentiveFP(
            node_feat_size=number_atom_features,
            edge_feat_size=number_bond_features,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
            graph_feat_size=graph_feat_size,
            n_tasks=out_size,
            dropout=dropout)

    def forward(self, g):
        preds = []
        for g_ in g:
            g_1, g_2 = g_
            node_feats_1 = g_1.ndata[self.nfeat_name]
            edge_feats_1 = g_1.edata[self.efeat_name]
            node_feats_2 = g_2.ndata[self.nfeat_name]
            edge_feats_2 = g_2.edata[self.efeat_name]

            out = self.model(g_1, node_feats_1, edge_feats_1, g_2, node_feats_2, edge_feats_2)

            if self.mode == 'classification':
                if self.n_tasks == 1:
                    logits = out.view(-1, self.n_classes)
                    softmax_dim = 1
                else:
                    logits = out.view(-1, self.n_tasks, self.n_classes)
                    softmax_dim = 2
                proba = F.softmax(logits, dim=softmax_dim)
                return proba, logits
            else:
                preds.append(out)
        # print('preds:', torch.tensor(preds))
        return torch.tensor(preds)

class mulAttentiveFPModel(TorchModel):
    def __init__(self,
               n_tasks: int,
               num_layers: int = 2,
               num_timesteps: int = 2,
               graph_feat_size: int = 200,
               dropout: float = 0.,
               mode: str = 'regression',
               number_atom_features: int = 30,
               number_bond_features: int = 11,
               n_classes: int = 2,
               self_loop: bool = True,
               **kwargs):

        model = mulAttentiveFP1Input(
            n_tasks=n_tasks,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
            graph_feat_size=graph_feat_size,
            dropout=dropout,
            mode=mode,
            number_atom_features=number_atom_features,
            number_bond_features=number_bond_features,
            n_classes=n_classes)
        if mode == 'regression':
            loss = L2Loss()
            output_types = ['prediction']
        else:
            loss = SparseSoftmaxCrossEntropy()
        # output_types = ['prediction', 'loss']
        super(mulAttentiveFPModel, self).__init__(
            model=model, loss=loss, output_types=output_types, **kwargs)

        self._self_loop = self_loop

    def _prepare_batch(self, batch):
        try:
            import dgl
        except:
            raise ImportError('This class requires dgl.')

        inputs, labels, weights = batch
        dgl_graphs_1 = [
            graph.to_dgl_graph(self_loop=self._self_loop).to(self.device) for graph in inputs[0]
        ]
        # inputs_1 = dgl.batch(dgl_graphs).to(self.device)

        dgl_graphs_2 = [
            graph.to_dgl_graph(self_loop=self._self_loop).to(self.device) for graph in inputs[1]
        ]
        # inputs_2 = dgl.batch(dgl_graphs).to(self.device)

        inputs = list(zip(dgl_graphs_1, dgl_graphs_2))

        # _, labels, weights = super(mulAttentiveFPModel, self)._prepare_batch(
        #     ([], labels, weights))

        # print('???', len(inputs), len(labels), len(weights))
        return inputs, labels, weights
        

def data_generator(mol_file, sol_file, datadir, epochs=50, shuffle=False):
    for epoch in range(epochs):
        mol_df = pd.read_csv(os.path.join(datadir,mol_file), encoding='utf-8')
        sol_df = pd.read_csv(os.path.join(datadir,sol_file), encoding='utf-8')

        mol_smiles = mol_df['smiles'].tolist()[:2]
        mol_labels = mol_df[TOXCAST_TASKS[0]].tolist()[:2]
        sol_smiles = sol_df['smiles'].tolist()[:2]
        sol_labels = sol_df[TOXCAST_TASKS[0]].tolist()[:2]

        assert mol_labels == sol_labels

        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
        mol_X = featurizer.featurize(mol_smiles)
        sol_X = featurizer.featurize(sol_smiles)

        # mol_labels = [np.array([x]) for x in mol_labels]
        mol_labels = torch.tensor(mol_labels)
        # print('labels', mol_labels)

        mol_weights = torch.ones_like(mol_labels)

        yield (mol_X, sol_X), mol_labels, mol_weights


if __name__=='__main__':
    datadir = './data/descriptor/'
    mol_file = "01-15-descriptor-train-feat.csv"
    sol_file = "01-15-descriptor-train-feat.csv"
    n_tasks = len(TOXCAST_TASKS)
    transformers = []

    config = {
      'train_epochs': 50,
      'eval_epochs': 1, 
      'batch_size': 8,
      'alpha': 1.0,
      'train_shuffle': True,
      'eval_shuffle': False,
      'device': 'cuda:0'
    }

    # model = dc.models.KerasModel(mulgcn(config['batch_size'], config['alpha']), loss=dc.models.losses.L2Loss())
    
    # model = dc.models.GraphConvModel(n_tasks=n_tasks, mode='regression')
    model = mulAttentiveFPModel(n_tasks=n_tasks, mode='regression')
    # model.fit(train_dataset, nb_epoch=50)

    model.fit_generator(data_generator(mol_file=mol_file, sol_file=sol_file, datadir=datadir, epochs=config['train_epochs']))
    metric = dc.metrics.Metric(dc.metrics.mean_absolute_error)
    # print('Training set score:', model.evaluate(train_dataset, [metric], transformers))
    # print('Test set score:', model.evaluate(test_dataset, [metric], transformers))
    print('Training set score:', model.evaluate_generator(data_generator(mol_file=mol_file, sol_file=sol_file, datadir=datadir, epochs=config['eval_epochs']), [metric], transformers))
    # print('Test set score:', model.evaluate_generator(data_generator(test_dataset, test_dataset, config['batch_size'], config['eval_epochs'], config['eval_shuffle']), [metric], transformers))

    exit(0)