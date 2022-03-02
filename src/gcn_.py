
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

from deepchem.models.torch_models.attentivefp import AttentiveFP, AttentiveFPModel

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

class mulgcn(keras.Model):
  def __init__(self, batch_size=128, alpha=0.5):
    super(mulgcn, self).__init__()
    self.alpha = alpha

    self.gc1_1 = GraphConv(128, activation_fn=tf.nn.tanh)
    self.batch_norm1_1 = layers.BatchNormalization()
    self.gp1_1 = GraphPool()

    self.gc1_2 = GraphConv(128, activation_fn=tf.nn.tanh)
    self.batch_norm1_2 = layers.BatchNormalization()
    self.gp1_2 = GraphPool()

    self.gc2_1 = GraphConv(128, activation_fn=tf.nn.tanh)
    self.batch_norm2_1 = layers.BatchNormalization()
    self.gp2_1 = GraphPool()

    self.gc2_2 = GraphConv(128, activation_fn=tf.nn.tanh)
    self.batch_norm2_2 = layers.BatchNormalization()
    self.gp2_2 = GraphPool()

    self.dense1 = layers.Dense(256, activation=tf.nn.tanh)
    self.batch_norm3 = layers.BatchNormalization()
    self.readout = GraphGather(batch_size=batch_size, activation_fn=tf.nn.tanh)

    self.dense2 = layers.Dense(1)
  def __call__(self, inputs, training=False):
    molecule, solvent = inputs[:len(inputs)//2], inputs[len(inputs)//2:]
    gc1_1_output = self.gc1_1(molecule)
    batch_norm1_1_output = self.batch_norm1_1(gc1_1_output)
    gp1_1_output = self.gp1_1([batch_norm1_1_output] + molecule[1:])

    gc1_2_output = self.gc1_2([gp1_1_output] + molecule[1:])
    batch_norm1_2_output = self.batch_norm1_2(gc1_2_output)
    gp1_2_output = self.gp1_2([batch_norm1_2_output] + molecule[1:])

    gc2_1_output = self.gc2_1(solvent)
    batch_norm2_1_output = self.batch_norm2_1(gc2_1_output)
    gp2_1_output = self.gp2_1([batch_norm2_1_output] + solvent[1:])

    gc2_2_output = self.gc2_2([gp2_1_output] + solvent[1:])
    batch_norm2_2_output = self.batch_norm2_2(gc2_2_output)
    gp2_2_output = self.gp2_2([batch_norm2_2_output] + solvent[1:])


    # print(type(gp1_2_output), gp1_2_output.shape)
    # print(type(gp2_2_output), gp2_2_output.shape)
    # exit(0)

    gp_output = gp1_2_output + gp2_2_output*self.alpha

    dense1_output = self.dense1(gp_output)
    batch_norm3_output = self.batch_norm3(dense1_output)
    readout_output = self.readout([batch_norm3_output] + molecule[1:] + solvent[1:])

    return self.dense2(readout_output)

def data_generator(mol_dataset, sol_dataset, batch_size=128, epochs=50, shuffle=False):
  mol_iter = mol_dataset.iterbatches(batch_size, epochs, deterministic=not shuffle, pad_batches=True)
  sol_iter = sol_dataset.iterbatches(batch_size, epochs, deterministic=not shuffle, pad_batches=True)

  # print('?????')

  for (mol, sol) in zip(mol_iter, sol_iter):
  # for ind, (X_b, y_b, w_b, ids_b) in enumerate(dataset.iterbatches(batch_size, epochs,
  #                                                                 deterministic=True, pad_batches=False)):
    mol_inputs = []
    sol_inputs = []

    X_b, y_b, w_b, ids_b = mol

    # print(y_b)
    # print(type(y_b))
    # print(type(y_b[0]))


    multiConvMol = ConvMol.agglomerate_mols(X_b)
    n_samples = np.array(X_b.shape[0])
    mol_inputs = [multiConvMol.get_atom_features(), multiConvMol.deg_slice, np.array(multiConvMol.membership), n_samples]
    for i in range(1, len(multiConvMol.get_deg_adjacency_lists())):
      mol_inputs.append(multiConvMol.get_deg_adjacency_lists()[i])
    # labels = [to_one_hot(y_b.flatten(), 2).reshape(-1, n_tasks, 2)]
    mol_labels = [y_b]
    mol_weights = [w_b]

    X_b, y_b, w_b, ids_b = sol
    multiConvMol = ConvMol.agglomerate_mols(X_b)
    n_samples = np.array(X_b.shape[0])
    sol_inputs = [multiConvMol.get_atom_features(), multiConvMol.deg_slice, np.array(multiConvMol.membership), n_samples]
    for i in range(1, len(multiConvMol.get_deg_adjacency_lists())):
      sol_inputs.append(multiConvMol.get_deg_adjacency_lists()[i])
    # labels = [to_one_hot(y_b.flatten(), 2).reshape(-1, n_tasks, 2)]
    sol_labels = [y_b]
    sol_weights = [w_b]      

    # print(len(mol_inputs), type(mol_inputs))
    # print(mol_inputs[0].shape)
    # for i in range(len(mol_inputs)):
    #   print(type(mol_inputs[i]))

    assert len(mol_labels) == len(sol_labels)
    for idx in range(len(mol_labels)):
      # print(mol_labels[idx], type(mol_labels[idx]))
      # print(sol_labels[idx])
      # print(np.array(sol_labels[idx]))
      assert (np.array(mol_labels[idx]) == np.array(sol_labels[idx])).all()

    # inputs = [np.array(xx) for xx in list(zip(mol_inputs, sol_inputs))]

    inputs = mol_inputs + sol_inputs

    # print(len(inputs))
    # print(type(inputs[0]))
    # print(inputs[0].shape)

    # xxx,yyy = inputs[0]
    # print(xxx.shape)

    # yield (mol_inputs, mol_labels, mol_weights)
    yield (inputs, mol_labels, mol_weights)


if __name__=='__main__':
    mol_file = "01-15-descriptor-train-feat.csv.gz"
    sol_file = "01-15-descriptor-train-feat.csv.gz"

    tasks, datasets, transformers = load_descriptor(data_dir='data/descriptor',  featurizer='GraphConv', filename=mol_file)
    train_dataset, valid_dataset, test_dataset = datasets
    n_tasks = len(tasks)
    # print(len(train_dataset))
    # print(len(valid_dataset))
    # print(len(test_dataset))

    # exit(0)

    config = {
      'train_epochs': 50,
      'eval_epochs': 1, 
      'batch_size': 8,
      'alpha': 1.0,
      'train_shuffle': True,
      'eval_shuffle': False
    }

    # model = dc.models.KerasModel(mulgcn(config['batch_size'], config['alpha']), loss=dc.models.losses.L2Loss())
    
    model = dc.models.GraphConvModel(n_tasks=n_tasks, mode='regression')
    # model = AttentiveFPModel(n_tasks=n_tasks, mode='regression')
    # model = dc.models.MPNNModel(n_tasks=n_tasks, mode='regression')
    # model.fit(train_dataset, nb_epoch=50)

    model.fit_generator(data_generator(train_dataset, train_dataset, config['batch_size'], config['train_epochs'], config['train_shuffle']))
    metric = dc.metrics.Metric(dc.metrics.mean_absolute_error)
    # print('Training set score:', model.evaluate(train_dataset, [metric], transformers))
    # print('Test set score:', model.evaluate(test_dataset, [metric], transformers))
    print('Training set score:', model.evaluate_generator(data_generator(train_dataset, train_dataset, config['batch_size'], config['eval_epochs'], config['eval_shuffle']), [metric], transformers))
    print('Test set score:', model.evaluate_generator(data_generator(test_dataset, test_dataset, config['batch_size'], config['eval_epochs'], config['eval_shuffle']), [metric], transformers))

    exit(0)