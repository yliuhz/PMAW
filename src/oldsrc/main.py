import sys
from itertools import islice

from IPython.display import SVG
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, MACCSkeys, Draw
from rdkit.Chem.Draw import DrawMorganBit, DrawMorganBits, DrawMorganEnv, IPythonConsole
import csv
from pprint import pprint
import os

'''
reference: 
    http://rdkit.chenzhaoqiang.com/basicManual.html
    http://rdkit.chenzhaoqiang.com/basicManual.html#id27
'''
dir = '1931/'
filenames = os.listdir('data/smile/' + dir)
# read csv file to smile_list
print(filenames)
empty_mol = Chem.MolFromSmiles('/')
method = 'maccs'  # topo morgan maccs
for csv_file_name in filenames:
    smile_list = []
    filepath = 'data/smile/' + dir + csv_file_name
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        # for row in islice(reader, 0, None):  # 不跳过第一行
        for row in islice(reader, 1, None):  # 跳过第一行
            if len(row) == 0:
                continue
            smile_list.append(Chem.MolFromSmiles(row[0]))

        # 拓扑指纹 Chem.RDKFingerprint(x)
        # MACCS指纹 MACCSkeys.GenMACCSKeys(x)
        # 摩根指纹（圆圈指纹）AllChem.GetMorganFingerprint(mol, 2)
        fps = []
        bi_list = []
        for x in smile_list:
            if x == empty_mol:
                bi_list.append(['/'])
                continue
            bi = {}
            temp = object()
            if method == 'topo':
                temp = Chem.RDKFingerprint(x, maxPath=2, bitInfo=bi)
            elif method == 'morgan':
                temp = AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=1024, bitInfo=bi)
            elif method == 'maccs':
                temp = MACCSkeys.GenMACCSKeys(x)
            fps.append(temp)
            bi_list.append([bi, temp.ToBitString()])

        header = [method + ' fingerprint', 'bit string']
        filepath = 'data/fp/' + dir + csv_file_name + '_' + method + '_fp.csv'
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            # writer.writerow(header)
            writer.writerows(bi_list)

# maccs_fps = [MACCSkeys.GenMACCSKeys(x) for x in smile_list]

# Atom Pairs

# topological torsions

# 摩根指纹（圆圈指纹）AllChem.GetMorganFingerprint(mol,2)
