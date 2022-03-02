import pandas as pd

smiles_file = '22-01-29-smiles-train.csv'

df = pd.read_csv(smiles_file, encoding='gb18030')

print(df.columns)

df_mol = df[['label', 'molecule']]
df_sol = df[['label', 'solvent']]

df_mol.to_csv('22-01-29-mol-smiles-train.csv', index=False, encoding='gb18030')
df_sol.to_csv('22-01-29-sol-smiles-train.csv', index=False, encoding='gb18030')

# 生成morgan+maccs数据

morgan_file = '22-01-29-morgan-train.csv'
maccs_file = '22-01-29-maccs-train.csv'

df_morgan = pd.read_csv(morgan_file, encoding='gb18030')
df_maccs = pd.read_csv(maccs_file, encoding='gb18030')

labels = df_morgan['label'].tolist()
molecule_morgan = df_morgan['molecule'].tolist()
solvent_morgan = df_morgan['solvent'].tolist()

molecule_maccs = df_maccs['molecule'].tolist()
solvent_maccs = df_maccs['solvent'].tolist()

df_morgan_maccs = pd.DataFrame({'label':labels, 'molecule_morgan': molecule_morgan, 'solvent_morgan': solvent_morgan, 'molecule_maccs': molecule_maccs, 'solvent_maccs': solvent_maccs})

df_morgan_maccs.to_csv('22-01-29-morgan-maccs-train.csv', index=False, encoding='gb18030')

###
morgan_file = '22-01-29-morgan-test-level-1.csv'
maccs_file = '22-01-29-maccs-test-level-1.csv'

df_morgan = pd.read_csv(morgan_file, encoding='gb18030')
df_maccs = pd.read_csv(maccs_file, encoding='gb18030')

labels = df_morgan['label'].tolist()
molecule_morgan = df_morgan['molecule'].tolist()
solvent_morgan = df_morgan['solvent'].tolist()

molecule_maccs = df_maccs['molecule'].tolist()
solvent_maccs = df_maccs['solvent'].tolist()

df_morgan_maccs = pd.DataFrame({'label':labels, 'molecule_morgan': molecule_morgan, 'solvent_morgan': solvent_morgan, 'molecule_maccs': molecule_maccs, 'solvent_maccs': solvent_maccs})

df_morgan_maccs.to_csv('22-01-29-morgan-maccs-test-level-1.csv', index=False, encoding='gb18030')

# 
smiles_file = '22-01-29-smiles-test-level-1.csv'

df = pd.read_csv(smiles_file, encoding='gb18030')

print(df.columns)

df_mol = df[['label', 'molecule']]
df_sol = df[['label', 'solvent']]

df_mol.to_csv('22-01-29-mol-smiles-test-level-1.csv', index=False, encoding='gb18030')
df_sol.to_csv('22-01-29-sol-smiles-test-level-1.csv', index=False, encoding='gb18030')