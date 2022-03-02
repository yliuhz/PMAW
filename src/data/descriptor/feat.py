
import pandas as pd

df = pd.read_csv('01-15-descriptor-train-smiles.csv', encoding='utf-8')

print(df.shape)

cols = df.columns
df_1 = df
df_2 = df[cols[-1]]

assert 'mol_smile' in df_1.columns.tolist()

df_1['smiles'] = df_1['mol_smile']
df_1 = df_1[['λabs (nm)', 'smiles']]

df_1.to_csv('01-15-descriptor-train-feat.csv', index=False, encoding='utf-8')
# df_2.to_csv('01-15-descriptor-train-label.csv', index=False, encoding='utf-8')