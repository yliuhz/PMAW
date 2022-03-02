
from email import header
import pandas as pd

# filename = 'Database_Train_OLEDpatch.csv'
filename = '01-15-descriptor-train.csv'

df = pd.read_csv(filename, encoding='gb18030')

print(len(df.columns))

new_cols = [str(df.columns[-1])] + list(df.columns[:-1])
print(new_cols)

df = df[new_cols]

# df = df.iloc[:, :-2]

print(len(df.columns))


df.to_csv(filename+'_cropped.csv', index=False, encoding='gb18030', header=False)