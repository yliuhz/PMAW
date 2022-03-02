from itertools import islice

import matplotlib.pyplot as plt
import csv

filepath = 'data/fp/sjn/R+B+Cmorgan_fp1202.csv'
# data_x = pd.DataFrame(columns=[str(i) for i in range(NUM_ATTR)])
data = list()
# data_y = pd.DataFrame(columns=['y'])
with open(filepath, 'r') as f:
    reader = csv.reader(f)
    num_attr = int()
    for row in islice(reader, 0, None):  # 不跳过第一行 # for row in islice(reader, 1, None):  # 跳过第一行
        if len(row) == 0:
            continue
        num_attr = len(row[0])
        data.append(int(row[1]))

plt.title('Distribution of values to be predicted')
plt.xlabel('value')
plt.ylabel('count')
plt.hist(data, bins=14)
plt.show()
