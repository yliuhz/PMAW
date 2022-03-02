# MLP
import csv
from itertools import islice
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, train_test_split
import pandas as pd


def bit2attr(bitstr) -> list:
    attr_vec = bitstr.split(' ')
    for i in range(len(attr_vec)):
        attr_vec[i] = float(attr_vec[i])
    return attr_vec

def mean_relative_error(y_pred, y_test):
    assert len(y_pred) == len(y_test)
    mre = 0.0
    for i in range(len(y_pred)):
        mre = mre + abs((y_pred[i] - y_test[i]) / y_test[i])
    mre = mre * 100/ len(y_pred)
    return mre

'''
1) 数据预处理
'''
NUM_ATTR = 6
# filepath = 'data/fp/sjn/R+B+Cmorgan_fp1202.csv'
filepath = 'data/fp/sjn/Merged.csv'
# data_x = pd.DataFrame(columns=[str(i) for i in range(NUM_ATTR)])
data = list()
# data_y = pd.DataFrame(columns=['y'])
with open(filepath, 'r') as f:
    reader = csv.reader(f)
    num_attr = int()
    for row in islice(reader, 0, None):  # 不跳过第一行 # for row in islice(reader, 1, None):  # 跳过第一行
        if len(row) == 0:
            continue
        
        # data_x.append(bit2attr(row[0]), ignore_index=True)
        # data_y.append([int(row[1])], ignore_index=True)
        temp = bit2attr(row[0])

        num_attr = len(temp)
        print(len(row[0]), NUM_ATTR)
        print(row[0])
        assert num_attr == NUM_ATTR

        temp.append(float(row[1]))
        print(temp)
        data.append(temp)

print(data)
random.shuffle(data)

print(data)

data = np.array(data)
data_x_df = pd.DataFrame(data[:, 0:NUM_ATTR])
data_y_df = pd.DataFrame(data[:, NUM_ATTR])

# 归一化
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(data_x_df)
x_trans1 = min_max_scaler.transform(data_x_df)

min_max_scaler.fit(data_y_df)
y_trans1 = min_max_scaler.transform(data_y_df)

'''
4) 训练模型
'''
from sklearn import metrics

# n_split = 10
mlp_scores = []
MAEs = []
plt.figure(figsize=(10, 6))
plt.xlabel('ground truth')
plt.ylabel('predicted')
plt.plot([400, 1100], [400, 1100], 'k--')
for i in range(10):
    # kf = KFold(n_splits=n_split, random_state=i, shuffle=True)
    # for train_in, test_in in kf.split(data_x_df):
    #     X_train = data_x_df.iloc[train_in, :]
    #     X_test = data_x_df.iloc[test_in, :]
    #     y_train = data_y_df.iloc[train_in]
    #     y_test = data_y_df.iloc[test_in]
    #     print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    X_train, X_test, y_train, y_test = train_test_split(x_trans1, y_trans1, test_size=0.2)
    # train_test_split 随机划分 random_state, 填0或不填，每次都会不一样

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    model_mlp = MLPRegressor(
        hidden_layer_sizes=(400, 200, 100), activation='relu', solver='lbfgs', alpha=0.0001,
        max_iter=5000,
        random_state=1, tol=0.0001, verbose=False, warm_start=False)
    model_mlp.fit(X_train, y_train)

    x1 = x_trans1.reshape(-1, NUM_ATTR)
    y = y_trans1.reshape(-1, 1)
    mlp_score = model_mlp.score(x1, y)
    print('sklearn多层感知器-回归模型得分', mlp_score)  # 预测正确/总数
    mlp_scores.append(mlp_score)

    result = model_mlp.predict(X_test)
    # plt.figure(figsize=(10, 6))
    # plt.xlabel('ground truth')
    # plt.ylabel('predicted')

    y_test = min_max_scaler.inverse_transform(y_test)
    result = result.reshape(-1, 1)
    result = min_max_scaler.inverse_transform(result)

    mae = mean_relative_error(y_test, result)
    MAEs.append(mae)
    # errstr = 'MAE = %.3f' % mae
    # plt.text(420, 750, errstr, fontsize=16)
    plt.plot(y_test, result, 'ro')


print('MRE', MAEs)
print('avg MRE', sum(MAEs) / len(MAEs))
print('max MRE', max(MAEs))
print('min MRE', min(MAEs))
errstr = 'MRE = %.2f%%' % (sum(MAEs) / len(MAEs))
plt.text(420, 750, errstr, fontsize=16)
print('mlp_score', mlp_scores)
plt.show()
plt.savefig('pics/maccs_fig.png')
print('avg mlp_score', sum(mlp_scores) / len(mlp_scores))
