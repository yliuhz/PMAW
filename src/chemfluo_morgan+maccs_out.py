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
from sklearn.utils import shuffle
from time import sleep
from base import bit2attr

from sklearn.ensemble import GradientBoostingRegressor

# def bit2attr(bitstr) -> list:
#     attr_vec = list()
#     for i in range(len(bitstr)):
#         attr_vec.append(int(bitstr[i]))
#     return attr_vec

def mean_relative_error(y_pred, y_test):
    assert len(y_pred) == len(y_test)
    mre = 0.0
    for i in range(len(y_pred)):
        mre = mre + abs((y_pred[i] - y_test[i]) / y_test[i])
    mre = mre * 100/ len(y_pred)
    return mre

Large_MRE_points = pd.DataFrame()
Large_MRE_X = []
Large_MRE_y_test = []
Large_MRE_y_pred = []
Large_MRE = []

'''
1) 数据预处理
'''
# filepath = 'data/fp/sjn/R+B+Cmorgan_fp1202.csv'
NUM_ATTR = 167

def read_bit(filepath):
    data = list()
    with open(filepath, 'r', encoding='gb18030') as f:
        reader = csv.reader(f)
        for row in islice(reader, 1, None):
            temp0 = bit2attr(row[1])
            temp0 = temp0 + bit2attr(row[2])

            temp = row[3].strip().split(' ')
            temp = [int(x) for x in temp]
            bits_1 = [0 for x in range(NUM_ATTR)]
            for t in temp:
                bits_1[t] = 1

            temp = row[4].strip().split(' ')
            temp = [int(x) for x in temp]

            bits_2 = [0 for x in range(NUM_ATTR)]
            for t in temp:
                bits_2[t] = 1

            bits = bits_1 + bits_2

            temp = bits
            temp.append(float(row[0]))

            temp = temp0 + temp

            data.append(temp)
    data = np.array(data)
    data = pd.DataFrame(data)
    return data

def read_bit_0816(filepath):
    data = list()
    with open(filepath, 'r', encoding='gb18030') as f:
        reader = csv.reader(f)
        for row in islice(reader, 1, None):
            temp0 = list() # Maccs
            for i in range(len(row)-3):
                r = row[i]
                temp0.append(int(r)) # Maccs
            
            temp1 = bit2attr(row[-3]) # Morgan
            temp2 = bit2attr(row[-2]) # Morgan

            temp = temp1 + temp2 + temp0
            temp.append(float(row[-1]))
            data.append(temp)
    data = np.array(data)
    data = pd.DataFrame(data)
    return data

# filepath = 'data/fp/sjn/R+B+Cmorgan_fp1202.csv'
filepath = 'data/database/22-01-29-morgan-maccs-train.csv'
# data_x = pd.DataFrame(columns=[str(i) for i in range(NUM_ATTR)])
test_filepath = "data/database/22-01-29-morgan-maccs-test-level-1.csv"

# [data_x_df, data_y_df] = read_bit(filepath)
data = read_bit(filepath)
data = shuffle(data)
data_x_df = pd.DataFrame(data.iloc[:, :-1])
data_y_df = pd.DataFrame(data.iloc[:, -1])

# 归一化
min_max_scaler_X = MinMaxScaler()
min_max_scaler_X.fit(data_x_df)
x_trans1 = min_max_scaler_X.transform(data_x_df)

min_max_scaler_y = MinMaxScaler()
min_max_scaler_y.fit(data_y_df)
y_trans1 = min_max_scaler_y.transform(data_y_df)

# [test_data_x_df, test_data_y_df] = read_bit(test_filepath)
test_data = read_bit(test_filepath)
test_data_x_df = pd.DataFrame(test_data.iloc[:, :-1])
test_data_y_df = pd.DataFrame(test_data.iloc[:, -1])

x_trans1_test = min_max_scaler_X.transform(test_data_x_df)
y_trans1_test = min_max_scaler_y.transform(test_data_y_df)

print(data_x_df.shape, data_y_df.shape)
print(test_data_x_df.shape, test_data_y_df.shape)

'''
3) 构建模型
'''

from keras.layers import MaxPooling1D, Conv1D, Dense, Flatten
from keras import models
from keras.optimizers import Adam, RMSprop, SGD

def buildModel():
    model = models.Sequential()

    l5 = Dense(512, activation='relu')
    l6 = Dense(128, activation='relu')
    l7 = Dense(30, activation='relu')
    l8 = Dense(1)

    layers = [l5, l6, l7, l8]
    for i in range(len(layers)):
        model.add(layers[i])

    adam = Adam(lr=1e-3)
    model.compile(optimizer=adam, loss='logcosh', metrics=['mae'])

    model_mlp = MLPRegressor(
        hidden_layer_sizes=(512, 128, 32), activation='relu', solver='lbfgs', alpha=0.0001,
        max_iter=5000,
        random_state=1, tol=0.0001, verbose=False, warm_start=False)

    return model

'''
4) 训练模型
'''
from sklearn import metrics

# n_split = 10
mlp_scores = []
MAEs = []
out_MAEs = []

in_y_test = []
in_y_pred = []
out_y_test = []
out_y_pred = []

X_train = x_trans1
y_train = y_trans1

model_mlp = GradientBoostingRegressor(learning_rate=0.05,
                                            max_depth=31,
                                            max_features=300,
                                            min_samples_leaf=20,
                                            n_estimators=1000,
                                            verbose=True)
        # model_mlp.fit(X_train, y_train, epochs=120, validation_data=(X_test, y_test), verbose=1)
model_mlp.fit(X_train, y_train)

# 外部验证
X_test = x_trans1_test
result = model_mlp.predict(x_trans1_test)

y_trans1_test = np.reshape(y_trans1_test, (-1, 1))
y_test = min_max_scaler_y.inverse_transform(y_trans1_test)
result = result.reshape(-1, 1)
result = min_max_scaler_y.inverse_transform(result)

mae = mean_relative_error(y_test, result)
out_MAEs.append(mae)

Large_MRE_X = [] ## Type of X_test??
Large_MRE_y_test = []
Large_MRE_y_pred = []
Large_MRE = []

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
X_test = min_max_scaler_X.inverse_transform(X_test)

for idx in range(len(y_test)):
    Large_MRE.append(mean_relative_error([result[idx]], [y_test[idx]])[0])
Large_MRE_y_test = list(np.reshape(y_test, (-1,)))
Large_MRE_y_pred = list(np.reshape(result, (-1,)))

temp = pd.DataFrame(X_test)
temp = pd.concat([temp, pd.DataFrame({'Real Value': Large_MRE_y_test}), pd.DataFrame({'Predicted Value': Large_MRE_y_pred}),
                      pd.DataFrame({'MRE': Large_MRE})], axis=1)
# temp = temp.sort_values(by='MRE', ascending=False)
temp.to_csv('Out/Large_MRE_out_points.csv', encoding='gb18030', index=False)

out_y_test.append(y_test)
out_y_pred.append(result)

## 白+绿纯色颜色映射
from pylab import *
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
clist = ['white', 'purple', 'black']
newcmp = LinearSegmentedColormap.from_list('chaos',clist)

# 外部验证图像

## 白+绿纯色颜色映射
out_y_pred = np.reshape(out_y_pred, (-1,))
out_y_test = np.reshape(out_y_test, (-1,))

xmin = out_y_test.min()
# xmin = min(xmin, out_y_pred.min())
xmax = out_y_test.max()
# xmax = max(xmax, out_y_pred.max())

fig = plt.figure(figsize=(14, 10))
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
# plt.grid(linestyle="--")
plt.xlabel('Real values for lambda(mm)', fontsize=20)
plt.ylabel('Predicted values for lambda(mm)', fontsize=20)
plt.yticks(size=16)
plt.xticks(size=16)
plt.plot([xmin, xmax], [xmin, xmax], ':', linewidth=1.5, color='gray')
print('MRE', out_MAEs)
print('avg MRE', sum(out_MAEs) / len(out_MAEs))
print('max MRE', max(out_MAEs))
print('min MRE', min(out_MAEs))

errstr = 'MRE=%.2f%%' % (sum(out_MAEs) / len(out_MAEs))
plt.text(xmin + 50, xmax - 130, errstr, fontsize=20, weight='bold')

# for i in range(len(in_y_pred)):
    # plt.scatter(in_y_test[i], in_y_pred[i], edgecolors='b')
hexf = plt.hexbin(out_y_test, out_y_pred, gridsize=20, extent=[xmin, xmax, xmin, xmax],
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
plt.savefig('pics/morgan-fig-out-mlp.png')
plt.show()

# plt.figure(figsize=(10, 6))
# plt.xlabel('ground truth')
# plt.ylabel('predicted')
# plt.plot([400, 1100], [400, 1100], 'k--')
# print('MRE', out_MAEs)
# print('avg MRE', sum(out_MAEs) / len(out_MAEs))
# print('max MRE', max(out_MAEs))
# print('min MRE', min(out_MAEs))
# errstr = 'MRE = %.2f%%' % (sum(out_MAEs) / len(out_MAEs))
# plt.text(420, 750, errstr, fontsize=16)
# for i in range(len(out_y_pred)):
#     plt.plot(out_y_test[i], out_y_pred[i], 'ro')
# print('mlp_score', mlp_scores)
# plt.savefig('pics/descriptor-fig-out.png')
# plt.show()
