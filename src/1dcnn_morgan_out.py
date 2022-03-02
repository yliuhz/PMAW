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

import tensorflow as tf

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
NUM_ATTR = 1024

def read_bit(filepath):
    data = list()
    # data_y = pd.DataFrame(columns=['y'])
    with open(filepath, 'r', encoding='gbk') as f:
        reader = csv.reader(f)
        num_attr = int()
        for row in islice(reader, 1, None):  # 不跳过第一行 # for row in islice(reader, 1, None):  # 跳过第一行
            if len(row) == 0:
                continue
            num_attr = len(row[1])
            # print('num_attrs: ', num_attr)
            assert num_attr == NUM_ATTR
            num_attr = len(row[2])
            # print('num_attrs_2: ', num_attr)
            assert num_attr == NUM_ATTR
            # data_x.append(bit2attr(row[0]), ignore_index=True)
            # data_y.append([int(row[1])], ignore_index=True)
            temp = bit2attr(row[1])
            temp = temp + bit2attr(row[2])
            temp.append(float(row[0]))
            data.append(temp)

    # random.shuffle(data) # 不打乱数据

    data = np.array(data)
    # data_x_df = pd.DataFrame(data[:, 0:2*NUM_ATTR])
    # data_y_df = pd.DataFrame(data[:, 2*NUM_ATTR])

    # return [data_x_df, data_y_df]
    data = pd.DataFrame(data)
    return data

# filepath = 'data/fp/sjn/R+B+Cmorgan_fp1202.csv'
filepath = 'data/database/22-01-29-morgan-train.csv'
# data_x = pd.DataFrame(columns=[str(i) for i in range(NUM_ATTR)])
test_filepath = "data/database/22-01-29-morgan-test-level-1.csv"

# [data_x_df, data_y_df] = read_bit(filepath)
data = read_bit(filepath)
data = shuffle(data)
data_x_df = pd.DataFrame(data.iloc[:, :-1])
data_y_df = pd.DataFrame(data.iloc[:, -1])

# 归一化
min_max_scaler_X = MinMaxScaler()
min_max_scaler_X.fit(data_x_df)
x_trans1 = min_max_scaler_X.transform(data_x_df)
x_trans1 = np.reshape(x_trans1, (x_trans1.shape[0], x_trans1.shape[1], 1))
# x_trans1 = np.reshape(x_trans1, (x_trans1.shape[0], x_trans1.shape[1]))

min_max_scaler_y = MinMaxScaler()
min_max_scaler_y.fit(data_y_df)
y_trans1 = min_max_scaler_y.transform(data_y_df)
# y_trans1 = np.reshape(y_trans1, (y_trans1.shape[0], 1, 1))
y_trans1 = np.reshape(y_trans1, (y_trans1.shape[0], 1))

# [test_data_x_df, test_data_y_df] = read_bit(test_filepath)
test_data = read_bit(test_filepath)
test_data_x_df = pd.DataFrame(test_data.iloc[:, :-1])
test_data_y_df = pd.DataFrame(test_data.iloc[:, -1])

x_trans1_test = min_max_scaler_X.transform(test_data_x_df)
y_trans1_test = min_max_scaler_y.transform(test_data_y_df)
x_trans1_test = np.reshape(x_trans1_test, (x_trans1_test.shape[0], x_trans1_test.shape[1], 1))
# y_trans1_test = np.reshape(y_trans1_test, (y_trans1_test.shape[0], 1, 1))
y_trans1_test = np.reshape(y_trans1_test, (y_trans1_test.shape[0], 1))

# print(data_x_df.shape, data_y_df.shape)
# print(test_data_x_df.shape, test_data_y_df.shape)
print(x_trans1.shape, y_trans1.shape)
print(x_trans1_test.shape, y_trans1_test.shape)
sleep(5)

'''
3) 构建模型
'''

from keras.layers import MaxPooling1D, Conv1D, Dense, Flatten, Dropout
from keras import models
from keras.optimizers import Adam, RMSprop, SGD

def buildModel():
    model = models.Sequential()

    l1 = Conv1D(6, 25, strides=1, activation='relu', use_bias=True, padding='same')
    l2 = MaxPooling1D(2, 2)
    l3 = Conv1D(16, 25, strides=1, activation='relu', use_bias=True, padding='same')
    l4 = MaxPooling1D(2, 2)
    l5 = Flatten()
    l6 = Dense(120, activation='relu')
    # l7 = Dropout(0.5)
    l8 = Dense(84, activation='relu')
    l9 = Dense(1, activation='linear')

    layers = [l1, l2, l3, l4, l5, l6, l8, l9]
    for i in range(len(layers)):
        model.add(layers[i])

    adam = Adam(lr=1e-3)
    model.compile(optimizer=adam, loss='logcosh', metrics=['mae'])

    return model

def scheduler(epoch, lr):
    if epoch > 0 and epoch % 500 == 0:
        return lr * 0.1
    else:
        return lr

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

callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
model_mlp = buildModel()
model_mlp.fit(X_train, y_train, epochs=2000, verbose=1, callbacks=[callback])

print(model_mlp.summary())
sleep(5)

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
clist = ['white', 'green', 'black']
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
plt.savefig('pics/morgan-fig-out-cnn.png')
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
