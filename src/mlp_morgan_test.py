# TIPS: only used to find the best params of mlp-morgan

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
            assert num_attr == NUM_ATTR
            num_attr = len(row[2])
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
sleep(5)

'''
3) 构建模型
'''

from keras.layers import MaxPooling1D, Conv1D, Dense, Flatten, Dropout
from keras import models
from keras.optimizers import Adam, RMSprop, SGD

def buildModel():
    model = models.Sequential()

    l5 = Dense(512, activation='relu')
    l6 = Dropout(rate=0.2)
    l6 = Dense(128, activation='relu')
    l7 = Dense(30, activation='relu')
    l8 = Dense(1)

    layers = [l5, l6, l7, l8]
    for i in range(len(layers)):
        model.add(layers[i])

    adam = Adam(lr=1e-3)
    model.compile(optimizer=adam, loss='logcosh', metrics=['mae', 'mape'])

    model_mlp = MLPRegressor(
        hidden_layer_sizes=(512, 128, 32), activation='relu', solver='lbfgs', alpha=0.0001,
        max_iter=5000,
        random_state=1, tol=0.0001, verbose=False, warm_start=False)

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

# 外部验证
X_test = x_trans1_test

y_trans1_test = np.reshape(y_trans1_test, (-1, 1))
y_test = y_trans1_test


callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
model_mlp = buildModel()
history = model_mlp.fit(X_train, y_train, epochs=2000, verbose=1, validation_data=(X_test, y_test), callbacks=[callback])

losses = history.history['loss']
eval_mres = history.history['val_mape']

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot([x for x in range(len(losses))], losses, 'b', label='loss')
ax1.set_ylabel('loss', color='b')
ax2.plot([x for x in range(len(eval_mres))], eval_mres, 'r', label='eval_mre')
ax2.set_ylabel('eval_mre', color='r')
ax1.set_xlabel('epochs')
plt.title('Training of MLP')
plt.savefig('pics/Training_of_MLP.png')

import os
outdir = 'Out/losses_and_mres'
os.makedirs(outdir, exist_ok=True)
with open(os.path.join(outdir, 'mlp_morgan.txt'), 'w') as f:
    f.write('loss\n')
    f.write(' '.join([str(x) for x in losses]))
    f.write('\n')
    f.write('mres\n')
    f.write(' '.join([str(x) for x in eval_mres]))