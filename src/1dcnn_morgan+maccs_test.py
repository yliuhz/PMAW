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
x_trans1 = np.reshape(x_trans1, (x_trans1.shape[0], x_trans1.shape[1], 1))

min_max_scaler_y = MinMaxScaler()
min_max_scaler_y.fit(data_y_df)
y_trans1 = min_max_scaler_y.transform(data_y_df)
y_trans1 = np.reshape(y_trans1, (y_trans1.shape[0], 1, 1))

# [test_data_x_df, test_data_y_df] = read_bit(test_filepath)
test_data = read_bit(test_filepath)
test_data_x_df = pd.DataFrame(test_data.iloc[:, :-1])
test_data_y_df = pd.DataFrame(test_data.iloc[:, -1])

x_trans1_test = min_max_scaler_X.transform(test_data_x_df)
y_trans1_test = min_max_scaler_y.transform(test_data_y_df)
x_trans1_test = np.reshape(x_trans1_test, (x_trans1_test.shape[0], x_trans1_test.shape[1], 1))
y_trans1_test = np.reshape(y_trans1_test, (y_trans1_test.shape[0], 1, 1))

print(data_x_df.shape, data_y_df.shape, x_trans1.shape, y_trans1.shape)
print(test_data_x_df.shape, test_data_y_df.shape, x_trans1_test.shape, y_trans1_test.shape)

'''
3) 构建模型
'''

from keras.layers import MaxPooling1D, Conv1D, Dense, Flatten, Dropout
from keras import models
from keras.optimizers import Adam, RMSprop, SGD

def buildModel():
    model = models.Sequential()

    l1 = Conv1D(6, 25, 1, activation='relu', use_bias=True, padding='same')
    l2 = MaxPooling1D(2, 2)
    l3 = Conv1D(16, 25, 1, activation='relu', use_bias=True, padding='same')
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
    model.compile(optimizer=adam, loss='logcosh', metrics=['mae', 'mape'])

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

# model_mlp = buildModel()
# model_mlp.fit(X_train, y_train, epochs=120, verbose=1)

# print(model_mlp.summary())
# sleep(5)

# 外部验证
X_test = x_trans1_test

y_trans1_test = np.reshape(y_trans1_test, (-1, 1))
y_test = y_trans1_test


callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
model_mlp = buildModel()
history = model_mlp.fit(X_train, y_train, epochs=2000, verbose=1, validation_data=(X_test, y_test))

losses = history.history['loss']
eval_mres = history.history['val_mape']

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot([x for x in range(len(losses))], losses, 'b', label='loss')
ax1.set_ylabel('loss', color='b')
ax2.plot([x for x in range(len(eval_mres))], eval_mres, 'r', label='eval_mre')
ax2.set_ylabel('eval_mre', color='r')
ax1.set_xlabel('epochs')
plt.title('Training of CNN')
plt.savefig('pics/Training_of_CNN.png')

import os
outdir = 'Out/losses_and_mres'
os.makedirs(outdir, exist_ok=True)
with open(os.path.join(outdir, '1dcnn_morgan+maccs.txt'), 'w') as f:
    f.write('loss\n')
    f.write(' '.join([str(x) for x in losses]))
    f.write('\n')
    f.write('mres\n')
    f.write(' '.join([str(x) for x in eval_mres]))
