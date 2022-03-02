# TIPS: only used to find the best params of cnn

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

import tensorflow as tf

def bit2attr(bitstr) -> list:
    attr_vec = list()
    for i in range(len(bitstr)):
        attr_vec.append(int(bitstr[i]))
    return attr_vec

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
filepath = 'data/database/22-01-29-descriptor-train.csv'

data = pd.read_csv(filepath, encoding='gb18030')
print(data.shape)
data = data.dropna()

print(data.shape)
data = shuffle(data)

data_x_df = data.drop(['label'], axis=1)
data_y_df = data[['label']]

# 归一化
min_max_scaler_X = MinMaxScaler()
min_max_scaler_X.fit(data_x_df)
x_trans1 = min_max_scaler_X.transform(data_x_df)
x_trans1 = np.reshape(x_trans1, (x_trans1.shape[0], x_trans1.shape[1], 1))

min_max_scaler_y = MinMaxScaler()
min_max_scaler_y.fit(data_y_df)
y_trans1 = min_max_scaler_y.transform(data_y_df)
y_trans1 = np.reshape(y_trans1, (y_trans1.shape[0], 1, 1))

test_filepath = "data/database/22-01-29-descriptor-test-level-1.csv"
test_data = pd.read_csv(test_filepath, encoding='gb18030')
print('test data: ', test_data.shape)

test_data_x_df = test_data.drop(['label'], axis=1)
test_data_y_df = test_data[['label']]
x_trans1_test = min_max_scaler_X.transform(test_data_x_df)
y_trans1_test = min_max_scaler_y.transform(test_data_y_df)
x_trans1_test = np.reshape(x_trans1_test, (x_trans1_test.shape[0], x_trans1_test.shape[1], 1))
y_trans1_test = np.reshape(y_trans1_test, (y_trans1_test.shape[0], 1, 1))

'''
3) 构建模型
'''

from keras.layers import MaxPooling1D, Conv1D, Dense, Flatten, Dropout, BatchNormalization, LayerNormalization
from keras import models
from keras.optimizers import Adam, RMSprop, SGD

def buildModel():
    model = models.Sequential()

    l1 = Conv1D(6, 25, 1, activation='relu', use_bias=True, padding='same')
    l2 = MaxPooling1D(2, 2)
    l3 = BatchNormalization(axis=-1)
    l4 = Conv1D(16, 25, 1, activation='relu', use_bias=True, padding='same')
    l5 = MaxPooling1D(2, 2)
    l6 = BatchNormalization(axis=-1)
    l7 = Flatten()
    l8 = Dense(120, activation='relu')
    l9 = Dropout(rate=0.1)
    l10 = BatchNormalization(axis=-1)
    l11 = LayerNormalization(axis=-1)
    l12 = Dense(84, activation='relu')
    l13 = Dense(1, activation='linear')

    layers = [l1, l2, l4, l5, l7, l8, l9, l11, l12, l13]
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
history = model_mlp.fit(X_train, y_train, epochs=1, verbose=1, validation_data=(X_test, y_test))
print(model_mlp.summary())
exit(0)

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
with open(os.path.join(outdir, '1dcnn_descriptor.txt'), 'w') as f:
    f.write('loss\n')
    f.write(' '.join([str(x) for x in losses]))
    f.write('\n')
    f.write('mres\n')
    f.write(' '.join([str(x) for x in eval_mres]))
