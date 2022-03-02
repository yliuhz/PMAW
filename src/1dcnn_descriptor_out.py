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

print(x_trans1.shape, y_trans1.shape)
print(x_trans1_test.shape, y_trans1_test.shape)

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

    layers = [l1, l2, l4, l5, l7, l8, l9, l12, l13]
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

callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
model_mlp = buildModel()
history = model_mlp.fit(X_train, y_train, epochs=2000, verbose=1, callbacks=[callback])

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

# 外部验证图像

## 白+绿纯色颜色映射
from pylab import *
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
clist = ['white', 'green', 'black']
newcmp = LinearSegmentedColormap.from_list('chaos',clist)

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
plt.savefig('pics/descriptor-fig-out-cnn.png')
plt.show()

