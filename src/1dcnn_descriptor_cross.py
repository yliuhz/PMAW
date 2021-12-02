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
from sklearn.model_selection import KFold

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
filepath = 'data/descriptor/0209/descriptor_train.csv'

data = pd.read_csv(filepath, encoding='gbk')
print(data.shape)
data = data.dropna()

print(data.shape)
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

# test_filepath = "data/descriptor/01-15-descriptor-test.csv"
# test_data = pd.read_csv(test_filepath, encoding='gbk')
# print('test data: ', test_data.shape)
# test_data = test_data.dropna()
# test_data_x_df = pd.DataFrame(test_data.iloc[:, :-1])
# test_data_y_df = pd.DataFrame(test_data.iloc[:, -1])
# x_trans1_test = min_max_scaler_X.transform(test_data_x_df)
# y_trans1_test = min_max_scaler_y.transform(test_data_y_df)
# x_trans1_test = np.reshape(x_trans1_test, (x_trans1_test.shape[0], x_trans1_test.shape[1], 1))
# y_trans1_test = np.reshape(y_trans1_test, (y_trans1_test.shape[0], 1, 1))

'''
3) 构建模型
'''

from keras.layers import MaxPooling1D, Conv1D, Dense, Flatten, Dropout, BatchNormalization
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
    l7 = Dropout(rate=0.1)
    l8 = Dense(84, activation='relu')
    l9 = Dense(1, activation='linear')

    layers = [l1, l2, l3, l4, l5, l6, l7, l8, l9]
    for i in range(len(layers)):
        model.add(layers[i])

    adam = Adam(lr=1e-3)
    model.compile(optimizer=adam, loss='logcosh', metrics=['mae'])

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

in_y_train_real = []
in_y_train_pred = []

for i in range(10):
    # kf = KFold(n_splits=n_split, random_state=i, shuffle=True)
    # for train_in, test_in in kf.split(data_x_df):
    #     X_train = data_x_df.iloc[train_in, :]
    #     X_test = data_x_df.iloc[test_in, :]
    #     y_train = data_y_df.iloc[train_in]
    #     y_test = data_y_df.iloc[test_in]
    #     print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    kf = KFold(n_splits=10, shuffle=True, random_state=i)

    for train_index, test_index in kf.split(x_trans1):
        X_train = x_trans1[train_index, :]
        y_train = y_trans1[train_index, :]
        X_test = x_trans1[test_index, :]
        y_test = y_trans1[test_index, :]

        # ## Initial: 0.2
        # X_train, X_test, y_train, y_test = train_test_split(x_trans1, y_trans1, test_size=0.1, shuffle=True, random_state=i)
        # # train_test_split 随机划分 random_state, 填0或不填，每次都会不一样

        print(X_train.shape, y_train.shape)
        print(X_test.shape, y_test.shape)

        # sleep(5)

        ## Initial: 400 200 100
        model_mlp = buildModel()
        model_mlp.fit(X_train, y_train, epochs=120, validation_data=(X_test, y_test), verbose=1)

        print(model_mlp.summary())

        # x1 = x_trans1.reshape(-1, NUM_ATTR)
        x1 = x_trans1
        # y = y_trans1.reshape(-1, 1)
        y = y_trans1
        # mlp_score = model_mlp.score(x1, y)
        #
        # print('sklearn多层感知器-回归模型得分', mlp_score)  # 预测正确/总数
        # mlp_scores.append(mlp_score)

        result = model_mlp.predict(X_test)
        # plt.figure(figsize=(10, 6))
        # plt.xlabel('ground truth')
        # plt.ylabel('predicted')

        y_test = np.reshape(y_test, (-1, 1))
        y_test = min_max_scaler_y.inverse_transform(y_test)
        # print('Result shape: ', result.shape)
        result = result.reshape(-1, 1)
        result = min_max_scaler_y.inverse_transform(result)

        # print(y_test.shape, result.shape)
        # print(result[:-20])
        mae = mean_relative_error(y_test, result)
        MAEs.append(mae)
        # errstr = 'MAE = %.3f' % mae
        # plt.text(420, 750, errstr, fontsize=16)
        # plt.plot(y_test, result, 'ro')

        Large_MRE_X = [] ## Type of X_test??
        Large_MRE_y_test = []
        Large_MRE_y_pred = []
        Large_MRE = []

        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
        X_test = min_max_scaler_X.inverse_transform(X_test)
        # for idx in range(len(y_test)):
        #     if mean_relative_error([result[idx]], [y_test[idx]]) > 5:
        #         Large_MRE_X.append(X_test[idx])
        #         Large_MRE_y_test.append(y_test[idx])
        #         Large_MRE_y_pred.append(result[idx])
        #         Large_MRE.append(mean_relative_error([result[idx]], [y_test[idx]]))

        for idx in range(len(y_test)):
            Large_MRE.append(mean_relative_error([result[idx]], [y_test[idx]])[0])
        Large_MRE_y_test = list(y_test)
        Large_MRE_y_pred = list(result)

        temp = pd.DataFrame(X_test)
        temp = pd.concat([temp, pd.DataFrame({'Real Value': Large_MRE_y_test}), pd.DataFrame({'Predicted Value': Large_MRE_y_pred}),
                          pd.DataFrame({'MRE': Large_MRE})], axis=1)
        temp = temp.sort_values(by='MRE', ascending=False)
        temp.to_csv('Out/Large_MRE_points' + str(i) + '.csv', encoding='gb18030', index=False)

        in_y_test = in_y_test + list(np.reshape(y_test, (-1,)))
        in_y_pred = in_y_pred + list(np.reshape(result, (-1,)))

        #
        result = model_mlp.predict(X_train)

        y_train = np.reshape(y_train, (-1, 1))
        y_train = min_max_scaler_y.inverse_transform(y_train)
        # print('Result shape: ', result.shape)
        result = result.reshape(-1, 1)
        result = min_max_scaler_y.inverse_transform(result)

        for c in y_train:
            in_y_train_real.append(c[0])
        for c in result:
            in_y_train_pred.append(c[0])


## 白+绿纯色颜色映射
from pylab import *
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
clist = ['white', 'green', 'black']
newcmp = LinearSegmentedColormap.from_list('chaos',clist)

in_y_pred = np.reshape(in_y_pred, (-1,))
in_y_test = np.reshape(in_y_test, (-1,))

xmin = in_y_test.min()
# xmin = min(xmin, in_y_pred.min())
xmax = in_y_test.max()
# xmax = max(xmax, in_y_pred.max())

fig = plt.figure(figsize=(14, 10))
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
# plt.grid(linestyle="--")
plt.xlabel('Real values for lambda(mm)', fontsize=20)
plt.ylabel('Predicted values for lambda(mm)', fontsize=20)
plt.yticks(size=16)
plt.xticks(size=16)
plt.plot([xmin, xmax], [xmin, xmax], ':', linewidth=1.5, color='gray')
print('MRE', MAEs)
print('avg MRE', sum(MAEs) / len(MAEs))
print('max MRE', max(MAEs))
print('min MRE', min(MAEs))

errstr = 'MRE=%.2f%%' % (sum(MAEs) / len(MAEs))
plt.text(xmin + 50, xmax - 130, errstr, fontsize=20, weight='bold')

cross_result = {'Real lambda': in_y_test, 'Predicted lambda': in_y_pred}
cross_result = pd.DataFrame(cross_result)
cross_result.to_csv('Out/cross_result_cnn.csv', index=False, encoding='gb18030')

# for i in range(len(in_y_pred)):
    # plt.scatter(in_y_test[i], in_y_pred[i], edgecolors='b')
hexf = plt.hexbin(in_y_test, in_y_pred, gridsize=20, extent=[xmin, xmax, xmin, xmax],
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
plt.savefig('pics/descriptor-fig-cnn.png')
plt.show()

cross_result = {'Real lambda': in_y_train_real, 'Predicted lambda': in_y_train_pred}
cross_result = pd.DataFrame(cross_result)
cross_result.to_csv('Out/cross_result_cnn_train.csv', index=False, encoding='gb18030')

