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
            num_attr = len(row[0])
            assert num_attr == NUM_ATTR
            num_attr = len(row[1])
            assert num_attr == NUM_ATTR
            # data_x.append(bit2attr(row[0]), ignore_index=True)
            # data_y.append([int(row[1])], ignore_index=True)
            temp = bit2attr(row[0])
            temp = temp + bit2attr(row[1])
            temp.append(float(row[2]))
            data.append(temp)

    # random.shuffle(data) # 不打乱数据

    data = np.array(data)
    data_x_df = pd.DataFrame(data[:, 0:2*NUM_ATTR])
    data_y_df = pd.DataFrame(data[:, 2*NUM_ATTR])

    return [data_x_df, data_y_df]

# filepath = 'data/fp/sjn/R+B+Cmorgan_fp1202.csv'
filepath = 'data/fp/sjn/01-15-morgan-train.csv'
# data_x = pd.DataFrame(columns=[str(i) for i in range(NUM_ATTR)])
# test_filepath = "data/fp/sjn/01-15-morgan-test-2.csv"

[data_x_df, data_y_df] = read_bit(filepath)

# 归一化
morgan_min_max_scaler_X = MinMaxScaler()
morgan_min_max_scaler_X.fit(data_x_df)
morgan_x_trans1 = morgan_min_max_scaler_X.transform(data_x_df)
# morgan_x_trans1 = np.reshape(morgan_x_trans1, (morgan_x_trans1.shape[0], morgan_x_trans1.shape[1], 1)) ##

morgan_min_max_scaler_y = MinMaxScaler()
morgan_min_max_scaler_y.fit(data_y_df)
morgan_y_trans1 = morgan_min_max_scaler_y.transform(data_y_df)
# morgan_y_trans1 = np.reshape(morgan_y_trans1, (morgan_y_trans1.shape[0], 1, 1)) ##

##
filepath = 'data/descriptor/01-15-descriptor-train.csv'

data = pd.read_csv(filepath, encoding='gbk')
print(data.shape)
data = data.dropna()

# print(data.shape)
# data = shuffle(data) # 不打乱数据

data_x_df = pd.DataFrame(data.iloc[:, :-1])
data_y_df = pd.DataFrame(data.iloc[:, -1])

# 归一化
min_max_scaler_X = MinMaxScaler()
min_max_scaler_X.fit(data_x_df)
x_trans1 = min_max_scaler_X.transform(data_x_df)
# x_trans1 = np.reshape(x_trans1, (x_trans1.shape[0], x_trans1.shape[1], 1))

min_max_scaler_y = MinMaxScaler()
min_max_scaler_y.fit(data_y_df)
y_trans1 = min_max_scaler_y.transform(data_y_df)
# y_trans1 = np.reshape(y_trans1, (y_trans1.shape[0], 1, 1))


# 打乱数据
# morgan_x_trans1, morgan_y_trans1
# x_trans1, y_trans1
temp = pd.DataFrame(morgan_x_trans1)
temp = pd.concat([temp, pd.DataFrame(x_trans1)], axis=1)
temp = pd.concat([temp, pd.DataFrame(y_trans1)], axis=1)
temp = shuffle(temp)
morgan_x_trans1 = temp.iloc[:, 0:2*NUM_ATTR]
x_trans1 = temp.iloc[:, 2*NUM_ATTR:-1]
morgan_y_trans1 = y_trans1 = temp.iloc[:, -1]

morgan_x_trans1 = np.array(morgan_x_trans1)
morgan_y_trans1 = np.array(morgan_y_trans1)
x_trans1 = np.array(x_trans1)
y_trans1 = np.array(y_trans1)

morgan_x_trans1 = np.reshape(morgan_x_trans1, (morgan_x_trans1.shape[0], morgan_x_trans1.shape[1], 1))
morgan_y_trans1 = np.reshape(morgan_y_trans1, (morgan_y_trans1.shape[0], 1, 1))
x_trans1 = np.reshape(x_trans1, (x_trans1.shape[0], x_trans1.shape[1], 1))
y_trans1 = np.reshape(y_trans1, (y_trans1.shape[0], 1, 1))

print(x_trans1.shape, y_trans1.shape)
print(morgan_x_trans1.shape, morgan_y_trans1.shape)
sleep(5)

'''
3) 构建模型
'''

from keras.layers import MaxPooling1D, Conv1D, Dense, Flatten, Input, concatenate, Dropout
from keras import models
from keras.optimizers import Adam, RMSprop, SGD
from keras.models import Model

def buildModel():
    inputA = Input(shape=(x_trans1.shape[1], 1), name='I1')
    inputB = Input(shape=(morgan_x_trans1.shape[1], 1), name='I2')

    x = Conv1D(6, 25, 1, activation='relu', use_bias=True, padding='same')(inputA)
    x = MaxPooling1D(2, 2)(x)
    x = Conv1D(16, 25, 1, activation='relu', use_bias=True, padding='same')(x)
    x = MaxPooling1D(2, 2)(x)
    x = Flatten()(x)
    x = Dense(120, activation='relu')(x)
    x = Model(inputs=inputA, outputs=x)

    y = Conv1D(6, 25, 1, activation='relu', use_bias=True, padding='same')(inputB)
    y = MaxPooling1D(2, 2)(y)
    y = Conv1D(16, 25, 1, activation='relu', use_bias=True, padding='same')(y)
    y = MaxPooling1D(2, 2)(y)
    y = Flatten()(y)
    y = Dense(120, activation='relu')(y)
    y = Model(inputs=inputB, outputs=y)

    combined = concatenate([x.output, y.output], axis=1)

    z = Dropout(rate=0.1)(combined)
    z = Dense(84, activation='relu')(z)
    z = Dense(1, activation='linear')(z)

    model = Model(inputs=[x.input, y.input], outputs=z)

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
        X_train_I1 = x_trans1[train_index, :]
        X_train_I2 = morgan_x_trans1[train_index, :]
        y_train = y_trans1[train_index, :]
        X_test_I1 = x_trans1[test_index, :]
        X_test_I2 = morgan_x_trans1[test_index, :]
        y_test = y_trans1[test_index, :]

        ## Initial: 0.2
        # X_train_I1, X_test_I1, X_train_I2, X_test_I2, y_train, y_test= train_test_split(x_trans1,
        #         morgan_x_trans1, y_trans1, test_size=0.1, shuffle=True, random_state=i)
        # train_test_split 随机划分 random_state, 填0或不填，每次都会不一样

        print(X_train_I1.shape, X_test_I1.shape)
        print(X_train_I2.shape, X_test_I2.shape)
        print(y_train.shape, y_test.shape)

        # sleep(5)

        ## Initial: 400 200 100
        model_mlp = buildModel()
        model_mlp.fit({'I1': X_train_I1, 'I2': X_train_I2}, y_train, epochs=120,
                      validation_data=({'I1': X_test_I1, 'I2': X_test_I2}, y_test), verbose=1)

        print(model_mlp.summary())
        sleep(5)


        # x1 = x_trans1.reshape(-1, NUM_ATTR)
        x1 = x_trans1
        # y = y_trans1.reshape(-1, 1)
        y = y_trans1
        # mlp_score = model_mlp.score(x1, y)
        #
        # print('sklearn多层感知器-回归模型得分', mlp_score)  # 预测正确/总数
        # mlp_scores.append(mlp_score)

        result = model_mlp.predict({'I1': X_test_I1, 'I2': X_test_I2})
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

        # Large_MRE_X = [] ## Type of X_test??
        # Large_MRE_y_test = []
        # Large_MRE_y_pred = []
        # Large_MRE = []

        # X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
        # X_test = min_max_scaler_X.inverse_transform(X_test)
        # for idx in range(len(y_test)):
        #     if mean_relative_error([result[idx]], [y_test[idx]]) > 5:
        #         Large_MRE_X.append(X_test[idx])
        #         Large_MRE_y_test.append(y_test[idx])
        #         Large_MRE_y_pred.append(result[idx])
        #         Large_MRE.append(mean_relative_error([result[idx]], [y_test[idx]]))

        # for idx in range(len(y_test)):
        #     Large_MRE.append(mean_relative_error([result[idx]], [y_test[idx]])[0])
        # Large_MRE_y_test = list(y_test)
        # Large_MRE_y_pred = list(result)

        # temp = pd.DataFrame(X_test)
        # temp = pd.concat([temp, pd.DataFrame({'Real Value': Large_MRE_y_test}), pd.DataFrame({'Predicted Value': Large_MRE_y_pred}),
        #                   pd.DataFrame({'MRE': Large_MRE})], axis=1)
        # temp = temp.sort_values(by='MRE', ascending=False)
        # temp.to_csv('Out/Large_MRE_points' + str(i) + '.csv', encoding='gb18030', index=False)

        in_y_test.append(y_test)
        in_y_pred.append(result)

        #
        result = model_mlp.predict({'I1': X_train_I1, 'I2': X_train_I2})

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
clist = ['white', 'orange', 'black']
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
cross_result.to_csv('Out/cross_result_multi.csv', index=False, encoding='gb18030')

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
plt.savefig('pics/descriptor-fig-mulinput.png')
plt.show()

cross_result = {'Real lambda': in_y_train_real, 'Predicted lambda': in_y_train_pred}
cross_result = pd.DataFrame(cross_result)
cross_result.to_csv('Out/cross_result_multi_train.csv', index=False, encoding='gb18030')