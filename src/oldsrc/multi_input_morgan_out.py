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
test_filepath = "data/fp/sjn/01-15-morgan-test.csv"

[data_x_df, data_y_df] = read_bit(filepath)

# 归一化
morgan_min_max_scaler_X = MinMaxScaler()
morgan_min_max_scaler_X.fit(data_x_df)
x_trans1 = morgan_min_max_scaler_X.transform(data_x_df)
morgan_x_trans1 = np.reshape(x_trans1, (x_trans1.shape[0], x_trans1.shape[1], 1)) ##

morgan_min_max_scaler_y = MinMaxScaler()
morgan_min_max_scaler_y.fit(data_y_df)
y_trans1 = morgan_min_max_scaler_y.transform(data_y_df)
morgan_y_trans1 = np.reshape(y_trans1, (y_trans1.shape[0], 1, 1)) ##

[test_data_x_df, test_data_y_df] = read_bit(test_filepath)
x_trans1_test = morgan_min_max_scaler_X.transform(test_data_x_df)
y_trans1_test = morgan_min_max_scaler_y.transform(test_data_y_df)
morgan_x_trans1_test = np.reshape(x_trans1_test, (x_trans1_test.shape[0], x_trans1_test.shape[1], 1))
morgan_y_trans1_test = np.reshape(y_trans1_test, (y_trans1_test.shape[0], 1, 1))

print(data_x_df.shape, data_y_df.shape)
print(test_data_x_df.shape, test_data_y_df.shape)
sleep(5)

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
x_trans1 = np.reshape(x_trans1, (x_trans1.shape[0], x_trans1.shape[1], 1))

min_max_scaler_y = MinMaxScaler()
min_max_scaler_y.fit(data_y_df)
y_trans1 = min_max_scaler_y.transform(data_y_df)
y_trans1 = np.reshape(y_trans1, (y_trans1.shape[0], 1, 1))

test_filepath = "data/descriptor/01-15-descriptor-test.csv"
test_data = pd.read_csv(test_filepath, encoding='gbk')
print('test data: ', test_data.shape) # 80，393
test_data = test_data.dropna()
test_data_x_df = pd.DataFrame(test_data.iloc[:, :-1])
test_data_y_df = pd.DataFrame(test_data.iloc[:, -1])
x_trans1_test = min_max_scaler_X.transform(test_data_x_df)
y_trans1_test = min_max_scaler_y.transform(test_data_y_df)

x_trans1_test = np.reshape(x_trans1_test, (x_trans1_test.shape[0], x_trans1_test.shape[1], 1))
y_trans1_test = np.reshape(y_trans1_test, (y_trans1_test.shape[0], 1, 1))

# 将输入的X特征合并为一个列表
# x_trans1 = np.concatenate((x_trans1, morgan_x_trans1), axis=1)
# x_trans1_test = np.concatenate((x_trans1_test, morgan_x_trans1_test), axis=1)

# morgan_x_trans1, morgan_y_trans1
# x_trans1, y_trans1
temp = pd.DataFrame(morgan_x_trans1)
temp = pd.concat([temp, x_trans1], axis=1)
temp = pd.concat([temp, y_trans1], axis=1)
temp = shuffle(temp)
morgan_x_trans1 = temp.iloc[:, 0:2*NUM_ATTR]
x_trans1 = temp.iloc[:, 2*NUM_ATTR:-1]
morgan_y_trans1 = y_trans1 = temp.iloc[:, -1]

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

X_train_I1 = x_trans1
X_train_I2 = morgan_x_trans1
y_train = y_trans1

model_mlp = buildModel()
model_mlp.fit({'I1': X_train_I1, 'I2': X_train_I2}, y_train, epochs=120, verbose=1)

print(model_mlp.summary())
sleep(5)

# 外部验证
X_test = x_trans1_test
print(x_trans1_test.shape, morgan_x_trans1_test.shape)
result = model_mlp.predict({'I1': x_trans1_test, 'I2': morgan_x_trans1_test})

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

for idx in range(len(y_test)):
    Large_MRE.append(mean_relative_error([result[idx]], [y_test[idx]])[0])
Large_MRE_y_test = list(y_test)
Large_MRE_y_pred = list(result)

temp = pd.DataFrame()
temp = pd.concat([temp, pd.DataFrame({'Real Value': Large_MRE_y_test}), pd.DataFrame({'Predicted Value': Large_MRE_y_pred}),
                      pd.DataFrame({'MRE': Large_MRE})], axis=1)
# temp = temp.sort_values(by='MRE', ascending=False)
temp.to_csv('Out/Large_MRE_out_points.csv', encoding='gb18030', index=False)

out_y_test.append(y_test)
out_y_pred.append(result)

## 白+绿纯色颜色映射
from pylab import *
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
clist = ['white', 'orange', 'black']
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
plt.savefig('pics/multi-fig-out-mulinput.png')
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
