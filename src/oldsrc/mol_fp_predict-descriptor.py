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
filepath = 'data/descriptor/12-21-1.csv'

data = pd.read_csv(filepath, encoding='gb18030')
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

min_max_scaler_y = MinMaxScaler()
min_max_scaler_y.fit(data_y_df)
y_trans1 = min_max_scaler_y.transform(data_y_df)

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

    ## Initial: 0.2
    X_train, X_test, y_train, y_test = train_test_split(x_trans1, y_trans1, test_size=0.2)
    # train_test_split 随机划分 random_state, 填0或不填，每次都会不一样

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    ## Initial: 400 200 100
    model_mlp = MLPRegressor(
        hidden_layer_sizes=(512, 128, 32), activation='relu', solver='lbfgs', alpha=0.0001,
        max_iter=5000,
        random_state=1, tol=0.0001, verbose=False, warm_start=False)
    model_mlp.fit(X_train, y_train)

    # x1 = x_trans1.reshape(-1, NUM_ATTR)
    x1 = x_trans1
    y = y_trans1.reshape(-1, 1)
    mlp_score = model_mlp.score(x1, y)
    print('sklearn多层感知器-回归模型得分', mlp_score)  # 预测正确/总数
    mlp_scores.append(mlp_score)

    result = model_mlp.predict(X_test)
    # plt.figure(figsize=(10, 6))
    # plt.xlabel('ground truth')
    # plt.ylabel('predicted')

    y_test = min_max_scaler_y.inverse_transform(y_test)
    result = result.reshape(-1, 1)
    result = min_max_scaler_y.inverse_transform(result)

    mae = mean_relative_error(y_test, result)
    MAEs.append(mae)
    # errstr = 'MAE = %.3f' % mae
    # plt.text(420, 750, errstr, fontsize=16)
    plt.plot(y_test, result, 'ro')

    Large_MRE_X = [] ## Type of X_test??
    Large_MRE_y_test = []
    Large_MRE_y_pred = []
    Large_MRE = []

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

print('MRE', MAEs)
print('avg MRE', sum(MAEs) / len(MAEs))
print('max MRE', max(MAEs))
print('min MRE', min(MAEs))
errstr = 'MRE = %.2f%%' % (sum(MAEs) / len(MAEs))
plt.text(420, 750, errstr, fontsize=16)
print('mlp_score', mlp_scores)
plt.savefig('pics/descriptor-fig.png')
plt.show()
print('avg mlp_score', sum(mlp_scores) / len(mlp_scores))
