# MLP
import csv
from itertools import islice
import random
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import multi_gpu_model
from keras import regularizers
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import pandas as pd


def bit2attr(bitstr) -> list:
    attr_vec = list()
    for i in range(len(bitstr)):
        attr_vec.append(int(bitstr[i]))
    return attr_vec


'''
1) 数据预处理
'''
NUM_ATTR = 1024
filepath = 'data/fp/sjn/R+B+Cmorgan_fp1202.csv'
data = list()
# data_y = pd.DataFrame(columns=['y'])
with open(filepath, 'r') as f:
    reader = csv.reader(f)
    num_attr = int()
    for row in islice(reader, 0, None):  # 不跳过第一行 # for row in islice(reader, 1, None):  # 跳过第一行
        if len(row) == 0:
            continue
        num_attr = len(row[0])
        assert num_attr == NUM_ATTR
        # data_x.append(bit2attr(row[0]), ignore_index=True)
        # data_y.append([int(row[1])], ignore_index=True)
        temp = bit2attr(row[0])
        temp.append(int(row[1]))
        print(temp)
        data.append(temp)

print(data)

random.shuffle(data)

print(data)

data = np.array(data)
data_x = data[:, 0:NUM_ATTR]
data_y = data[:, NUM_ATTR]

'''
2) 加载数据
'''
test_rate = 0.2
test_n = round(test_rate * len(data_x))
train_n = len(data_x) - test_n

x_train = data_x[:train_n][:]
y_train = data_y[:train_n][:]
x_valid = data_x[train_n:][:]
y_valid = data_y[train_n:][:]

# 转成DataFrame格式方便数据处理
x_train_pd = pd.DataFrame(x_train)
y_train_pd = pd.DataFrame(np.array(y_train))
x_valid_pd = pd.DataFrame(np.array(x_valid))
y_valid_pd = pd.DataFrame(np.array(y_valid))
print(x_train_pd.shape)
print(x_train_pd.head(5))
print('-----------------------------------------------------------------')
print(y_train_pd.head(5))

'''
3) 数据归一化
'''
# 训练集归一化
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(x_train_pd)
x_train = min_max_scaler.transform(x_train_pd)

min_max_scaler.fit(y_train_pd)
y_train = min_max_scaler.transform(y_train_pd)

# 验证集归一化
min_max_scaler.fit(x_valid_pd)
x_valid = min_max_scaler.transform(x_valid_pd)

min_max_scaler.fit(y_valid_pd)
y_valid = min_max_scaler.transform(y_valid_pd)

'''
4) 训练模型
'''
# 单CPU or GPU版本，若有GPU则自动切换
model = Sequential()  # 初始化，很重要！
model.add(Dense(units=10,  # 输出大小
                activation='relu',  # 激励函数
                input_shape=(x_train_pd.shape[1],)  # 输入大小, 也就是列的大小
                )
          )

model.add(Dropout(0.2))  # 丢弃神经元链接概率

model.add(Dense(units=15,
                # kernel_regularizer=regularizers.l2(0.01),  # 施加在权重上的正则项
                # activity_regularizer=regularizers.l1(0.01),  # 施加在输出上的正则项
                activation='relu'  # 激励函数
                # bias_regularizer=keras.regularizers.l1_l2(0.01)  # 施加在偏置向量上的正则项
                )
          )

model.add(Dense(units=1,
                activation='linear'  # 线性激励函数 回归一般在输出层用这个激励函数
                )
          )

print(model.summary())  # 打印网络层次结构

model.compile(loss='mse',  # 损失均方误差
              optimizer='adam',  # 优化器
              )

n_split = 10
for i in range(10):
    history = model.fit(x_train, y_train,
                        epochs=50,  # 迭代次数
                        batch_size=50,  # 每次用来梯度下降的批处理数据大小
                        verbose=2,  # verbose：日志冗长度，int：冗长度，0：不输出训练过程，1：输出训练进度，2：输出每一个epoch
                        validation_data=(x_valid, y_valid)  # 验证集
                        )

'''
5) 训练过程可视化
'''
import matplotlib.pyplot as plt

# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

'''
6) 保存模型 & 模型可视化 & 加载模型
'''
from keras.utils import plot_model
from keras.models import load_model

# 保存模型
model.save('model_MLP.h5')  # 生成模型文件 'my_model.h5'

# 模型可视化 需要安装pydot pip install pydot
plot_model(model, to_file='model_MLP.png', show_shapes=True)

# 加载模型
model = load_model('model_MLP.h5')

'''
7) 模型的预测功能
'''
# 预测
y_new = model.predict(x_valid)
# 反归一化还原原始量纲
min_max_scaler.fit(y_valid_pd)
y_new = min_max_scaler.inverse_transform(y_new)
print(min_max_scaler.inverse_transform(y_valid) - y_new)
