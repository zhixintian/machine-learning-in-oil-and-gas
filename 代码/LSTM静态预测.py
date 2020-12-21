# coding=utf-8
"""'
利用keras的LSTM预测油井产量,实现时间序列静态预测
"""
import os
import random as rn

import numpy as np
import tensorflow as tf

# 防止数字随机化，使得实验可以复现
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)

# 强制TensorFlow使用单线程，多线程是结果不可复现的一个潜在因素。
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

# `tf.set_random_seed()`将会以 TensorFlow 为后端，在一个明确的初始状态下生成固定随机数字。
from keras import backend as K

tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

###################################################################################
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# 1、将时间序列数据转化为监督问题数据
from pandas import DataFrame
from pandas import concat
from pandas import Series

# 将数据变为监督数据
def series_to_supervised(data, look_back=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]  # 若为列表则为1，若为二维数组则为它的列数

    df = DataFrame(data)
    cols, names = list(), list()

    # input sequence (t-n, ... t-1)
    for i in range(look_back, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]

    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]

    # put it all together
    agg = concat(cols, axis=1)  # 按列连接起来
    agg.columns = names  # 添加列名

    # drop rows with NaN values（去除含有NaN的行）
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# 2、导入，处理数据集

# 创建差分序列
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]  # 得到相邻两个数值的差值，后面的减去前面的
        diff.append(value)
    return Series(diff)


# 反转差值,得到原来的值
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

# 归一化处理数据
def cs_to_sl(dropnan=True):
    df = pd.read_excel(r'C:\Users\dell-hy\Desktop\数据\youcang.xlsx')
    if dropnan:
        df.dropna(inplace=True)

    values = df.iloc[:, 1].values
    values1 = df.iloc[:, 2].values
    values2 = df.iloc[:, 3].values

    mc = MinMaxScaler(feature_range=(-1, 1))
    mc1 = MinMaxScaler(feature_range=(-1, 1))
    mc2 = MinMaxScaler(feature_range=(-1, 1))

    df_std = mc.fit_transform(values.reshape(-1, 1))
    df_std1 = mc1.fit_transform(values1.reshape(-1, 1))
    df_std2 = mc2.fit_transform(values2.reshape(-1, 1))
    df_std3 = np.concatenate((df_std1, df_std2), axis=1)

    data = np.concatenate((df_std3, df_std), axis=1)
    # data = series_to_supervised(df_std, 1, 1)
    # data.drop(data.columns[3], axis=1, inplace=True) # 删除不需要预测的列
    print(data)
    print('有监督数据集大小:', data.shape, '目标数据集大小:', values.shape)
    return mc, data, values


# 3、划分数据集
def train_test(data):

    letrain = int(data.shape[0] * 0.6) #  训练集长度
    levalid = int(data.shape[0] * 0.8) #  验证集长度
    letest = int(data.shape[0]) #  测试集长度

    data_train = data[:letrain, :]
    data_valid = data[letrain:levalid, :]
    data_test = data[levalid:, :]

    train_X, train_y = data_train[:, :-1], data_train[:, -1]
    valid_X, valid_y = data_valid[:, :-1], data_valid[:, -1]
    test_X, test_y = data_test[:, :-1], data_test[:, -1]

    # [samples, timesteps, features]，改变数据的维数
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    valid_X = valid_X.reshape((valid_X.shape[0], 1, valid_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    print('训练集大小:', train_X.shape, train_y.shape)
    print('验证集大小:', valid_X.shape, valid_y.shape)
    print('测试集大小:', test_X.shape, test_y.shape)
    # print(data)
    return train_X, train_y, test_X, test_y, letrain, levalid, letest, valid_X, valid_y


# 4、建立模型
from keras.layers import Input, Dense, LSTM, Dropout
from keras.models import Model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def fit_network(train_X, train_y, test_X, test_y, mc, values, letrain, levalid, letest, valid_X, valid_y):
    output_dim = 1
    batch_size = 1
    epochs = 50
    hidden_size = 5

    X = Input(shape=[train_X.shape[1], train_X.shape[2], ])
    h = LSTM(hidden_size, activation='relu')(X)
    # h = LSTM(hidden_size, activation='relu')(h)
    # h = Dropout(0.3)(h)
    Y = Dense(output_dim)(h)
    model = Model(X, Y)
    model.compile(loss='mean_squared_error', optimizer='adam')
    # fit network
    history = model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, validation_data=(valid_X, valid_y),
                        verbose=0, shuffle=False)
    # 画出误差图
    plt.plot(history.history['loss'], label='训练集误差')
    plt.plot(history.history['val_loss'], label='验证集误差')
    plt.xlabel('迭代次数')
    plt.ylabel('误差')
    plt.legend()
    # 得到预测值
    y_test_pred = model.predict(test_X)
    y_valid_pred = model.predict(valid_X)
    y_train_pred = model.predict(train_X)

    # 反归一化
    inv_y_pred3 = mc.inverse_transform(y_test_pred)
    inv_y_pred2 = mc.inverse_transform(y_valid_pred)
    inv_y_pred1 = mc.inverse_transform(y_train_pred)

    test_y = test_y.reshape((len(test_y), 1))

    inv_y3 = mc.inverse_transform(test_y)
    real_values = values[0:] #  所有的真实值，由于时间步为1，所以从第二个数开始
    inv_values = np.concatenate((inv_y_pred1, inv_y_pred2), axis=0) #  训练集加上验证集的预测值
    inv_values1 = np.concatenate((inv_values, inv_y_pred3), axis=0) #  所有的预测值

    # 计算测试集的均方误差与R2
    rmse = np.sqrt(mean_squared_error(inv_y3, inv_y_pred3))
    r2 = r2_score(inv_y3, inv_y_pred3)

    # 计算相对误差
    error = []

    for i in range(0, len(inv_y3)):
        error.append(abs(inv_y3[i] - inv_y_pred3[i]) / inv_y3[i])
    error = np.reshape(error, (1, len(inv_y3)))
    mean_error = np.mean(error)

    # 计算预测累计产量
    for_cumulative = []
    for i in range(len(inv_values1)):
        for_cul = np.sum(inv_values1[0:i])
        for_cumulative.append(for_cul)

    # 计算实际累计产量
    act_cumulative = []
    for i in range(len(real_values)):
        act_cul = np.sum(real_values[0:i])
        act_cumulative.append(act_cul)

    # 结果可视化
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.figure()

    plt.scatter(range(len(values)), real_values, c='r', s=5, label='真实值')
    plt.plot(range(0, letrain), inv_y_pred1, c='blue', label='训练集')
    plt.plot(range(letrain, levalid), inv_y_pred2, c='black', label='验证集')
    plt.plot(range(levalid, letest), inv_y_pred3, c='green', label='测试集')

    plt.axvline(x=letrain, color='k', linestyle='--')  # 在训练集与验证集之间添加一个线
    plt.axvline(x=levalid, color='k', linestyle='--')  # 在验证集与测试集之间添加一个线

    plt.xlabel('时间')
    plt.ylabel('日产气' + r'$(10^4 m^3)$')
    plt.legend()

    print('测试集RMSE: %.3f' % rmse)
    print('测试集r2: %.3f' % r2)
    # print('实际值:', real_values)
    # print('预测值:', inv_values1)
    # print('累积产量预测值:', for_cumulative[0])
    # print('累积产量实际值:', act_cumulative[0])
    print('测试集相对误差:',error)
    print('测试集最后一侧迭代的平均相对误差:', mean_error)

    plt.show()

if __name__ == '__main__':
    mc, data, values = cs_to_sl()
    train_X, train_y, test_X, test_y, letrain, levalid, letest, valid_X, valid_y = train_test(data)
    fit_network(train_X, train_y, test_X, test_y, mc, values, letrain, levalid, letest, valid_X, valid_y)
