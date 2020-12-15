# coding=utf-8
"""'
对LSTM进行超参数寻优
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
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor

# 1、将时间序列数据转化为监督问题数据

df = pd.read_excel(r'C:\Users\dell-hy\Desktop\数据\youcang.xlsx')
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

print(data)

# 划分数据集

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

# 建立模型
from keras.layers import Input, Dense, LSTM
from keras.models import Model

def Neural_Network(epochs = 50, neurons = 5, batch_size = 1, learn_rate=0.01, output_dim = 1):

    X = Input(shape=[train_X.shape[1], train_X.shape[2], ])
    h = LSTM(neurons, activation='relu')(X)
    Y = Dense(output_dim)(h)
    model = Model(X, Y)

    optimizer = tf.train.AdamOptimizer(learn_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mse'])
    return model

model = KerasRegressor(build_fn=Neural_Network, verbose=0, batch_size=train_X.shape[0], epochs=100)

learn_rate = [0.001, 0.003, 0.006, 0.01, 0.03, 0.06]
# dropout_rate = [0.0, 0.2, 0.4, 0.6, 0.8]
neurons = [5, 6, 7, 8, 9]
# batch_size = [1, 2, 3, 4, 5]
# epochs = [1000, 1500]
param_grid = dict(neurons=neurons, learn_rate=learn_rate)


grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', n_jobs=1, cv=2)
grid_result = grid.fit(valid_X, valid_y)

print("Best: %f using %s" % (abs(grid_result.best_score_), grid_result.best_params_))

score_means = abs(grid_result.cv_results_['mean_test_score'])
score_stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, std, param in zip(score_means, score_stds, params):
    print("score_mean: %f (score_stds: %f) with: %r" % (mean, std, param))

# 画出参数变化结果图
scores = np.array(score_means).reshape(len(learn_rate), len(neurons))
print(scores)

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

for i, value in enumerate(neurons):
    plt.plot(learn_rate, scores[:, i], label='neurons: ' + str(value))

plt.legend()
plt.grid()
plt.xlabel('learn_rate')
plt.ylabel('neg_mean_squared_error')

plt.show()
