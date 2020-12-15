# coding=utf-8
''''
对BP神经网络自动进行超参数搜索
'''
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import regularizers
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 1、读取并查看数据
data = pd.read_excel(r'C:\Users\dell-hy\Desktop\相关性.xlsx')
data.dropna(inplace=True)

print(data.shape)
# 2、数据划分
X, y = data.iloc[:, 1:13].values, data.iloc[:, 15].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
print(X.shape, y.shape, X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# 3、数据归一化，只能使用X_train进行fit()

# 正态分布归一化
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# 最大最小归一化，feature_range可以调整范围
# mc=MinMaxScaler()
# mc.fit(X_train)
# X_train_std=mc.transform(X_train)
# X_test_std=mc.transform(X_test)

# 对y做归一化处理
mc = StandardScaler()
mc.fit(y_train.reshape(-1, 1))
y_train_std = mc.transform(y_train.reshape(-1, 1))
y_test_std = mc.transform(y_test.reshape(-1, 1))

# 4、建立模型
''''
from keras.layers import Input, Dense, Dropout
from keras.models import Model

input_dim = X_train_std.shape[1]
hidden_1 = 40
hidden_2 = 40
hidden_3 = 40
hidden_4 = 40
hidden_5 = 40
output_dim = 1
batch_size = 1
epochs = 2000
learning_rate = 0.01
dropout_rate = 0.5
a = 0.001

x = Input(shape=[input_dim, ])
h = Dense(hidden_1, activation='relu', kernel_regularizer=regularizers.l2(a))(x)
h = Dropout(dropout_rate)(h)
h = Dense(hidden_2, activation='relu', kernel_regularizer=regularizers.l2(a))(h)
h = Dense(hidden_3, activation='relu', kernel_regularizer=regularizers.l2(a))(h)
h = Dense(hidden_4, activation='relu', kernel_regularizer=regularizers.l2(a))(h)
h = Dense(hidden_5, activation='relu', kernel_regularizer=regularizers.l2(a))(h)
Y = Dense(output_dim)(h)

optimizer = tf.train.AdamOptimizer(learning_rate)
model = Model(x, Y)
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])
'''
from keras.models import Sequential
from keras.layers import Dense, Dropout


def Neural_Network(learn_rate=0.01, dropout_rate=0.0, neurons=40, l2_rate=0.05):
    model = Sequential()
    model.add(Dense(neurons, activation='relu', kernel_regularizer=regularizers.l2(l2_rate),
                    input_dim=X_train_std.shape[1]))
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons, activation='relu', kernel_regularizer=regularizers.l2(l2_rate)))
    model.add(Dense(neurons, activation='relu', kernel_regularizer=regularizers.l2(l2_rate)))
    model.add(Dense(neurons, activation='relu', kernel_regularizer=regularizers.l2(l2_rate)))
    model.add(Dense(neurons, activation='relu', kernel_regularizer=regularizers.l2(l2_rate)))
    model.add(Dense(1))
    optimizer = tf.train.AdamOptimizer(learn_rate)
    # optimizer = SGD(lr=learning_rate, decay=learning_rate / nb_epoch, momentum=0.9, nesterov=True)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
    return model


model1 = KerasRegressor(build_fn=Neural_Network, verbose=0, batch_size=X_train.shape[0], epochs=500)

learn_rate = [0.001, 0.003, 0.006, 0.01, 0.03, 0.06]
# dropout_rate = [0.0, 0.2, 0.4, 0.6, 0.8]
neurons = [40, 50, 60, 70, 80, 90, 100]
# batch_size = [1, 2, 3, 4, 5]
# epochs = [1000, 1500]
param_grid = dict(neurons=neurons, learn_rate=learn_rate)
''''
param_grid = {
    'learn_rate': range(0.01, 0.1, 0.01),
    'neruons': range(40, 100, 10)
}
'''
grid = GridSearchCV(estimator=model1, param_grid=param_grid, scoring='neg_mean_squared_error', n_jobs=1, cv=5)
grid_result = grid.fit(X_train_std, y_train_std)
print("Best: %f using %s" % (abs(grid_result.best_score_), grid_result.best_params_))

score_means = abs(grid_result.cv_results_['mean_test_score'])
score_stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, std, param in zip(score_means, score_stds, params):
    print("score_mean: %f (score_stds: %f) with: %r" % (mean, std, param))

# 画出参数变化结果图
scores = np.array(score_means).reshape(len(learn_rate), len(neurons))
print(scores)
for i, value in enumerate(neurons):
    plt.plot(learn_rate, scores[:, i], label='neurons: ' + str(value))
plt.legend()
plt.grid(linestyle='-.')
plt.xlabel('learn_rate')
plt.ylabel('neg_mean_squared_error')
plt.show()
