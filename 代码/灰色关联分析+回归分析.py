# coding=utf-8
"""'
灰色关联分析自动选择参数组合并且计算在每一个参数组合的条件下预测数据集的预测EUR
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.metrics import r2_score

# 导入训练数据集并处理
data = pd.read_excel(r'C:\Users\dell-hy\Desktop\数据\灰色分析数据\灰色相关分析1.xlsx')
data.dropna(inplace=True)
print(data.shape)  # 删掉所有空值的行

eur = data.iloc[:, -1].values  # 得到360井均日产那列

data1 = data.iloc[:, 1:-1]  # 只保留输入参数列
# 以下二者均获得数据框的列名列表
# print(list(data1))
# print(data1.columns.values.tolist())
# print(data1.columns)

zuhe = []

# 设置每次选择的参数数量
num_canshu = 8

for i,j,k,l,m,n,o,p in combinations(list(data1), num_canshu):
    zuhe.append((i,j,k,l,m,n,o,p))
print('组合有：', zuhe)
print('组合数为：', len(zuhe))
print('\n')
print('------------------------------------------------------------------------------------------------')

# 得到w所在的数据列
x = data.iloc[:, 1:].T

# 1、数据均值化处理
x_mean = x.mean(axis=1)
for i in range(x.index.size):
    x.iloc[i, :] = x.iloc[i, :] / x_mean[i]

# 2、提取参考队列和比较队列
ck = x.iloc[-1, :]  # 参考队列
cp = x.iloc[0:-1, :]

# 比较队列与参考队列相减
t = pd.DataFrame()
for j in range(cp.index.size):
    temp = pd.Series(cp.iloc[j, :] - ck)
    t = t.append(temp, ignore_index=True)

# 求最大差和最小差
mmax = t.abs().max().max()
mmin = t.abs().min().min()
rho = 0.1  # 分辨系数的大小

# 3、求关联系数
ksi = ((mmin + rho * mmax) / (abs(t) + rho * mmax))
# print('关联系数：', ksi)

# 4、求关联度
r = ksi.sum(axis=1) / ksi.columns.size
r1 = r.to_frame()
r1 = r1.T
print('关联度：', r)

# 拟合精度
R2 = []
R2_ = []

# 将原始数据集中含气量的那一列同乘以-1
data['含气量'] = data['含气量'] * (-1)

# 将原始数据集中TOC的那一列同乘以-1
#data['TOC'] = data['TOC'] * (-1)
#data['孔隙度'] = data['孔隙度'] * (-1)

for i in range(len(zuhe)):
    df = data[list(zuhe[i])]  # 将所选参数的列组成新的数据列

    # 将数据列进行均值化处理
    df = df.T
    x_mean = abs(df.mean(axis=1))
    for j in range(df.index.size):
        df.iloc[j, :] = df.iloc[j, :] / x_mean[j]

    # print(df) # 得到均值化后的数据列， 行为参数名

    r1.columns = list(data1)

    # print(r1) #  将关联度转化成了数据列的形式，列为参数名
    r2 = r1[list(zuhe[i])]
    print(r2)
    print('\n')

    d = np.sum(r2.values)

    r3 = r2.values.tolist()

    w = []  # 指标权重的计算
    for ii in range(r2.columns.size - (num_canshu - 1)):
        w.append(r3[ii] / d)

    print('指标权重：', w)
    print('\n')

    pre2 = np.dot(w, df)
    # print(pre2)
    # print('无量纲产能指标：', pre2)

    # 计算无量纲产能指标与首年日产或者EUR之间的关系式,注：是利用所有数据进行回归的，没有划分数据集
    from sklearn.linear_model import LinearRegression

    lin = LinearRegression()
    model = lin.fit(pre2.T, eur)

    # 拟合函数的系数
    a = lin.coef_
    b = lin.intercept_

    print(a,b)

    pre_eur = lin.predict(pre2.T)

    print('训练数据集的拟合R2', r2_score(eur, pre_eur))

    R2.append(r2_score(eur, pre_eur))

    ####################################### 利用类比法进行预测####################################################

    # 选取该参数组合下的训练数据集的均值
    x_mean = x_mean[list(zuhe[i])]


    # print(x_mean.values)

    # 定义拟合函数
    def fun(x, a=a, b=b):
        return a * x + b

    # 导入预测数据
    data2 = pd.read_excel(r'C:\Users\dell-hy\Desktop\数据\灰色分析数据\类比法数据.xlsx')
    data2 = data2[list(zuhe[i])]
    data2.dropna(inplace=True)

    print('包含该参数组合的预测数据集中井的数量为：', data2.shape[0])
    print('\n')

    # 将预测数据集中含气量的那一列同乘以-1

    if '含气量' in zuhe[i]:
        data2['含气量'] = data2['含气量'] * (-1)

    #if 'TOC' in zuhe[i]:
        #data2['TOC'] = data2['TOC'] * (-1)

    #if '孔隙度' in zuhe[i]:
        #data2['孔隙度'] = data2['孔隙度'] * (-1)

    # 对预测数据均值化处理

    df2 = data2.iloc[:, :].T

    for j in range(df.index.size):
        df2.iloc[j, :] = df2.iloc[j, :] / x_mean[j]

    # 计算无量纲产能指标QI
    QI = np.dot(w, df2)

    # print('无量纲产能指标为：', QI)

    eur_pre = []
    for jj in range(len(QI)):
        eur_pre.append(fun(QI[jj]))

    print('eur的预测值为：', eur_pre)
    print('\n', '------------------------------------------------------------------------------------------------',
          '\n')

df = pd.DataFrame({'拟合精度': R2})
df.to_excel('C:/Users/dell-hy/Desktop/精度1.xlsx')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.figure()
plt.plot(np.arange(1, len(R2) + 1), R2)
plt.scatter(np.arange(1, len(R2) + 1), R2, c='r', s=10)
plt.axhline(y=np.max(R2), color='green', linestyle='--')  # 画一条水平虚线
plt.xlabel('n=%.0f时的参数组合' % num_canshu)  # n保留零位小数
plt.ylabel('R^2')

print('最大的R2为：', np.max(R2))
# 输出最大值对应的原来序列的索引值
R2 = np.array(R2)
zuihao = R2.argsort()[-1]
print('最好参数组合对应的原来组合数的索引值为：', zuihao)

n = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12', 'P13', 'P14', 'P15']
R2_ = [10524.07865733, 10491.85882282,  9455.06662517, 10251.04490432,
        8992.07538847,  9023.81240127,  8036.19109403,  5488.90148871,
        8717.8922875 ,  6105.11939758,  5995.51456669, 10691.22958064,
        9451.32538754, 10354.34584336,  8651.06696889]
plt.figure()
plt.plot(n, R2_)
plt.scatter(n, R2_, c='r', s=12)
plt.xlabel('井号') #
plt.ylabel('预测EUR')


print('训练集中最好的参数组合为：', zuhe[zuihao])

plt.show()
