# coding: utf-8

# In[1]:


# EM算法训练LDS参数
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 只输出 FATAL（致命的）

''''
# 按照月份统计出注入量与生产量，并导出excel表
data1 = pd.read_excel(r'C:/Users\dell-hy\Desktop\OFM数据库\X10-2.xlsx')
grouped = data1.groupby('date')
tongji1 = grouped['monthwinj'].sum()
tongji2 = grouped['monthliquid'].sum()
tongji3 = grouped['monthoil'].sum()
print(tongji2)
df = pd.DataFrame({'monthwinj': tongji1, 'monthliquid': tongji2, 'monthoil': tongji3})
df.to_excel('C:/Users/dell-hy/Desktop/统计.xlsx')
'''


data = pd.read_excel(r'C:\Users\dell-hy\Desktop\扶余lds\扶余统计.xlsx')
#data = pd.read_csv(open(r'C:/Users\dell-hy\Desktop\LDS-2020.07\拟合结果\理想模型\4五点法面积\面积井网4.csv'))

# 删除数据为零的行
data = data[data.monthwinj != 0]
#data = data[data.monthoil != 0]
print(data.shape)
# 只保留注入量大于产出量的日期
data = data[data.bizhi < 1]
#data = data[data.chazhi > 1000]
print(data.shape)

u = data.iloc[:, 1].values.tolist()
y = data.iloc[:, 2].values.tolist()  # 导入输入与输出


# In[2]:


# 初始化参数 数字“1”均应为随机数,randn函数返回一个或一组样本，具有标准正态分布
A = np.random.randn()
B = np.random.randn()
C = np.random.randn()
D = np.random.randn()
E = np.random.randn()
T = np.random.randn()
u.insert(0, np.random.randn())
y.insert(0, None)  # i,y起点在1
yb = []
V = [np.random.randn()]
x = [u[0]+np.random.randn()]
M = 50  # 总遍历最大次数
Q = len(y)  # 总数据量
N = int(len(y) * 0.75)  # 训练集数据


# In[3]:


# E步-第一次循环
# 前向阶段(B-1ato1e,B-3)
x_m = ["null"]  # _为-，m为^，b为~
V_m = ["null"]
K = ["null"]
xm = [x[0]]
Vm = [x[0]]
for n in range(1, N):

    x_m.append(A*xm[n-1]+B*u[n-1])
    V_m.append(A*Vm[n-1]*A+E)
    K.append(V_m[n]*C*(C*V_m[n]*C+T)**-1)
    xm.append(x_m[n]+K[n]*(y[n]-C*x_m[n]-D*u[n]))
    Vm.append((1-K[n]*C)*V_m[n])
x_m.append(A*x_m[N-1]+B*u[N-1])
V_m.append(A*V_m[N-1]*A+E)


# In[4]:


# 后向阶段(B-2ato2c)
J = [0] * N
xb = [0] * N
Vb = [0] * N
xb.append(x_m[N])
Vb.append(V_m[N])
J[0] = "null"
xb[0] = "null"
Vb[0] = "null"
for n in range(N-1, 0, -1):
    J[n] = Vm[n]*A*V_m[n+1]**-1
    xb[n] = xm[n]+J[n]*(xb[n+1]-x_m[n+1])
    Vb[n] = Vm[n]+J[n]*(Vb[n+1]-V_m[n+1])*J[n]


# In[5]:


# E{xn]
def Ex(n, xb):
    return xb[n]


# E{xnxn-1]
def Exx(n, J, Vb, xb):
    return J[n-1]*Vb[n]+xb[n]*xb[n-1]


# E{xnxn]
def Ex2(n, Vb, xb):
    return Vb[n]+xb[n]*xb[n]


# EE{xn]
def EEx(xb, i, j):
    sum = 0
    for n in range(i, j):
        sum = sum + Ex(n, xb)
    return sum


# EE{xnxn-1]
def EExx(J, Vb, xb, i, j):
    sum = 0
    for n in range(i, j):
        sum = sum + Exx(n, J, Vb, xb)
    return sum


# EE{xnxn]
def EEx2(Vb, xb, i, j):
    sum = 0
    for n in range(i, j):
        sum = sum + Ex2(n, Vb, xb)
    return sum


# Sum
def Sum(t, i, j, w, xb=False, Vb=False, J=False):
    sum = 0
    if t == 0:  # u/y的求和
        for n in range(i, j):
            sum = sum + w[n]
    elif t == 1:  # u/y平方的求和
        for n in range(i, j):
            sum = sum + w[n]*w[n]
    elif t == 2:  # u与y乘积的求和
        for n in range(i, j):
            sum = sum + w[n]*xb[n]
    elif t == 3:  # u/y与E{xn]乘积的求和
        for n in range(i, j):
            sum = sum + w[n]*Ex(n, xb)
    elif t == 4:  # u/y与E{xnxn-1]乘积的求和
        for n in range(i, j):
            sum = sum + w[n]*Exx(n, J, Vb, xb)
    elif t == 5:  # u/y与E{xnxn]乘积的求和
        for n in range(i, j):
            sum = sum + w[n]*Ex2(n, Vb, xb)
    elif t == 6:  # u/y与E{xn+1]乘积的求和
        for n in range(i, j):
            sum = sum + w[n]*Ex(n+1, xb)
    return sum


# 判断是否收敛
def ifsl(new, old):
    if abs(new-old) <= 0.0001:
        return True
    else:
        return False


# In[6]:


A_old = A
B_old = B
C_old = C
D_old = D
E_old = E
T_old = T
for m in range(1, M):
    # M步
    # 求u0,V0
    u[0] = Ex(1, xb)
    V[0] = Ex2(1, Vb, xb)-Ex(1, xb)*Ex(1, xb)
    # 求A,B
    X = np.matrix([EExx(J, Vb, xb, 2, N), Sum(3, 1, N-1, u, xb)])
    Y = np.matrix([[EEx2(Vb, xb, 1, N-1), Sum(3, 1, N-1, u, xb)],
                   [Sum(3, 1, N-1, u, xb), Sum(1, 1, N-1, u)]])
    Y = Y.astype(np.float)
    Z = X*np.linalg.inv(Y)
    A_old = A
    B_old = B
    A = Z[0, 0]
    B = Z[0, 1]
    # 求C,D
    X = np.matrix([Sum(3, 1, N, y, xb), Sum(2, 1, N, y, u)])
    Y = np.matrix([[EEx2(Vb, xb, 1, N), Sum(3, 1, N, u, xb)],
                   [Sum(3, 1, N, u, xb), Sum(1, 1, N, u)]])
    Y = Y.astype(np.float)
    Z = X*np.linalg.inv(Y)
    C_old = C
    if C == 0:
        break
    D_old = D
    C = Z[0, 0]
    D = Z[0, 1]
    # 求E,T
    E_old = E
    T_old = T
    E = 1/(N-1)*(EEx2(Vb, xb, 2, N)+A*A*EEx2(Vb, xb, 1, N-1)+B*B*Sum(1, 1, N-1, u) -
                 2*A*EExx(J, Vb, xb, 2, N)-2*B*Sum(6, 1, N-1, u, xb)+2*A*B*Sum(3, 1, N-1, u, xb))
    T = 1/(N)*(Sum(1, 1, N, y)+C*C*EEx2(Vb, xb, 1, N)+D*D*Sum(1, 1, N, u) +
               2*C*D*Sum(3, 1, N, u, xb)-2*C*Sum(3, 1, N, y, xb)-2*D*Sum(2, 1, N, y, u))
    print(m, A, B, C, D, E, T)
    # E步
    # 前向阶段(B-1ato1e,B-3)
    for n in range(1, N):
        x_m[n] = A*xm[n-1]+B*u[n-1]
        V_m[n] = A*Vm[n-1]*A+E
        K[n] = V_m[n]*C*(C*V_m[n]*C+T)**-1
        xm[n] = x_m[n]+K[n]*(y[n]-C*x_m[n]-D*u[n])
        Vm[n] = (1-K[n]*C)*V_m[n]
    x_m[N] = A*x_m[N-1]+B*u[N-1]
    V_m[N] = A*V_m[N-1]*A+E
    # 后向阶段(B-2ato2c)
    xb[N] = x_m[N]
    Vb[N] = V_m[N]
    for n in range(N-1, 0, -1):
        J[n] = Vm[n]*A*V_m[n+1]**-1
        xb[n] = xm[n]+J[n]*(xb[n+1]-x_m[n+1])
        Vb[n] = Vm[n]+J[n]*(Vb[n+1]-V_m[n+1])*J[n]
    if ifsl(A, A_old) and ifsl(B, B_old) and ifsl(C, C_old) and ifsl(D, D_old): # and ifsl(E, E_old) and ifsl(T, T_old):
        break


# In[7]:
# 卡尔曼滤波加平滑，即训练集预测值
# 前向阶段(B-1ato1e,B-3)
for n in range(1, N):
    x_m[n] = A*xm[n-1]+B*u[n-1]
    V_m[n] = A*Vm[n-1]*A+E
    K[n] = V_m[n]*C*(C*V_m[n]*C+T)**-1
    xm[n] = x_m[n]+K[n]*(y[n]-C*x_m[n]-D*u[n])
    Vm[n] = (1-K[n]*C)*V_m[n]
x_m[N] = A*x_m[N-1]+B*u[N-1]
V_m[N] = A*V_m[N-1]*A+E
# 后向阶段(B-2ato2c)
xb[N] = x_m[N]
Vb[N] = V_m[N]
for n in range(N-1, 0, -1):
    J[n] = Vm[n]*A*V_m[n+1]**-1
    xb[n] = xm[n]+J[n]*(xb[n+1]-x_m[n+1])
    Vb[n] = Vm[n]+J[n]*(Vb[n+1]-V_m[n+1])*J[n]
    yb.append(C * xb[n] + D * u[n])
yb.append(y[0])
yb.reverse()

# xb，Vb导出csv
# dataframe = pd.DataFrame({'xb': xb, 'Vb': Vb})
# dataframe.to_csv("test.csv", index=False, sep=',')


# In[8]:


# LDS预测产量，测试集预测值
for n in range(N, Q):
    xb.append(A*xb[n-1]+B*u[n-1])
    yb.append(C*xb[n]+D*u[n])

# 计算训练集的平均绝对百分比误差

error = []
for i in range(1, N):
    error.append(abs(yb[i] - y[i]) / y[i])

print('训练集结果的平均绝对百分比误差为：%.5f' % (np.mean(error) * 100))

# 计算测试集的平均绝对百分比误差

error1 = []
num1 = 0
num2 = 0
num3 = 0

for i in range(N, Q):
    error1.append(abs(yb[i] - y[i]) / y[i])

for i in range(len(error1)):
    if error1[i] < 0.1:  # 相对误差小于0.1
        num1 = num1 + 1
    elif error1[i] < 0.2:  # 相对误差大于0.1小于0.2
        num2 = num2 + 1
    else:  # 相对误差大于0.2
        num3 = num3 + 1
print(error1)
print('测试集结果的平均绝对百分比误差为：%.5f' % (np.mean(error1) * 100))

# In[9]:

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
#画出注入与产出的关系图
plt.figure()
u[0] = None
plt.plot(u, c='green', label='月注入量')
plt.plot(y, c='red', label='月产油量')
plt.xlabel('时间')
plt.ylabel('注入量或产油量')
plt.legend(loc='best')

# 画出注入量与产出量之间的散点图
plt.figure()
plt.scatter(u, y, c='r', s=12)
plt.xlabel('月注入量') #
plt.ylabel('月产油量')

# 画出相对误差饼状图
labels = ['相对误差 < 10%', '10% < 相对误差 < 20%', '相对误差 > 20%']
values = [num1, num2, num3]
explode=[0.01, 0.01, 0.01] #  设定各项距离圆心n个半径
# 同时显示数值和占比的饼图
def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
    return my_autopct

plt.figure(figsize=(6, 6))
plt.pie(values, explode=explode, labels=labels, autopct=make_autopct(values), shadow=True)
#plt.legend(loc='best')

# 画出产出/注入比值散点 分布图
plt.figure()
bizhi = data.iloc[:, 3].values.tolist()
plt.scatter(np.arange(len(bizhi)), bizhi, c='green', s=12)
plt.xlabel('时间')
plt.ylabel('产油量与注入量的比值')

# 绘出预测结果图
plt.figure()

plt.plot(y, label='origin')
plt.plot(yb, label='predict')
plt.xlabel('时间')
plt.ylabel('产油量')
plt.legend()
yb[0] = -0
plt.vlines(N, 0, max(yb), linestyle="--")
# plt.ylim(1980000, 2120000)

# 导出预测值与真实值
df1 = pd.DataFrame({'实际值': y, '预测值': yb})
df1.to_excel('C:/Users/dell-hy/Desktop/预测结果1.xlsx')
plt.show()
